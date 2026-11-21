"""SLMGR constrained beam generation over the official serving ID mapping."""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from pathlib import Path
import shutil
import tempfile
from threading import Lock
from typing import Any, cast

from filelock import FileLock
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import T5Config, T5PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions

from examples.experimental.generative_retriever.model.model_loading import import_slmv3, require_directory
from examples.experimental.generative_retriever.utils.paths import absolute_path

PrefixConstraint = Callable[[int, torch.Tensor], list[int]]


class SortedU32Constraint:
    """Zero-copy prefix constraint over sorted packed uint32 docids."""

    def __init__(self, unique_u32: Any, code_dim: int = 256) -> None:
        self._u32 = np.array(unique_u32, dtype=np.uint32, copy=True)
        self._code_dim = code_dim
        self._root_tokens = list(range(2, code_dim + 2))
        p24 = self._u32 >> np.uint32(8)
        keep24 = np.empty(len(p24), dtype=bool)
        keep24[0] = True
        keep24[1:] = p24[1:] != p24[:-1]
        u_p24 = p24[keep24]
        starts = np.flatnonzero(keep24).astype(np.int64)
        ends = np.empty_like(starts)
        ends[:-1] = starts[1:]
        ends[-1] = len(self._u32)

        p16 = u_p24 >> np.uint32(8)
        keep16 = np.empty(len(p16), dtype=bool)
        keep16[0] = True
        keep16[1:] = p16[1:] != p16[:-1]
        u_p16 = p16[keep16]

        self._level1: dict[int, list[int]] = {}
        for val in u_p16.tolist():
            b0 = (int(val) >> 8) & 0xFF
            b1 = int(val) & 0xFF
            self._level1.setdefault(b0 + 2, []).append(b1 + 2 + code_dim)

        l2_mask = np.zeros((code_dim * code_dim, code_dim), dtype=bool)
        l2_mask[u_p24 >> np.uint32(8), u_p24 & np.uint32(0xFF)] = True
        l1_mask = np.zeros((code_dim, code_dim), dtype=bool)
        l1_mask[u_p16 >> np.uint32(8), u_p16 & np.uint32(0xFF)] = True
        l0_mask = l1_mask.any(axis=1)

        self._l0_cpu = torch.from_numpy(l0_mask)
        self._l1_cpu = torch.from_numpy(l1_mask)
        self._l2_cpu = torch.from_numpy(l2_mask)
        self._u_p24_cpu = torch.from_numpy(u_p24.astype(np.int64))
        self._starts_cpu = torch.from_numpy(starts)
        self._ends_cpu = torch.from_numpy(ends)
        self._low8_cpu = torch.from_numpy((self._u32 & np.uint32(0xFF)).astype(np.int64))
        self._gpu_tables: dict[torch.device, tuple[torch.Tensor, ...]] = {}
        self._lock = Lock()

    def _tables_for(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        if device.type == "cpu":
            return (
                self._l0_cpu,
                self._l1_cpu,
                self._l2_cpu,
                self._u_p24_cpu,
                self._starts_cpu,
                self._ends_cpu,
                self._low8_cpu,
            )
        cached = self._gpu_tables.get(device)
        if cached is not None:
            return cached
        with self._lock:
            cached = self._gpu_tables.get(device)
            if cached is None:
                with torch.cuda.device(device):
                    cached = (
                        self._l0_cpu.to(device=device),
                        self._l1_cpu.to(device=device),
                        self._l2_cpu.to(device=device),
                        self._u_p24_cpu.to(device=device),
                        self._starts_cpu.to(device=device),
                        self._ends_cpu.to(device=device),
                        self._low8_cpu.to(device=device),
                    )
                self._gpu_tables[device] = cached
            return cached

    def _fill_level3_mask(
        self,
        mask_2d: torch.Tensor,
        c0: torch.Tensor,
        c1: torch.Tensor,
        c2: torch.Tensor,
        u_p24: torch.Tensor,
        starts: torch.Tensor,
        ends: torch.Tensor,
        low8: torch.Tensor,
    ) -> None:
        d = self._code_dim
        valid_prefix = (c0 >= 0) & (c0 < d) & (c1 >= 0) & (c1 < d) & (c2 >= 0) & (c2 < d)
        q_p24 = ((c0.clamp(0, d - 1) << 16) | (c1.clamp(0, d - 1) << 8) | c2.clamp(0, d - 1))
        idx = torch.searchsorted(u_p24, q_p24)
        idx_clamped = idx.clamp(max=u_p24.shape[0] - 1)
        matched = valid_prefix & (idx < u_p24.shape[0]) & (u_p24[idx_clamped] == q_p24)
        s = torch.where(matched, starts[idx_clamped], 0)
        e = torch.where(matched, ends[idx_clamped], 0)
        lengths = e - s
        if lengths.numel() > 0:
            offsets = torch.arange(d, device=mask_2d.device).unsqueeze(0)
            valid_off = offsets < lengths.unsqueeze(1)
            gather_pos = (s.unsqueeze(1) + offsets).clamp(max=low8.shape[0] - 1)
            vals = low8[gather_pos] + (2 + 3 * d)
            row_ids = torch.arange(mask_2d.shape[0], device=mask_2d.device).unsqueeze(1).expand_as(vals)
            mask_2d[row_ids[valid_off], vals[valid_off]] = True

    def build_step_mask(self, flat_seq: torch.Tensor, vocab_size: int) -> torch.Tensor:
        l0, l1, l2, u_p24, starts, ends, low8 = self._tables_for(flat_seq.device)
        d = self._code_dim
        rows, plen = flat_seq.shape
        mask = torch.zeros((rows, vocab_size), dtype=torch.bool, device=flat_seq.device)
        if plen == 1:
            mask[:, 2 : 2 + d] = l0.unsqueeze(0)
        elif plen == 2:
            c0 = flat_seq[:, 1] - 2
            valid = (c0 >= 0) & (c0 < d)
            mask[valid, 2 + d : 2 + 2 * d] = l1[c0[valid]]
        elif plen == 3:
            c0 = flat_seq[:, 1] - 2
            c1 = flat_seq[:, 2] - (2 + d)
            valid = (c0 >= 0) & (c0 < d) & (c1 >= 0) & (c1 < d)
            p16 = (c0[valid] << 8) | c1[valid]
            mask[valid, 2 + 2 * d : 2 + 3 * d] = l2[p16]
        elif plen == 4:
            c0 = flat_seq[:, 1] - 2
            c1 = flat_seq[:, 2] - (2 + d)
            c2 = flat_seq[:, 3] - (2 + 2 * d)
            self._fill_level3_mask(mask, c0, c1, c2, u_p24, starts, ends, low8)
        return mask

    def build_response_mask(self, responses: torch.Tensor, vocab_size: int) -> torch.Tensor:
        l0, l1, l2, u_p24, starts, ends, low8 = self._tables_for(responses.device)
        d = self._code_dim
        rows = responses.shape[0]
        mask = torch.zeros((rows, 4, vocab_size), dtype=torch.bool, device=responses.device)
        c0 = responses[:, 0] - 2
        c1 = responses[:, 1] - (2 + d)
        c2 = responses[:, 2] - (2 + 2 * d)
        mask[:, 0, 2 : 2 + d] = l0.unsqueeze(0)
        v1 = (c0 >= 0) & (c0 < d)
        mask[v1, 1, 2 + d : 2 + 2 * d] = l1[c0[v1]]
        v2 = v1 & (c1 >= 0) & (c1 < d)
        mask[v2, 2, 2 + 2 * d : 2 + 3 * d] = l2[(c0[v2] << 8) | c1[v2]]
        self._fill_level3_mask(mask[:, 3, :], c0, c1, c2, u_p24, starts, ends, low8)
        return mask

    def __call__(self, batch_idx: int, prefix_beam: torch.Tensor) -> list[int]:
        del batch_idx
        prefix = prefix_beam.tolist()
        plen = len(prefix)
        if plen == 1:
            return self._root_tokens
        if plen == 2:
            return list(self._level1.get(int(prefix[1]), ()))
        if plen == 3:
            b0 = int(prefix[1]) - 2
            b1 = int(prefix[2]) - 2 - self._code_dim
            if not (0 <= b0 < self._code_dim and 0 <= b1 < self._code_dim):
                return []
            lo = np.uint32((b0 << 24) | (b1 << 16))
            hi = np.uint32((b0 << 24) | (b1 << 16) | 0xFFFF)
            i0 = int(np.searchsorted(self._u32, lo, side="left"))
            i1 = int(np.searchsorted(self._u32, hi, side="right"))
            if i0 >= i1:
                return []
            b2_vals = np.unique((self._u32[i0:i1] >> np.uint32(8)) & np.uint32(0xFF))
            offset2 = 2 + 2 * self._code_dim
            return [int(v) + offset2 for v in b2_vals.tolist()]
        if plen == 4:
            b0 = int(prefix[1]) - 2
            b1 = int(prefix[2]) - 2 - self._code_dim
            b2 = int(prefix[3]) - 2 - 2 * self._code_dim
            if not (0 <= b0 < self._code_dim and 0 <= b1 < self._code_dim and 0 <= b2 < self._code_dim):
                return []
            lo = np.uint32((b0 << 24) | (b1 << 16) | (b2 << 8))
            hi = np.uint32((b0 << 24) | (b1 << 16) | (b2 << 8) | 0xFF)
            i0 = int(np.searchsorted(self._u32, lo, side="left"))
            i1 = int(np.searchsorted(self._u32, hi, side="right"))
            if i0 >= i1:
                return []
            offset3 = 2 + 3 * self._code_dim
            return [int(v & 0xFF) + offset3 for v in self._u32[i0:i1].tolist()]
        return []


@lru_cache(maxsize=1)
def load_constraint(mapping_path: str, code_num: int, code_dim: int) -> PrefixConstraint:
    mapping = absolute_path(mapping_path)
    if code_num != 4 or code_dim != 256:
        raise ValueError("Columnar u32 docid index requires code_num=4 and code_dim=256")
    local_u32 = Path(tempfile.gettempdir()) / f"gr_docid_unique_{mapping.name}.u32.npy"
    if not local_u32.is_file():
        with FileLock(f"{local_u32}.lock"):
            if not local_u32.is_file():
                tmp_u32 = local_u32.with_suffix(".tmp.npy")
                shutil.copyfile(mapping / "docid_unique.u32.npy", tmp_u32)
                tmp_u32.replace(local_u32)
    unique_u32 = np.load(local_u32)
    constraint = SortedU32Constraint(unique_u32, code_dim=code_dim)
    if len(constraint._level1) != code_dim:
        raise ValueError("Official mapping must cover the initial codebook used by SLMGR")
    return constraint


def _make_grouped_xattn_forward(mod: Any, orig_forward: Any):
    def _forward(
        hidden_states: torch.Tensor,
        mask: torch.Tensor | None = None,
        key_value_states: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        past_key_values: Any = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ):
        if (
            key_value_states is None
            or past_key_values is not None
            or output_attentions
            or key_value_states.shape[0] == hidden_states.shape[0]
        ):
            return orig_forward(
                hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                **kwargs,
            )
        bsz_total, q_len, _ = hidden_states.shape
        g_cnt, k_len, _ = key_value_states.shape
        if g_cnt <= 0 or bsz_total % g_cnt != 0:
            return orig_forward(
                hidden_states,
                mask=mask,
                key_value_states=key_value_states,
                position_bias=position_bias,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                **kwargs,
            )
        grp = bsz_total // g_cnt
        n_heads = int(mod.n_heads)
        d_kv = int(mod.key_value_proj_dim)
        q = mod.q(hidden_states).view(g_cnt, grp * q_len, n_heads, d_kv).transpose(1, 2)
        k = mod.k(key_value_states).view(g_cnt, k_len, n_heads, d_kv).transpose(1, 2)
        v = mod.v(key_value_states).view(g_cnt, k_len, n_heads, d_kv).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(3, 2))
        if position_bias is None:
            position_bias = torch.zeros((1, n_heads, 1, k_len), device=scores.device, dtype=scores.dtype)
            if mask is not None:
                position_bias = position_bias + mask[:, :, :1, :k_len]
        scores = scores + position_bias
        probs = F.softmax(scores.float(), dim=-1).type_as(scores)
        probs = F.dropout(probs, p=mod.dropout, training=mod.training)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz_total, q_len, n_heads * d_kv)
        return (mod.o(out), position_bias)

    return _forward


def install_grouped_cross_attention(decoder: nn.Module) -> None:
    """Patch T5 decoder blocks so ``[B*G, q_len, d]`` queries attend to unexpanded ``[B, k_len, d]`` encoder states."""
    for block in getattr(decoder, "block", ()):
        layers = getattr(block, "layer", ())
        if len(layers) < 2:
            continue
        attn = getattr(layers[1], "EncDecAttention", None)
        if attn is None or getattr(attn, "_gr_grouped_xattn", False):
            continue
        attn.forward = _make_grouped_xattn_forward(attn, attn.forward)
        attn._gr_grouped_xattn = True


def has_grouped_cross_attention(decoder: nn.Module) -> bool:
    return any(
        getattr(getattr(block.layer[1], "EncDecAttention", None), "_gr_grouped_xattn", False)
        for block in getattr(decoder, "block", ())
        if len(getattr(block, "layer", ())) >= 2
    )


def build_generation_model(
    *,
    slm_root: str,
    config: T5Config,
    projection: nn.Module,
    decoder: nn.Module,
    lm_head: nn.Module,
) -> nn.Module:
    """Attach existing policy tensors to the official generation class."""
    slmv3 = import_slmv3(require_directory(slm_root, "SLMv3 source"))
    source_class = slmv3["T5MultiWithGivenQueryDocEncoderDSI"]
    install_grouped_cross_attention(decoder)

    class GenerationHead(source_class):
        def __init__(self) -> None:
            T5PreTrainedModel.__init__(self, config)
            self.docEnc2decoder = projection
            self.decoder = decoder
            self.lm_head = lm_head
            self.running_args = slmv3["RunArguments"]()
            self.margin = self.running_args.margin

    return GenerationHead().eval()


def build_step_constraint_mask(
    constraint: PrefixConstraint,
    flat_seq: torch.Tensor,
    vocab_size: int,
    prefix_cache: dict[tuple[int, ...], list[int]],
    device: torch.device,
) -> torch.Tensor:
    if isinstance(constraint, SortedU32Constraint):
        return constraint.build_step_mask(flat_seq, vocab_size)
    mask_cpu = torch.zeros((flat_seq.shape[0], vocab_size), dtype=torch.bool)
    for row_idx, seq_row in enumerate(flat_seq.detach().cpu().tolist()):
        key = tuple(seq_row)
        allowed = prefix_cache.get(key)
        if allowed is None:
            allowed = constraint(0, torch.tensor(seq_row, dtype=torch.long))
            if len(prefix_cache) < 262_144:
                prefix_cache[key] = allowed
        if allowed:
            mask_cpu[row_idx, allowed] = True
    return mask_cpu.to(device=device, non_blocking=True)


def sample_kool_sbs_beams(
    *,
    step: int,
    mask_3d: torch.Tensor,
    cum_scores: torch.Tensor,
    step_log_probs: torch.Tensor,
    cand_cum_3d: torch.Tensor,
    beam_gumbels: torch.Tensor,
    next_k: int,
    temperature: float,
    step_temps: tuple[float, ...] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_temps is not None:
        t_cur = step_temps[step]
        scaled_3d = (
            cum_scores.unsqueeze(-1) / step_temps[step - 1] + step_log_probs / t_cur
            if step > 0
            else cand_cum_3d / t_cur
        )
    else:
        scaled_3d = cand_cum_3d / temperature if temperature != 1.0 else cand_cum_3d
    gumbel_3d = -torch.empty_like(scaled_3d).exponential_().clamp_min_(1e-38).log()
    raw_g = torch.where(mask_3d, scaled_3d + gumbel_3d, float("-inf"))
    if step > 0:
        max_g = raw_g.max(dim=-1, keepdim=True).values
        delta = torch.where(mask_3d, raw_g - max_g, float("-inf"))
        shifted = torch.where(mask_3d, -raw_g + torch.log(-torch.expm1(delta)), float("inf"))
        cond_g = -torch.logaddexp(-beam_gumbels.unsqueeze(-1).expand_as(raw_g), shifted)
        cond_g = torch.where(mask_3d & torch.isfinite(max_g), cond_g, float("-inf"))
    else:
        cond_g = raw_g
    flat_g = cond_g.reshape(mask_3d.shape[0], -1)
    return torch.topk(flat_g, k=next_k, dim=-1)


def assert_unique_beam_tokens(tokens: torch.Tensor, num_return_sequences: int, code_num: int) -> None:
    if num_return_sequences <= 1:
        return
    if code_num == 4:
        packed = (tokens[:, :, 0] << 33) | (tokens[:, :, 1] << 22) | (tokens[:, :, 2] << 11) | tokens[:, :, 3]
        sorted_packed = torch.sort(packed, dim=1).values
        if bool((sorted_packed[:, 1:] == sorted_packed[:, :-1]).any().item()):
            raise RuntimeError("SLMGR constrained beam tree returned duplicate IDs")
        return
    if any(len(set(map(tuple, rows.tolist()))) != num_return_sequences for rows in tokens):
        raise RuntimeError("SLMGR constrained beam tree returned duplicate IDs")


@torch.no_grad()
def generate_beams_tree(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_beams: int,
    num_return_sequences: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    step_temperatures: tuple[float, ...] | None = None,
    query_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if step_temperatures is not None and (len(step_temperatures) != code_num or any(t <= 0 for t in step_temperatures)):
        raise ValueError(f"step_temperatures must contain {code_num} positive floats, got {step_temperatures!r}")
    total_queries = hidden_states.shape[0]
    device = hidden_states.device
    all_tokens: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    prefix_cache: dict[tuple[int, ...], list[int]] = {}
    grouped_xattn = has_grouped_cross_attention(model.decoder)

    for q_start in range(0, total_queries, query_chunk_size):
        q_end = min(q_start + query_chunk_size, total_queries)
        hs_chunk = hidden_states[q_start:q_end]
        mask_chunk = attention_mask[q_start:q_end]
        batch_size = hs_chunk.shape[0]
        projected = model.docEnc2decoder(hs_chunk)

        sequences = torch.zeros((batch_size, 1, 1), dtype=torch.long, device=device)
        token_scores = torch.empty((batch_size, 1, 0), dtype=torch.float32, device=device)
        cum_scores = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)
        beam_gumbels = torch.zeros((batch_size, 1), dtype=torch.float32, device=device)

        for step in range(code_num):
            cur_beams = sequences.shape[1]
            flat_seq = sequences.reshape(batch_size * cur_beams, step + 1)
            rep_proj = projected if (grouped_xattn or cur_beams == 1) else projected.repeat_interleave(cur_beams, dim=0)
            rep_mask = mask_chunk if (grouped_xattn or cur_beams == 1) else mask_chunk.repeat_interleave(cur_beams, dim=0)
            dec_out = model.decoder(
                input_ids=flat_seq,
                encoder_hidden_states=rep_proj,
                encoder_attention_mask=rep_mask,
                use_cache=False,
                return_dict=True,
            )
            logits = model.lm_head(dec_out.last_hidden_state[:, -1, :]).float()
            vocab_size = logits.shape[-1]
            mask_gpu = build_step_constraint_mask(constraint, flat_seq, vocab_size, prefix_cache, device)
            mask_3d = mask_gpu.reshape(batch_size, cur_beams, vocab_size)
            step_log_probs = F.log_softmax(logits.masked_fill(~mask_gpu, float("-inf")), dim=-1).reshape(
                batch_size, cur_beams, vocab_size
            )
            step_log_probs = step_log_probs.masked_fill(~mask_3d, float("-inf"))
            cand_cum_3d = cum_scores.unsqueeze(-1) + step_log_probs
            cand_cum = cand_cum_3d.reshape(batch_size, cur_beams * vocab_size)
            valid_counts = (cand_cum > float("-inf")).sum(dim=-1)
            target_k = num_beams if step > 0 else min(num_beams, 256)
            next_k = min(target_k, int(valid_counts.min().item()))
            if next_k <= 0:
                raise RuntimeError(f"Constrained beam tree has no valid continuations at step {step}")
            if do_sample:
                beam_gumbels, top_idx = sample_kool_sbs_beams(
                    step=step,
                    mask_3d=mask_3d,
                    cum_scores=cum_scores,
                    step_log_probs=step_log_probs,
                    cand_cum_3d=cand_cum_3d,
                    beam_gumbels=beam_gumbels,
                    next_k=next_k,
                    temperature=temperature,
                    step_temps=step_temperatures,
                )
            else:
                _, top_idx = torch.topk(cand_cum, k=next_k, dim=-1)
            beam_idx = top_idx // vocab_size
            tok_idx = top_idx % vocab_size
            cum_scores = cand_cum.gather(1, top_idx)
            chosen_step_lp = step_log_probs.reshape(batch_size, -1).gather(1, top_idx)
            prev_seq = sequences.gather(1, beam_idx.unsqueeze(-1).expand(batch_size, next_k, step + 1))
            sequences = torch.cat([prev_seq, tok_idx.unsqueeze(-1)], dim=-1)
            prev_scores = token_scores.gather(1, beam_idx.unsqueeze(-1).expand(batch_size, next_k, step))
            token_scores = torch.cat([prev_scores, chosen_step_lp.unsqueeze(-1)], dim=-1)

        if sequences.shape[1] < num_return_sequences:
            raise RuntimeError(
                f"Constrained beam tree produced {sequences.shape[1]} beams, fewer than {num_return_sequences}"
            )
        all_tokens.append(sequences[:, :num_return_sequences, 1:])
        all_scores.append(token_scores[:, :num_return_sequences, :])

    tokens = torch.cat(all_tokens, dim=0)
    scores = torch.cat(all_scores, dim=0)
    if not torch.isfinite(scores).all():
        raise RuntimeError("SLMGR constrained beam tree output has non-finite token logprobs")
    assert_unique_beam_tokens(tokens, num_return_sequences, code_num)
    return tokens, scores


@torch.no_grad()
def generate_beams(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_beams: int,
    num_return_sequences: int,
    do_sample: bool = False,
    temperature: float = 1.0,
    step_temperatures: tuple[float, ...] | None = None,
    query_chunk_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not 1 <= num_return_sequences <= num_beams:
        raise ValueError("Require 1 <= num_return_sequences <= num_beams")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if num_beams > 256:
        return generate_beams_tree(
            model,
            hidden_states,
            attention_mask,
            constraint,
            code_num=code_num,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            temperature=temperature,
            step_temperatures=step_temperatures if do_sample else None,
            query_chunk_size=query_chunk_size,
        )
    sample_kwargs: dict[str, Any] = (
        {"do_sample": True, "temperature": temperature, "top_p": 1.0, "top_k": 0}
        if do_sample
        else {}
    )
    output = model.generate(
        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=cast(Any, hidden_states),
            hidden_states=cast(Any, hidden_states),
            attentions=cast(Any, attention_mask),
        ),
        attention_mask=attention_mask,
        max_length=code_num + 1,
        min_length=code_num + 1,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        prefix_allowed_tokens_fn=constraint,
        early_stopping=True,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        decoder_start_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
        **sample_kwargs,
    )
    tokens = output.sequences[:, 1:]
    scores = model.compute_transition_scores(
        output.sequences, output.scores, getattr(output, "beam_indices", None), normalize_logits=True,
    )
    expected = (hidden_states.shape[0] * num_return_sequences, code_num)
    if tokens.shape != expected or scores.shape != expected or not torch.isfinite(scores).all():
        raise RuntimeError("SLMGR constrained beam output has invalid shape or token logprobs")
    for sequence in output.sequences:
        for position in range(code_num):
            if sequence[position + 1].item() not in constraint(0, sequence[:position + 1]):
                raise RuntimeError("SLMGR beam returned a token outside the official prefix constraint")
    tokens = tokens.reshape(hidden_states.shape[0], num_return_sequences, code_num)
    if any(len(set(map(tuple, rows.tolist()))) != num_return_sequences for rows in tokens):
        raise RuntimeError("SLMGR constrained beam returned duplicate IDs")
    return tokens, scores.reshape_as(tokens)


@torch.no_grad()
def generate_samples(
    model: Any,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
    constraint: PrefixConstraint,
    *,
    code_num: int,
    num_return_sequences: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_return_sequences <= 0:
        raise ValueError("num_return_sequences must be positive")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    output = model.generate(
        encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=cast(Any, hidden_states),
            hidden_states=cast(Any, hidden_states),
            attentions=cast(Any, attention_mask),
        ),
        attention_mask=attention_mask,
        max_length=code_num + 1,
        min_length=code_num + 1,
        do_sample=True,
        temperature=temperature,
        top_p=1.0,
        num_beams=1,
        num_return_sequences=num_return_sequences,
        prefix_allowed_tokens_fn=constraint,
        forced_bos_token_id=None,
        forced_eos_token_id=None,
        decoder_start_token_id=0,
        return_dict_in_generate=True,
        output_scores=True,
    )
    tokens = output.sequences[:, 1:]
    scores = model.compute_transition_scores(
        output.sequences, output.scores, normalize_logits=True,
    )
    expected = (hidden_states.shape[0] * num_return_sequences, code_num)
    if tokens.shape != expected or scores.shape != expected or not torch.isfinite(scores).all():
        raise RuntimeError("SLMGR constrained sampling output has invalid shape or token logprobs")
    for sequence in output.sequences:
        for position in range(code_num):
            if sequence[position + 1].item() not in constraint(0, sequence[:position + 1]):
                raise RuntimeError("SLMGR sampling returned a token outside the official prefix constraint")
    return (
        tokens.reshape(hidden_states.shape[0], num_return_sequences, code_num),
        scores.reshape(hidden_states.shape[0], num_return_sequences, code_num),
    )
