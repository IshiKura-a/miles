"""Fixed-step GR prompt source with checkpoint cursors independent of async prefetch."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import threading
from typing import Any

import torch

from examples.experimental.generative_retriever.utils.paths import absolute_path
from miles.rollout.data_source import RolloutDataSourceWithBuffer
from miles.utils.types import Sample


class GRRolloutDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args: Any) -> None:
        if args.save and args.num_rollout is not None and args.num_rollout % args.update_weights_interval:
            raise ValueError("The final GR checkpoint must coincide with a rollout weight update")
        super().__init__(args)

    def get_samples(self, num_samples: int) -> list[list[Sample]]:
        samples = self._get_samples_from_buffer(num_samples)
        remaining = num_samples - len(samples)
        if remaining == 0:
            return samples
        if self.dataset is None:
            raise ValueError("GR rollout requires a global prompt dataset")
        if self.sample_offset + remaining <= len(self.dataset):
            prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + remaining]
            self.sample_offset += remaining
        else:
            prompt_samples = self.dataset.samples[self.sample_offset :]
            remaining -= len(prompt_samples)
            self.epoch_id += 1
            if self.args.rollout_shuffle:
                self.dataset.shuffle(self.epoch_id)
            prompt_samples += self.dataset.samples[:remaining]
            self.sample_offset = remaining

        group_size = self.args.n_samples_per_prompt
        for prompt_sample in prompt_samples:
            group_index = self.sample_group_index
            base_index = self.sample_index
            base_metadata = prompt_sample.metadata
            group = [
                Sample(
                    group_index=group_index,
                    index=base_index + rank,
                    prompt=prompt_sample.prompt,
                    label=prompt_sample.label,
                    metadata=dict(base_metadata) if (rank == 0 and base_metadata) else {},
                )
                for rank in range(group_size)
            ]
            self.sample_group_index += 1
            self.sample_index += group_size
            samples.append(group)
        return samples

    def pop_prompt_batch(self, num_samples: int) -> tuple[list[str], int, int]:
        """Advance dataset cursors identically to ``get_samples`` without allocating ``num_samples * G`` Samples."""
        buffered = self._get_samples_from_buffer(num_samples)
        prompts: list[str] = [str(grp[0].prompt) for grp in buffered]
        remaining = num_samples - len(prompts)
        if remaining > 0:
            if self.dataset is None:
                raise ValueError("GR rollout requires a global prompt dataset")
            if self.sample_offset + remaining <= len(self.dataset):
                prompt_samples = self.dataset.samples[self.sample_offset : self.sample_offset + remaining]
                self.sample_offset += remaining
            else:
                prompt_samples = self.dataset.samples[self.sample_offset :]
                remaining -= len(prompt_samples)
                self.epoch_id += 1
                if self.args.rollout_shuffle:
                    self.dataset.shuffle(self.epoch_id)
                prompt_samples += self.dataset.samples[:remaining]
                self.sample_offset = remaining
            prompts.extend(str(ps.prompt) for ps in prompt_samples)
        start_group_idx = self.sample_group_index
        start_sample_idx = self.sample_index
        non_buffered = num_samples - len(buffered)
        self.sample_group_index += non_buffered
        self.sample_index += non_buffered * self.args.n_samples_per_prompt
        return prompts, start_group_idx, start_sample_idx

    def completed_position(self, rollout_id: int) -> int:
        if self.dataset is None:
            raise ValueError("GR resume requires a global prompt dataset")
        position = (rollout_id + 1) * self.args.rollout_batch_size
        if position > len(self.dataset):
            raise ValueError("GR rollout checkpoint exceeds the prompt dataset")
        return position

    def save(self, rollout_id: int) -> None:
        position = self.completed_position(rollout_id)
        if self.epoch_id != 0 or self.sample_offset < position or self.buffer:
            raise ValueError("GR rollout cannot checkpoint a wrapped or buffered prompt dataset")
        save_root = absolute_path(self.args.save)
        path = save_root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        payload = {
            "sample_offset": position,
            "epoch_id": 0,
            "sample_group_index": position,
            "sample_index": position * self.args.n_samples_per_prompt,
            "metadata": self.metadata,
        }
        if bool(self.args.temp_stage_save):
            fd, tmp_str = tempfile.mkstemp(prefix=f"gr_ds_s{rollout_id}_", suffix=".pt")
            os.close(fd)
            tmp_path = Path(tmp_str)
            torch.save(payload, tmp_path)

            def _sync_ds() -> None:
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    dst_tmp = path.with_suffix(".pt.tmp")
                    shutil.copyfile(tmp_path, dst_tmp)
                    dst_tmp.replace(path)
                finally:
                    tmp_path.unlink(missing_ok=True)

            threading.Thread(target=_sync_ds, name=f"gr-ds-sync-{rollout_id}", daemon=False).start()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, path)

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None:
            return
        if rollout_id is None:
            raise ValueError("GR resume requires a completed rollout ID")
        path = Path(self.args.load) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"GR rollout checkpoint is missing: {path}")
        super().load(rollout_id)
        position = self.completed_position(rollout_id)
        if self.epoch_id != 0 or self.sample_offset < position or self.buffer:
            raise ValueError("GR rollout checkpoint has an invalid prompt cursor")
        self.sample_offset = self.sample_group_index = position
        self.sample_index = position * self.args.n_samples_per_prompt
