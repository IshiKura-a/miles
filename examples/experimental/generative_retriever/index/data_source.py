"""Fixed-step GR prompt source with checkpoint cursors independent of async prefetch."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import torch

from miles.rollout.data_source import RolloutDataSourceWithBuffer


class GRRolloutDataSource(RolloutDataSourceWithBuffer):
    def __init__(self, args: Any) -> None:
        if args.save and args.num_rollout is not None and args.num_rollout % args.update_weights_interval:
            raise ValueError("The final GR checkpoint must coincide with a rollout weight update")
        super().__init__(args)
        self.pending_mapping_checkpoints: list[int] = []

    def _completed_position(self, rollout_id: int) -> int:
        if self.dataset is None:
            raise ValueError("GR resume requires a global prompt dataset")
        position = (rollout_id + 1) * self.args.rollout_batch_size
        if position > len(self.dataset):
            raise ValueError("GR rollout checkpoint exceeds the prompt dataset")
        return position

    def save(self, rollout_id: int) -> None:
        position = self._completed_position(rollout_id)
        if self.epoch_id != 0 or self.sample_offset < position or self.buffer:
            raise ValueError("GR rollout cannot checkpoint a wrapped or buffered prompt dataset")
        save_root = Path(self.args.save).expanduser().resolve()
        path = save_root / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "sample_offset": position,
                "epoch_id": 0,
                "sample_group_index": position,
                "sample_index": position * self.args.n_samples_per_prompt,
                "metadata": self.metadata,
            },
            path,
        )
        self.pending_mapping_checkpoints.append(rollout_id)

    def sync_mapping(self, rollout_id: int) -> None:
        reward_url = self.args.generative_retriever_reward_url.removesuffix("/score")
        pending = [step for step in self.pending_mapping_checkpoints if step <= rollout_id]
        with httpx.Client(timeout=600.0, trust_env=False) as client:
            for step in sorted(set(pending + [rollout_id])):
                response = client.post(f"{reward_url}/commit_step_updates", json={"rollout_id": step})
                response.raise_for_status()
                if step in pending:
                    output_dir = Path(self.args.save) / f"iter_{step + 1:07d}" / "mapping"
                    response = client.post(f"{reward_url}/save_mapping", json={"output_dir": str(output_dir)})
                    response.raise_for_status()
                    self.pending_mapping_checkpoints.remove(step)

    def load(self, rollout_id: int | None = None) -> None:
        if self.args.load is None:
            return
        if rollout_id is None:
            raise ValueError("GR resume requires a completed rollout ID")
        path = Path(self.args.load) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"GR rollout checkpoint is missing: {path}")
        super().load(rollout_id)
        position = self._completed_position(rollout_id)
        if self.epoch_id != 0 or self.sample_offset < position or self.buffer:
            raise ValueError("GR rollout checkpoint has an invalid prompt cursor")
        self.sample_offset = self.sample_group_index = position
        self.sample_index = position * self.args.n_samples_per_prompt


def sync_mapping_on_weight_update(args: Any, data_source: GRRolloutDataSource, rollout_id: int) -> None:
    data_source.sync_mapping(rollout_id)
