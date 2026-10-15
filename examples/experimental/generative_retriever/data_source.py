"""Fixed-step GR prompt source with checkpoint cursors independent of async prefetch."""

from __future__ import annotations

from pathlib import Path

import torch

from miles.rollout.data_source import RolloutDataSourceWithBuffer


class GRRolloutDataSource(RolloutDataSourceWithBuffer):
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
        path = Path(self.args.save) / "rollout" / f"global_dataset_state_dict_{rollout_id}.pt"
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
