"""Protein-level dataset splitting."""

from __future__ import annotations

from dataclasses import dataclass

from protein_hmm.types import ProteinRecord
from protein_hmm.utils.random_state import get_rng


@dataclass(slots=True)
class ProteinLevelSplitter:
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_state: int | None = None

    def split(self, records: list[ProteinRecord]) -> dict[str, list[ProteinRecord]]:
        if not records:
            return {"train": [], "val": [], "test": []}
        total = self.train_fraction + self.val_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-8:
            raise ValueError("Split fractions must sum to 1.0.")

        rng = get_rng(self.random_state)
        grouped: dict[str, list[ProteinRecord]] = {}
        for record in records:
            grouped.setdefault(record.family, []).append(record)

        splits = {"train": [], "val": [], "test": []}
        for family_records in grouped.values():
            shuffled = list(family_records)
            rng.shuffle(shuffled)
            n_items = len(shuffled)
            n_test = min(int(round(n_items * self.test_fraction)), max(n_items - 1, 0))
            n_val = min(int(round(n_items * self.val_fraction)), max(n_items - n_test - 1, 0))
            n_train = n_items - n_val - n_test
            if n_train <= 0:
                n_train = 1
                if n_val > 0:
                    n_val -= 1
                elif n_test > 0:
                    n_test -= 1

            splits["train"].extend(shuffled[:n_train])
            splits["val"].extend(shuffled[n_train:n_train + n_val])
            splits["test"].extend(shuffled[n_train + n_val:])
        return splits
