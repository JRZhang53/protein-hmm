"""Sequence encoders for amino acids and structure labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from protein_hmm.constants import AMINO_ACIDS, DSSP_COLLAPSE_MAP, DSSP_LABELS


@dataclass(slots=True)
class CategoricalEncoder:
    vocabulary: tuple[str, ...]
    unknown_token: str | None = None
    token_to_id: dict[str, int] = field(init=False)
    id_to_token: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        tokens = list(self.vocabulary)
        if self.unknown_token is not None and self.unknown_token not in tokens:
            tokens.append(self.unknown_token)
        self.vocabulary = tuple(tokens)
        self.token_to_id = {token: index for index, token in enumerate(self.vocabulary)}
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

    @property
    def size(self) -> int:
        return len(self.vocabulary)

    def encode(self, sequence: str) -> np.ndarray:
        encoded: list[int] = []
        for token in sequence:
            token = token.upper()
            if token not in self.token_to_id:
                if self.unknown_token is None:
                    raise ValueError(f"Unknown token '{token}' for vocabulary {self.vocabulary}.")
                token = self.unknown_token
            encoded.append(self.token_to_id[token])
        return np.asarray(encoded, dtype=int)

    def encode_many(self, sequences: list[str]) -> list[np.ndarray]:
        return [self.encode(sequence) for sequence in sequences]

    def decode(self, encoded: np.ndarray) -> str:
        return "".join(self.id_to_token[int(index)] for index in np.asarray(encoded, dtype=int))


class AminoAcidEncoder(CategoricalEncoder):
    def __init__(self, allow_unknown: bool = False) -> None:
        super().__init__(vocabulary=AMINO_ACIDS, unknown_token="X" if allow_unknown else None)


class StructureLabelEncoder(CategoricalEncoder):
    def __init__(self) -> None:
        super().__init__(vocabulary=DSSP_LABELS)

    def normalize_labels(self, labels: str) -> str:
        return "".join(DSSP_COLLAPSE_MAP.get(label.upper(), "C") for label in labels)
