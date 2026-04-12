"""Project-wide biological constants."""

from __future__ import annotations

AMINO_ACIDS: tuple[str, ...] = tuple("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_TO_INDEX: dict[str, int] = {
    residue: index for index, residue in enumerate(AMINO_ACIDS)
}
INDEX_TO_AMINO_ACID: dict[int, str] = {
    index: residue for residue, index in AMINO_ACID_TO_INDEX.items()
}
VALID_AMINO_ACIDS: frozenset[str] = frozenset(AMINO_ACID_TO_INDEX)

DSSP_LABELS: tuple[str, ...] = ("H", "E", "C")
DSSP_COLLAPSE_MAP: dict[str, str] = {
    "H": "H",
    "G": "H",
    "I": "H",
    "E": "E",
    "B": "E",
    "T": "C",
    "S": "C",
    "C": "C",
    "-": "C",
    " ": "C",
}

KYTE_DOOLITTLE_HYDROPHOBICITY: dict[str, float] = {
    "A": 1.8,
    "C": 2.5,
    "D": -3.5,
    "E": -3.5,
    "F": 2.8,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "K": -3.9,
    "L": 3.8,
    "M": 1.9,
    "N": -3.5,
    "P": -1.6,
    "Q": -3.5,
    "R": -4.5,
    "S": -0.8,
    "T": -0.7,
    "V": 4.2,
    "W": -0.9,
    "Y": -1.3,
}

POLAR_RESIDUES: frozenset[str] = frozenset({"N", "Q", "S", "T", "Y", "C", "H", "D", "E", "K", "R"})
CHARGED_RESIDUES: frozenset[str] = frozenset({"D", "E", "K", "R", "H"})
