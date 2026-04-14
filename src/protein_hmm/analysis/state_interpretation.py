"""Summaries for interpreting latent states biologically."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from protein_hmm.constants import AMINO_ACIDS, CHARGED_RESIDUES, DSSP_LABELS, KYTE_DOOLITTLE_HYDROPHOBICITY, POLAR_RESIDUES


def background_distribution(sequences: Iterable[np.ndarray], alphabet_size: int = 20) -> np.ndarray:
    counts = np.ones(alphabet_size, dtype=float)
    for sequence in sequences:
        for symbol in np.asarray(sequence, dtype=int):
            counts[int(symbol)] += 1.0
    return counts / counts.sum()


def state_enrichment(emission_probs: np.ndarray, background: np.ndarray | None = None) -> np.ndarray:
    emission_probs = np.asarray(emission_probs, dtype=float)
    if background is None:
        background = np.full(emission_probs.shape[1], 1.0 / emission_probs.shape[1], dtype=float)
    background = np.asarray(background, dtype=float)
    return np.log2(np.clip(emission_probs, 1e-12, None) / np.clip(background[None, :], 1e-12, None))


def state_hydrophobicity(emission_probs: np.ndarray) -> np.ndarray:
    weights = np.asarray([KYTE_DOOLITTLE_HYDROPHOBICITY[residue] for residue in AMINO_ACIDS], dtype=float)
    return np.asarray(emission_probs, dtype=float) @ weights


def summarize_states(emission_probs: np.ndarray, top_n: int = 5) -> list[dict[str, object]]:
    emission_probs = np.asarray(emission_probs, dtype=float)
    hydrophobicity = state_hydrophobicity(emission_probs)
    summaries: list[dict[str, object]] = []
    for state_index, row in enumerate(emission_probs):
        top_indices = np.argsort(row)[::-1][:top_n]
        top_residues = [AMINO_ACIDS[index] for index in top_indices]
        polar_mass = float(sum(row[index] for index, residue in enumerate(AMINO_ACIDS) if residue in POLAR_RESIDUES))
        charged_mass = float(sum(row[index] for index, residue in enumerate(AMINO_ACIDS) if residue in CHARGED_RESIDUES))
        summaries.append(
            {
                "state": state_index,
                "top_residues": top_residues,
                "hydrophobicity": float(hydrophobicity[state_index]),
                "polar_mass": polar_mass,
                "charged_mass": charged_mass,
            }
        )
    return summaries


def state_label_enrichment(
    decoded_states: list[list[int]],
    label_sequences: list[str],
    num_states: int,
    label_order: tuple[str, ...] = DSSP_LABELS,
) -> np.ndarray:
    counts = np.ones((num_states, len(label_order)), dtype=float)
    label_to_index = {label: index for index, label in enumerate(label_order)}
    for states, labels in zip(decoded_states, label_sequences):
        if len(states) != len(labels):
            raise ValueError("Decoded state paths and labels must have equal length.")
        for state, label in zip(states, labels):
            if label in label_to_index:
                counts[int(state), label_to_index[label]] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def family_state_prevalence(decoded_sequences: dict[str, list[list[int]]], num_states: int) -> dict[str, np.ndarray]:
    prevalence: dict[str, np.ndarray] = {}
    for family, paths in decoded_sequences.items():
        counts = Counter(state for path in paths for state in path)
        total = sum(counts.values()) or 1
        prevalence[family] = np.asarray([counts.get(state, 0) / total for state in range(num_states)], dtype=float)
    return prevalence
