from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.analysis.evaluation import annotation_baselines, training_diagnostics
from protein_hmm.types import ProteinRecord, TrainingHistory


class EvaluationTests(unittest.TestCase):
    def test_annotation_baselines_include_residue_majority(self) -> None:
        train = [
            ProteinRecord("p1", "famA", "AAAC", labels="HHHC"),
            ProteinRecord("p2", "famB", "CCCA", labels="CCCH"),
        ]
        test = [
            ProteinRecord("p3", "famA", "ACAC", labels="HCHC"),
        ]
        payload = annotation_baselines(train, test)
        baseline_names = {row["name"] for row in payload["baselines"]}
        self.assertIn("global_majority_label", baseline_names)
        self.assertIn("family_majority_label", baseline_names)
        self.assertIn("residue_majority_label", baseline_names)

        residue_baseline = next(row for row in payload["baselines"] if row["name"] == "residue_majority_label")
        self.assertAlmostEqual(residue_baseline["q3"], 1.0)

    def test_training_diagnostics_reports_per_observation_delta(self) -> None:
        history = TrainingHistory(log_likelihoods=[-100.0, -90.0, -89.99], converged=False, iterations=3)
        diagnostics = training_diagnostics(history, num_observations=100)
        self.assertAlmostEqual(diagnostics["last_delta"], 0.01)
        self.assertAlmostEqual(diagnostics["last_delta_per_observation"], 0.0001)
        self.assertTrue(diagnostics["near_converged_per_observation"])


if __name__ == "__main__":
    unittest.main()
