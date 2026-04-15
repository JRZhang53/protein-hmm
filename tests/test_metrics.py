from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.analysis.metrics import adjusted_rand_index, bic_score, q3_accuracy, segment_overlap_score


class MetricTests(unittest.TestCase):
    def test_adjusted_rand_index_is_permutation_invariant(self) -> None:
        self.assertAlmostEqual(adjusted_rand_index("AABBCC", "XXYYZZ"), 1.0)

    def test_q3_accuracy_collapses_secondary_structure_variants(self) -> None:
        self.assertAlmostEqual(q3_accuracy("HGIB", "HHHE"), 1.0)

    def test_segment_overlap_score_is_one_for_identical_segments(self) -> None:
        self.assertAlmostEqual(segment_overlap_score("HHHCCEEE", "HHHCCEEE"), 1.0)

    def test_bic_score_runs(self) -> None:
        self.assertGreater(bic_score(-10.0, 5, 20), 0.0)


if __name__ == "__main__":
    unittest.main()
