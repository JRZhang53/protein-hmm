from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.data.alignment import AlignmentError, ResidueAligner
from protein_hmm.types import ProteinRecord


class AlignmentTests(unittest.TestCase):
    def test_gap_trim_alignment(self) -> None:
        aligner = ResidueAligner()
        labels = aligner.align_labels("ACDE", "A-CD-E", "HHEECC")
        self.assertEqual(labels, "HEEC")

    def test_alignment_error_on_mismatch(self) -> None:
        aligner = ResidueAligner()
        with self.assertRaises(AlignmentError):
            aligner.align_labels("ACDE", "ABDE", "HHHH")

    def test_align_record_preserves_metadata(self) -> None:
        aligner = ResidueAligner()
        record = ProteinRecord("p1", "famA", "ACDE", metadata={"source": "toy"})
        aligned = aligner.align_record(record, "ACDE", "HEEC")
        self.assertEqual(aligned.labels, "HEEC")
        self.assertEqual(aligned.metadata["source"], "toy")


if __name__ == "__main__":
    unittest.main()
