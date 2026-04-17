from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.data.alignment import AlignmentError, ResidueAligner
from protein_hmm.types import AlignedProteinRecord, ProteinRecord


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

    def test_aligned_record_round_trip_to_dict(self) -> None:
        record = AlignedProteinRecord(
            "p1",
            "famA",
            "ACDE",
            labels="HEEC",
            metadata={"source": "toy"},
            structure_sequence="ACDE",
            alignment_score=0.95,
        )
        payload = record.to_dict()
        restored = AlignedProteinRecord.from_dict(payload)
        self.assertEqual(restored.structure_sequence, "ACDE")
        self.assertEqual(restored.alignment_score, 0.95)
        self.assertEqual(restored.metadata["source"], "toy")


if __name__ == "__main__":
    unittest.main()
