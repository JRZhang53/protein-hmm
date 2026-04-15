from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.data.encoding import AminoAcidEncoder, StructureLabelEncoder


class EncodingTests(unittest.TestCase):
    def test_amino_acid_round_trip(self) -> None:
        encoder = AminoAcidEncoder()
        encoded = encoder.encode("ACDE")
        self.assertEqual(encoded.tolist(), [0, 1, 2, 3])
        self.assertEqual(encoder.decode(encoded), "ACDE")

    def test_unknown_residue_raises_without_unknown_token(self) -> None:
        encoder = AminoAcidEncoder()
        with self.assertRaises(ValueError):
            encoder.encode("ACXZ")

    def test_structure_label_encoder_collapses_dssp_variants(self) -> None:
        encoder = StructureLabelEncoder()
        self.assertEqual(encoder.normalize_labels("HGIBTS"), "HHHECC")


if __name__ == "__main__":
    unittest.main()
