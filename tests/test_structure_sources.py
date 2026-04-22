from __future__ import annotations

import unittest

from support import bootstrap

bootstrap()

from protein_hmm.data.dssp import parse_dssp_line
from protein_hmm.data.pfam_seed import PfamSeedRecord, parse_seed_sequence_id
from protein_hmm.data.sifts import (
    PfamChainMapping,
    SiftsResidueMapping,
    UniProtChainMapping,
    select_best_structure_mapping,
)
from protein_hmm.data.structure_annotations import _build_annotation_strings


class StructureSourceTests(unittest.TestCase):
    def test_parse_seed_sequence_id(self) -> None:
        accession, start, end = parse_seed_sequence_id("Q8U2T8_PYRFU/7-144")
        self.assertEqual(accession, "Q8U2T8")
        self.assertEqual((start, end), (7, 144))

    def test_parse_dssp_line(self) -> None:
        line = "    9   14 A Q  H  > S+     0   0   57      1,-0.3     4,-2.7     2,-0.2     5,-0.2"
        residue = parse_dssp_line(line)
        self.assertIsNotNone(residue)
        assert residue is not None
        self.assertEqual(residue.chain_id, "A")
        self.assertEqual(residue.pdb_resnum, "14")
        self.assertEqual(residue.residue, "Q")
        self.assertEqual(residue.secondary_structure, "H")

    def test_select_best_structure_mapping_prefers_full_domain_cover(self) -> None:
        record = PfamSeedRecord(
            protein_id="Q8U2T8_PYRFU/7-144",
            accession="Q8U2T8",
            family="PF00210",
            family_name="Ferritin",
            start=7,
            end=144,
            sequence="A" * 138,
        )
        pfam_mappings = [
            PfamChainMapping("2x17", "Z", "Q8U2T8", "PF00210", 1.0),
            PfamChainMapping("1abc", "A", "Q8U2T8", "PF00210", 1.0),
        ]
        uniprot_mappings = [
            UniProtChainMapping("2x17", "Z", "Q8U2T8", 1, 173, 0, 172),
            UniProtChainMapping("1abc", "A", "Q8U2T8", 50, 130, 49, 129),
        ]
        best = select_best_structure_mapping(record, pfam_mappings, uniprot_mappings)
        self.assertIsNotNone(best)
        assert best is not None
        self.assertEqual(best.pdb_id, "2x17")
        self.assertTrue(best.covers_domain)

    def test_build_annotation_strings_falls_back_to_sifts_secondary_structure(self) -> None:
        record = PfamSeedRecord(
            protein_id="p1/1-3",
            accession="P1",
            family="PF00000",
            family_name="toy",
            start=1,
            end=3,
            sequence="ACD",
        )
        residue_map = {
            1: SiftsResidueMapping("P1", "A", 1, "A", "1", "A", True, "H"),
            2: SiftsResidueMapping("P1", "A", 2, "C", "2", "C", True, "E"),
            3: SiftsResidueMapping("P1", "A", 3, "D", "3", "D", True, "T"),
        }
        structure_sequence, labels, observed_count, fallback_count = _build_annotation_strings(
            record=record,
            residue_map=residue_map,
            dssp_map={},
            chain_id="A",
        )
        self.assertEqual(structure_sequence, "ACD")
        self.assertEqual(labels, "HET")
        self.assertEqual(observed_count, 3)
        self.assertEqual(fallback_count, 3)


if __name__ == "__main__":
    unittest.main()
