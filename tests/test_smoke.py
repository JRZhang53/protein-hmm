from __future__ import annotations

import unittest

from support import bootstrap

ROOT = bootstrap()

from protein_hmm.analysis.metrics import q3_accuracy
from protein_hmm.data.encoding import AminoAcidEncoder
from protein_hmm.data.loaders import load_split_records, save_split_records
from protein_hmm.data.preprocessing import DatasetPreprocessor
from protein_hmm.data.splits import ProteinLevelSplitter
from protein_hmm.models.discrete_hmm import DiscreteHMM
from protein_hmm.models.semi_supervised_hmm import SemiSupervisedHMM
from protein_hmm.types import ProteinRecord


class SmokeTests(unittest.TestCase):
    def test_toy_pipeline_end_to_end(self) -> None:
        records = [
            ProteinRecord("p1", "famA", "AAAAA", labels="HHHHH"),
            ProteinRecord("p2", "famA", "AAAAV", labels="HHHHC"),
            ProteinRecord("p3", "famB", "VVVVV", labels="EEEEE"),
            ProteinRecord("p4", "famB", "VVVVA", labels="EEEEC"),
        ]
        preprocessor = DatasetPreprocessor(min_length=4)
        cleaned = preprocessor.clean(records)
        splitter = ProteinLevelSplitter(train_fraction=0.5, val_fraction=0.0, test_fraction=0.5, random_state=1)
        splits = splitter.split(cleaned)

        encoder = AminoAcidEncoder()
        temp_path = ROOT / "results" / "runs" / "smoke_test_workspace"
        temp_path.mkdir(parents=True, exist_ok=True)
        save_split_records(temp_path, splits)
        reloaded = load_split_records(temp_path)
        train_sequences = [encoder.encode(record.sequence) for record in reloaded["train"]]
        test_sequences = [encoder.encode(record.sequence) for record in reloaded["test"]]

        model = DiscreteHMM(num_states=2, max_iter=5, random_state=2)
        model.fit(train_sequences)
        model_path = temp_path / "unsupervised.json"
        model.save(model_path)
        loaded_model = DiscreteHMM.load(model_path)
        decoded = loaded_model.decode(test_sequences[0], protein_id=reloaded["test"][0].protein_id)
        self.assertEqual(len(decoded.states), len(test_sequences[0]))

        reference = SemiSupervisedHMM(max_iter=5, random_state=2)
        reference.fit(
            train_sequences,
            [record.labels or "" for record in reloaded["train"]],
        )
        predicted_labels = reference.predict_labels(test_sequences[0])
        self.assertEqual(len(predicted_labels), len(reloaded["test"][0].labels or ""))
        self.assertGreaterEqual(
            q3_accuracy(reloaded["test"][0].labels or "", predicted_labels),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()
