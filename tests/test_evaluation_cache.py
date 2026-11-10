"""Check portable file hashes and reuse of saved entailment scores."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import evaluate_entailment as evaluation


class EvaluationCacheTests(unittest.TestCase):
    def test_line_endings_do_not_change_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_bytes(b"first line\nsecond line\n")
            original = evaluation.file_hash(path)
            path.write_bytes(b"first line\r\nsecond line\r\n")
            self.assertEqual(evaluation.file_hash(path), original)

    def test_changed_content_changes_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.txt"
            path.write_bytes(b"original\n")
            original = evaluation.file_hash(path)
            path.write_bytes(b"changed\n")
            self.assertNotEqual(evaluation.file_hash(path), original)

    def test_second_run_reuses_cache_after_line_endings_change(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            labels = project / "labels.jsonl"
            passages = project / "passages.jsonl"
            sources = [project / "factcheck.py", project / "factchecking_main.py",
                       labels, passages]
            for path in sources:
                path.write_bytes(b"example content\nsecond line\n")

            cache = project / "results" / "scores.json"
            table = project / "results" / "thresholds.csv"
            examples = [
                SimpleNamespace(fact="supported claim", passages=[], label="S"),
                SimpleNamespace(fact="irrelevant claim", passages=[], label="IR"),
            ]
            checker = Mock(total_sentences=2, pruned_sentences=0, model_calls=2)
            checker.score_fact.side_effect = [0.9, 0.1]
            tokenizer = Mock()
            model = Mock()
            output = io.StringIO()

            with patch.multiple(
                evaluation,
                PROJECT=project, LABELS_PATH=labels, PASSAGES_PATH=passages,
                RESULTS_DIR=cache.parent, CACHE_PATH=cache, TABLE_PATH=table,
                read_passages=Mock(return_value={}),
                read_fact_examples=Mock(return_value=examples),
                AutoTokenizer=tokenizer,
                AutoModelForSequenceClassification=model,
                EntailmentModel=Mock(),
                EntailmentFactChecker=Mock(return_value=checker),
                tqdm=lambda items, **kwargs: items,
            ), contextlib.redirect_stdout(output):
                evaluation.main()
                saved_scores = cache.read_bytes()
                saved_table = table.read_bytes()
                self.assertNotIn(b"\r\n", saved_table)

                for path in sources:
                    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
                evaluation.main()

            self.assertIn("Using saved scores; no model inference needed.", output.getvalue())
            self.assertEqual(checker.score_fact.call_count, 2)
            tokenizer.from_pretrained.assert_called_once()
            model.from_pretrained.assert_called_once()
            self.assertEqual(cache.read_bytes(), saved_scores)
            self.assertEqual(table.read_bytes(), saved_table)


if __name__ == "__main__":
    unittest.main()
