"""Check sentence windows using fixed scores, without downloading DeBERTa."""

import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from factcheck import EntailmentFactChecker
from report import window_test
from report.window_test import WindowFactChecker, evaluate


class WindowExperimentTests(unittest.TestCase):
    def test_pair_can_support_a_claim_when_singles_do_not(self):
        model = Mock()
        model.check_entailment.side_effect = lambda premise, hypothesis: (
            0.9 if premise == "Maya invented a radio. She lived in Ohio." else 0.1
        )
        fact = "Maya invented a radio and lived in Ohio."
        passages = [{"text": "Maya invented a radio. She lived in Ohio."}]
        baseline = EntailmentFactChecker(model)
        windows = WindowFactChecker(model)
        self.assertEqual(baseline.predict(fact, passages), "NS")
        self.assertEqual(windows.predict(fact, passages), "S")
        self.assertEqual(windows.model_calls, 2)
        self.assertEqual(windows.window_model_calls, 1)

    def test_only_adjacent_sentences_are_joined(self):
        model = Mock()
        model.check_entailment.return_value = 0.1
        checker = WindowFactChecker(model, use_pruning=False)
        checker.predict("claim", [{"text": "First sentence. Second sentence. Third sentence."}])
        premises = [call.kwargs["premise"] for call in model.check_entailment.call_args_list]
        self.assertEqual(premises, [
            "First sentence.", "Second sentence.", "Third sentence.",
            "First sentence. Second sentence.", "Second sentence. Third sentence.",
        ])

    def test_windows_do_not_cross_passages(self):
        model = Mock()
        model.check_entailment.return_value = 0.1
        checker = WindowFactChecker(model, use_pruning=False)
        checker.predict("claim", [{"text": "First sentence."}, {"text": "Second sentence."}])
        self.assertEqual(checker.total_windows, 0)
        self.assertEqual(checker.window_model_calls, 0)
        self.assertEqual(checker.model_calls, 2)

    def test_zero_overlap_windows_are_pruned(self):
        model = Mock()
        checker = WindowFactChecker(model)
        self.assertEqual(checker.predict(
            "Radio invented in Ohio.", [{"text": "Volcanoes erupt. Lava flows."}]
        ), "NS")
        self.assertEqual(checker.total_windows, 1)
        self.assertEqual(checker.pruned_windows, 1)
        model.check_entailment.assert_not_called()

    def test_empty_evidence_is_not_supported(self):
        model = Mock()
        checker = WindowFactChecker(model)
        for passages in [[], [{"text": "  "}]]:
            with self.subTest(passages=passages):
                self.assertEqual(checker.predict("claim", passages), "NS")
        model.check_entailment.assert_not_called()

    def test_evaluation_counts_ir_as_not_supported_and_includes_window_calls(self):
        checker = Mock(model_calls=2, window_model_calls=1)
        checker.predict.side_effect = ["S", "NS", "NS"]
        examples = [SimpleNamespace(fact="claim", passages=[], label=label)
                    for label in ["S", "NS", "IR"]]
        result = evaluate(checker, examples)
        self.assertEqual(result["correct"], 3)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["model_calls"], 3)

    def test_main_prints_both_variants(self):
        model = Mock()
        model.check_entailment.side_effect = lambda premise, hypothesis: (
            0.9 if premise == "Maya invented a radio. She lived in Ohio." else 0.1
        )
        examples = [
            SimpleNamespace(
                fact="Maya invented a radio and lived in Ohio.", label="S",
                passages=[{"text": "Maya invented a radio. She lived in Ohio."}],
            ),
            SimpleNamespace(fact="Radio", label="IR", passages=[]),
        ]
        output = io.StringIO()
        with patch.multiple(
            window_test,
            read_passages=Mock(return_value={}),
            read_fact_examples=Mock(return_value=examples),
            AutoTokenizer=Mock(),
            AutoModelForSequenceClassification=Mock(),
            EntailmentModel=Mock(return_value=model),
        ), contextlib.redirect_stdout(output):
            window_test.main()
        self.assertIn("50.00%", output.getvalue())
        self.assertIn("100.00%", output.getvalue())
        self.assertIn("Max window", output.getvalue())


if __name__ == "__main__":
    unittest.main()
