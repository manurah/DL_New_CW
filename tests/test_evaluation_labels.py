"""Check evaluation labels without importing the optional ML dependencies.

These checks exercise the actual evaluator functions, extracted with ast.
They do not run either classifier or replace full driver reproduction.
"""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


PROJECT = Path(__file__).resolve().parents[1]


def load_function(filename, name, namespace=None):
    tree = ast.parse((PROJECT / filename).read_text(encoding="utf-8"))
    function = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    module = ast.Module(body=[function], type_ignores=[])
    scope = dict(namespace or {})
    exec(compile(module, str(PROJECT / filename), "exec"), scope)
    return scope[name]


class RecordedPredictions:
    def __init__(self, predictions):
        self.predictions = iter(predictions)
        self.calls = 0

    def predict(self, fact, passages):
        self.calls += 1
        return next(self.predictions)


class EvaluationLabelTests(unittest.TestCase):
    def setUp(self):
        self.evaluate = load_function("threshold_sweep.py", "evaluate")

    def test_ir_is_included_as_not_supported(self):
        examples = [SimpleNamespace(fact="claim", passages=[], label="IR")]
        for prediction, expected in [("NS", 1.0), ("S", 0.0)]:
            with self.subTest(prediction=prediction):
                checker = RecordedPredictions([prediction])
                self.assertEqual(self.evaluate(checker, examples), expected)
                self.assertEqual(checker.calls, 1)

    def test_mixed_labels_use_all_examples(self):
        examples = [
            SimpleNamespace(fact="claim", passages=[], label=label)
            for label in ["S", "NS", "IR"]
        ]
        checker = RecordedPredictions(["S", "S", "NS"])
        self.assertAlmostEqual(self.evaluate(checker, examples), 2 / 3)
        self.assertEqual(checker.calls, 3)

    def test_matches_actual_driver_on_all_saved_records(self):
        saved = json.loads(
            (PROJECT / "results" / "entailment_pruning_005_scores.json")
            .read_text(encoding="utf-8")
        )
        records = saved["records"]
        examples = [
            SimpleNamespace(
                fact=row["fact"], passages=row["passages"],
                label=row["original_label"]
            )
            for row in records
        ]
        predictions = ["S" if row["score"] >= 0.38 else "NS" for row in records]
        captured = []
        driver_evaluate = load_function(
            "factchecking_main.py", "predict_two_classes",
            {"List": list, "FactExample": SimpleNamespace,
             "tqdm": lambda items: items,
             "print_eval_stats": lambda matrix, labels: captured.append(matrix)}
        )
        driver_checker = RecordedPredictions(predictions)
        driver_evaluate(examples, driver_checker)
        matrix = captured[0]
        driver_total = sum(sum(row) for row in matrix)
        driver_accuracy = (matrix[0][0] + matrix[1][1]) / driver_total
        helper_checker = RecordedPredictions(predictions)
        self.assertEqual(len(examples), 221)
        self.assertEqual(matrix, [[88, 24], [10, 99]])
        self.assertEqual(self.evaluate(helper_checker, examples), driver_accuracy)
        self.assertEqual(helper_checker.calls, driver_checker.calls)
        self.assertEqual(helper_checker.calls, 221)


if __name__ == "__main__":
    unittest.main()
