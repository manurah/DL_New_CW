"""Unit tests for word-overlap preprocessing, scores and predictions."""

import unittest
from unittest.mock import patch

from factcheck import WordRecallThresholdFactChecker


class WordOverlapPreprocessingTests(unittest.TestCase):
    def setUp(self):
        self.checker = WordRecallThresholdFactChecker()

    def test_lowercases_words(self):
        self.assertEqual(self.checker.preprocess("OHIO"), {"ohio"})

    def test_removes_punctuation(self):
        self.assertEqual(
            self.checker.preprocess("Ohio, London."), {"ohio", "london"}
        )

    def test_removes_stopwords(self):
        self.assertEqual(
            self.checker.preprocess("The radio is in Ohio."), {"radio", "ohio"}
        )

    def test_stems_word_variants(self):
        self.assertEqual(
            self.checker.preprocess("invented inventing invention"), {"invent"}
        )

    def test_tokenization_toggle(self):
        # Leave punctuation untouched so the two tokenizers can be compared.
        for use_nltk, expected in [(True, {"ohio", "."}), (False, {"ohio."})]:
            with self.subTest(use_nltk=use_nltk):
                checker = WordRecallThresholdFactChecker(
                    use_nltk=use_nltk,
                    remove_stopwords=False,
                    use_stemming=False,
                )
                self.assertEqual(checker.preprocess("Ohio."), expected)

    def test_stopword_removal_can_be_disabled(self):
        checker = WordRecallThresholdFactChecker(remove_stopwords=False)
        self.assertEqual(checker.preprocess("the radio"), {"the", "radio"})

    def test_stemming_can_be_disabled(self):
        checker = WordRecallThresholdFactChecker(use_stemming=False)
        self.assertEqual(checker.preprocess("inventing"), {"inventing"})


class WordOverlapScoringTests(unittest.TestCase):
    def test_scores_match_hand_calculated_values(self):
        fact = "radio metal patent"
        passage = "radio metal detector device"
        # Two shared words, three fact words, four passage words, five in union.
        expected_scores = {
            "recall": 2 / 3,
            "jaccard": 2 / 5,
            "f1": 4 / 7,
            "rouge": 4 / 7,
        }
        for scorer, expected in expected_scores.items():
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                score = checker._compute_score(
                    set(fact.split()), set(passage.split()), fact, passage
                )
                self.assertAlmostEqual(score, expected)

    def test_empty_fact_scores_zero(self):
        for scorer in ["recall", "jaccard", "f1", "rouge"]:
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                self.assertEqual(
                    checker._compute_score(set(), {"radio"}, "", "radio"), 0.0
                )

    def test_empty_passage_scores_zero(self):
        for scorer in ["recall", "jaccard", "f1", "rouge"]:
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                self.assertEqual(
                    checker._compute_score({"radio"}, set(), "radio", ""), 0.0
                )

    def test_unrelated_words_score_zero(self):
        for scorer in ["recall", "jaccard", "f1", "rouge"]:
            with self.subTest(scorer=scorer):
                checker = WordRecallThresholdFactChecker(scorer=scorer)
                self.assertEqual(
                    checker._compute_score({"radio"}, {"metal"}, "radio", "metal"),
                    0.0,
                )

    def test_unknown_scorer_raises_value_error(self):
        checker = WordRecallThresholdFactChecker(scorer="unknown")
        with self.assertRaises(ValueError):
            checker._compute_score({"radio"}, {"radio"}, "radio", "radio")


class WordOverlapPredictionTests(unittest.TestCase):
    def setUp(self):
        self.checker = WordRecallThresholdFactChecker()
        self.fact = "The radio was invented in Ohio."

    def test_threshold_boundary(self):
        passages = [{"title": "Radio", "text": self.fact}]
        for score, expected in [(0.649999, "NS"), (0.65, "S"), (0.650001, "S")]:
            with self.subTest(score=score):
                # Fix the score to isolate the default threshold decision.
                with patch.object(self.checker, "_compute_score", return_value=score):
                    self.assertEqual(self.checker.predict(self.fact, passages), expected)

    def test_one_supporting_passage_is_enough(self):
        passages = [
            {"title": "Volcano", "text": "Volcanoes release molten rock."},
            {"title": "Radio", "text": self.fact},
        ]
        self.assertEqual(self.checker.predict(self.fact, passages), "S")

    def test_no_passages_is_not_supported(self):
        self.assertEqual(self.checker.predict(self.fact, []), "NS")

    def test_empty_fact_is_not_supported(self):
        passages = [{"title": "Radio", "text": self.fact}]
        self.assertEqual(self.checker.predict("", passages), "NS")


if __name__ == "__main__":
    unittest.main()
