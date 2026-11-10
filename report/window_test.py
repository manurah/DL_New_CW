"""Compare single-sentence evidence with singles plus adjacent sentence pairs.

Run from the repository root: python report/window_test.py
Uses CPU, the existing thresholds and the same 221 development examples.
The submitted classifier and its defaults are unchanged.
"""

import os
import sys
from pathlib import Path
from time import perf_counter

import nltk
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from factcheck import EntailmentFactChecker, EntailmentModel
from factchecking_main import read_fact_examples, read_passages


class WindowFactChecker(EntailmentFactChecker):
    def __init__(self, ent_model, **kwargs):
        super().__init__(ent_model, **kwargs)
        self.total_windows = 0
        self.pruned_windows = 0
        self.window_model_calls = 0

    def score_fact(self, fact, passages):
        # Keep the original single-sentence evidence, then try neighbouring pairs.
        best_score = super().score_fact(fact, passages)
        fact_words = self.overlap_checker.preprocess(fact) if self.use_pruning else None

        for passage in passages:
            text = " ".join(passage["text"].split())
            sentences = [s.strip() for s in nltk.sent_tokenize(text) if s.strip()]
            for first, second in zip(sentences, sentences[1:]):
                window = first + " " + second
                self.total_windows += 1

                if self.use_pruning and fact_words:
                    words = self.overlap_checker.preprocess(window)
                    overlap = self.overlap_checker._compute_score(
                        fact_words, words, fact, window
                    )
                    if overlap < self.pruning_threshold:
                        self.pruned_windows += 1
                        continue

                score = self.ent_model.check_entailment(premise=window, hypothesis=fact)
                self.window_model_calls += 1
                best_score = max(best_score, score)

        return best_score


def evaluate(checker, examples):
    correct = 0
    start = perf_counter()
    for example in tqdm(examples, desc=type(checker).__name__):
        gold = "NS" if example.label == "IR" else example.label
        correct += checker.predict(example.fact, example.passages) == gold
    return {
        "correct": correct,
        "total": len(examples),
        "model_calls": checker.model_calls + getattr(checker, "window_model_calls", 0),
        "seconds": perf_counter() - start,
    }


def main():
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    lookup = read_passages(str(PROJECT / "data/passages_bm25_ChatGPT_humfacts.jsonl"))
    examples = read_fact_examples(str(PROJECT / "data/dev_labeled_ChatGPT.jsonl"), lookup)
    if not examples:
        raise ValueError("No examples were loaded.")

    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    wrapper = EntailmentModel(model, tokenizer, cuda=False)
    baseline = EntailmentFactChecker(wrapper)
    windows = WindowFactChecker(wrapper)
    print(f"Model: {name}; device: CPU; examples: {len(examples)}")
    print(f"Decision threshold: {baseline.threshold}; pruning: {baseline.pruning_threshold}")

    results = [(1, evaluate(baseline, examples)), (2, evaluate(windows, examples))]
    print("\nMax window   Correct     Accuracy   Model calls   Seconds")
    for size, result in results:
        count = f"{result['correct']}/{result['total']}"
        accuracy = result["correct"] / result["total"]
        print(f"{size:<12} {count:<11} {accuracy:<10.2%} "
              f"{result['model_calls']:<13} {result['seconds']:.2f}")
    print("Window 2 includes all single sentences plus adjacent pairs within each passage.")
    print("Times exclude model loading; both variants use the existing 512-token limit.")


if __name__ == "__main__":
    main()
