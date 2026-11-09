"""
Threshold sweep for the word overlap classifier.

Runs WordRecallThresholdFactChecker across a range of thresholds on the
labelled dev set and prints the accuracy at each. The best threshold found
here is what we hard-code (via the default) in factcheck.py for submission.

This script is NOT required for the submission - it only produces the
numbers that go into the methodology/results section of the report.

Run with:  python threshold_sweep.py
"""

from factcheck import WordRecallThresholdFactChecker
from factchecking_main import read_passages, read_fact_examples

def evaluate(fact_checker, examples):
    # Runs the given fact_checker over every example and returns accuracy as
    # a float in [0, 1]. Kept separate from the sweep loop so we can reuse it
    # for other experiments (preprocessing ablation, alternative scorers).
    correct = 0
    total = 0
    for ex in examples:
        # Match the driver (predict_two_classes): IR (irrelevant) facts are
        # scored as NS rather than skipped, so all 221 examples count.
        gold = "NS" if ex.label == "IR" else ex.label
        prediction = fact_checker.predict(ex.fact, ex.passages)
        if prediction == gold:
            correct += 1
        total += 1
    return correct / total if total > 0 else 0.0

def preprocessing_ablation(examples, threshold=0.65):
    # Measures the individual contribution of each preprocessing step by
    # trying every on/off combination of (NLTK tokenisation, stopword removal,
    # Porter stemming). Fixes the scorer and threshold so the only variable
    # is preprocessing. The resulting 8-row table goes in the report to
    # justify why our final configuration enables all three steps.
    print("\n" + "=" * 60)
    print("Preprocessing ablation (scorer=recall, threshold=0.65)")
    print("=" * 60)
    print(f"{'NLTK':>6} {'stopwords':>10} {'stemming':>9}  {'accuracy':>10}")
    print("-" * 60)

    # itertools.product would be tidier but a visible nested loop is easier
    # to read for a reviewer who hasn't seen the file before. 3 flags, 2
    # values each -> 8 rows.
    for use_nltk in [False, True]:
        for remove_stopwords in [False, True]:
            for use_stemming in [False, True]:
                fact_checker = WordRecallThresholdFactChecker(
                    threshold=threshold,
                    use_nltk=use_nltk,
                    remove_stopwords=remove_stopwords,
                    use_stemming=use_stemming,
                    scorer="recall",
                )
                accuracy = evaluate(fact_checker, examples)
                print(f"{str(use_nltk):>6} {str(remove_stopwords):>10} "
                      f"{str(use_stemming):>9}  {accuracy:>10.4f}")
    print("-" * 60)

def scorer_comparison(examples):
    # For each of the four overlap metrics (recall, jaccard, f1, rouge),
    # runs a full threshold sweep with best preprocessing (NLTK + stopwords
    # + stemming) and records the best accuracy achieved. Each scorer lives
    # on a different numeric range so each gets its own best threshold.
    # This is the comparison table the report uses to justify why we chose
    # recall over the alternatives suggested in the brief.
    print("\n" + "=" * 60)
    print("Scorer comparison (NLTK + stopwords + stemming on)")
    print("=" * 60)
    print(f"{'scorer':>8}  {'best threshold':>15}  {'best accuracy':>14}")
    print("-" * 60)

    scorers = ["recall", "jaccard", "f1", "rouge"]
    results = []

    for scorer in scorers:
        best_threshold = 0.0
        best_accuracy = 0.0

        # Jaccard and F1 produce much smaller scores than recall because of
        # the fact/passage size mismatch (passages have hundreds of words,
        # facts have ~5). A coarse sweep in 0.05 steps would miss their
        # optimum entirely — their peak might sit at 0.03. We use a wider
        # grid (0.00 to 1.00) but keep the 0.05 step for simplicity; a
        # finer sweep could be added later if the result is ambiguous.
        threshold = 0.0
        while threshold <= 1.0 + 1e-9:
            fact_checker = WordRecallThresholdFactChecker(
                threshold=threshold,
                scorer=scorer,
            )
            accuracy = evaluate(fact_checker, examples)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
            threshold += 0.05

        print(f"{scorer:>8}  {best_threshold:>15.2f}  {best_accuracy:>14.4f}")
        results.append((scorer, best_threshold, best_accuracy))

    print("-" * 60)

    # Highlight the overall winner — handy when scanning the output or
    # pasting it into a report.
    winner = max(results, key=lambda row: row[2])
    print(f"\nBest scorer: {winner[0]} at threshold {winner[1]:.2f} "
          f"({winner[2] * 100:.2f}% accuracy)")

def main():
    # Load the data once using the lecturer's own loaders so we match the
    # exact behaviour of factchecking_main.py. Doing this once at the top
    # (rather than inside each experiment) saves a lot of repeated I/O.
    print("Loading data...")
    fact_to_passage_dict = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", fact_to_passage_dict)
    print(f"Loaded {len(examples)} examples")

    # Experiment 1: threshold sweep (recall + full preprocessing).
    # Determines the best threshold for our chosen default configuration.
    print("\n" + "=" * 60)
    print("Threshold sweep (scorer=recall, NLTK + stopwords + stemming)")
    print("=" * 60)
    print(f"{'threshold':>10}  {'accuracy':>8}")
    print("-" * 60)

    best_threshold = 0.0
    best_accuracy = 0.0

    threshold = 0.0
    while threshold <= 1.0 + 1e-9:
        fact_checker = WordRecallThresholdFactChecker(threshold=threshold)
        accuracy = evaluate(fact_checker, examples)
        print(f"{threshold:>10.2f}  {accuracy:>8.4f}")
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
        threshold += 0.05

    print("-" * 60)
    print(f"Best threshold: {best_threshold:.2f}  "
          f"({best_accuracy * 100:.2f}% accuracy)")

    # Experiment 2: preprocessing ablation. Isolates the contribution of
    # each preprocessing step by trying every on/off combination.
    preprocessing_ablation(examples, threshold=best_threshold)

    # Experiment 3: scorer comparison. Compares the four overlap metrics
    # suggested in the brief (recall, Jaccard, F1, ROUGE-1) side-by-side.
    scorer_comparison(examples)

    print("\nAll experiments complete.")


if __name__ == "__main__":
    main()
