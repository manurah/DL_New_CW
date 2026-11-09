import csv
import hashlib
import json
from pathlib import Path
from time import perf_counter

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples


# Locate files relative to this script, regardless of the working directory.
PROJECT = Path(__file__).resolve().parent
LABELS_PATH = PROJECT / "data" / "dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = (
    PROJECT / "data" / "passages_bm25_ChatGPT_humfacts.jsonl"
)

# Keep pruning fixed while evaluating the entailment decision threshold.
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
PRUNING_THRESHOLD = 0.05

# Store experiment outputs separately from the supplied source data.
RESULTS_DIR = PROJECT / "results"
CACHE_PATH = RESULTS_DIR / "entailment_pruning_005_scores.json"
TABLE_PATH = RESULTS_DIR / "entailment_pruning_005_thresholds.csv"


def file_hash(path):
    # Fingerprint files so changed code or data cannot silently reuse old scores.
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # Record the settings and source files that determine these predictions.
    signature = {
        "model": MODEL_NAME,
        "device": "cpu",
        "pruning_threshold": PRUNING_THRESHOLD,
        "factcheck_sha256": file_hash(PROJECT / "factcheck.py"),
        "driver_sha256": file_hash(PROJECT / "factchecking_main.py"),
        "labels_sha256": file_hash(LABELS_PATH),
        "passages_sha256": file_hash(PASSAGES_PATH)
    }

    saved_run = None

    # Reuse saved scores only if the code, data and settings still match.
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as file:
            candidate = json.load(file)

        if candidate["signature"] == signature:
            saved_run = candidate
            print("Using saved scores; no model inference needed.")
        else:
            print("Code, data or settings changed. Recomputing scores.")

    if saved_run is None:
        # Load exactly the same examples as the coursework driver.
        passage_lookup = read_passages(str(PASSAGES_PATH))
        examples = read_fact_examples(
            str(LABELS_PATH),
            passage_lookup
        )

        if not examples:
            raise ValueError("No examples were loaded.")

        print(f"Loaded {len(examples)} examples.")

        # Load the checkpoint once and use CPU inference for this experiment.
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME
        )
        wrapper = EntailmentModel(model, tokenizer, cuda=False)

        checker = EntailmentFactChecker(
            wrapper,
            threshold=0.5,
            pruning_threshold=PRUNING_THRESHOLD,
            use_pruning=True
        )

        records = []
        start = perf_counter()

        for index, example in enumerate(
            tqdm(examples, desc="Scoring claims")
        ):
            # Compute the maximum entailment probability once per claim.
            # Gold labels are not passed into the scoring method.
            score = checker.score_fact(
                example.fact,
                example.passages
            )

            # Match the driver's evaluation rule: IR becomes NS.
            gold = "NS" if example.label == "IR" else example.label

            # Preserve full evidence for later inspection.
            records.append({
                "example_id": index,
                "fact": example.fact,
                "original_label": example.label,
                "gold": gold,
                "score": score,
                "passages": example.passages
            })

        elapsed_seconds = perf_counter() - start

        saved_run = {
            "signature": signature,
            "evaluation_seconds": elapsed_seconds,
            "total_sentences": checker.total_sentences,
            "pruned_sentences": checker.pruned_sentences,
            "model_calls": checker.model_calls,
            "records": records
        }

        # Save scores so threshold comparisons never need model reruns.
        with CACHE_PATH.open("w", encoding="utf-8") as file:
            json.dump(saved_run, file, indent=2, ensure_ascii=False)

        print(f"Scores saved to: {CACHE_PATH}")

    records = saved_run["records"]
    total = len(records)

    # Evaluate thresholds from 0.01 to 1.00 using the saved probabilities.
    # Exclude zero: a zero score also represents no retained support.
    rows = []

    for step in range(1, 101):
        threshold = step / 100
        tp = tn = fp = fn = 0

        for record in records:
            prediction = (
                "S" if record["score"] >= threshold else "NS"
            )

            if record["gold"] == "S":
                if prediction == "S":
                    tp += 1
                else:
                    fn += 1
            else:
                if prediction == "S":
                    fp += 1
                else:
                    tn += 1

        rows.append({
            "threshold": threshold,
            "correct": tp + tn,
            "total": total,
            "accuracy": (tp + tn) / total,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn
        })

    # Export every tested threshold, not just the best result.
    with TABLE_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    baseline = next(row for row in rows if row["threshold"] == 0.5)

    # If several cutoffs tie, prefer the one closest to the initial 0.5.
    best = max(
        rows,
        key=lambda row: (
            row["correct"],
            -abs(row["threshold"] - 0.5)
        )
    )

    print("\nResults with pruning fixed at 0.05:")

    for name, row in [("Initial threshold", baseline), ("Best tested", best)]:
        print(
            f"{name}: threshold={row['threshold']:.2f}, "
            f"accuracy={row['correct']}/{total} "
            f"({row['accuracy']:.2%}), "
            f"FP={row['false_positives']}, "
            f"FN={row['false_negatives']}"
        )

    print(
        "\nScoring time, excluding model loading: "
        f"{saved_run['evaluation_seconds'] / 60:.2f} minutes"
    )
    print("Evidence sentences:", saved_run["total_sentences"])
    print("Pruned sentences:", saved_run["pruned_sentences"])
    print("DeBERTa calls:", saved_run["model_calls"])
    print("Threshold table:", TABLE_PATH)


if __name__ == "__main__":
    main()