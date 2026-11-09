"""Export the entailment model's errors (false positives / false negatives)
to a CSV so the group can examine and categorise them by hand.

Not part of the submission. Mirrors EntailmentFactChecker.score_fact in
CM3604-factchecker/factcheck.py exactly, but also records which evidence
sentences produced the scores.

Run from the repo folder:
    venv/bin/python report/export_errors.py   (from the repo root)
"""
import csv
import os
import sys

import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.getcwd())
from factcheck import EntailmentFactChecker, EntailmentModel  # noqa: E402
from factchecking_main import read_fact_examples, read_passages  # noqa: E402

TOP_K = 3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entailment_errors.csv")


def scored_sentences(checker, fact, passages):
    """Same logic as EntailmentFactChecker.score_fact, returning every
    (score, sentence, passage_title) that reached DeBERTa."""
    fact_words = checker.overlap_checker.preprocess(fact) if checker.use_pruning else None
    out = []
    for passage in passages:
        text = " ".join(passage["text"].split())
        if not text:
            continue
        for sentence in nltk.sent_tokenize(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            if checker.use_pruning and fact_words:
                sw = checker.overlap_checker.preprocess(sentence)
                if checker.overlap_checker._compute_score(fact_words, sw, fact, sentence) < checker.pruning_threshold:
                    continue
            score = checker.ent_model.check_entailment(premise=sentence, hypothesis=fact)
            out.append((score, sentence, passage["title"]))
    out.sort(key=lambda t: t[0], reverse=True)
    return out


def main():
    passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)

    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name)
    checker = EntailmentFactChecker(EntailmentModel(model, tok, False))

    rows, correct = [], 0
    for i, ex in enumerate(examples, 1):
        gold = "NS" if ex.label == "IR" else ex.label
        ranked = scored_sentences(checker, ex.fact, ex.passages)
        best = ranked[0][0] if ranked else 0.0
        pred = "S" if best >= checker.threshold else "NS"
        if pred == gold:
            correct += 1
            continue
        row = {
            "error_type": "FP" if pred == "S" else "FN",
            "example_index": i,
            "fact": ex.fact,
            "gold_label": gold,
            "original_label": ex.label,
            "predicted_label": pred,
            "max_entailment_score": round(best, 4),
            "threshold": checker.threshold,
            "sentences_scored": len(ranked),
            "passage_titles": "; ".join(sorted({p["title"] for p in ex.passages})),
        }
        for k in range(TOP_K):
            s = ranked[k] if k < len(ranked) else (None, "", "")
            row[f"top{k + 1}_score"] = "" if s[0] is None else round(s[0], 4)
            row[f"top{k + 1}_sentence"] = s[1]
        row["category (group fills in)"] = ""
        row["notes (group fills in)"] = ""
        rows.append(row)
        print(f"[{i}/{len(examples)}] {row['error_type']}: {ex.fact}", flush=True)

    rows.sort(key=lambda r: (r["error_type"], r["example_index"]))
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    fp = sum(r["error_type"] == "FP" for r in rows)
    print(f"\nAccuracy check: {correct}/{len(examples)} = {correct / len(examples):.4f}")
    print(f"False positives: {fp}  False negatives: {len(rows) - fp}")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
