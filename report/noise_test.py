"""Test passage noise removal for Part 2 (not part of the submission).

Variant raw:    current factcheck.py behaviour (whitespace normalisation only)
Variant tags:   split on <s>/</s> markup first, then sentence-split each segment
Variant clean:  as tags, plus strip leading section headers like
                "Biography.:Inventions." from each segment
Pruning 0.05 and max P(entailment), as in the submitted system.

    venv/bin/python report/noise_test.py   (from the repo root)
"""
import json
import os
import re
import sys
import time

import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.getcwd())
from factcheck import EntailmentModel, WordRecallThresholdFactChecker  # noqa: E402
from factchecking_main import read_fact_examples, read_passages  # noqa: E402

HEADER = re.compile(r"^(?:[A-Z][\w'’ ()-]{0,40}\.(?::[\w'’ ,()-]{1,60}\.)*)\s+(?=[A-Z\"(-])")


def segments(text, variant):
    if variant == "raw":
        return [" ".join(text.split())]
    segs = [" ".join(s.split()) for s in re.split(r"</?s>", text)]
    if variant == "clean":
        segs = [HEADER.sub("", s, count=1) if re.match(r"^[A-Z][\w'’ ()-]{0,40}\.:", s) or re.match(r"^[A-Z][\w ]{0,25}\. [A-Z]", s) else s for s in segs]
    return [s for s in segs if s.strip()]


def main():
    ex = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl"))
    golds = ["NS" if e.label == "IR" else e.label for e in ex]
    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    em = EntailmentModel(AutoModelForSequenceClassification.from_pretrained(name), AutoTokenizer.from_pretrained(name), False)
    ov = WordRecallThresholdFactChecker(scorer="recall")
    out = {}
    for variant in ["raw", "tags", "clean"]:
        t0, maxes, calls, sents = time.perf_counter(), [], 0, 0
        for e in ex:
            fw = ov.preprocess(e.fact)
            best = 0.0
            for p in e.passages:
                for seg in segments(p["text"], variant):
                    for s in nltk.sent_tokenize(seg):
                        s = s.strip()
                        if not s:
                            continue
                        sents += 1
                        if fw and ov._compute_score(fw, ov.preprocess(s), e.fact, s) < 0.05:
                            continue
                        calls += 1
                        best = max(best, em.check_entailment(premise=s, hypothesis=e.fact))
            maxes.append(best)
        curve = [(t / 100, sum((m >= t / 100) == (g == "S") for m, g in zip(maxes, golds)) / len(golds)) for t in range(1, 100)]
        at = dict(curve)[0.38]
        best = max(curve, key=lambda r: r[1])
        out[variant] = {"acc_at_0.38": at, "best": best, "calls": calls, "sentences": sents,
                        "seconds": time.perf_counter() - t0, "curve": curve}
        print(variant, {k: v for k, v in out[variant].items() if k != "curve"}, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "noise_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
