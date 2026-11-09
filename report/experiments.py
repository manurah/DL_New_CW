"""Report experiments for Parts 1 and 2 (not part of the submission).

Uses the driver's evaluation semantics: all 221 facts, IR mapped to NS.

Part 1: threshold sweep per scorer, preprocessing ablation.
Part 2: scores every passage sentence once (no pruning) with all three NLI
probabilities, plus each sentence's word-recall overlap. From that cache it
evaluates pruning thresholds, decision rules and premise granularity
offline, and times real runs with and without pruning.

    venv/bin/python report/experiments.py   (from the repo root)
"""
import json
import os
import sys
import time

import nltk
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.getcwd())
from factcheck import EntailmentFactChecker, EntailmentModel, WordRecallThresholdFactChecker  # noqa: E402
from factchecking_main import read_fact_examples, read_passages  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiments_results.json")
torch.set_num_threads(max(1, os.cpu_count() or 1))


def gold(ex):
    return "NS" if ex.label == "IR" else ex.label


def acc(preds, golds):
    return sum(p == g for p, g in zip(preds, golds)) / len(golds)


def part1(examples, golds, res):
    grid = [round(x * 0.05, 2) for x in range(21)]
    fine = [round(x * 0.01, 2) for x in range(101)]
    sweep = {}
    for scorer in ["recall", "jaccard", "f1", "rouge"]:
        fc = WordRecallThresholdFactChecker(scorer=scorer)
        scores = []
        for ex in examples:
            fw = fc.preprocess(ex.fact)
            best = 0.0
            for p in ex.passages:
                best = max(best, fc._compute_score(fw, fc.preprocess(p["text"]), ex.fact, p["text"]))
            scores.append(best)
        rows = [(t, acc(["S" if s >= t else "NS" for s in scores], golds)) for t in fine]
        bt, ba = max(rows, key=lambda r: (r[1], -abs(r[0] - 0.5)))
        sweep[scorer] = {"best_threshold": bt, "best_accuracy": ba,
                         "curve": [(t, a) for t, a in rows if t in grid]}
        print("P1", scorer, bt, round(ba, 4), flush=True)
    res["part1_scorers"] = sweep

    ablation = []
    for use_nltk in [False, True]:
        for sw in [False, True]:
            for stem in [False, True]:
                fc = WordRecallThresholdFactChecker(use_nltk=use_nltk, remove_stopwords=sw, use_stemming=stem)
                scores = []
                for ex in examples:
                    fw = fc.preprocess(ex.fact)
                    scores.append(max([fc._compute_score(fw, fc.preprocess(p["text"]), ex.fact, p["text"]) for p in ex.passages] or [0.0]))
                at065 = acc(["S" if s >= 0.65 else "NS" for s in scores], golds)
                best = max(((t, acc(["S" if s >= t else "NS" for s in scores], golds)) for t in fine), key=lambda r: r[1])
                ablation.append({"nltk": use_nltk, "stopwords": sw, "stem": stem,
                                 "acc_at_0.65": at065, "best_threshold": best[0], "best_acc": best[1]})
                print("P1 ablation", use_nltk, sw, stem, round(at065, 4), best, flush=True)
    res["part1_ablation"] = ablation


def sentences_of(ex):
    for p in ex.passages:
        text = " ".join(p["text"].split())
        if not text:
            continue
        for s in nltk.sent_tokenize(text):
            s = s.strip()
            if s:
                yield s


def nli_probs(model, tok, premise, hypothesis):
    with torch.no_grad():
        enc = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=512)
        return torch.softmax(model(**enc).logits, dim=-1)[0].tolist()


def part2(examples, golds, res):
    name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).eval()
    l2i = model.config.label2id
    ei, ni, ci = l2i["entailment"], l2i["neutral"], l2i["contradiction"]
    ov = WordRecallThresholdFactChecker(scorer="recall")

    # Cache: per fact, list of (overlap, p_ent, p_neu, p_con) per sentence
    cache, t0 = [], time.perf_counter()
    for i, ex in enumerate(examples):
        fw = ov.preprocess(ex.fact)
        rows = []
        for s in sentences_of(ex):
            o = ov._compute_score(fw, ov.preprocess(s), ex.fact, s) if fw else 1.0
            pr = nli_probs(model, tok, s, ex.fact)
            rows.append((o, pr[ei], pr[ni], pr[ci]))
        cache.append(rows)
        if i % 20 == 0:
            print(f"P2 cache {i}/{len(examples)} {time.perf_counter() - t0:.0f}s", flush=True)
    unpruned_secs = time.perf_counter() - t0
    n_sent = sum(len(r) for r in cache)

    # Passage-level premises (whole chunk, truncated to 512 tokens)
    pas = []
    for ex in examples:
        pas.append([nli_probs(model, tok, " ".join(p["text"].split()), ex.fact)[ei] for p in ex.passages if p["text"].strip()])

    def kept(rows, pt):
        return [r for r in rows if r[0] >= pt]

    def threshold_curve(maxes):
        out = [(t / 100, acc(["S" if m >= t / 100 else "NS" for m in maxes], golds)) for t in range(1, 100)]
        return out

    # Pruning sweep with threshold rule
    prune = []
    for pt in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]:
        maxes = [max([r[1] for r in kept(rows, pt)] or [0.0]) for rows in cache]
        calls = sum(len(kept(rows, pt)) for rows in cache)
        curve = threshold_curve(maxes)
        at038 = acc(["S" if m >= 0.38 else "NS" for m in maxes], golds)
        best = max(curve, key=lambda r: r[1])
        lost = sum(1 for rows, g in zip(cache, golds) if g == "S"
                   and max([r[1] for r in rows] or [0]) >= 0.38 and max([r[1] for r in kept(rows, pt)] or [0]) < 0.38)
        prune.append({"prune": pt, "model_calls": calls, "frac_calls": calls / n_sent,
                      "acc_at_0.38": at038, "best_threshold": best[0], "best_acc": best[1],
                      "supported_lost_to_pruning": lost})
        print("P2 prune", prune[-1], flush=True)

    # Decision rules at pruning 0.05
    pt = 0.05
    k = [kept(rows, pt) for rows in cache]
    rules = {}
    argmax_any = ["S" if any(r[1] >= max(r[2], r[3]) for r in rows) else "NS" for rows in k]
    rules["discrete_argmax_any_sentence"] = {"acc": acc(argmax_any, golds)}
    ent = [max([r[1] for r in rows] or [0.0]) for rows in k]
    c = threshold_curve(ent)
    rules["max_p_entail_threshold"] = {"acc_at_0.38": acc(["S" if m >= 0.38 else "NS" for m in ent], golds),
                                       "best": max(c, key=lambda r: r[1]), "curve": c}
    diff = [max([r[1] - r[3] for r in rows] or [-1.0]) for rows in k]
    cd = [(t / 100, acc(["S" if d >= t / 100 else "NS" for d in diff], golds)) for t in range(-50, 100)]
    rules["max_(p_entail-p_contra)_threshold"] = {"best": max(cd, key=lambda r: r[1])}
    nonneu = [max([r[1] / (r[1] + r[3] + 1e-9) for r in rows if r[1] >= 0.05] or [0.0]) for rows in k]
    cn = threshold_curve(nonneu)
    rules["entail_vs_contra_ratio_threshold"] = {"best": max(cn, key=lambda r: r[1])}
    meanent = [float(np.mean([r[1] for r in rows])) if rows else 0.0 for rows in k]
    cm = threshold_curve(meanent)
    rules["mean_p_entail_threshold"] = {"best": max(cm, key=lambda r: r[1])}
    res["part2_rules"] = rules
    print("P2 rules", {kk: (v.get("acc") or v.get("best")) for kk, v in rules.items()}, flush=True)

    # Premise granularity
    pmax = [max(v or [0.0]) for v in pas]
    cp = threshold_curve(pmax)
    res["part2_granularity"] = {
        "sentence_unpruned": max(threshold_curve([max([r[1] for r in rows] or [0.0]) for rows in cache]), key=lambda r: r[1]),
        "passage_chunk": max(cp, key=lambda r: r[1]),
        "passage_chunk_at_0.38": acc(["S" if m >= 0.38 else "NS" for m in pmax], golds),
        "passage_calls": sum(len(v) for v in pas),
    }
    print("P2 granularity", res["part2_granularity"], flush=True)

    res["part2_prune"] = prune
    res["part2_counts"] = {"sentences": n_sent, "unpruned_cache_seconds": unpruned_secs}

    # Real timed runs through the group's class
    em = EntailmentModel(model, tok, False)
    for use_pruning in [True, False]:
        fc = EntailmentFactChecker(em, use_pruning=use_pruning)
        t = time.perf_counter()
        preds = [fc.predict(ex.fact, ex.passages) for ex in examples]
        res[f"timed_pruning_{use_pruning}"] = {"seconds": time.perf_counter() - t, "acc": acc(preds, golds),
                                               "model_calls": fc.model_calls, "pruned": fc.pruned_sentences,
                                               "total_sentences": fc.total_sentences}
        print("P2 timed", use_pruning, res[f"timed_pruning_{use_pruning}"], flush=True)


def main():
    passages = read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl")
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl", passages)
    golds = [gold(e) for e in examples]
    res = {"n": len(examples), "n_S": golds.count("S"), "n_NS": golds.count("NS"),
           "label_counts": {l: sum(e.label == l for e in examples) for l in ("S", "NS", "IR")}}
    part1(examples, golds, res)
    json.dump(res, open(OUT, "w"), indent=1)
    part2(examples, golds, res)
    json.dump(res, open(OUT, "w"), indent=1)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
