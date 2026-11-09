# Part 2 handoff — from Sandali to Dilshan

Part 1 (Word Overlap) is complete and merged into main. This doc gives you everything you need to start Part 2 without re-reading my code.

## My final numbers

- Accuracy on main driver: **81.0%** (6 points above the 75% floor)
- Chosen threshold: **0.65**
- Best scorer: **recall** (compared against Jaccard, F1 overlap, ROUGE-1 in `threshold_sweep.py`)
- Preprocessing: NLTK tokenisation + English stopword removal + Porter stemming

## What you can use for pruning

My `WordRecallThresholdFactChecker` in `factcheck.py` is already set up to be reused. For your pruning step:

```python
from factcheck import WordRecallThresholdFactChecker

# Create a pruner. The threshold here does NOT matter for pruning —
# you'll compare scores directly, not use the S/NS output.
pruner = WordRecallThresholdFactChecker()

# For each fact + passage pair you're about to feed to DeBERTa:
fact_words = pruner.preprocess(fact)
passage_words = pruner.preprocess(passage["text"])
score = pruner._compute_score(fact_words, passage_words, fact, passage["text"])

# Skip this pair if the overlap is clearly too low to be entailed.
PRUNE_THRESHOLD = 0.1   # starting point — tune this
if score < PRUNE_THRESHOLD:
    continue   # don't call DeBERTa on this one
```

### Important notes

- **Use a LOW threshold for pruning (~0.1), not my classification threshold of 0.65.** The brief says "if there is very low word overlap, then entailment is extremely unlikely." Pruning is aggressive filtering, not classifying.
- My preprocessing + recall scorer are baked in as defaults — you don't need to pass any arguments.
- Measure both runtime and accuracy with/without pruning. The rubric rewards "measurable improvements in efficiency and/or predictive performance."

## What the rubric wants from Part 2

Target: 83% accuracy minimum.

Grade-A descriptors:
1. **Decision strategy** — try both discrete entailment (argmax) and probability threshold on the entailment class. Compare. Pick the better one and justify.
2. **Sentence splitting** — chunk each passage into sentences before running DeBERTa. NLTK's `sent_tokenize` works.
3. **Pruning** — use my scorer as above. Report accuracy and runtime with vs without pruning.
4. **Memory discipline** — wrap inference in `torch.no_grad()`, call `model.eval()`, batch sensibly, keep tensors off GPU after use.
5. **Experimental analysis** — numbers justifying each choice.

## File coordination

We both edit `factcheck.py`:
- You own: `EntailmentFactChecker`, `EntailmentModel.check_entailment`
- I own: `WordRecallThresholdFactChecker`, `_compute_score`, `preprocess`

**Please WhatsApp me before you commit** so we don't collide. Pull latest from main before you start.

## Reference material

- My code: `factcheck.py` (classes mentioned above)
- My experiments: `threshold_sweep.py` — reusable `evaluate()` function you can import for your own sweeps
- README: setup steps if you haven't done them yet
- Brief: Part 2 section pages 4–5

Ping me if the pruning integration gets fiddly — I know how my code expects to be called and can save you time.

Sandali