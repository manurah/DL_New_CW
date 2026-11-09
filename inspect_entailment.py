from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)
from factcheck import EntailmentModel


# Load the same pretrained checkpoint used by the coursework driver.
model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Wrap the loaded model using our implementation from factcheck.py.
# This test runs on the CPU.
entailment_model = EntailmentModel(model, tokenizer, cuda=False)

# Keep the claim fixed while changing the evidence relationship.
claim = "Maya was born in London."

examples = [
    ("Support", "Maya was born in London in 1985."),
    ("Contradiction", "Maya was not born in London."),
    ("Neutral", "Maya enjoys playing the piano.")
]

# Check that the wrapper returns a valid Python probability for each pair.
for description, evidence in examples:
    score = entailment_model.check_entailment(evidence, claim)

    assert isinstance(score, float), "Expected a Python float"
    assert 0.0 <= score <= 1.0, "Probability must be between 0 and 1"

    print(f"{description}: entailment probability = {score:.4f}")

from factcheck import EntailmentFactChecker


# Use the wrapper already created earlier in this script.
checker = EntailmentFactChecker(
    entailment_model,
    threshold=0.5
)

# Each case contains a name, evidence passages and an expected label.
# These teaching examples are separate from the coursework dataset.
checker_examples = [
    (
        "Supporting evidence",
        [
            {
                "title": "Example",
                "text": (
                    "Maya enjoys music. "
                    "Maya was born in London in 1985."
                )
            }
        ],
        "S"
    ),
    (
        "Contradicting evidence",
        [
            {
                "title": "Example",
                "text": "Maya was not born in London."
            }
        ],
        "NS"
    ),
    (
        "Unrelated evidence",
        [
            {
                "title": "Example",
                "text": "Maya enjoys playing the piano."
            }
        ],
        "NS"
    ),
    (
        "No evidence",
        [],
        "NS"
    )
]

# Check the complete sentence-processing and binary-decision pipeline.
for description, passages, expected in checker_examples:
    prediction = checker.predict(claim, passages)

    print(
        f"{description}: predicted={prediction}, "
        f"expected={expected}"
    )

    assert prediction == expected, (
        f"{description}: expected {expected}, got {prediction}"
    )

# Summarize filtering during the complete-checker tests.
print("Evidence sentences:", checker.total_sentences)
print("Pruned sentences:", checker.pruned_sentences)
print("DeBERTa calls:", checker.model_calls)

# Every evidence sentence must be either skipped or evaluated.
assert checker.total_sentences == (
    checker.pruned_sentences + checker.model_calls
)

# Record the counters before testing clearly unrelated evidence.
before_pruned = checker.pruned_sentences
before_calls = checker.model_calls

# This sentence shares no content words with the birthplace claim.
unrelated_passages = [
    {
        "title": "Example",
        "text": "Volcanoes release molten rock."
    }
]

prediction = checker.predict(
    "Maya was born in London.",
    unrelated_passages
)

print("Zero-overlap prediction:", prediction)

# The sentence should be filtered without calling DeBERTa.
assert prediction == "NS"
assert checker.pruned_sentences == before_pruned + 1
assert checker.model_calls == before_calls

print("Zero-overlap pruning check passed.")