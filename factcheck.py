# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rouge_score import rouge_scorer


def _ensure_nltk_data():
    # Download the NLTK resources we use if they are missing, so both modes
    # run on a fresh environment without a manual nltk.download() step.
    for resource, path in [("punkt", "tokenizers/punkt"),
                           ("punkt_tab", "tokenizers/punkt_tab"),
                           ("stopwords", "corpora/stopwords")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


_ensure_nltk_data()


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        # Reuse the model and tokenizer already loaded by the driver.
        # This avoids loading a new model for every prediction.
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

        # Set prediction behaviour, including disabling training-time dropout.
        self.model.eval()

        # Read the entailment position from the model's actual configuration.
        self.entailment_index = self.model.config.label2id["entailment"]

    def check_entailment(self, premise: str, hypothesis: str) -> float:
        # Disable gradient tracking because we are predicting, not training.
        with torch.no_grad():
            # Encode evidence first and claim second.
            # Limit the combined input to 512 tokens, including special tokens.
            inputs = self.tokenizer(
                premise,
                hypothesis,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )

            # Match the inputs to the GPU when the driver uses --cuda.
            if self.cuda:
                inputs = {
                    key: value.to("cuda")
                    for key, value in inputs.items()
                }

            # Run the model and obtain its three raw class scores.
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Convert logits into probabilities across the three classes.
            probabilities = torch.softmax(logits, dim=-1)

            # Select the first input pair's entailment probability.
            # .item() converts the single tensor value into a Python number.
            entailment_probability = probabilities[
                0, self.entailment_index
            ].item()

        # Release references to temporary tensors before returning.
        # The returned Python number does not retain these tensors.
        del inputs, outputs, logits, probabilities
        gc.collect()

        return entailment_probability

class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.65, use_nltk: bool = True,
                 remove_stopwords: bool = True, use_stemming: bool = True, 
                 scorer: str = "recall"):
        # Threshold is kept as a field so we can tune it later from one place
        # rather than hunting through the predict() body. The default 0.65
        # was chosen by sweeping thresholds on the dev set (see the report).
        self.threshold = threshold

        # Toggles for the preprocessing experiments. Keeping these as flags
        # (rather than hard-coding one choice) lets us switch each step on and
        # off from the driver and measure its individual impact. The results
        # of that ablation go in the report comparison table.
        self.use_nltk = use_nltk
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming

        # Which overlap metric to use. Options: "recall", "jaccard", "f1",
        # "rouge". Default is "recall" because our class name is literally
        # WordRecallThresholdFactChecker and the sweep showed it works well
        # here — the fact/passage size mismatch disadvantages Jaccard and F1,
        # which both penalise differences in set size. Alternatives are
        # exposed so the report can compare them side-by-side.
        self.scorer = scorer

        # Load the stopword set once per instance rather than on every predict()
        # call — looking it up is cheap but still worth caching since predict
        # runs 221 times. Set-based membership check is O(1).
        self.stopword_set = set(stopwords.words("english"))

        # Built once and reused; constructing it on every call is slow.
        self._rouge = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

        # Porter stemmer is stateless so one instance is reused across all calls.
        self.stemmer = PorterStemmer()

    def preprocess(self, text: str) -> set:
        # One function handles tokenisation, stopword removal and stemming
        # based on the toggles set in __init__. Returning a set here (rather
        # than later in predict) keeps the preprocessing logic in one place.

        # Step A: lowercase. Still a sensible default even with better
        # tokenisation — Wikipedia starts sentences with capitals that would
        # otherwise mismatch mid-passage occurrences of the same word.
        text = text.lower()

        # Step B: tokenise. NLTK handles punctuation properly
        # ("Cleveland." -> ["Cleveland", "."]), which plain split does not.
        # The toggle lets us compare both for the experiment table.
        if self.use_nltk:
            tokens = nltk.word_tokenize(text)
        else:
            tokens = text.split()

        # Step C: drop stopwords and bare punctuation. Stopwords like "is",
        # "the", "has" inflate recall by matching every passage; punctuation
        # tokens from NLTK (".", ",") carry no meaning either.
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stopword_set and t.isalnum()]

        # Step D: stem. Reduces morphological variants to a shared root so
        # "invented", "inventing", "invention" all collapse to "invent".
        # Helps when the fact and the passage use different word forms.
        if self.use_stemming:
            tokens = [self.stemmer.stem(t) for t in tokens]

        # Return as a set — recall is defined on word types, not tokens.
        return set(tokens)

    def _compute_score(self, fact_words: set, passage_words: set,
                       fact_text: str, passage_text: str) -> float:
        # Returns a real-valued overlap score in [0, 1] using the metric
        # chosen by self.scorer. Four metrics are supported so the report
        # can compare them side-by-side; "recall" is our chosen final metric.
        #
        # Both the pre-tokenised sets AND the raw text strings are passed
        # in because ROUGE wants to tokenise the strings itself (it has its
        # own tokeniser and would get confused by stemmed tokens like
        # "marri" that aren't real words).

        # Empty fact -> no information to match on, treat as unsupported.
        if len(fact_words) == 0:
            return 0.0

        overlap = fact_words & passage_words

        if self.scorer == "recall":
            # Recall = fraction of the fact's words appearing in the passage.
            # Only cares about fact coverage — passage size doesn't matter.
            return len(overlap) / len(fact_words)

        elif self.scorer == "jaccard":
            # Jaccard = intersection over union of the two word sets.
            # Fair when the two sets are similar in size; here passages are
            # much larger than facts, so Jaccard scores will be very small
            # even for genuine matches. We expect this to underperform.
            union = fact_words | passage_words
            return len(overlap) / len(union) if len(union) > 0 else 0.0

        elif self.scorer == "f1":
            # F1 = harmonic mean of precision and recall.
            # Precision = how many of the passage's words are in the fact
            # (always tiny here because passages are long).
            # Same size-mismatch problem as Jaccard, included for completeness.
            if len(passage_words) == 0:
                return 0.0
            precision = len(overlap) / len(passage_words)
            recall = len(overlap) / len(fact_words)
            if precision + recall == 0:
                return 0.0
            return 2 * precision * recall / (precision + recall)

        elif self.scorer == "rouge":
            # ROUGE-1 from the rouge-score library. We feed it the raw strings
            # (not our preprocessed sets) because its own tokeniser handles
            # casing and punctuation internally, and stemmed tokens would
            # confuse it. Returns a precision/recall/f-measure triple;
            # we take f-measure as the comparable single number.
            result = self._rouge.score(fact_text, passage_text)
            return result["rouge1"].fmeasure

        else:
            # Fail loudly on a typo rather than silently defaulting.
            raise ValueError(
                f"Unknown scorer: {self.scorer!r}. "
                f"Expected one of: 'recall', 'jaccard', 'f1', 'rouge'."
            )
    
    def predict(self, fact: str, passages: List[dict]) -> str:
        # Step 1: run the fact through the shared preprocessing pipeline
        # (tokenise -> drop stopwords -> stem, controlled by the toggles).
        fact_words = self.preprocess(fact)

        # Step 2: for each passage, compute an overlap score using whichever
        # scoring method is configured (recall / jaccard / f1 / rouge).
        # We take the max rather than the mean because a fact only needs ONE
        # supporting passage. Averaging would unfairly punish a fact that is
        # strongly supported by passage 1 but unrelated to passages 2 and 3.
        best_score = 0.0
        for passage in passages:
            # Preprocess the passage text the same way as the fact so both
            # sides share a vocabulary. The brief tells us to use "text" and
            # ignore "title".
            passage_text = passage["text"]
            passage_words = self.preprocess(passage_text)

            # _compute_score needs both the preprocessed sets AND the raw
            # strings — most scorers use the sets, but ROUGE wants to do its
            # own tokenisation on the originals.
            score = self._compute_score(fact_words, passage_words,
                                        fact, passage_text)

            if score > best_score:
                best_score = score

        # Step 3: convert the real-valued score into a binary S/NS decision.
        # Threshold is tuned per scorer (see the report) because different
        # metrics live on different numeric ranges — a Jaccard of 0.1 might
        # be a strong match while a recall of 0.1 is weak.
        if best_score >= self.threshold:
            return "S"
        else:
            return "NS"


class EntailmentFactChecker(FactChecker):
    def __init__(
        self,
        ent_model,
        threshold: float = 0.38,
        pruning_threshold: float = 0.05,
        use_pruning: bool = True
    ):
        # Reuse the entailment model loaded by the original driver.
        self.ent_model = ent_model

        # This cutoff converts the final entailment score into S or NS.
        self.threshold = threshold

        # This separate cutoff filters weak evidence before model inference.
        # 0.05 drops only sentences sharing no content word with the fact;
        # it removes ~71% of model calls for a 0.5-point accuracy cost.
        self.pruning_threshold = pruning_threshold
        self.use_pruning = use_pruning

        # Reuse Part 1's existing preprocessing and recall scoring.
        # We do not change its code or call its S/NS prediction method.
        self.overlap_checker = (
            WordRecallThresholdFactChecker(scorer="recall")
            if use_pruning
            else None
        )

        # These counters measure how much evidence reaches DeBERTa.
        self.total_sentences = 0
        self.pruned_sentences = 0
        self.model_calls = 0

    def score_fact(self, fact: str, passages: List[dict]) -> float:
        # No evidence means no support has been established.
        best_score = 0.0

        # Preprocess the claim once for overlap comparisons.
        fact_words = (
            self.overlap_checker.preprocess(fact)
            if self.use_pruning
            else None
        )

        for passage in passages:
            # Normalize whitespace while preserving the original meaning.
            text = " ".join(passage["text"].split())

            if not text:
                continue

            # Evaluate individual evidence sentences.
            for sentence in nltk.sent_tokenize(text):
                sentence = sentence.strip()

                if not sentence:
                    continue

                self.total_sentences += 1

                # If preprocessing leaves no claim tokens, avoid pruning:
                # there is no useful lexical basis for filtering.
                if self.use_pruning and fact_words:
                    sentence_words = (
                        self.overlap_checker.preprocess(sentence)
                    )

                    # Calculate recall using Part 1's existing method.
                    overlap_score = (
                        self.overlap_checker._compute_score(
                            fact_words,
                            sentence_words,
                            fact,
                            sentence
                        )
                    )

                    if overlap_score < self.pruning_threshold:
                        self.pruned_sentences += 1

                        # Skip only this sentence; continue checking others.
                        continue

                # DeBERTa receives natural text, not stemmed tokens.
                score = self.ent_model.check_entailment(
                    premise=sentence,
                    hypothesis=fact
                )
                self.model_calls += 1

                # Retain the strongest support across the evidence.
                best_score = max(best_score, score)

        return best_score

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Convert the maximum evidence score into the driver's expected label.
        best_score = self.score_fact(fact, passages)

        if best_score >= self.threshold:
            return "S"

        return "NS"
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
