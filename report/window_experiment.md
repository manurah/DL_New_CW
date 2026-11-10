# Sentence-window experiment

Run from the repository root with the project environment activated:

```bash
python report/window_test.py
```

The script loads the same DeBERTa checkpoint and development data as the main
driver. It compares:

- Maximum window 1: the original single-sentence classifier.
- Maximum window 2: all original sentences plus each adjacent pair within the
  same passage. Sentences from different passages are never joined.

Both variants use the existing decision threshold (0.38), word-recall pruning
threshold (0.05), and 512-token input limit. Pair scores are combined with the
single-sentence scores using the maximum. Pruning is applied to the combined
pair text before inference. The model is loaded once and runs on CPU with at
most four threads.

The output table reports correct predictions, accuracy, model calls and elapsed
seconds. Window 2's call count includes both its single-sentence and pair calls.
Each variant runs separately; times exclude model loading. Longer pairs may be
truncated by the existing tokenizer limit. Results are development-set results,
not an independent test-set estimate.

## Validation and remaining run

Unit tests in `tests/test_window_experiment.py` exercise adjacent-pair creation,
passage boundaries, pruning, empty evidence, IR handling, call counts and the
printed comparison using fixed test scores. These checks do not measure the
pretrained model's accuracy.

The full experiment was attempted in the cloud environment on 10 November 2026.
The network proxy rejected the Hugging Face model download with HTTP 403, so
there are no measured window-comparison results yet. Run the command above on
an environment with the checkpoint cached or with access to Hugging Face and
its download destinations, then retain the printed table for the team's report.

The experiment does not modify `factcheck.py`, its defaults, the driver, source
data, or the existing result files.
