# CM3604-Deep-Learning-group-coursework---fact-checking-ChatGPT-outputs

Group coursework for CM3604 Deep Learning (Robert Gordon University). Fact-checking ChatGPT biographical claims against Wikipedia passages, based on FActScore (Min et al., EMNLP 2023).

**Deadline:** 11 November 2026, 23:59

## Team

- Member A — Sandali
- Member B — Dilshan
- Member C — Sachinthana
- Member D — Manura

## Setup

1. Install **Python 3.10.11** from https://www.python.org/downloads/release/python-31011/ (tick "Add to PATH" during install, custom location `C:\Python310`). Keep any newer Python you already have — the `py` launcher picks which one to use.

2. Clone this repo via GitHub Desktop or: git clone https://github.com/SandaliBalasooriya/CM3604-factchecker.git

3. Open PowerShell in the cloned repo folder (Shift + right-click → "Open in Terminal").

4. Create and activate the virtual environment:
				py -3.10 -m venv venv
				venv\Scripts\activate

5. Install pinned requirements: pip install -r requirements.txt

6. Download NLTK and spaCy data:
				python -c “import nltk; nltk.download(‘punkt’); nltk.download(‘punkt_tab’); 
				nltk.download(‘stopwords’); nltk.download(‘wordnet’)”
				python -m spacy download en_core_web_sm

7. (Optional) If you want to run Part 2 on GPU, swap torch for the CUDA version:
				pip uninstall torch torchvision -y
				pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
				


Note: this is a 0.0.1 patch deviation from the pinned `torch==2.5.0` — document in report if used.

## Running the two modes

    python factchecking_main.py --mode word_overlap   # 179/221 = 81.0%  (target 75%)
    python factchecking_main.py --mode entailment     # 187/221 = 84.6%  (target 83%), ~2-3 min on CPU

Add `--cuda` to the entailment command if PyTorch is installed with CUDA support.
Missing NLTK data (punkt, punkt_tab, stopwords) is downloaded automatically on first run.

## File structure

- `factcheck.py` — Part 1 (`WordRecallThresholdFactChecker`) and Part 2 (`EntailmentModel`, `EntailmentFactChecker`); `factchecking_main.py` is the unmodified framework driver
- `data/` — the labelled dev set and retrieved Wikipedia passages
- `requirements.txt` — pinned package versions (Python 3.10)
- `report/` — experiment scripts and results behind the written report's tables, and the entailment error export used for Part 3
- `threshold_sweep.py`, `evaluate_entailment.py`, `inspect_entailment.py`, `results/` — development experiments (not needed to run the two modes)
