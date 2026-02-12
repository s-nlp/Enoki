# Enoki: Dependency-Guided Claim Decomposition for Efficient Multi-Level Hallucination Detection

This repository contains the implementation of a hallucination detection system that employs dependency parsing to extract subject-predicate-argument triples from text. The system supports evaluation across multiple granularities: sentence-level, entity-level, and span-level detection.

## Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_trf
```

## Usage

End-to-end example of extracting facts and verifying them against source context:

```python
from fact_extractor import FactExtractor
from nli import ModernBERTEncoderNLI, hallucination_prob_from_nli

# Initialize extractor and NLI checker
extractor = FactExtractor()
nli_checker = ModernBERTEncoderNLI()

# Source context and generated text
context = "Lanny Flaherty (July 27, 1942 – February 18, 2024) was an American actor."
generated = "Lanny Flaherty was born on July 27, 1949, in Pensacola, Florida. Flaherty was an American actor."

# Extract incremental fact groups
fact_groups = extractor.extract_granular_facts(generated)

# Process each fact group
for group in fact_groups:
    hypotheses = [str(fact) for fact in group.facts]
    nli_scores = nli_checker.check_batch(context, hypotheses, max_length=2048)

    for fact, delta, score in zip(group.facts, group.deltas, nli_scores):
        hal_prob = hallucination_prob_from_nli(score, mode="default")
        print(f"{fact} | delta: '{delta}' | hal_prob: {hal_prob:.3f}")

# Output shows detected hallucinations:
# Lanny Flaherty | was born on | July | delta: 'July' | hal_prob: 0.013
# Lanny Flaherty | was born on | July 27 | delta: '27' | hal_prob: 0.019
# Lanny Flaherty | was born on | July 27, 1949 | delta: '1949' | hal_prob: 0.998
# Lanny Flaherty | was born | Pensacola (prep: in) | delta: 'Pensacola' | hal_prob: 0.931
# Lanny Flaherty | was born | Pensacola, Florida (prep: in) | delta: 'Pensacola, Florida' | hal_prob: 0.987
# Lanny Flaherty | was born | Florida (prep: in) | delta: 'Florida' | hal_prob: 0.885
# Flaherty | was | an American actor | delta: 'an American actor' | hal_prob: 0.038

# The system correctly identifies hallucinations:
# - Birth year: 1949 (should be 1942) - detected with 0.998 confidence
# - Birth location: Pensacola, Florida (not mentioned in context) - detected with 0.987 confidence
#
# While correctly validating accurate facts:
# - Birth month/day: July 27 (correct) - hal_prob: 0.019
# - Profession: American actor (correct) - hal_prob: 0.038
```

## Running Evaluations

The system provides a unified CLI for running evaluations across different benchmarks and baseline methods.

### Dataset Preparation

The system expects datasets in the following directory structure:

```
data/
├── felm/
│   ├── wk/              # FELM Wiki subset
│   ├── science/         # FELM Science subset
│   └── writing_rec/     # FELM Writing subset
├── factcheckbench/
│   └── factcheck-GPT-benchmark.jsonl
├── halluentity/
│   └── halluentity_contexts.csv
└── mushroom/
    └── mushroom.en-tst.v1.extra.with_context.jsonl
```

**Note:** PsiloQA and RAGTruth are loaded automatically from HuggingFace and do not require manual download.

When running evaluations, specify `--data-dir data` to use this structure.

### Sentence-Level Evaluation

```bash
# FELM (all subsets: wk, science, writing_rec)
python enoki_cli.py evaluate sentence --dataset felm --method modernbert --data-dir data

# FactCheckBench
python enoki_cli.py evaluate sentence --dataset factcheckbench --method modernbert --data-dir data
```

### Entity-Level Evaluation

```bash
# HalluEntity
python enoki_cli.py evaluate entity --dataset halluentity --method modernbert --data-dir data
```

### Span-Level Evaluation

```bash
# All span datasets (PsiloQA, Mushroom, RAGTruth)
python enoki_cli.py evaluate span --method modernbert --all --data-dir data
```

### Running All Baselines

To evaluate all baseline methods (ModernBERT, AlignScore, Qwen-8B) across all datasets:

```bash
# Sentence-level baselines
for method in modernbert alignscore qwen_8b; do
    python enoki_cli.py evaluate sentence --dataset felm --method $method --data-dir data
    python enoki_cli.py evaluate sentence --dataset factcheckbench --method $method --data-dir data
done

# Entity-level baselines
for method in modernbert alignscore qwen_8b; do
    python enoki_cli.py evaluate entity --dataset halluentity --method $method --data-dir data
done

# Span-level baselines
for method in modernbert alignscore qwen_8b; do
    python enoki_cli.py evaluate span --method $method --all --data-dir data
done
```

### Available Options

- `--method`: NLI method (`modernbert`, `alignscore`, `qwen_06b`, `qwen_4b`, `qwen_8b`)
- `--max-length`: Maximum sequence length (default: 2048)
- `--threshold`: Classification threshold (default: 0.5)
- `--coref`: Enable coreference resolution
- `--force-recompute`: Bypass cache and recompute results
- `--cache-dir`: Directory for caching intermediate results (default: `cache`)
- `--output-dir`: Directory for evaluation results (default: `eval_results`)

## Output Format

Results are saved to `eval_results/` with timestamped filenames containing:
- Performance metrics (Precision, Recall, F1, Accuracy, AUROC)
- Per-sample predictions and gold labels
- Configuration parameters
- Runtime statistics
