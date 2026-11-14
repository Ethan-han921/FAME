# FAME-Benchmark

**FAME: A Benchmark for Fine-Grained Entity and Attribute Matching**

FAME is a benchmark for evaluating entity matching and semi-structured attribute matching over heterogeneous records.

Existing entity matching benchmarks primarily focus on determining whether two records refer to the same real-world entity. FAME extends this evaluation to the attribute level by identifying corresponding information within matched records, including complex mappings across different attribute structures.

## Benchmark Overview

FAME evaluates two related tasks:

- **Record Matching (RM):** Determine whether two records refer to the same real-world entity.
- **Attribute Matching (AM):** Identify corresponding attributes or groups of attributes between matching records.

The benchmark supports four types of attribute correspondences: **1:1, 1:N, N:1, and N:M**, allowing attributes to be split, combined, or represented in different structures.

## Datasets

FAME includes three datasets from different domains:

| Dataset | Domain | Development Pairs | Test Pairs | Attribute Correspondences |
|---|---|---:|---:|---:|
| Semi-Heter | Books | 300 | 564 | 768 |
| Semi-Rel | Movies | 300 | 587 | 925 |
| NYC | Organization Funding | 300 | 1,000 | 2,151 |

The datasets are constructed from existing entity matching benchmarks and extended with heterogeneous record structures, value transformations, and manually reviewed attribute correspondence annotations.

## Baseline Approaches

We evaluate several baseline approaches across three categories:

- **Non-LLM:** TF-IDF and Sentence-Transformer.
- **LLM-based:** LLM-One, LLM-Evidence, LLM-Two-Stage, and LLM-Two-Stage Few-shot.
- **Hybrid:** Embedding-Based Hybrid and LLM-Refined Hybrid.

These methods are evaluated using record matching metrics and attribute matching metrics under different evaluation settings.

## Evaluation

FAME provides three evaluation settings for attribute matching:

- **Predicted-Positive AM:** Evaluates attribute correspondences for record pairs predicted as matches.
- **Conditional AM:** Evaluates attribute correspondences for correctly predicted matching pairs.
- **Oracle-EM AM:** Evaluates attribute matching using ground-truth record matching decisions.

## Quick Start

### 1. Environment Setup

Choose **one** of the following options:

#### Option A: Using venv (Python built-in)

```bash
# Create virtual environment
python -m venv fame

# Activate virtual environment
source fame/bin/activate        # Linux/Mac
# or: fame\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Using Conda

```bash
# Create conda environment with Python 3.9+
conda create -n fame python=3.9

# Activate conda environment
conda activate fame

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Run LLM Baseline

Navigate to the LLM baseline directory:

```bash
cd src/LLM_baseline
```

#### Single Dataset & Mode

```bash
# Usage: python pipeline.py --dataset <DATASET> --mode <MODE> --model <MODEL>

# Datasets: nyc, semi_rel, semi_heter
# Modes: two_stage, whole, evidence
# Models: gpt-5.6-luna, gpt-4o, gpt-4o-mini, etc.
```

**Example commands:**

```bash
# NYC dataset
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna
python pipeline.py --dataset nyc --mode whole --model gpt-5.6-luna
python pipeline.py --dataset nyc --mode evidence --model gpt-5.6-luna

# Semi-Rel dataset (Movies)
python pipeline.py --dataset semi_rel --mode two_stage --model gpt-5.6-luna
python pipeline.py --dataset semi_rel --mode whole --model gpt-5.6-luna
python pipeline.py --dataset semi_rel --mode evidence --model gpt-5.6-luna

# Semi-Heter dataset (Books)
python pipeline.py --dataset semi_heter --mode two_stage --model gpt-5.6-luna
python pipeline.py --dataset semi_heter --mode whole --model gpt-5.6-luna
python pipeline.py --dataset semi_heter --mode evidence --model gpt-5.6-luna
```

#### Pipeline Modes

| Mode | Description |
|------|-------------|
| `two_stage` | Stage 1: Record Matching → Stage 2: Attribute Matching (separate LLM calls) |
| `whole` | Single LLM call for both Record and Attribute Matching |
| `evidence` | Evidence-driven approach with intermediate reasoning |

#### Additional Options

```bash
# Limit number of pairs (for testing)
python pipeline.py --dataset nyc --mode two_stage --model gpt-4o-mini --limit 10

# Run only Stage 1 (Record Matching)
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna --stage1-only

# Run only Stage 2 (Attribute Matching, assumes all pairs match)
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna --stage2-only
```

#### Few-Shot Learning (LLM-Two-Stage Few-shot)

The few-shot approach uses examples from the development set to improve Stage 2 (Attribute Matching) performance.

**Step 1:** Run Stage 1 to get record matching predictions:

```bash
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna --stage1-only
# Output: logs/nyc_two_stage/<run_folder>/stage1_em/matched_pairs.json
```

**Step 2:** Run Stage 2 with few-shot examples on Stage 1 results:

```bash
python pipeline.py --dataset nyc --stage2-only \
    --data-file logs/nyc_two_stage/<run_folder>/stage1_em/matched_pairs.json \
    --few-shot-file ../../data/few_shot_nyc.json \
    --few-shot-num 5 \
    --model gpt-5.6-luna
```

**Few-shot files** (located in `data/`):
- `few_shot_nyc.json` - NYC dataset examples
- `few_shot_semi_rel.json` - Semi-Rel dataset examples
- `few_shot_semi_heter.json` - Semi-Heter dataset examples

| Parameter | Description |
|-----------|-------------|
| `--stage2-only` | Skip Stage 1, run only Attribute Matching |
| `--data-file` | Path to `matched_pairs.json` from Stage 1 |
| `--few-shot-file` | Path to few-shot examples config file |
| `--few-shot-num` | Number of few-shot examples (default: 5) |

### 4. Run Non-LLM Baselines

Navigate to the Non-LLM baseline directory:

```bash
cd src/Non_LLM_baseline
```

#### TF-IDF Baseline

Uses TF-IDF vectorization + cosine similarity for both EM and AM.

```bash
python tfidf_baseline.py \
    --test_data ../../data/Test_dataset/transoformd_dataset/nyc_combined.json \
    --ground_truth ../../data/Test_dataset/nyc_AT_level_GT/results_verified_manually.json \
    --output ./output_tfidf_nyc.json \
    --em_threshold 0.2 \
    --am_threshold 0.2
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--em_threshold` | Record matching similarity threshold | 0.1 |
| `--am_threshold` | Attribute matching similarity threshold | 0.5 |
| `--max_group_size` | Maximum attribute group size (None=dynamic) | None |

#### Sentence-Transformer Baseline

Uses pre-trained sentence embeddings (all-mpnet-base-v2) for both EM and AM.

```bash
python sentence-transformer.py \
    --test_data ../../data/Test_dataset/transoformd_dataset/nyc_combined.json \
    --ground_truth ../../data/Test_dataset/nyc_AT_level_GT/results_verified_manually.json \
    --output ./output_embedding_nyc.json \
    --em_threshold 0.6 \
    --similarity_threshold 0.2 \
    --group_threshold 0.2 \
    --top_k 2 \
    --nn_max_size 2
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--em_threshold` | Record matching similarity threshold | 0.7 |
| `--similarity_threshold` | AM candidate similarity threshold | 0.5 |
| `--group_threshold` | AM group selection threshold | 0.6 |
| `--top_k` | Top-k candidates per attribute | 3 |
| `--enable_nn` / `--disable_nn` | Enable/disable N-N correspondences | enabled |
| `--nn_max_size` | Max size per side for N-N groups | 4 |

#### Dataset Paths

| Dataset | Test Data | Ground Truth |
|---------|-----------|--------------|
| NYC | `data/Test_dataset/transoformd_dataset/nyc_combined.json` | `data/Test_dataset/nyc_AT_level_GT/results_verified_manually.json` |
| Semi-Rel | `data/Test_dataset/transoformd_dataset/semi_rel_combined.json` | `data/Test_dataset/semi_re_AT_level_GT/results_verified_manually.json` |
| Semi-Heter | `data/Test_dataset/transoformd_dataset/semi_heter_combined.json` | `data/Test_dataset/semi_heter_AT_level_GT/results_verified_manually.json` |

### 5. Run Hybrid Baselines

The hybrid approaches combine embedding-based methods with LLM refinement.

Navigate to the Hybrid baseline directory:

```bash
cd src/Hyber_baseline
```

#### Embedding-Based Hybrid

This hybrid baseline uses a two-stage approach:
- **Stage 1 (Record Matching):** Uses the LLM two-stage baseline's Stage 1 to identify matching record pairs.
- **Stage 2 (Attribute Matching):** Uses Sentence-Transformer embeddings with heuristic candidate generation and greedy selection.

**Step 1:** Run Stage 1 (Record Matching) using LLM baseline:

```bash
cd src/LLM_baseline
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna --stage1-only
# Output: logs/nyc_two_stage/<run_folder>/stage1_em/matched_pairs.json
```

**Step 2:** Run Stage 2 (Attribute Matching) using Embedding-Based method:

```bash
cd src/Hyber_baseline
python Embedding_Based.py \
    --test_data <path_to_matched_pairs.json_from_stage1> \
    --ground_truth ../../data/Test_dataset/nyc_AT_level_GT/results_verified_manually.json \
    --output ./output_embedding_based_nyc.json \
    --similarity_threshold 0.5 \
    --top_k 3 \
    --group_threshold 0.6
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--similarity_threshold` | Single attribute similarity threshold | 0.5 |
| `--top_k` | Top-k candidates per attribute | 3 |
| `--group_threshold` | Group selection threshold | 0.6 |
| `--enable_nn` / `--disable_nn` | Enable/disable N-N correspondences | enabled |
| `--nn_max_size` | Max size per side for N-N groups | 4 |

#### LLM-Refined Hybrid

This hybrid baseline uses a two-stage approach:
- **Stage 1 (Record Matching):** Uses the LLM two-stage baseline's Stage 1 to identify matching record pairs.
- **Stage 2 (Attribute Matching):** Uses sentence embeddings to generate attribute correspondence candidates, then sends them to LLM for final selection.

**Step 1:** Run Stage 1 (Record Matching) using LLM baseline:

```bash
cd src/LLM_baseline
python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna --stage1-only
# Output: logs/nyc_two_stage/<run_folder>/stage1_em/matched_pairs.json
```

**Step 2:** Run Stage 2 (Attribute Matching) using Embedding + LLM:

```bash
cd src/Hyber_baseline
python "LLM-Refined Hybrid_baseline.py" \
    --test_data <path_to_matched_pairs.json_from_stage1> \
    --ground_truth ../../data/Test_dataset/nyc_AT_level_GT/results_verified_manually.json \
    --output ./output_hybrid_nyc.json \
    --similarity_threshold 0.5 \
    --llm_model gpt-5.6-luna
```

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--test_data` | Path to matched pairs from Stage 1 | (required) |
| `--ground_truth` | Path to ground truth file | (required) |
| `--output` | Output file path | (required) |
| `--similarity_threshold` | Embedding similarity threshold for candidate generation | 0.5 |
| `--llm_model` | LLM model for refinement | gpt-5.6-luna |
| `--log_dir` | Custom log directory | logs/embedding_llm_TIMESTAMP |

### 6. Results

Results are saved to `src/LLM_baseline/logs/` with the following structure:

```
logs/
└── <dataset>_<mode>/
    └── <dataset>_<model>_<timestamp>/
        ├── final_results.json      # Overall metrics
        ├── stage1_em/              # Record matching results
        └── stage2_attr/            # Attribute matching results
```

## TODO
- [x] Release the test datasets with ground-truth annotations.
- [x] Release the LLM-based baseline implementations.
- [x] Release the non-LLM and hybrid baseline implementations.
- [ ] Release the evaluation scripts and documentation.

**The remaining code and resources will be released soon.**
