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

- ## TODO
- 
- [x] Release the test datasets with ground-truth annotations.
- [x] Release the LLM-based baseline implementations.
- [ ] Release the non-LLM and hybrid baseline implementations.
- [ ] Release the evaluation scripts and documentation.

**The remaining code and resources will be released soon.**
