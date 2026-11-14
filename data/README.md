# Data Directory Structure

This directory contains the datasets used in the FAME benchmark.

## Overview

```
data/
├── Dev_dataset/                    # Development set (300 pairs per dataset)
├── Test_dataset/                   # Test set
├── few_shot_nyc.json              # Few-shot config for NYC
├── few_shot_semi_rel.json         # Few-shot config for Semi-Rel
└── few_shot_semi_heter.json       # Few-shot config for Semi-Heter
```

## Test Dataset

```
Test_dataset/
├── transoformd_dataset/           # Record pairs for matching
│   ├── nyc_combined.json          # NYC: 1000 pairs
│   ├── semi_rel_combined.json     # Semi-Rel: 587 pairs
│   ├── semi_heter_combined.json   # Semi-Heter: 564 pairs
│   ├── nyc/
│   │   ├── nyc_combined.json
│   │   ├── nyc_left.json          # Left records (1000)
│   │   └── nyc_right.json         # Right records (1000)
│   ├── semi_rel/
│   │   ├── semi_rel_combined.json
│   │   ├── semi_rel_left.json     # Left records (587)
│   │   └── semi_rel_right.json    # Right records (587)
│   └── semi_heter/
│       ├── semi_heter_combined.json
│       ├── semi_heter_left.json   # Left records (564)
│       └── semi_heter_right.json  # Right records (564)
├── nyc_AT_level_GT/               # NYC attribute-level ground truth
│   └── results_verified_manually.json
├── semi_re_AT_level_GT/           # Semi-Rel attribute-level ground truth
│   └── results_verified_manually.json
└── semi_heter_AT_level_GT/        # Semi-Heter attribute-level ground truth
    └── results_verified_manually.json
```

## Development Dataset

```
Dev_dataset/
├── transformed_dataset/           # Record pairs for development
│   ├── nyc_dev.json              # NYC dev pairs (all labels)
│   ├── nyc_dev_label1.json       # NYC dev pairs (label=1 only)
│   ├── semi_rel_dev.json
│   ├── semi_rel_dev_label1.json
│   ├── semi_heter_dev.json
│   └── semi_heter_dev_label1.json
├── nyc_AT_level_GT/
│   └── nyc_dev_AM.json           # NYC dev attribute matching ground truth
├── semi_re_AT_level_GT/
│   └── semi_rel_dev_AM.json
└── semi_heter_AT_level_GT/
    └── semi_heter_dev_AM.json
```

## File Formats

### Combined JSON (Record Pairs)

Each `*_combined.json` file contains an array of record pairs:

```json
[
  {
    "left_right_pair": "id1_id2",
    "left": { "_id": "id1", "attr1": "value1", ... },
    "right": { "_id": "id2", "attr1": "value1", ... },
    "label": 1
  },
  ...
]
```

- `label`: 1 = match, 0 = non-match

### Left/Right JSON (Separate Records)

Each `*_left.json` and `*_right.json` contains an array of individual records:

```json
[
  { "_id": "id1", "attr1": "value1", ... },
  { "_id": "id2", "attr1": "value1", ... },
  ...
]
```

### Ground Truth JSON (Attribute Matching)

Each ground truth file contains attribute correspondences:

```json
[
  {
    "pair_id": "id1_id2",
    "output": [
      [["left_attr1"], ["right_attr1"]],
      [["left_attr2", "left_attr3"], ["right_attr2"]]
    ]
  },
  ...
]
```

- Supports 1:1, 1:N, N:1, and N:M attribute correspondences

## Datasets Summary

| Dataset | Domain | Dev Pairs | Test Pairs | Attribute Correspondences |
|---------|--------|-----------|------------|---------------------------|
| Semi-Heter | Books | 300 | 564 | 768 |
| Semi-Rel | Movies | 300 | 587 | 925 |
| NYC | Organization Funding | 300 | 1,000 | 2,151 |
