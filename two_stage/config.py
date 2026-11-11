"""
Configuration for Two-Stage GEM Pipeline
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
TWO_STAGE_ROOT = Path(__file__).parent
DATA_ROOT = TWO_STAGE_ROOT / 'data' / 'Test_dataset'

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================

DATASETS = {
    'nyc': {
        'name': 'NYC Funding',
        'domain': 'organization which apply to the service categories/source',
        # Entity Matching data (1000 pairs)
        'em_data_file': DATA_ROOT / 'transoformd_dataset' / 'nyc_combined.json',
        'em_gt_file': None,  # Will use 'label' field in data file
        # Attribute Matching ground truth (500 entries)
        # 'attr_gt_file': DATA_ROOT / 'nyc_AT_level_GT' / 'results_verified_manually.json',
        'attr_gt_file': DATA_ROOT / 'nyc_AT_level_GT' / 'results_verified_manually.json',
    },
    'semi_rel': {
        'name': 'Semi-Rel (Movie)',
        'domain': 'movie',
        # Entity Matching data (437 pairs)
        'em_data_file': DATA_ROOT / 'transoformd_dataset' / 'semi_rel_combined.json',
        'em_gt_file': None,
        # Attribute Matching ground truth (183 entries)
        'attr_gt_file': DATA_ROOT / 'semi_re_AT_level_GT' / 'results_verified_manually.json',
    },
    'semi_heter': {
        'name': 'Semi-Heter (Book)',
        'domain': 'book',
        # Entity Matching data (414 pairs)
        'em_data_file': DATA_ROOT / 'transoformd_dataset' / 'semi_heter_combined.json',
        'em_gt_file': None,
        # Attribute Matching ground truth (159 entries)
        'attr_gt_file': DATA_ROOT / 'semi_heter_AT_level_GT' / 'results_verified_manually.json',
    }
}

# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

DEFAULT_MODEL = "gpt-5.6-luna"

MODEL_CONFIGS = {
    'gpt-5.4-2026-03-05': {
        'em_max_tokens': 10,
        'em_temperature': 0.0,
        'attr_max_tokens': 500,
        'attr_temperature': 0.0,
    },
    'gpt-5.4-mini-2026-03-17': {
        'em_max_tokens': 10,
        'em_temperature': 0.0,
        'attr_max_tokens': 500,
        'attr_temperature': 0.0,
    },
    'gpt-4o-mini': {
        'em_max_tokens': 10,
        'em_temperature': 0.0,
        'attr_max_tokens': 500,
        'attr_temperature': 0.0,
    },
    'gpt-4o': {
        'em_max_tokens': 10,
        'em_temperature': 0.0,
        'attr_max_tokens': 500,
        'attr_temperature': 0.0,
    },
    'gpt-4-turbo': {
        'em_max_tokens': 10,
        'em_temperature': 0.0,
        'attr_max_tokens': 500,
        'attr_temperature': 0.0,
    },
    'gpt-5.6-luna': {
        'em_max_tokens': 10,
        'em_temperature': None,
        'attr_max_tokens': 1500,
        'attr_temperature': None,
    }
}

# =============================================================================
# LOGGING CONFIGURATIONS
# =============================================================================

LOG_DIR = TWO_STAGE_ROOT / 'logs'
