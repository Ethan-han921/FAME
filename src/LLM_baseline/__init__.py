"""
Two-Stage GEM Pipeline

Stage 1: Entity Matching (binary classification)
Stage 2: Attribute-Level Matching (for matched pairs)

Supported datasets:
- nyc: NYC Government Funding
- semi_rel: Semi-Relational (Movie)
- semi_heter: Semi-Heterogeneous (Book)
"""

from .pipeline import run_pipeline
from .config import DATASETS, DEFAULT_MODEL
from .prompts import EM_PROMPTS, ATTR_PROMPTS

__all__ = ['run_pipeline', 'DATASETS', 'DEFAULT_MODEL', 'EM_PROMPTS', 'ATTR_PROMPTS']
