#!/usr/bin/env python
"""
GEM Pipeline - Two modes available:
1. Two-Stage: Entity Matching → Attribute Matching (for matched pairs)
2. Whole-Stage: Combined EM + Attribute Matching in single call

Usage:
    # Two-Stage mode (default)
    python pipeline.py --dataset nyc
    python pipeline.py --dataset nyc --mode two_stage

    # Whole-Stage mode
    python pipeline.py --dataset nyc --mode whole

    # Other options
    python pipeline.py --dataset all
    python pipeline.py --dataset nyc --limit 10  # for testing
"""

import json
import os
import sys
import csv
import logging
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from config import DATASETS, DEFAULT_MODEL, MODEL_CONFIGS, LOG_DIR
from prompts import EM_PROMPTS, ATTR_PROMPTS, WHOLE_STAGE_PROMPTS, EVIDENCE_DRIVEN_PROMPTS
import random


# =============================================================================
# UTILITIES
# =============================================================================

def setup_logging(log_dir: Path, dataset_name: str, model: str, mode: str = 'two_stage', prefix: str = '') -> Tuple[logging.Logger, Path]:
    """Setup logging with session directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create mode-specific directory: logs/{prefix}_{dataset}_{mode}/{dataset}_{model}_{timestamp}
    mode_suffix_map = {
        'two_stage': 'two_stage',
        'whole': 'whole',
        'evidence': 'evidence'
    }
    mode_suffix = mode_suffix_map.get(mode, mode)
    folder_name = f'{prefix}_{dataset_name}_{mode_suffix}' if prefix else f'{dataset_name}_{mode_suffix}'
    mode_dir = log_dir / folder_name
    mode_dir.mkdir(parents=True, exist_ok=True)

    session_dir = mode_dir / f'{dataset_name}_{model}_{timestamp}'
    session_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories for each stage
    (session_dir / 'stage1_em').mkdir(exist_ok=True)
    (session_dir / 'stage2_attr').mkdir(exist_ok=True)

    logger = logging.getLogger(f'pipeline_{dataset_name}')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(session_dir / 'run.log', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger, session_dir


def serialize_record(record: dict) -> str:
    """Serialize record to JSON string, excluding internal IDs."""
    record_copy = {k: v for k, v in record.items() if k.lower() not in ['id', '_id']}
    return json.dumps(record_copy, ensure_ascii=False, indent=2)


def extract_id(record: dict) -> str:
    """Extract ID from record, handling different field names (_id, id, ID)."""
    for key in ['_id', 'id', 'ID']:
        if key in record:
            return str(record[key])
    return "unknown"


def load_data(data_file: Path) -> List[Dict]:
    """Load data from JSON file."""
    with open(data_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_attr_ground_truth(gt_file: Path) -> Dict[Tuple[str, str], List[str]]:
    """Load attribute-level ground truth and build index."""
    gt_index = {}
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    for item in gt_data:
        key = (str(item['left_id']), str(item['right_id']))
        gt_index[key] = item.get('output', [])
    return gt_index


def load_few_shot_examples(few_shot_file: Path, num_examples: int = 5, seed: int = 42) -> List[Dict]:
    """
    Load few-shot examples from a JSON file containing:
    - pairs_file: path to record pairs (label1.json)
    - gt_file: path to AM ground truth (_AM.json)

    Or directly from a pre-built examples file.

    Returns: List of {left, right, output} dicts
    """
    with open(few_shot_file, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # If it's a config file with pairs_file and gt_file
    if 'pairs_file' in config and 'gt_file' in config:
        pairs_path = Path(config['pairs_file'])
        gt_path = Path(config['gt_file'])

        # Load pairs
        with open(pairs_path, 'r', encoding='utf-8') as f:
            pairs = json.load(f)

        # Load GT and build index
        gt_index = load_attr_ground_truth(gt_path)

        # Build examples with GT
        examples = []
        for pair in pairs:
            left_id = str(pair.get('left_id')) if pair.get('left_id') else extract_id(pair['left'])
            right_id = str(pair.get('right_id')) if pair.get('right_id') else extract_id(pair['right'])
            key = (left_id, right_id)

            if key in gt_index:
                examples.append({
                    'left': pair['left'],
                    'right': pair['right'],
                    'output': gt_index[key]
                })

        # Random sample
        random.seed(seed)
        if len(examples) > num_examples:
            examples = random.sample(examples, num_examples)

        return examples

    # If it's already a list of examples
    elif isinstance(config, list):
        random.seed(seed)
        if len(config) > num_examples:
            config = random.sample(config, num_examples)
        return config

    else:
        raise ValueError(f"Invalid few-shot file format: {few_shot_file}")


def build_few_shot_prompt(examples: List[Dict]) -> str:
    """
    Build few-shot examples string for Stage 2 prompt.

    Returns formatted string with examples.
    """
    if not examples:
        return ""

    few_shot_str = "\n\n=== FEW-SHOT EXAMPLES ===\n"

    for i, ex in enumerate(examples, 1):
        left_str = serialize_record(ex['left'])
        right_str = serialize_record(ex['right'])

        # Format output
        output_formatted = []
        for corr in ex['output']:
            if isinstance(corr, str):
                output_formatted.append(corr)
            else:
                output_formatted.append(json.dumps(corr, ensure_ascii=False))

        output_json = json.dumps({"output": ex['output']}, ensure_ascii=False, indent=2)

        few_shot_str += f"""
--- Example {i} ---
Left record:
{left_str}

Right record:
{right_str}

Output:
{output_json}
"""

    few_shot_str += "\n=== END OF EXAMPLES ===\n"
    return few_shot_str


# =============================================================================
# STAGE 1: ENTITY MATCHING
# =============================================================================

def call_em_llm(client: OpenAI, system_prompt: str, user_prompt: str,
                model: str, config: Dict) -> Tuple[int, str, Optional[str]]:
    """
    Call LLM for entity matching (binary classification).
    Returns: (prediction, raw_response, error_message)
    """
    try:
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_completion_tokens": config.get('em_max_tokens', 10),
        }
        em_temp = config.get('em_temperature')
        if em_temp is not None:
            api_params["temperature"] = em_temp
        response = client.chat.completions.create(**api_params)
        answer = response.choices[0].message.content.strip()
        pred = 1 if '1' in answer else 0
        return pred, answer, None
    except Exception as e:
        error_msg = str(e)
        return -1, "", error_msg


def run_entity_matching(
    data: List[Dict],
    client: OpenAI,
    dataset_name: str,
    model: str,
    config: Dict,
    session_dir: Path,
    logger: logging.Logger
) -> Tuple[List[Dict], Dict]:
    """
    Stage 1: Run entity matching on all pairs.
    Returns: (predictions, metrics)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 1: ENTITY MATCHING")
    logger.info("=" * 60)

    prompts = EM_PROMPTS[dataset_name]
    system_prompt = prompts['system']
    user_template = prompts['user']

    predictions = []
    stage_dir = session_dir / 'stage1_em'

    # Initialize log files
    prompt_log_path = stage_dir / 'prompt.log'
    response_log_path = stage_dir / 'response.log'
    error_log_path = stage_dir / 'error.log'

    # Clear/create log files
    prompt_log_path.write_text("", encoding='utf-8')
    response_log_path.write_text("", encoding='utf-8')
    error_log_path.write_text("", encoding='utf-8')

    y_true, y_pred = [], []

    for i, pair in enumerate(data):
        left_id = str(pair.get('left_id')) if pair.get('left_id') else extract_id(pair['left'])
        right_id = str(pair.get('right_id')) if pair.get('right_id') else extract_id(pair['right'])
        pair_id = f"{left_id}_{right_id}"

        left_str = serialize_record(pair['left'])
        right_str = serialize_record(pair['right'])

        user_prompt = user_template.replace('{left_record}', left_str).replace('{right_record}', right_str)

        # Log prompt (system + user for each pair, matching original format)
        with open(prompt_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Pair: left_id={left_id}, right_id={right_id}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n\n")
            f.write(f"[USER PROMPT]\n{user_prompt}\n")

        pred, raw_response, error_msg = call_em_llm(client, system_prompt, user_prompt, model, config)

        # Log response
        with open(response_log_path, 'a', encoding='utf-8') as f:
            f.write(f"left_id={left_id}, right_id={right_id} | response='{raw_response}'\n")

        # Log error if any
        if error_msg:
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {pair_id} | {error_msg}\n")
            logger.warning(f"  [{i+1:3d}/{len(data)}] {pair_id} → ERROR: {error_msg}")

        # Get ground truth label if available
        label = pair.get('label', None)

        predictions.append({
            'left_id': left_id,
            'right_id': right_id,
            'left': pair['left'],
            'right': pair['right'],
            'pred': pred,
            'label': label
        })

        if label is not None:
            y_true.append(label)
            y_pred.append(pred)

        status = "✓" if (label is None or pred == label) else "✗"
        logger.info(f"  [{i+1:3d}/{len(data)}] {left_id}_{right_id} → pred={pred} {status}")

    # Calculate metrics if ground truth available
    metrics = {}
    if y_true:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        metrics = {
            'total': len(y_true),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        logger.info("\n" + "-" * 40)
        logger.info("Stage 1 Results:")
        logger.info(f"  Total: {metrics['total']}")
        logger.info(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall:    {recall:.2%}")
        logger.info(f"  F1 Score:  {f1:.2%}")
        logger.info(f"  Accuracy:  {accuracy:.2%}")

    # Save results - JSON
    with open(stage_dir / 'predictions.json', 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Save results - CSV (compatible with original format)
    with open(stage_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['left_id', 'right_id', 'label'])  # 'label' here means prediction
        for p in predictions:
            writer.writerow([p['left_id'], p['right_id'], p['pred']])

    if metrics:
        with open(stage_dir / 'metrics.json', 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

    # Extract matched pairs (pred=1)
    matched_pairs = [p for p in predictions if p['pred'] == 1]
    with open(stage_dir / 'matched_pairs.json', 'w', encoding='utf-8') as f:
        json.dump(matched_pairs, f, indent=2, ensure_ascii=False)

    # Save matched pairs - CSV
    with open(stage_dir / 'matched_pairs.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['left_id', 'right_id'])
        for p in matched_pairs:
            writer.writerow([p['left_id'], p['right_id']])

    logger.info(f"\n  Matched pairs (pred=1): {len(matched_pairs)}")

    return predictions, metrics


# =============================================================================
# STAGE 2: ATTRIBUTE-LEVEL MATCHING
# =============================================================================

def call_attr_llm(client: OpenAI, system_prompt: str, user_prompt: str,
                  model: str, config: Dict) -> Tuple[str, Optional[str]]:
    """
    Call LLM for attribute matching.
    Returns: (raw_response, error_message)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=config.get('attr_max_tokens', 500),
            temperature=config.get('attr_temperature', 0.0)
        )
        return response.choices[0].message.content.strip(), None
    except Exception as e:
        return "", str(e)


def parse_attr_output(raw_response: str) -> List[str]:
    """Parse LLM output for attribute matching."""
    try:
        parsed = json.loads(raw_response)
        raw_output = parsed.get('output', [])
        output_list = []

        # Handle single correspondence case
        if len(raw_output) == 2 and isinstance(raw_output[0], list) and isinstance(raw_output[1], list):
            if all(isinstance(x, str) for x in raw_output[0]) and all(isinstance(x, str) for x in raw_output[1]):
                return [json.dumps(raw_output, ensure_ascii=False)]

        for item in raw_output:
            if isinstance(item, str):
                output_list.append(item)
            elif isinstance(item, list):
                output_list.append(json.dumps(item, ensure_ascii=False))
        return output_list
    except:
        return []


def normalize_correspondence(corr_str: str) -> Optional[Tuple[Tuple, Tuple]]:
    """Normalize correspondence for comparison."""
    try:
        if isinstance(corr_str, str):
            corr = json.loads(corr_str)
        else:
            corr = corr_str
        left = tuple(sorted([x.lower().strip() for x in corr[0]]))
        right = tuple(sorted([x.lower().strip() for x in corr[1]]))
        return (left, right)
    except:
        return None


def normalize_correspondence_strict(corr_str: str) -> Optional[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    Normalize correspondence for strict comparison (Joint Exact-Match).

    Preserves:
    - Case sensitivity
    - Spaces and full nested paths
    - Group structure (no pairwise expansion)

    Ignores:
    - Order of attributes within each side
    - Order of correspondence groups (handled at set level)

    Returns:
        Tuple of (sorted_left_attrs, sorted_right_attrs) or None if parse fails
    """
    try:
        if isinstance(corr_str, str):
            corr = json.loads(corr_str)
        else:
            corr = corr_str

        if not isinstance(corr, list) or len(corr) != 2:
            return None

        left_attrs = corr[0]
        right_attrs = corr[1]

        if not isinstance(left_attrs, list) or not isinstance(right_attrs, list):
            return None

        # Sort within each side but preserve original strings (case, spaces, etc.)
        left = tuple(sorted([str(x) for x in left_attrs]))
        right = tuple(sorted([str(x) for x in right_attrs]))

        return (left, right)
    except (json.JSONDecodeError, TypeError, KeyError, IndexError):
        return None


def normalize_am_set(output_list: List[str]) -> Optional[frozenset]:
    """
    Normalize a list of correspondence strings into a frozenset for comparison.

    Args:
        output_list: List of JSON-encoded correspondence strings

    Returns:
        frozenset of normalized correspondences, or None if any parse fails
    """
    if output_list is None:
        return frozenset()

    normalized = set()
    for corr_str in output_list:
        norm = normalize_correspondence_strict(corr_str)
        if norm is None:
            # Parse failed - return None to indicate invalid output
            return None
        normalized.add(norm)

    return frozenset(normalized)


def evaluate_joint_exact_match(
    all_gt_pairs: List[Dict],
    em_predictions: Dict[Tuple[str, str], int],
    am_predictions: Dict[Tuple[str, str], List[str]],
    am_gt_index: Dict[Tuple[str, str], List[str]],
    logger: logging.Logger
) -> Dict:
    """
    Evaluate Joint Exact-Match Accuracy.

    A pair gets score 1 only if:
    1. EM prediction matches EM ground truth
    2. AM prediction set exactly equals AM ground truth set

    Args:
        all_gt_pairs: All ground truth pairs with 'left_id', 'right_id', 'label'
        em_predictions: Dict mapping (left_id, right_id) -> em_pred
        am_predictions: Dict mapping (left_id, right_id) -> list of correspondence strings
        am_gt_index: Dict mapping (left_id, right_id) -> list of GT correspondence strings
        logger: Logger instance

    Returns:
        Dict with metrics and diagnostics
    """
    if len(all_gt_pairs) == 0:
        raise ValueError("N=0: No ground truth pairs provided for evaluation")

    diagnostics = []
    joint_correct = 0
    total = len(all_gt_pairs)

    # Track issues
    missing_predictions = []
    duplicate_ids = set()
    invalid_am_outputs = []

    # Check for duplicate IDs in GT
    seen_ids = set()
    for pair in all_gt_pairs:
        left_id = str(pair.get('left_id', ''))
        right_id = str(pair.get('right_id', ''))
        key = (left_id, right_id)
        if key in seen_ids:
            duplicate_ids.add(key)
        seen_ids.add(key)

    if duplicate_ids:
        logger.warning(f"  Duplicate ID pairs in GT: {list(duplicate_ids)[:5]}...")

    for pair in all_gt_pairs:
        left_id = str(pair.get('left_id', ''))
        right_id = str(pair.get('right_id', ''))
        key = (left_id, right_id)

        # Get ground truth EM label
        em_label = pair.get('label')
        if em_label is None:
            # Skip pairs without ground truth label
            continue

        # Get EM prediction
        em_pred = em_predictions.get(key)
        if em_pred is None:
            missing_predictions.append(key)
            diag = {
                'left_id': left_id,
                'right_id': right_id,
                'em_correct': False,
                'am_exact_match': False,
                'joint_exact_match': False,
                'reason': 'EM prediction missing'
            }
            diagnostics.append(diag)
            continue

        em_correct = (em_pred == em_label)

        # Get AM ground truth and prediction
        am_gt = am_gt_index.get(key, [])
        am_pred = am_predictions.get(key, [])

        # Validate: For true negatives (label=0), GT AM should be empty
        if em_label == 0 and len(am_gt) > 0:
            raise ValueError(
                f"Invalid GT: Pair ({left_id}, {right_id}) has label=0 but non-empty AM ground truth. "
                "True negatives should have empty AM."
            )

        # Normalize AM sets
        am_gt_set = normalize_am_set(am_gt)
        am_pred_set = normalize_am_set(am_pred)

        # Check for invalid AM output
        am_parse_error = False
        if am_pred_set is None:
            invalid_am_outputs.append(key)
            am_parse_error = True
            am_pred_set = None  # Will cause am_exact_match to be False

        # Determine AM exact match
        if em_label == 1 and em_pred == 1:
            # Both positive: compare AM sets
            if am_parse_error:
                am_exact_match = False
                reason = 'AM parse error'
            elif am_pred_set == am_gt_set:
                am_exact_match = True
                reason = None
            else:
                am_exact_match = False
                # Determine specific reason
                if am_gt_set is None:
                    reason = 'AM GT parse error'
                else:
                    missing = am_gt_set - (am_pred_set or frozenset())
                    extra = (am_pred_set or frozenset()) - am_gt_set
                    if missing and extra:
                        reason = f'AM missing {len(missing)} groups, extra {len(extra)} groups'
                    elif missing:
                        reason = f'AM missing {len(missing)} groups'
                    elif extra:
                        reason = f'AM extra {len(extra)} groups'
                    else:
                        reason = 'AM mismatch (unknown)'

        elif em_label == 1 and em_pred == 0:
            # FN: EM wrong, AM not run
            am_exact_match = False
            reason = 'EM predicted 0 for positive pair'

        elif em_label == 0 and em_pred == 1:
            # FP: EM wrong
            am_exact_match = False
            reason = 'EM predicted 1 for negative pair'

        else:  # em_label == 0 and em_pred == 0
            # TN: Both AM should be empty
            # For y_pred=0, AM is not run, so treat as empty
            am_pred_effective = am_pred_set if am_pred_set is not None else frozenset()
            am_gt_effective = am_gt_set if am_gt_set is not None else frozenset()

            if am_pred_effective == frozenset() and am_gt_effective == frozenset():
                am_exact_match = True
                reason = None
            else:
                am_exact_match = False
                reason = 'TN but AM not empty'

        # Joint exact match
        joint_match = em_correct and am_exact_match
        if joint_match:
            joint_correct += 1

        diag = {
            'left_id': left_id,
            'right_id': right_id,
            'em_label': em_label,
            'em_pred': em_pred,
            'em_correct': em_correct,
            'am_exact_match': am_exact_match,
            'joint_exact_match': joint_match,
            'reason': reason
        }
        diagnostics.append(diag)

    # Report issues
    if missing_predictions:
        logger.warning(f"  Missing EM predictions: {len(missing_predictions)} pairs")
    if invalid_am_outputs:
        logger.warning(f"  Invalid AM outputs (parse failed): {len(invalid_am_outputs)} pairs")

    joint_accuracy = joint_correct / total if total > 0 else 0

    metrics = {
        'joint_exact_match_correct': joint_correct,
        'joint_exact_match_total': total,
        'joint_exact_match_accuracy': joint_accuracy,
        'missing_predictions': len(missing_predictions),
        'invalid_am_outputs': len(invalid_am_outputs),
        'duplicate_gt_ids': len(duplicate_ids)
    }

    logger.info("\n" + "-" * 40)
    logger.info("Joint Exact-Match Results:")
    logger.info(f"  Correct: {joint_correct}/{total}")
    logger.info(f"  Accuracy: {joint_accuracy:.2%}")
    if missing_predictions:
        logger.info(f"  Missing predictions: {len(missing_predictions)}")
    if invalid_am_outputs:
        logger.info(f"  Invalid AM outputs: {len(invalid_am_outputs)}")

    return {
        'metrics': metrics,
        'diagnostics': diagnostics
    }


def run_attribute_matching(
    matched_pairs: List[Dict],
    client: OpenAI,
    dataset_name: str,
    model: str,
    config: Dict,
    attr_gt_index: Dict[Tuple[str, str], List[str]],
    session_dir: Path,
    logger: logging.Logger,
    few_shot_examples: Optional[List[Dict]] = None
) -> Tuple[List[Dict], Dict]:
    """
    Stage 2: Run attribute matching on matched pairs.
    Returns: (results, metrics)
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: ATTRIBUTE-LEVEL MATCHING")
    logger.info("=" * 60)
    logger.info(f"  Processing {len(matched_pairs)} matched pairs...")

    if len(matched_pairs) == 0:
        logger.info("  No matched pairs to process.")
        return [], {}

    prompts = ATTR_PROMPTS[dataset_name]
    system_prompt = prompts['system']
    user_template = prompts['user']

    # Add few-shot examples to system prompt if provided
    if few_shot_examples:
        few_shot_str = build_few_shot_prompt(few_shot_examples)
        system_prompt = system_prompt + few_shot_str
        logger.info(f"  Using {len(few_shot_examples)} few-shot examples")

    results = []
    stage_dir = session_dir / 'stage2_attr'

    # Initialize log files
    prompt_log_path = stage_dir / 'prompt.log'
    response_log_path = stage_dir / 'response.log'
    error_log_path = stage_dir / 'error.log'

    # Clear/create log files
    prompt_log_path.write_text("", encoding='utf-8')
    response_log_path.write_text("", encoding='utf-8')
    error_log_path.write_text("", encoding='utf-8')

    for i, pair in enumerate(matched_pairs):
        left_id = str(pair['left_id'])
        right_id = str(pair['right_id'])
        pair_id = f"{left_id}_{right_id}"

        left_str = serialize_record(pair['left'])
        right_str = serialize_record(pair['right'])

        user_prompt = user_template.replace('{LEFT_RECORD}', left_str).replace('{RIGHT_RECORD}', right_str)

        # Log prompt (system + user for each pair, matching original format)
        with open(prompt_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Pair: left_id={left_id}, right_id={right_id}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n\n")
            f.write(f"[USER PROMPT]\n{user_prompt}\n")

        response, error_msg = call_attr_llm(client, system_prompt, user_prompt, model, config)
        output = parse_attr_output(response)

        # Log response
        with open(response_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{pair_id}]\n{response}\n\n")

        # Log error if any
        if error_msg:
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {pair_id} | {error_msg}\n")
            logger.warning(f"  [{i+1:3d}/{len(matched_pairs)}] {pair_id} → ERROR: {error_msg}")

        # Get ground truth
        gt_key = (left_id, right_id)
        gt = attr_gt_index.get(gt_key, [])

        results.append({
            'left_id': left_id,
            'right_id': right_id,
            'output': output,
            'ground_truth': gt,
            'em_label': pair.get('label'),
            'em_pred': pair.get('pred')
        })

        logger.info(f"  [{i+1:3d}/{len(matched_pairs)}] {pair_id} → {len(output)} correspondences")

    # Evaluate - End-to-End (all pred=1 pairs)
    e2e_metrics = evaluate_attribute_matching(results, logger, mode='end_to_end')

    # Evaluate - Conditional (only TP: pred=1 and label=1)
    tp_results = [r for r in results if r.get('em_label') == 1]
    if tp_results:
        cond_metrics = evaluate_attribute_matching(tp_results, logger, mode='conditional')
    else:
        cond_metrics = {}
        logger.info("\n" + "-" * 40)
        logger.info("Stage 2 Results - Conditional (TP only):")
        logger.info("  No TP pairs to evaluate")

    # Combined metrics
    metrics = {
        'end_to_end': e2e_metrics,
        'conditional': cond_metrics
    }

    # Save results
    with open(stage_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(stage_dir / 'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    return results, metrics


def evaluate_attribute_matching(results: List[Dict], logger: logging.Logger, mode: str = 'end_to_end') -> Dict:
    """
    Evaluate attribute matching results.

    Args:
        results: List of AM results
        logger: Logger instance
        mode: 'end_to_end' (all pred=1) or 'conditional' (only TP: pred=1 and label=1)
    """
    total_tp, total_fp, total_fn = 0, 0, 0

    # Calculate total GT correspondences
    total_gt_correspondences = sum(len(item.get('ground_truth', [])) for item in results)

    for item in results:
        pred_output = item.get('output', [])
        gt_output = item.get('ground_truth', [])

        pred_set = set(filter(None, [normalize_correspondence(p) for p in pred_output]))
        gt_set = set(filter(None, [normalize_correspondence(g) for g in gt_output]))

        tp = len(pred_set & gt_set)
        fp = len(pred_set - gt_set)
        fn = len(gt_set - pred_set)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        item['eval'] = {'tp': tp, 'fp': fp, 'fn': fn}

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0

    exact_match = sum(1 for r in results if r['eval']['fp'] == 0 and r['eval']['fn'] == 0)
    pair_accuracy = exact_match / len(results) if results else 0

    metrics = {
        'total_pairs': len(results),
        'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'pair_accuracy': pair_accuracy,
        'exact_matches': exact_match,
        'total_gt_correspondences': total_gt_correspondences
    }

    mode_label = "End-to-End" if mode == 'end_to_end' else "Conditional (TP only)"
    logger.info("\n" + "-" * 40)
    logger.info(f"Stage 2 Results - {mode_label}:")
    logger.info(f"  Total pairs: {metrics['total_pairs']}")
    logger.info(f"  Total GT correspondences: {total_gt_correspondences}")
    logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
    logger.info(f"  Precision:     {precision:.2%}")
    logger.info(f"  Recall:        {recall:.2%}")
    logger.info(f"  F1 Score:      {f1:.2%}")
    logger.info(f"  Accuracy:      {accuracy:.2%}")
    logger.info(f"  Pair Accuracy: {pair_accuracy:.2%} [{exact_match}/{len(results)}]")

    return metrics


# =============================================================================
# WHOLE-STAGE MODE (Combined EM + Attribute Matching)
# =============================================================================

def call_whole_stage_llm(client: OpenAI, system_prompt: str, user_prompt: str,
                          model: str, config: Dict) -> Tuple[Dict, str, Optional[str]]:
    """
    Call LLM for whole-stage (combined EM + Attribute Matching).
    Returns: (parsed_result, raw_response, error_message)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=config.get('attr_max_tokens', 500),
            temperature=config.get('attr_temperature', 0.0)
        )
        raw_response = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            parsed = json.loads(raw_response)
            return parsed, raw_response, None
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed, raw_response, None
            return {"match": 0, "output": []}, raw_response, "JSON parse error"
    except Exception as e:
        return {"match": 0, "output": []}, "", str(e)


def run_whole_stage(
    data: List[Dict],
    client: OpenAI,
    dataset_name: str,
    model: str,
    config: Dict,
    attr_gt_index: Dict[Tuple[str, str], List[str]],
    session_dir: Path,
    logger: logging.Logger
) -> Tuple[List[Dict], Dict, Dict, Dict]:
    """
    Whole-Stage: Combined EM + Attribute Matching in single call.
    Returns: (results, em_metrics, attr_metrics, joint_metrics)
    """
    logger.info("\n" + "=" * 60)
    logger.info("WHOLE-STAGE: COMBINED EM + ATTRIBUTE MATCHING")
    logger.info("=" * 60)

    prompts = WHOLE_STAGE_PROMPTS[dataset_name]
    system_prompt = prompts['system']
    user_template = prompts['user']

    results = []
    stage_dir = session_dir / 'whole_stage'
    stage_dir.mkdir(exist_ok=True)

    # Initialize log files
    prompt_log_path = stage_dir / 'prompt.log'
    response_log_path = stage_dir / 'response.log'
    error_log_path = stage_dir / 'error.log'

    prompt_log_path.write_text("", encoding='utf-8')
    response_log_path.write_text("", encoding='utf-8')
    error_log_path.write_text("", encoding='utf-8')

    # For EM metrics
    y_true, y_pred = [], []

    for i, pair in enumerate(data):
        left_id = str(pair.get('left_id')) if pair.get('left_id') else extract_id(pair['left'])
        right_id = str(pair.get('right_id')) if pair.get('right_id') else extract_id(pair['right'])
        pair_id = f"{left_id}_{right_id}"

        left_str = serialize_record(pair['left'])
        right_str = serialize_record(pair['right'])

        user_prompt = user_template.replace('{LEFT_RECORD}', left_str).replace('{RIGHT_RECORD}', right_str)

        # Log prompt
        with open(prompt_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Pair: left_id={left_id}, right_id={right_id}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n\n")
            f.write(f"[USER PROMPT]\n{user_prompt}\n")

        # Call LLM
        parsed, raw_response, error_msg = call_whole_stage_llm(
            client, system_prompt, user_prompt, model, config
        )

        # Log response
        with open(response_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{pair_id}]\n{raw_response}\n\n")

        # Log error if any
        if error_msg:
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {pair_id} | {error_msg}\n")
            logger.warning(f"  [{i+1:3d}/{len(data)}] {pair_id} → ERROR: {error_msg}")

        # Extract results
        em_pred = parsed.get('match', 0)
        attr_output = parsed.get('output', [])

        # Normalize attr_output
        output_list = []
        for item in attr_output:
            if isinstance(item, str):
                output_list.append(item)
            elif isinstance(item, list):
                output_list.append(json.dumps(item, ensure_ascii=False))

        # Get ground truth
        em_label = pair.get('label', None)
        gt_key = (left_id, right_id)
        attr_gt = attr_gt_index.get(gt_key, [])

        results.append({
            'left_id': left_id,
            'right_id': right_id,
            'em_pred': em_pred,
            'em_label': em_label,
            'attr_output': output_list,
            'attr_gt': attr_gt
        })

        # Track EM metrics
        if em_label is not None:
            y_true.append(em_label)
            y_pred.append(em_pred)

        status = "✓" if (em_label is None or em_pred == em_label) else "✗"
        logger.info(f"  [{i+1:3d}/{len(data)}] {pair_id} → match={em_pred} {status}, {len(output_list)} correspondences")

    # Calculate EM metrics
    em_metrics = {}
    if y_true:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        em_metrics = {
            'total': len(y_true),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        logger.info("\n" + "-" * 40)
        logger.info("Entity Matching Results:")
        logger.info(f"  Total: {em_metrics['total']}")
        logger.info(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall:    {recall:.2%}")
        logger.info(f"  F1 Score:  {f1:.2%}")

    # Calculate Attribute metrics - both End-to-End and Conditional
    attr_metrics = {}

    # End-to-End: all pred=1 pairs
    e2e_results = [r for r in results if r['em_pred'] == 1]
    e2e_metrics = {}
    if e2e_results:
        total_tp, total_fp, total_fn = 0, 0, 0
        for item in e2e_results:
            pred_output = item.get('attr_output', [])
            gt_output = item.get('attr_gt', [])
            pred_set = set(filter(None, [normalize_correspondence(p) for p in pred_output]))
            gt_set = set(filter(None, [normalize_correspondence(g) for g in gt_output]))
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            item['attr_eval_e2e'] = {'tp': tp, 'fp': fp, 'fn': fn}

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        exact_match = sum(1 for r in e2e_results if r.get('attr_eval_e2e', {}).get('fp', 1) == 0 and r.get('attr_eval_e2e', {}).get('fn', 1) == 0)
        pair_accuracy = exact_match / len(e2e_results) if e2e_results else 0

        e2e_metrics = {
            'total_pairs': len(e2e_results),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'pair_accuracy': pair_accuracy, 'exact_matches': exact_match
        }

        logger.info("\n" + "-" * 40)
        logger.info("Attribute Matching Results - End-to-End (pred=1):")
        logger.info(f"  Total pairs: {e2e_metrics['total_pairs']}")
        logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"  Precision:     {precision:.2%}")
        logger.info(f"  Recall:        {recall:.2%}")
        logger.info(f"  F1 Score:      {f1:.2%}")
        logger.info(f"  Accuracy:      {accuracy:.2%}")
        logger.info(f"  Pair Accuracy: {pair_accuracy:.2%} [{exact_match}/{len(e2e_results)}]")

    # Conditional: only TP (pred=1 AND label=1)
    cond_results = [r for r in results if r['em_pred'] == 1 and r.get('em_label') == 1]
    cond_metrics = {}
    if cond_results:
        total_tp, total_fp, total_fn = 0, 0, 0
        for item in cond_results:
            pred_output = item.get('attr_output', [])
            gt_output = item.get('attr_gt', [])
            pred_set = set(filter(None, [normalize_correspondence(p) for p in pred_output]))
            gt_set = set(filter(None, [normalize_correspondence(g) for g in gt_output]))
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            item['attr_eval_cond'] = {'tp': tp, 'fp': fp, 'fn': fn}

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        exact_match = sum(1 for r in cond_results if r.get('attr_eval_cond', {}).get('fp', 1) == 0 and r.get('attr_eval_cond', {}).get('fn', 1) == 0)
        pair_accuracy = exact_match / len(cond_results) if cond_results else 0

        cond_metrics = {
            'total_pairs': len(cond_results),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'pair_accuracy': pair_accuracy, 'exact_matches': exact_match
        }

        logger.info("\n" + "-" * 40)
        logger.info("Attribute Matching Results - Conditional (TP only):")
        logger.info(f"  Total pairs: {cond_metrics['total_pairs']}")
        logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"  Precision:     {precision:.2%}")
        logger.info(f"  Recall:        {recall:.2%}")
        logger.info(f"  F1 Score:      {f1:.2%}")
        logger.info(f"  Accuracy:      {accuracy:.2%}")
        logger.info(f"  Pair Accuracy: {pair_accuracy:.2%} [{exact_match}/{len(cond_results)}]")

    attr_metrics = {
        'end_to_end': e2e_metrics,
        'conditional': cond_metrics
    }

    # Save results
    with open(stage_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(stage_dir / 'em_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(em_metrics, f, indent=2)

    with open(stage_dir / 'attr_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(attr_metrics, f, indent=2)

    # Save predictions CSV
    with open(stage_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['left_id', 'right_id', 'em_pred', 'em_label', 'num_correspondences'])
        for r in results:
            writer.writerow([r['left_id'], r['right_id'], r['em_pred'], r.get('em_label', ''), len(r['attr_output'])])

    # Calculate Joint Exact-Match Accuracy
    joint_metrics = {}
    if y_true:  # Only if we have ground truth
        # Build prediction dictionaries
        em_predictions_dict = {}
        am_predictions_dict = {}
        for r in results:
            key = (r['left_id'], r['right_id'])
            em_predictions_dict[key] = r['em_pred']
            # For em_pred=0, AM is not run, so use empty list
            am_predictions_dict[key] = r['attr_output'] if r['em_pred'] == 1 else []

        # Build GT pairs list
        all_gt_pairs = []
        for r in results:
            all_gt_pairs.append({
                'left_id': r['left_id'],
                'right_id': r['right_id'],
                'label': r.get('em_label')
            })

        try:
            joint_result = evaluate_joint_exact_match(
                all_gt_pairs=all_gt_pairs,
                em_predictions=em_predictions_dict,
                am_predictions=am_predictions_dict,
                am_gt_index=attr_gt_index,
                logger=logger
            )
            joint_metrics = joint_result['metrics']

            # Save joint metrics
            with open(stage_dir / 'joint_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(joint_metrics, f, indent=2)

            # Save diagnostics
            with open(stage_dir / 'joint_diagnostics.json', 'w', encoding='utf-8') as f:
                json.dump(joint_result['diagnostics'], f, indent=2, ensure_ascii=False)

        except ValueError as e:
            logger.error(f"Joint Exact-Match evaluation failed: {e}")

    return results, em_metrics, attr_metrics, joint_metrics


# =============================================================================
# EVIDENCE-DRIVEN MODE
# Key idea: Analyze attribute-level evidence FIRST, then derive EM decision
# The matched attributes serve as supporting evidence for the decision
# =============================================================================

def call_evidence_driven_llm(client: OpenAI, system_prompt: str, user_prompt: str,
                              model: str, config: Dict) -> Tuple[Dict, str, Optional[str]]:
    """
    Call LLM for evidence-driven reasoning.
    Returns: (parsed_result, raw_response, error_message)
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_completion_tokens=config.get('attr_max_tokens', 500),
            temperature=config.get('attr_temperature', 0.0)
        )
        raw_response = response.choices[0].message.content.strip()

        # Parse JSON response
        try:
            parsed = json.loads(raw_response)
            return parsed, raw_response, None
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return parsed, raw_response, None
            return {"match": 0, "evidence": []}, raw_response, "JSON parse error"
    except Exception as e:
        return {"match": 0, "evidence": []}, "", str(e)


def run_evidence_driven(
    data: List[Dict],
    client: OpenAI,
    dataset_name: str,
    model: str,
    config: Dict,
    attr_gt_index: Dict[Tuple[str, str], List[str]],
    session_dir: Path,
    logger: logging.Logger
) -> Tuple[List[Dict], Dict, Dict, Dict]:
    """
    Evidence-Driven: Analyze attribute evidence to derive EM decision.

    Workflow:
        Record Pair
            ↓
        Attribute-level evidence reasoning
            ↓
            ├── Entity Matching prediction
            └── Matched attribute pairs (evidence)

    Returns: (results, em_metrics, attr_metrics, joint_metrics)
    """
    logger.info("\n" + "=" * 60)
    logger.info("EVIDENCE-DRIVEN: ATTRIBUTE EVIDENCE → EM DECISION")
    logger.info("=" * 60)

    prompts = EVIDENCE_DRIVEN_PROMPTS[dataset_name]
    system_prompt = prompts['system']
    user_template = prompts['user']

    results = []
    stage_dir = session_dir / 'evidence_driven'
    stage_dir.mkdir(exist_ok=True)

    # Initialize log files
    prompt_log_path = stage_dir / 'prompt.log'
    response_log_path = stage_dir / 'response.log'
    error_log_path = stage_dir / 'error.log'

    prompt_log_path.write_text("", encoding='utf-8')
    response_log_path.write_text("", encoding='utf-8')
    error_log_path.write_text("", encoding='utf-8')

    # For EM metrics
    y_true, y_pred = [], []

    for i, pair in enumerate(data):
        left_id = str(pair.get('left_id')) if pair.get('left_id') else extract_id(pair['left'])
        right_id = str(pair.get('right_id')) if pair.get('right_id') else extract_id(pair['right'])
        pair_id = f"{left_id}_{right_id}"

        left_str = serialize_record(pair['left'])
        right_str = serialize_record(pair['right'])

        user_prompt = user_template.replace('{LEFT_RECORD}', left_str).replace('{RIGHT_RECORD}', right_str)

        # Log prompt
        with open(prompt_log_path, 'a', encoding='utf-8') as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"Pair: left_id={left_id}, right_id={right_id}\n")
            f.write(f"{'=' * 60}\n")
            f.write(f"[SYSTEM PROMPT]\n{system_prompt}\n\n")
            f.write(f"[USER PROMPT]\n{user_prompt}\n")

        # Call LLM
        parsed, raw_response, error_msg = call_evidence_driven_llm(
            client, system_prompt, user_prompt, model, config
        )

        # Log response
        with open(response_log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{pair_id}]\n{raw_response}\n\n")

        # Log error if any
        if error_msg:
            with open(error_log_path, 'a', encoding='utf-8') as f:
                f.write(f"{datetime.now().isoformat()} | ERROR | {pair_id} | {error_msg}\n")
            logger.warning(f"  [{i+1:3d}/{len(data)}] {pair_id} → ERROR: {error_msg}")

        # Extract results - note: 'evidence' field instead of 'output'
        em_pred = parsed.get('match', 0)
        evidence_output = parsed.get('evidence', [])

        # Normalize evidence output (same format as attr_output)
        output_list = []
        for item in evidence_output:
            if isinstance(item, str):
                output_list.append(item)
            elif isinstance(item, list):
                output_list.append(json.dumps(item, ensure_ascii=False))

        # Get ground truth
        em_label = pair.get('label', None)
        gt_key = (left_id, right_id)
        attr_gt = attr_gt_index.get(gt_key, [])

        results.append({
            'left_id': left_id,
            'right_id': right_id,
            'em_pred': em_pred,
            'em_label': em_label,
            'evidence': output_list,  # attribute pairs as evidence
            'attr_gt': attr_gt
        })

        # Track EM metrics
        if em_label is not None:
            y_true.append(em_label)
            y_pred.append(em_pred)

        status = "✓" if (em_label is None or em_pred == em_label) else "✗"
        logger.info(f"  [{i+1:3d}/{len(data)}] {pair_id} → match={em_pred} {status}, {len(output_list)} evidence pairs")

    # Calculate EM metrics
    em_metrics = {}
    if y_true:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(y_true)

        em_metrics = {
            'total': len(y_true),
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        }

        logger.info("\n" + "-" * 40)
        logger.info("Entity Matching Results:")
        logger.info(f"  Total: {em_metrics['total']}")
        logger.info(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        logger.info(f"  Precision: {precision:.2%}")
        logger.info(f"  Recall:    {recall:.2%}")
        logger.info(f"  F1 Score:  {f1:.2%}")

    # Calculate Attribute metrics - both End-to-End and Conditional
    attr_metrics = {}

    # End-to-End: all pred=1 pairs (evidence serves as attribute correspondences)
    e2e_results = [r for r in results if r['em_pred'] == 1]
    e2e_metrics = {}
    if e2e_results:
        total_tp, total_fp, total_fn = 0, 0, 0
        for item in e2e_results:
            pred_output = item.get('evidence', [])
            gt_output = item.get('attr_gt', [])
            pred_set = set(filter(None, [normalize_correspondence(p) for p in pred_output]))
            gt_set = set(filter(None, [normalize_correspondence(g) for g in gt_output]))
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            item['attr_eval_e2e'] = {'tp': tp, 'fp': fp, 'fn': fn}

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        exact_match = sum(1 for r in e2e_results if r.get('attr_eval_e2e', {}).get('fp', 1) == 0 and r.get('attr_eval_e2e', {}).get('fn', 1) == 0)
        pair_accuracy = exact_match / len(e2e_results) if e2e_results else 0

        e2e_metrics = {
            'total_pairs': len(e2e_results),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'pair_accuracy': pair_accuracy, 'exact_matches': exact_match
        }

        logger.info("\n" + "-" * 40)
        logger.info("Attribute Evidence Results - End-to-End (pred=1):")
        logger.info(f"  Total pairs: {e2e_metrics['total_pairs']}")
        logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"  Precision:     {precision:.2%}")
        logger.info(f"  Recall:        {recall:.2%}")
        logger.info(f"  F1 Score:      {f1:.2%}")
        logger.info(f"  Accuracy:      {accuracy:.2%}")
        logger.info(f"  Pair Accuracy: {pair_accuracy:.2%} [{exact_match}/{len(e2e_results)}]")

    # Conditional: only TP (pred=1 AND label=1)
    cond_results = [r for r in results if r['em_pred'] == 1 and r.get('em_label') == 1]
    cond_metrics = {}
    if cond_results:
        total_tp, total_fp, total_fn = 0, 0, 0
        for item in cond_results:
            pred_output = item.get('evidence', [])
            gt_output = item.get('attr_gt', [])
            pred_set = set(filter(None, [normalize_correspondence(p) for p in pred_output]))
            gt_set = set(filter(None, [normalize_correspondence(g) for g in gt_output]))
            tp = len(pred_set & gt_set)
            fp = len(pred_set - gt_set)
            fn = len(gt_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn
            item['attr_eval_cond'] = {'tp': tp, 'fp': fp, 'fn': fn}

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
        exact_match = sum(1 for r in cond_results if r.get('attr_eval_cond', {}).get('fp', 1) == 0 and r.get('attr_eval_cond', {}).get('fn', 1) == 0)
        pair_accuracy = exact_match / len(cond_results) if cond_results else 0

        cond_metrics = {
            'total_pairs': len(cond_results),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
            'precision': precision, 'recall': recall, 'f1': f1,
            'accuracy': accuracy, 'pair_accuracy': pair_accuracy, 'exact_matches': exact_match
        }

        logger.info("\n" + "-" * 40)
        logger.info("Attribute Evidence Results - Conditional (TP only):")
        logger.info(f"  Total pairs: {cond_metrics['total_pairs']}")
        logger.info(f"  TP={total_tp}, FP={total_fp}, FN={total_fn}")
        logger.info(f"  Precision:     {precision:.2%}")
        logger.info(f"  Recall:        {recall:.2%}")
        logger.info(f"  F1 Score:      {f1:.2%}")
        logger.info(f"  Accuracy:      {accuracy:.2%}")
        logger.info(f"  Pair Accuracy: {pair_accuracy:.2%} [{exact_match}/{len(cond_results)}]")

    attr_metrics = {
        'end_to_end': e2e_metrics,
        'conditional': cond_metrics
    }

    # Save results
    with open(stage_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(stage_dir / 'em_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(em_metrics, f, indent=2)

    with open(stage_dir / 'attr_metrics.json', 'w', encoding='utf-8') as f:
        json.dump(attr_metrics, f, indent=2)

    # Save predictions CSV
    with open(stage_dir / 'predictions.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['left_id', 'right_id', 'em_pred', 'em_label', 'num_evidence_pairs'])
        for r in results:
            writer.writerow([r['left_id'], r['right_id'], r['em_pred'], r.get('em_label', ''), len(r['evidence'])])

    # Calculate Joint Exact-Match Accuracy
    joint_metrics = {}
    if y_true:  # Only if we have ground truth
        # Build prediction dictionaries
        em_predictions_dict = {}
        am_predictions_dict = {}
        for r in results:
            key = (r['left_id'], r['right_id'])
            em_predictions_dict[key] = r['em_pred']
            # For em_pred=0, evidence is still output but for AM purposes treat as empty
            am_predictions_dict[key] = r['evidence'] if r['em_pred'] == 1 else []

        # Build GT pairs list
        all_gt_pairs = []
        for r in results:
            all_gt_pairs.append({
                'left_id': r['left_id'],
                'right_id': r['right_id'],
                'label': r.get('em_label')
            })

        try:
            joint_result = evaluate_joint_exact_match(
                all_gt_pairs=all_gt_pairs,
                em_predictions=em_predictions_dict,
                am_predictions=am_predictions_dict,
                am_gt_index=attr_gt_index,
                logger=logger
            )
            joint_metrics = joint_result['metrics']

            # Save joint metrics
            with open(stage_dir / 'joint_metrics.json', 'w', encoding='utf-8') as f:
                json.dump(joint_metrics, f, indent=2)

            # Save diagnostics
            with open(stage_dir / 'joint_diagnostics.json', 'w', encoding='utf-8') as f:
                json.dump(joint_result['diagnostics'], f, indent=2, ensure_ascii=False)

        except ValueError as e:
            logger.error(f"Joint Exact-Match evaluation failed: {e}")

    return results, em_metrics, attr_metrics, joint_metrics


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    dataset_name: str,
    model: str = DEFAULT_MODEL,
    limit: Optional[int] = None,
    mode: str = 'two_stage',
    data_file: Optional[str] = None,
    stage1_only: bool = False,
    stage2_only: bool = False,
    log_prefix: str = '',
    few_shot_file: Optional[str] = None,
    few_shot_num: int = 5
) -> Dict:
    """
    Run the GEM pipeline.

    Args:
        dataset_name: Name of the dataset (nyc, semi_rel, semi_heter)
        model: LLM model to use
        limit: Limit number of pairs (for testing)
        mode: 'two_stage', 'whole', or 'evidence' - pipeline mode
        data_file: Custom data file path (overrides dataset config)
        stage1_only: If True, only run Stage 1 (Entity Matching)
        stage2_only: If True, only run Stage 2 (Attribute Matching), skip EM
        log_prefix: Prefix for log folder name (e.g., "gold")

    Returns:
        Dictionary with results
    """
    # Load environment
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not set")
        return {}

    client = OpenAI(api_key=api_key)

    # Get dataset config
    if dataset_name not in DATASETS:
        print(f"Unknown dataset: {dataset_name}")
        print(f"Available: {list(DATASETS.keys())}")
        return {}

    dataset_cfg = DATASETS[dataset_name]
    model_cfg = MODEL_CONFIGS.get(model, MODEL_CONFIGS[DEFAULT_MODEL])

    # Setup logging (auto-creates {dataset}_{mode} folder)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger, session_dir = setup_logging(LOG_DIR, dataset_name, model, mode, log_prefix)

    mode_name_map = {
        'two_stage': 'TWO-STAGE',
        'whole': 'WHOLE-STAGE',
        'evidence': 'EVIDENCE-DRIVEN'
    }
    mode_name = mode_name_map.get(mode, mode.upper())
    logger.info("=" * 60)
    logger.info(f"{mode_name} GEM PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Dataset:   {dataset_cfg['name']}")
    logger.info(f"Model:     {model}")
    logger.info(f"Mode:      {mode}")
    logger.info(f"Log dir:   {session_dir}")

    # Load data (use custom data_file if provided)
    actual_data_file = Path(data_file) if data_file else dataset_cfg['em_data_file']
    data = load_data(actual_data_file)
    if limit:
        data = data[:limit]
        logger.info(f"Limited to {limit} pairs")
    logger.info(f"Loaded {len(data)} pairs from {actual_data_file}")

    # Load attribute-level ground truth
    attr_gt_index = {}
    if dataset_cfg['attr_gt_file']:
        attr_gt_index = load_attr_ground_truth(dataset_cfg['attr_gt_file'])
        logger.info(f"Loaded {len(attr_gt_index)} attribute GT entries")

    # Load few-shot examples if provided
    few_shot_examples = None
    if few_shot_file:
        few_shot_examples = load_few_shot_examples(Path(few_shot_file), num_examples=few_shot_num)
        logger.info(f"Loaded {len(few_shot_examples)} few-shot examples from {few_shot_file}")

    if mode == 'two_stage':
        # =========================================================================
        # TWO-STAGE MODE
        # =========================================================================

        if stage2_only:
            # Skip Stage 1, treat all input pairs as matched
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 2 ONLY MODE - Skipping Entity Matching")
            logger.info("=" * 60)
            em_predictions = []
            em_metrics = {}
            matched_pairs = []
            for pair in data:
                left_id = str(pair.get('left_id')) if pair.get('left_id') else extract_id(pair['left'])
                right_id = str(pair.get('right_id')) if pair.get('right_id') else extract_id(pair['right'])
                matched_pairs.append({
                    'left_id': left_id,
                    'right_id': right_id,
                    'left': pair['left'],
                    'right': pair['right'],
                    'pred': 1,
                    'label': pair.get('label', 1)
                })
            logger.info(f"  Treating all {len(matched_pairs)} pairs as matched")
        else:
            # STAGE 1: Entity Matching
            em_predictions, em_metrics = run_entity_matching(
                data, client, dataset_name, model, model_cfg, session_dir, logger
            )

            # Filter matched pairs (pred=1)
            matched_pairs = [p for p in em_predictions if p['pred'] == 1]

        # If stage1_only, skip Stage 2
        if stage1_only:
            logger.info("\n" + "=" * 60)
            logger.info("STAGE 1 ONLY - SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Total input pairs:      {len(data)}")
            logger.info(f"Stage 1 matched (pred=1): {len(matched_pairs)}")

            if em_metrics:
                logger.info(f"\nStage 1 (Entity Matching):")
                logger.info(f"  F1: {em_metrics['f1']:.2%}, Precision: {em_metrics['precision']:.2%}, Recall: {em_metrics['recall']:.2%}")

            final_results = {
                'dataset': dataset_name,
                'model': model,
                'mode': 'stage1_only',
                'total_pairs': len(data),
                'matched_pairs': len(matched_pairs),
                'stage1_em': em_metrics,
                'stage2_attr': None
            }

            with open(session_dir / 'final_results.json', 'w', encoding='utf-8') as f:
                json.dump(final_results, f, indent=2)

            return final_results

        # STAGE 2: Attribute-Level Matching
        attr_results, attr_metrics = run_attribute_matching(
            matched_pairs, client, dataset_name, model, model_cfg, attr_gt_index, session_dir, logger,
            few_shot_examples=few_shot_examples
        )

        # Calculate Joint Exact-Match Accuracy
        joint_metrics = {}
        if em_metrics:  # Only if we have ground truth
            # Build prediction dictionaries from both stages
            em_predictions_dict = {}
            am_predictions_dict = {}

            for p in em_predictions:
                key = (p['left_id'], p['right_id'])
                em_predictions_dict[key] = p['pred']
                # Default to empty AM for pairs not predicted as match
                am_predictions_dict[key] = []

            # Update AM predictions for matched pairs
            for r in attr_results:
                key = (r['left_id'], r['right_id'])
                am_predictions_dict[key] = r.get('output', [])

            # Build GT pairs list
            all_gt_pairs = []
            for p in em_predictions:
                if p.get('label') is not None:
                    all_gt_pairs.append({
                        'left_id': p['left_id'],
                        'right_id': p['right_id'],
                        'label': p['label']
                    })

            if all_gt_pairs:
                try:
                    joint_result = evaluate_joint_exact_match(
                        all_gt_pairs=all_gt_pairs,
                        em_predictions=em_predictions_dict,
                        am_predictions=am_predictions_dict,
                        am_gt_index=attr_gt_index,
                        logger=logger
                    )
                    joint_metrics = joint_result['metrics']

                    # Save joint metrics
                    with open(session_dir / 'joint_metrics.json', 'w', encoding='utf-8') as f:
                        json.dump(joint_metrics, f, indent=2)

                    # Save diagnostics
                    with open(session_dir / 'joint_diagnostics.json', 'w', encoding='utf-8') as f:
                        json.dump(joint_result['diagnostics'], f, indent=2, ensure_ascii=False)

                except ValueError as e:
                    logger.error(f"Joint Exact-Match evaluation failed: {e}")

        # FINAL SUMMARY
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total input pairs:      {len(data)}")
        logger.info(f"Stage 1 matched (pred=1): {len(matched_pairs)}")
        logger.info(f"Stage 2 processed:      {len(attr_results)}")

        if em_metrics:
            logger.info(f"\nStage 1 (Entity Matching):")
            logger.info(f"  F1: {em_metrics['f1']:.2%}, Precision: {em_metrics['precision']:.2%}, Recall: {em_metrics['recall']:.2%}")

        if attr_metrics:
            e2e = attr_metrics.get('end_to_end', {})
            cond = attr_metrics.get('conditional', {})
            if e2e:
                logger.info(f"\nStage 2 (Attribute Matching) - End-to-End:")
                logger.info(f"  F1: {e2e['f1']:.2%}, Precision: {e2e['precision']:.2%}, Recall: {e2e['recall']:.2%}, Accuracy: {e2e.get('accuracy', 0):.2%}")
            if cond:
                logger.info(f"\nStage 2 (Attribute Matching) - Conditional (TP only):")
                logger.info(f"  F1: {cond['f1']:.2%}, Precision: {cond['precision']:.2%}, Recall: {cond['recall']:.2%}, Accuracy: {cond.get('accuracy', 0):.2%}")

        if joint_metrics:
            logger.info(f"\nJoint Exact-Match:")
            logger.info(f"  Accuracy: {joint_metrics['joint_exact_match_accuracy']:.2%} [{joint_metrics['joint_exact_match_correct']}/{joint_metrics['joint_exact_match_total']}]")

        # Save final results
        final_results = {
            'dataset': dataset_name,
            'model': model,
            'mode': mode,
            'total_pairs': len(data),
            'matched_pairs': len(matched_pairs),
            'stage1_em': em_metrics,
            'stage2_attr': attr_metrics,
            'joint_metrics': joint_metrics
        }

    elif mode == 'whole':
        # =========================================================================
        # WHOLE-STAGE MODE
        # =========================================================================
        results, em_metrics, attr_metrics, joint_metrics = run_whole_stage(
            data, client, dataset_name, model, model_cfg, attr_gt_index, session_dir, logger
        )

        matched_count = sum(1 for r in results if r['em_pred'] == 1)

        # FINAL SUMMARY
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total input pairs:      {len(data)}")
        logger.info(f"Predicted matches:      {matched_count}")

        if em_metrics:
            logger.info(f"\nEntity Matching:")
            logger.info(f"  F1: {em_metrics['f1']:.2%}, Precision: {em_metrics['precision']:.2%}, Recall: {em_metrics['recall']:.2%}")

        if attr_metrics:
            e2e = attr_metrics.get('end_to_end', {})
            cond = attr_metrics.get('conditional', {})
            if e2e:
                logger.info(f"\nAttribute Matching - End-to-End (pred=1):")
                logger.info(f"  F1: {e2e['f1']:.2%}, Precision: {e2e['precision']:.2%}, Recall: {e2e['recall']:.2%}, Accuracy: {e2e.get('accuracy', 0):.2%}")
            if cond:
                logger.info(f"\nAttribute Matching - Conditional (TP only):")
                logger.info(f"  F1: {cond['f1']:.2%}, Precision: {cond['precision']:.2%}, Recall: {cond['recall']:.2%}, Accuracy: {cond.get('accuracy', 0):.2%}")

        if joint_metrics:
            logger.info(f"\nJoint Exact-Match:")
            logger.info(f"  Accuracy: {joint_metrics['joint_exact_match_accuracy']:.2%} [{joint_metrics['joint_exact_match_correct']}/{joint_metrics['joint_exact_match_total']}]")

        # Save final results
        final_results = {
            'dataset': dataset_name,
            'model': model,
            'mode': mode,
            'total_pairs': len(data),
            'matched_pairs': matched_count,
            'em_metrics': em_metrics,
            'attr_metrics': attr_metrics,
            'joint_metrics': joint_metrics
        }

    else:  # mode == 'evidence'
        # =========================================================================
        # EVIDENCE-DRIVEN MODE
        # =========================================================================
        results, em_metrics, attr_metrics, joint_metrics = run_evidence_driven(
            data, client, dataset_name, model, model_cfg, attr_gt_index, session_dir, logger
        )

        matched_count = sum(1 for r in results if r['em_pred'] == 1)

        # FINAL SUMMARY
        logger.info("\n" + "=" * 60)
        logger.info("FINAL SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total input pairs:      {len(data)}")
        logger.info(f"Predicted matches:      {matched_count}")

        if em_metrics:
            logger.info(f"\nEntity Matching (Evidence-Driven):")
            logger.info(f"  F1: {em_metrics['f1']:.2%}, Precision: {em_metrics['precision']:.2%}, Recall: {em_metrics['recall']:.2%}")

        if attr_metrics:
            e2e = attr_metrics.get('end_to_end', {})
            cond = attr_metrics.get('conditional', {})
            if e2e:
                logger.info(f"\nAttribute Evidence - End-to-End (pred=1):")
                logger.info(f"  F1: {e2e['f1']:.2%}, Precision: {e2e['precision']:.2%}, Recall: {e2e['recall']:.2%}, Accuracy: {e2e.get('accuracy', 0):.2%}")
            if cond:
                logger.info(f"\nAttribute Evidence - Conditional (TP only):")
                logger.info(f"  F1: {cond['f1']:.2%}, Precision: {cond['precision']:.2%}, Recall: {cond['recall']:.2%}, Accuracy: {cond.get('accuracy', 0):.2%}")

        if joint_metrics:
            logger.info(f"\nJoint Exact-Match:")
            logger.info(f"  Accuracy: {joint_metrics['joint_exact_match_accuracy']:.2%} [{joint_metrics['joint_exact_match_correct']}/{joint_metrics['joint_exact_match_total']}]")

        # Save final results
        final_results = {
            'dataset': dataset_name,
            'model': model,
            'mode': mode,
            'total_pairs': len(data),
            'matched_pairs': matched_count,
            'em_metrics': em_metrics,
            'attr_metrics': attr_metrics,
            'joint_metrics': joint_metrics
        }

    with open(session_dir / 'final_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nResults saved to: {session_dir}")

    return final_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='GEM Pipeline (Two-Stage, Whole-Stage, or Evidence-Driven)')
    parser.add_argument('--dataset', '-d', required=True,
                        choices=['nyc', 'semi_rel', 'semi_heter', 'all'],
                        help='Dataset to run')
    parser.add_argument('--model', '-m', default=DEFAULT_MODEL,
                        help=f'Model to use (default: {DEFAULT_MODEL})')
    parser.add_argument('--limit', '-l', type=int, default=None,
                        help='Limit number of pairs (for testing)')
    parser.add_argument('--mode', choices=['two_stage', 'whole', 'evidence'], default='two_stage',
                        help='Pipeline mode: two_stage (default), whole, or evidence')
    parser.add_argument('--data-file', type=str, default=None,
                        help='Custom data file path (overrides dataset config)')
    parser.add_argument('--stage1-only', action='store_true',
                        help='Only run Stage 1 (Entity Matching) - only for two_stage mode')
    parser.add_argument('--stage2-only', action='store_true',
                        help='Only run Stage 2 (Attribute Matching) - skip EM, treat all pairs as matched')
    parser.add_argument('--log-prefix', type=str, default='',
                        help='Prefix for log folder name (e.g., "gold" -> gold_nyc_two_stage/)')
    parser.add_argument('--few-shot-file', type=str, default=None,
                        help='JSON file for few-shot examples (Stage 2). Format: {"pairs_file": "...", "gt_file": "..."}')
    parser.add_argument('--few-shot-num', type=int, default=5,
                        help='Number of few-shot examples to use (default: 5)')
    args = parser.parse_args()

    if args.dataset == 'all':
        all_results = {}
        for ds in ['nyc', 'semi_rel', 'semi_heter']:
            print(f"\n{'#' * 60}")
            print(f"# Dataset: {ds} | Mode: {args.mode}")
            print(f"{'#' * 60}\n")
            results = run_pipeline(ds, args.model, args.limit, args.mode,
                                   args.data_file, args.stage1_only, args.stage2_only, args.log_prefix,
                                   args.few_shot_file, args.few_shot_num)
            if results:
                all_results[ds] = results

        # Print summary table
        if all_results:
            print(f"\n{'=' * 80}")
            print(f"SUMMARY (All Datasets) - Mode: {args.mode}")
            print(f"{'=' * 80}")
            print(f"{'Dataset':<12} {'EM F1':<10} {'EM Prec':<10} {'EM Rec':<10} {'Attr F1':<10} {'Attr Prec':<10} {'Attr Rec':<10}")
            print("-" * 80)
            for ds, r in all_results.items():
                # Handle both two_stage and whole mode result formats
                em = r.get('stage1_em') or r.get('em_metrics') or {}
                attr = r.get('stage2_attr') or r.get('attr_metrics') or {}
                print(f"{ds:<12} "
                      f"{em.get('f1', 0):<10.2%} "
                      f"{em.get('precision', 0):<10.2%} "
                      f"{em.get('recall', 0):<10.2%} "
                      f"{attr.get('f1', 0):<10.2%} "
                      f"{attr.get('precision', 0):<10.2%} "
                      f"{attr.get('recall', 0):<10.2%}")
    else:
        run_pipeline(args.dataset, args.model, args.limit, args.mode,
                     args.data_file, args.stage1_only, args.stage2_only, args.log_prefix,
                     args.few_shot_file, args.few_shot_num)


if __name__ == '__main__':
    main()
