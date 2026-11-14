"""
Baseline: TF-IDF + Cosine Similarity Matching

Model: TF-IDF vectorization + Cosine similarity
Features: Pure statistical method, no neural networks, fast and interpretable

EM: Record serialization -> TF-IDF -> Cosine Similarity
AM: Two-stage matching
    Stage 1: Single attribute value TF-IDF matching (1-to-1)
    Stage 2: Attribute group TF-IDF matching (N-to-M)
"""

import json
import numpy as np
from itertools import combinations
from typing import List, Dict, Tuple, Optional, FrozenSet
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Configuration parameters"""
    em_threshold: float = 0.1  # EM threshold
    am_threshold: float = 0.5  # AM threshold
    max_group_size: Optional[int] = None  # None = dynamic
    ngram_range: Tuple[int, int] = (1, 2)  # TF-IDF n-gram range
    min_df: int = 1  # Minimum document frequency
    max_df: float = 1.0  # Maximum document frequency


def flatten_record(record: dict, prefix: str = '') -> Dict[str, str]:
    """Flatten nested record to {attr_path: value}"""
    result = {}
    for key, value in record.items():
        if key.lower() in ['_id', 'id']:
            continue
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_record(value, full_key))
        elif value is not None and value != '':
            result[full_key] = str(value)
    return result


def serialize_record(record: dict) -> str:
    """Serialize complete record for EM"""
    flat = flatten_record(record)
    lines = [f"{key}: {value}" for key, value in flat.items()]
    return " ".join(lines)


def serialize_attribute_group(
    attrs: Tuple[str, ...],
    record_flat: Dict[str, str]
) -> str:
    """Serialize attribute group"""
    parts = []
    for attr in sorted(attrs):
        value = record_flat.get(attr, '')
        if value:
            parts.append(f"{attr}: {value}")
    return " ".join(parts)


def canonicalize_correspondence(corr) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Convert correspondence to unordered form for comparison"""
    if isinstance(corr, str):
        corr = json.loads(corr)
    left = frozenset(x.lower().strip() for x in corr[0])
    right = frozenset(x.lower().strip() for x in corr[1])
    return (left, right)


# =============================================================================
# TF-IDF Matcher
# =============================================================================

class TFIDFMatcher:
    """
    TF-IDF Matching for GEM

    Uses TF-IDF vectorization + Cosine similarity for:
    - Entity Matching: Complete record similarity
    - Attribute Matching: Attribute/attribute group similarity
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

    def compute_tfidf_similarity(self, texts1: List[str], texts2: List[str]) -> np.ndarray:
        """
        Compute TF-IDF Cosine similarity matrix between two groups of texts

        Returns:
            similarity_matrix: shape (len(texts1), len(texts2))
        """
        # Merge all texts for TF-IDF fitting
        all_texts = texts1 + texts2

        # Handle empty texts
        all_texts = [t if t.strip() else "EMPTY" for t in all_texts]

        vectorizer = TfidfVectorizer(
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            max_df=self.config.max_df,
            lowercase=True,
            token_pattern=r'(?u)\b\w+\b'  # Include single-character tokens
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(all_texts)
        except ValueError:
            # If vocabulary is empty, return zero matrix
            return np.zeros((len(texts1), len(texts2)))

        # Split back into two groups
        tfidf1 = tfidf_matrix[:len(texts1)]
        tfidf2 = tfidf_matrix[len(texts1):]

        # Compute similarity matrix
        sim_matrix = cosine_similarity(tfidf1, tfidf2)

        return sim_matrix

    # =========================================================================
    # Entity Matching
    # =========================================================================

    def entity_matching(self, left: dict, right: dict) -> Tuple[int, float]:
        """
        Entity Matching: Determine if two records match

        Returns:
            (pred_label, similarity_score)
        """
        left_text = serialize_record(left)
        right_text = serialize_record(right)

        sim_matrix = self.compute_tfidf_similarity([left_text], [right_text])
        similarity = float(sim_matrix[0, 0])

        pred_label = 1 if similarity >= self.config.em_threshold else 0

        return pred_label, similarity

    # =========================================================================
    # Attribute Matching
    # =========================================================================

    def attribute_matching(
        self,
        left: dict,
        right: dict
    ) -> Tuple[List[List[List[str]]], Dict]:
        """
        Attribute Matching: Two-stage matching

        Stage 1: Single attribute matching - compare values only
        Stage 2: Group matching - for remaining attributes
        """
        left_flat = flatten_record(left)
        right_flat = flatten_record(right)

        if not left_flat or not right_flat:
            return [], {"left_attrs": 0, "right_attrs": 0}

        correspondences = []
        used_left = set()
        used_right = set()

        # =====================================================================
        # Stage 1: Single attribute matching - compare values only
        # =====================================================================
        left_attrs = list(left_flat.keys())
        right_attrs = list(right_flat.keys())

        left_values = [str(left_flat[a]) for a in left_attrs]
        right_values = [str(right_flat[a]) for a in right_attrs]

        # Compute value similarity matrix
        sim_matrix = self.compute_tfidf_similarity(left_values, right_values)

        # Collect all candidate pairs
        single_scores = []
        for i, l_attr in enumerate(left_attrs):
            for j, r_attr in enumerate(right_attrs):
                score = sim_matrix[i, j]
                single_scores.append({
                    'left': l_attr,
                    'right': r_attr,
                    'score': score
                })

        # Sort by score descending, greedy selection
        single_scores.sort(key=lambda x: -x['score'])

        for item in single_scores:
            if item['score'] < self.config.am_threshold:
                continue
            if item['left'] in used_left or item['right'] in used_right:
                continue

            correspondences.append([[item['left']], [item['right']]])
            used_left.add(item['left'])
            used_right.add(item['right'])

        # =====================================================================
        # Stage 2: Group matching - for remaining attributes
        # =====================================================================
        remaining_left = [a for a in left_attrs if a not in used_left]
        remaining_right = [a for a in right_attrs if a not in used_right]

        if remaining_left and remaining_right:
            max_size = self.config.max_group_size or max(
                len(remaining_left), len(remaining_right))

            # Generate group candidates
            left_candidates = []
            for size in range(1, min(max_size, len(remaining_left)) + 1):
                for combo in combinations(remaining_left, size):
                    left_candidates.append(combo)

            right_candidates = []
            for size in range(1, min(max_size, len(remaining_right)) + 1):
                for combo in combinations(remaining_right, size):
                    right_candidates.append(combo)

            if left_candidates and right_candidates:
                # Serialize attribute groups
                left_texts = [serialize_attribute_group(
                    c, left_flat) for c in left_candidates]
                right_texts = [serialize_attribute_group(
                    c, right_flat) for c in right_candidates]

                # Filter empty
                valid_left = [(c, t) for c, t in zip(
                    left_candidates, left_texts) if t.strip()]
                valid_right = [(c, t) for c, t in zip(
                    right_candidates, right_texts) if t.strip()]

                if valid_left and valid_right:
                    left_candidates, left_texts = zip(*valid_left)
                    right_candidates, right_texts = zip(*valid_right)

                    # Compute similarity
                    sim_matrix = self.compute_tfidf_similarity(
                        list(left_texts), list(right_texts))

                    group_scores = []
                    for i, l_cand in enumerate(left_candidates):
                        for j, r_cand in enumerate(right_candidates):
                            score = sim_matrix[i, j]
                            if score >= self.config.am_threshold:
                                group_scores.append({
                                    'left': frozenset(l_cand),
                                    'right': frozenset(r_cand),
                                    'score': score,
                                    'size': len(l_cand) + len(r_cand)
                                })

                    # Prefer smaller groups, then by score
                    group_scores.sort(key=lambda x: (x['size'], -x['score']))

                    for item in group_scores:
                        if item['left'] & used_left or item['right'] & used_right:
                            continue
                        correspondences.append(
                            [list(item['left']), list(item['right'])])
                        used_left |= item['left']
                        used_right |= item['right']

        stats = {
            "left_attrs": len(left_flat),
            "right_attrs": len(right_flat),
            "stage1_matches": len([c for c in correspondences if len(c[0]) == 1 and len(c[1]) == 1]),
            "stage2_matches": len([c for c in correspondences if len(c[0]) > 1 or len(c[1]) > 1]),
            "output_correspondences": len(correspondences)
        }

        return correspondences, stats

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def predict(self, pair: dict) -> dict:
        """Full Pipeline: EM -> AM"""
        left = pair['left']
        right = pair['right']

        left_id = str(left.get('_id') or left.get('id') or left.get(
            'ID') or left.get('left_id', ''))
        right_id = str(right.get('_id') or right.get('id') or right.get(
            'ID') or right.get('right_id', ''))
        pair_id = f"{left_id}_{right_id}"

        em_pred, em_score = self.entity_matching(left, right)

        result = {
            "pair_id": pair_id,
            "em_pred": em_pred,
            "em_score": round(em_score, 4),
            "am_pred": [],
            "am_stats": {}
        }

        if em_pred == 1:
            am_pred, am_stats = self.attribute_matching(left, right)
            result["am_pred"] = am_pred
            result["am_stats"] = am_stats

        return result


# =============================================================================
# Evaluation
# =============================================================================

class Evaluator:
    """End-to-end evaluator"""

    @staticmethod
    def evaluate_end_to_end(
        predictions: List[dict],
        ground_truth: List[dict],
        test_data: List[dict] = None
    ) -> dict:
        """End-to-end evaluation - includes three AM metrics

        Args:
            predictions: Prediction results
            ground_truth: AM GT (only pairs with label=1)
            test_data: Original test data (contains labels for all pairs)
        """
        # Build GT index for AM correspondences
        gt_by_id = {}
        total_gt_correspondences = 0
        for gt in ground_truth:
            pair_id = gt.get('pair_id') or gt.get('left_right_pair') or \
                f"{gt.get('left_id', '')}_{gt.get('right_id', '')}"
            gt_by_id[pair_id] = gt
            total_gt_correspondences += len(gt.get('output', []))

        # Build label index from test_data
        label_by_id = {}
        if test_data:
            for item in test_data:
                pair_id = item.get('left_right_pair', '')
                label_by_id[pair_id] = item.get('label', 1)

        em_tp, em_fp, em_fn, em_tn = 0, 0, 0, 0

        # End-to-End AM: All pairs with pred=1
        e2e_am_tp, e2e_am_fp, e2e_am_fn = 0, 0, 0
        e2e_total_pairs = 0
        e2e_exact_matches = 0

        # Conditional AM: Only TP pairs (pred=1 AND label=1)
        cond_am_tp, cond_am_fp, cond_am_fn = 0, 0, 0
        cond_total_pairs = 0
        cond_exact_matches = 0

        # Joint Exact Match: EM correct AND AM exact match
        joint_correct = 0
        total_pairs = 0

        # Iterate over all predictions (not just GT)
        for pred in predictions:
            pair_id = pred['pair_id']
            total_pairs += 1

            # Get true label from test_data, if not available infer from GT
            if pair_id in label_by_id:
                true_label = label_by_id[pair_id]
            elif pair_id in gt_by_id:
                true_label = 1  # In GT means positive sample
            else:
                true_label = 0  # Not in GT means negative sample

            pred_label = pred['em_pred']

            # Get AM ground truth (only pairs with label=1 have it)
            true_corrs = set()
            if true_label == 1 and pair_id in gt_by_id:
                gt = gt_by_id[pair_id]
                if 'output' in gt:
                    true_corrs = {canonicalize_correspondence(
                        c) for c in gt['output']}

            pred_corrs = set()
            if pred_label == 1:
                pred_corrs = {canonicalize_correspondence(
                    c) for c in pred['am_pred']}

            # EM statistics
            if pred_label == 1 and true_label == 1:
                em_tp += 1
            elif pred_label == 1 and true_label == 0:
                em_fp += 1
            elif pred_label == 0 and true_label == 1:
                em_fn += 1
            else:
                em_tn += 1

            # AM statistics - per pair
            pair_tp = len(pred_corrs & true_corrs)
            pair_fp = len(pred_corrs - true_corrs)
            pair_fn = len(true_corrs - pred_corrs)
            pair_exact = (pair_fp == 0 and pair_fn == 0)

            # End-to-End AM: All pairs with pred=1 (including FP)
            if pred_label == 1:
                e2e_am_tp += pair_tp
                e2e_am_fp += pair_fp
                e2e_am_fn += pair_fn
                e2e_total_pairs += 1
                if pair_exact:
                    e2e_exact_matches += 1

            # Conditional AM: Only TP (pred=1 AND label=1)
            if pred_label == 1 and true_label == 1:
                cond_am_tp += pair_tp
                cond_am_fp += pair_fp
                cond_am_fn += pair_fn
                cond_total_pairs += 1
                if pair_exact:
                    cond_exact_matches += 1

            # Joint Exact Match
            em_correct = (pred_label == true_label)
            if em_correct:
                if true_label == 0:
                    # TN: EM correct, no AM needed
                    joint_correct += 1
                elif pair_exact:
                    # TP with exact AM match
                    joint_correct += 1

        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        # EM metrics
        em_precision = safe_div(em_tp, em_tp + em_fp)
        em_recall = safe_div(em_tp, em_tp + em_fn)
        em_f1 = safe_div(2 * em_precision * em_recall,
                         em_precision + em_recall)
        em_accuracy = safe_div(em_tp + em_tn, total_pairs)

        # End-to-End AM metrics
        e2e_precision = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fp)
        e2e_recall = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fn)
        e2e_f1 = safe_div(2 * e2e_precision * e2e_recall,
                          e2e_precision + e2e_recall)
        e2e_accuracy = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fp + e2e_am_fn)
        e2e_pair_accuracy = safe_div(e2e_exact_matches, e2e_total_pairs)

        # Conditional AM metrics
        cond_precision = safe_div(cond_am_tp, cond_am_tp + cond_am_fp)
        cond_recall = safe_div(cond_am_tp, cond_am_tp + cond_am_fn)
        cond_f1 = safe_div(2 * cond_precision * cond_recall,
                           cond_precision + cond_recall)
        cond_accuracy = safe_div(
            cond_am_tp, cond_am_tp + cond_am_fp + cond_am_fn)
        cond_pair_accuracy = safe_div(cond_exact_matches, cond_total_pairs)

        # Joint Exact Match
        joint_accuracy = safe_div(joint_correct, total_pairs)

        return {
            "EM": {
                "TP": em_tp, "FP": em_fp, "FN": em_fn, "TN": em_tn,
                "Precision": round(em_precision, 4),
                "Recall": round(em_recall, 4),
                "F1": round(em_f1, 4),
                "Accuracy": round(em_accuracy, 4)
            },
            "AM_EndToEnd": {
                "TP": e2e_am_tp, "FP": e2e_am_fp, "FN": e2e_am_fn,
                "Precision": round(e2e_precision, 4),
                "Recall": round(e2e_recall, 4),
                "F1": round(e2e_f1, 4),
                "Accuracy": round(e2e_accuracy, 4),
                "Pair_Accuracy": round(e2e_pair_accuracy, 4),
                "Exact_Matches": e2e_exact_matches,
                "Total_Pairs": e2e_total_pairs
            },
            "AM_Conditional": {
                "TP": cond_am_tp, "FP": cond_am_fp, "FN": cond_am_fn,
                "Precision": round(cond_precision, 4),
                "Recall": round(cond_recall, 4),
                "F1": round(cond_f1, 4),
                "Accuracy": round(cond_accuracy, 4),
                "Pair_Accuracy": round(cond_pair_accuracy, 4),
                "Exact_Matches": cond_exact_matches,
                "Total_Pairs": cond_total_pairs,
                "Total_GT_Correspondences": total_gt_correspondences
            },
            "Joint_Exact_Match": {
                "Correct": joint_correct,
                "Total": total_pairs,
                "Accuracy": round(joint_accuracy, 4)
            }
        }


# =============================================================================
# Run Script
# =============================================================================

def run_baseline(
    test_data_path: str,
    gt_path: str,
    output_path: str,
    em_threshold: float = 0.1,
    am_threshold: float = 0.5,
    max_group_size: Optional[int] = None
):
    """Run TF-IDF baseline"""
    print("Loading data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    print(f"  Test pairs: {len(test_data)}")
    print(f"  Ground truth: {len(ground_truth)}")

    config = Config(
        em_threshold=em_threshold,
        am_threshold=am_threshold,
        max_group_size=max_group_size
    )

    matcher = TFIDFMatcher(config)

    print("\nRunning predictions...")
    predictions = []
    for i, pair in enumerate(test_data):
        result = matcher.predict(pair)
        predictions.append(result)

        # Detailed output
        label = pair.get('label', 1)
        pred = result['em_pred']
        status = "✓" if pred == label else "✗"
        if pred == 1:
            num_corr = len(result['am_pred'])
            print(
                f"  [{i+1:3d}/{len(test_data)}] {result['pair_id']} -> pred={pred} {status}, {num_corr} correspondences")
        else:
            print(
                f"  [{i+1:3d}/{len(test_data)}] {result['pair_id']} -> pred={pred} {status}")

    print("\nEvaluating...")
    metrics = Evaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    print("\n" + "=" * 60)
    print("Results: TF-IDF + Cosine Similarity Matching")
    print("=" * 60)
    print(f"\nEM Threshold: {em_threshold}")
    print(f"AM Threshold: {am_threshold}")
    max_group_str = "dynamic" if config.max_group_size is None else str(
        config.max_group_size)
    print(f"Max Group Size: {max_group_str}")

    print(f"\nEntity Matching:")
    print(f"  Precision: {metrics['EM']['Precision']}")
    print(f"  Recall:    {metrics['EM']['Recall']}")
    print(f"  F1:        {metrics['EM']['F1']}")
    print(f"  Accuracy:  {metrics['EM']['Accuracy']}")

    print(f"\nAttribute Matching (End-to-End):")
    print(f"  Precision:      {metrics['AM_EndToEnd']['Precision']}")
    print(f"  Recall:         {metrics['AM_EndToEnd']['Recall']}")
    print(f"  F1:             {metrics['AM_EndToEnd']['F1']}")
    print(f"  Accuracy:       {metrics['AM_EndToEnd']['Accuracy']}")
    print(
        f"  Pair Accuracy:  {metrics['AM_EndToEnd']['Pair_Accuracy']} [{metrics['AM_EndToEnd']['Exact_Matches']}/{metrics['AM_EndToEnd']['Total_Pairs']}]")

    print(f"\nAttribute Matching (Conditional - TP only):")
    print(f"  Precision:      {metrics['AM_Conditional']['Precision']}")
    print(f"  Recall:         {metrics['AM_Conditional']['Recall']}")
    print(f"  F1:             {metrics['AM_Conditional']['F1']}")
    print(f"  Accuracy:       {metrics['AM_Conditional']['Accuracy']}")
    print(
        f"  Pair Accuracy:  {metrics['AM_Conditional']['Pair_Accuracy']} [{metrics['AM_Conditional']['Exact_Matches']}/{metrics['AM_Conditional']['Total_Pairs']}]")
    print(
        f"  Total GT Correspondences: {metrics['AM_Conditional']['Total_GT_Correspondences']}")

    print(f"\nJoint Exact-Match:")
    print(
        f"  Accuracy:       {metrics['Joint_Exact_Match']['Accuracy']} [{metrics['Joint_Exact_Match']['Correct']}/{metrics['Joint_Exact_Match']['Total']}]")

    output = {
        "config": {
            "method": "TF-IDF + Cosine Similarity",
            "em_threshold": em_threshold,
            "am_threshold": am_threshold,
            "max_group_size": config.max_group_size if config.max_group_size else "dynamic"
        },
        "metrics": metrics,
        "predictions": predictions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")

    return output


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline: TF-IDF + Cosine Similarity Matching"
    )
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--em_threshold", type=float, default=0.1)
    parser.add_argument("--am_threshold", type=float, default=0.5)
    parser.add_argument("--max_group_size", type=int, default=None)

    args = parser.parse_args()

    run_baseline(
        test_data_path=args.test_data,
        gt_path=args.ground_truth,
        output_path=args.output,
        em_threshold=args.em_threshold,
        am_threshold=args.am_threshold,
        max_group_size=args.max_group_size
    )
