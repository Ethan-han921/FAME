"""
Two-Stage Grid Search for Baseline Hyperparameters

Stage 1: Independently select EM parameters (based on EM.F1)
Stage 2: Independently select AM parameters (based on Oracle AM, using GT EM=1 pairs)
Stage 3: Combine optimal parameters, evaluate full pipeline

Usage:
    python grid_search.py --baseline tfidf --dev_data dev.json --dev_gt dev_gt.json --output results/
    python grid_search.py --baseline embedding_hybrid --dev_data dev.json --dev_gt dev_gt.json --output results/
    python grid_search.py --baseline embedding_threshold --dev_data dev.json --dev_gt dev_gt.json --output results/
"""

import json
import os
import argparse
from itertools import product
from datetime import datetime
from typing import List, Dict, Any, Tuple

# Import three baselines
from tfidf_baseline import TFIDFMatcher, Config as TFIDFConfig, Evaluator as TFIDFEvaluator
from embedding_hybrid_baseline import EmbeddingHybridMatcher, Config as HybridConfig, Evaluator as HybridEvaluator
from embedding_threshold_baseline import EmbeddingThresholdMatcher, Config as ThresholdConfig, Evaluator as ThresholdEvaluator


# =============================================================================
# Search Space Definition (separate EM and AM)
# =============================================================================

EM_SEARCH_SPACE = {
    "tfidf": {
        "em_threshold": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    },
    "embedding_hybrid": {
        "em_threshold": [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    "embedding_threshold": {}  # embedding_threshold has no EM stage
}

AM_SEARCH_SPACE = {
    "tfidf": {
        "am_threshold": [0.2, 0.4, 0.6, 0.8, 1.0]
    },
    "embedding_hybrid": {
        "similarity_threshold": [0.2, 0.4, 0.6, 0.8, 1.0],
        "group_threshold": [0.2, 0.4, 0.6, 0.8, 1.0],
        "top_k": [2, 3, 4],
        "enable_nn": [True],
        "nn_max_size": [2, 3, 4]
    },
    "embedding_threshold": {
        "similarity_threshold": [0.2, 0.4, 0.6, 0.8, 1.0],
        "group_threshold": [0.2, 0.4, 0.6, 0.8, 1.0],
        "top_k": [1, 2, 3, 4, 5],
        "enable_nn": [True],
        "nn_max_size": [2, 3, 4, 5]
    }
}

# AM stage default parameters (used during Stage 1 EM search)
AM_DEFAULTS = {
    "tfidf": {"am_threshold": 0.5},
    "embedding_hybrid": {
        "similarity_threshold": 0.5,
        "group_threshold": 0.5,
        "top_k": 3,
        "enable_nn": True,
        "nn_max_size": 3
    },
    "embedding_threshold": {
        "similarity_threshold": 0.5,
        "group_threshold": 0.5,
        "top_k": 3,
        "enable_nn": True,
        "nn_max_size": 3
    }
}

# EM stage default parameters (used during Stage 2 AM search)
EM_DEFAULTS = {
    "tfidf": {"em_threshold": 0.5},
    "embedding_hybrid": {"em_threshold": 0.5},
    "embedding_threshold": {}
}


# =============================================================================
# Helper Functions
# =============================================================================

def generate_param_combinations(search_space: Dict[str, List]) -> List[Dict]:
    """Generate all parameter combinations"""
    if not search_space:
        return [{}]

    keys = list(search_space.keys())
    values = list(search_space.values())

    combinations = []
    for combo in product(*values):
        param_dict = dict(zip(keys, combo))
        combinations.append(param_dict)

    return combinations


def filter_positive_pairs(data: List[dict], ground_truth: List[dict]) -> Tuple[List[dict], List[dict]]:
    """
    Filter pairs with label=1 (for Oracle AM evaluation)

    Note:
    - Each pair in data has a 'label' field
    - ground_truth only contains AM correspondences for label=1 pairs
    - Need to match data and GT via left_id/right_id

    Returns:
        (filtered_data, filtered_gt) - Only pairs with label=1
    """
    # Build GT index by pair_id
    gt_by_id = {}
    for gt in ground_truth:
        left_id = str(gt.get('left_id', ''))
        right_id = str(gt.get('right_id', ''))
        pair_id = f"{left_id}_{right_id}"
        gt_by_id[pair_id] = gt

    filtered_data = []
    filtered_gt = []

    for d in data:
        if d.get("label", 0) == 1:
            # Find corresponding GT
            left_id = str(d.get('left', {}).get(
                'id', d.get('left', {}).get('ID', '')))
            right_id = str(d.get('right', {}).get('id', d.get(
                'right', {}).get('ID', d.get('right', {}).get('Id', ''))))
            pair_id = f"{left_id}_{right_id}"

            if pair_id in gt_by_id:
                filtered_data.append(d)
                filtered_gt.append(gt_by_id[pair_id])

    return filtered_data, filtered_gt


def get_metric_value(metrics: Dict, key: str) -> float:
    """Extract specified metric value from metrics"""
    parts = key.split(".")
    value = metrics
    for part in parts:
        value = value[part]
    return value


# =============================================================================
# Baseline Runners
# =============================================================================

def run_tfidf_with_params(test_data: List[dict], ground_truth: List[dict], params: Dict) -> Dict:
    """Run TF-IDF baseline with specified parameters"""
    config = TFIDFConfig(
        em_threshold=params.get("em_threshold", 0.5),
        am_threshold=params.get("am_threshold", 0.5)
    )

    matcher = TFIDFMatcher(config)

    predictions = []
    for pair in test_data:
        result = matcher.predict(pair)
        predictions.append(result)

    metrics = TFIDFEvaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    return metrics


def run_tfidf_am_only(test_data: List[dict], ground_truth: List[dict], params: Dict) -> Dict:
    """
    TF-IDF: Evaluate AM stage only (Oracle EM)
    Assume all pairs are EM=1, directly evaluate AM
    """
    config = TFIDFConfig(
        em_threshold=0.0,  # Let all pairs pass EM
        am_threshold=params.get("am_threshold", 0.5)
    )

    matcher = TFIDFMatcher(config)

    predictions = []
    for pair in test_data:
        result = matcher.predict(pair)
        # Force set pred=1 (Oracle EM)
        result["pred"] = 1
        predictions.append(result)

    metrics = TFIDFEvaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    return metrics


def run_embedding_hybrid_with_params(
    test_data: List[dict],
    ground_truth: List[dict],
    params: Dict,
    matcher: EmbeddingHybridMatcher = None
) -> Tuple[Dict, EmbeddingHybridMatcher]:
    """Run Embedding Hybrid baseline with specified parameters"""
    config = HybridConfig(
        em_threshold=params.get("em_threshold", 0.5),
        similarity_threshold=params.get("similarity_threshold", 0.5),
        group_threshold=params.get("group_threshold", 0.5),
        top_k=params.get("top_k", 3),
        enable_nn=params.get("enable_nn", True),
        nn_max_size=params.get("nn_max_size", 3)
    )

    if matcher is None:
        matcher = EmbeddingHybridMatcher(config)
    else:
        matcher.config = config

    predictions = []
    for pair in test_data:
        result = matcher.predict(pair)
        predictions.append(result)

    metrics = HybridEvaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    return metrics, matcher


def run_embedding_hybrid_am_only(
    test_data: List[dict],
    ground_truth: List[dict],
    params: Dict,
    matcher: EmbeddingHybridMatcher = None
) -> Tuple[Dict, EmbeddingHybridMatcher]:
    """
    Embedding Hybrid: Evaluate AM stage only (Oracle EM)
    """
    config = HybridConfig(
        em_threshold=0.0,  # Let all pairs pass EM
        similarity_threshold=params.get("similarity_threshold", 0.5),
        group_threshold=params.get("group_threshold", 0.5),
        top_k=params.get("top_k", 3),
        enable_nn=params.get("enable_nn", True),
        nn_max_size=params.get("nn_max_size", 3)
    )

    if matcher is None:
        matcher = EmbeddingHybridMatcher(config)
    else:
        matcher.config = config

    predictions = []
    for pair in test_data:
        result = matcher.predict(pair)
        result["pred"] = 1  # Oracle EM
        predictions.append(result)

    metrics = HybridEvaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    return metrics, matcher


def run_embedding_threshold_with_params(
    test_data: List[dict],
    ground_truth: List[dict],
    params: Dict,
    matcher: EmbeddingThresholdMatcher = None
) -> Tuple[Dict, EmbeddingThresholdMatcher]:
    """Run Embedding Threshold baseline with specified parameters (AM only)"""
    config = ThresholdConfig(
        similarity_threshold=params.get("similarity_threshold", 0.5),
        group_threshold=params.get("group_threshold", 0.5),
        top_k=params.get("top_k", 3),
        enable_nn=params.get("enable_nn", True),
        nn_max_size=params.get("nn_max_size", 3)
    )

    if matcher is None:
        matcher = EmbeddingThresholdMatcher(config)
    else:
        matcher.config = config

    predictions = []
    for pair in test_data:
        result = matcher.predict(pair)
        predictions.append(result)

    metrics = ThresholdEvaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    return metrics, matcher


# =============================================================================
# Two-Stage Grid Search
# =============================================================================

def two_stage_grid_search(
    baseline: str,
    dev_data: List[dict],
    dev_gt: List[dict],
    output_dir: str = None
) -> Dict:
    """
    Two-Stage Grid Search

    Stage 1: Search EM parameters (based on EM.F1)
    Stage 2: Search AM parameters (based on Oracle AM, using GT EM=1 pairs)
    Stage 3: Combine optimal parameters, evaluate full pipeline
    """
    print(f"\n{'='*70}")
    print(f"Two-Stage Grid Search: {baseline}")
    print(f"{'='*70}")
    print(f"Dev set size: {len(dev_data)}")

    # Filter GT EM=1 pairs (for Stage 2)
    positive_data, positive_gt = filter_positive_pairs(dev_data, dev_gt)
    print(f"GT EM=1 pairs (for Oracle AM): {len(positive_data)}")

    result = {
        "baseline": baseline,
        "dev_set_size": len(dev_data),
        "positive_pairs_size": len(positive_data),
        "timestamp": datetime.now().isoformat()
    }

    # Preload model
    matcher = None

    # =========================================================================
    # Stage 1: Search EM parameters
    # =========================================================================
    em_search_space = EM_SEARCH_SPACE[baseline]
    best_em_params = {}
    best_em_f1 = -1
    stage1_results = []

    if em_search_space:  # tfidf and embedding_hybrid have EM stage
        print(f"\n{'='*70}")
        print(f"Stage 1: EM Parameter Search")
        print(f"{'='*70}")
        print(f"Search space: {em_search_space}")
        print(f"Selection metric: EM.F1")

        em_combinations = generate_param_combinations(em_search_space)
        print(f"Total combinations: {len(em_combinations)}")

        am_defaults = AM_DEFAULTS[baseline]

        for i, em_params in enumerate(em_combinations):
            # Merge EM parameters and AM default parameters
            full_params = {**em_params, **am_defaults}

            if baseline == "tfidf":
                metrics = run_tfidf_with_params(dev_data, dev_gt, full_params)
            elif baseline == "embedding_hybrid":
                metrics, matcher = run_embedding_hybrid_with_params(
                    dev_data, dev_gt, full_params, matcher)

            em_f1 = get_metric_value(metrics, "EM.F1")

            stage1_results.append({
                "params": em_params,
                "em_f1": em_f1,
                "metrics": metrics
            })

            is_best = em_f1 > best_em_f1
            if is_best:
                best_em_f1 = em_f1
                best_em_params = em_params

            print(f"  [{i+1}/{len(em_combinations)}] {em_params} -> EM.F1={em_f1:.4f}" +
                  (" *BEST*" if is_best else ""))

        print(f"\n--- Stage 1 Result ---")
        print(f"  Best EM params: {best_em_params}")
        print(f"  Best EM.F1: {best_em_f1:.4f}")

        result["stage1"] = {
            "search_space": em_search_space,
            "best_params": best_em_params,
            "best_em_f1": best_em_f1,
            "all_results": stage1_results
        }
    else:
        print(f"\n[Stage 1 skipped - {baseline} has no EM stage]")
        result["stage1"] = {"skipped": True, "reason": "No EM stage"}

    # =========================================================================
    # Stage 2: Search AM parameters (Oracle AM)
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Stage 2: AM Parameter Search (Oracle EM)")
    print(f"{'='*70}")

    am_search_space = AM_SEARCH_SPACE[baseline]
    print(f"Search space: {am_search_space}")
    print(f"Using GT EM=1 pairs: {len(positive_data)}")
    print(f"Selection metric: AM.Recall (primary), AM.F1 (secondary)")

    am_combinations = generate_param_combinations(am_search_space)
    print(f"Total combinations: {len(am_combinations)}")

    best_am_params = {}
    best_am_score = (-1, -1)  # (recall, f1)
    stage2_results = []

    for i, am_params in enumerate(am_combinations):
        # Use Oracle EM (only evaluate AM on GT EM=1 pairs)
        if baseline == "tfidf":
            metrics = run_tfidf_am_only(positive_data, positive_gt, am_params)
        elif baseline == "embedding_hybrid":
            metrics, matcher = run_embedding_hybrid_am_only(
                positive_data, positive_gt, am_params, matcher)
        elif baseline == "embedding_threshold":
            metrics, matcher = run_embedding_threshold_with_params(
                positive_data, positive_gt, am_params, matcher)

        am_recall = get_metric_value(metrics, "AM_Conditional.Recall")
        am_f1 = get_metric_value(metrics, "AM_Conditional.F1")
        am_score = (am_recall, am_f1)

        stage2_results.append({
            "params": am_params,
            "am_recall": am_recall,
            "am_f1": am_f1,
            "metrics": metrics
        })

        is_best = am_score > best_am_score
        if is_best:
            best_am_score = am_score
            best_am_params = am_params

        print(f"  [{i+1:4d}/{len(am_combinations)}] {am_params}")
        print(f"       -> AM.Recall={am_recall:.4f}, AM.F1={am_f1:.4f}" +
              (" *BEST*" if is_best else ""))

    print(f"\n--- Stage 2 Result ---")
    print(f"  Best AM params: {best_am_params}")
    print(f"  Best AM.Recall: {best_am_score[0]:.4f}")
    print(f"  Best AM.F1: {best_am_score[1]:.4f}")

    result["stage2"] = {
        "search_space": am_search_space,
        "best_params": best_am_params,
        "best_am_recall": best_am_score[0],
        "best_am_f1": best_am_score[1],
        "all_results": stage2_results
    }

    # =========================================================================
    # Stage 3: Combine optimal parameters, evaluate full pipeline
    # =========================================================================
    print(f"\n{'='*70}")
    print(f"Stage 3: Full Pipeline Evaluation")
    print(f"{'='*70}")

    final_params = {**best_em_params, **best_am_params}
    print(f"Combined params: {final_params}")

    if baseline == "tfidf":
        final_metrics = run_tfidf_with_params(dev_data, dev_gt, final_params)
    elif baseline == "embedding_hybrid":
        final_metrics, _ = run_embedding_hybrid_with_params(
            dev_data, dev_gt, final_params, matcher)
    elif baseline == "embedding_threshold":
        final_metrics, _ = run_embedding_threshold_with_params(
            dev_data, dev_gt, final_params, matcher)

    print(f"\n--- Final Pipeline Results (Dev) ---")
    print(f"\n  EM:")
    print(f"    Precision: {final_metrics['EM']['Precision']:.4f}")
    print(f"    Recall:    {final_metrics['EM']['Recall']:.4f}")
    print(f"    F1:        {final_metrics['EM']['F1']:.4f}")
    print(f"    Accuracy:  {final_metrics['EM']['Accuracy']:.4f}")
    print(f"\n  AM (Conditional):")
    print(f"    Precision: {final_metrics['AM_Conditional']['Precision']:.4f}")
    print(f"    Recall:    {final_metrics['AM_Conditional']['Recall']:.4f}")
    print(f"    F1:        {final_metrics['AM_Conditional']['F1']:.4f}")
    print(
        f"    Pair Acc:  {final_metrics['AM_Conditional']['Pair_Accuracy']:.4f}")
    print(f"\n  Joint Exact Match:")
    print(
        f"    Accuracy:  {final_metrics['Joint_Exact_Match']['Accuracy']:.4f}")

    result["stage3"] = {
        "final_params": final_params,
        "final_metrics": final_metrics
    }

    # =========================================================================
    # Save results
    # =========================================================================
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(
            output_dir,
            f"two_stage_search_{baseline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")

    return result


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Two-Stage Grid Search for Baseline Hyperparameters")

    parser.add_argument("--baseline", type=str, required=True,
                        choices=["tfidf", "embedding_hybrid",
                                 "embedding_threshold"],
                        help="Baseline to tune")
    parser.add_argument("--dev_data", type=str, required=True,
                        help="Dev set data (JSON)")
    parser.add_argument("--dev_gt", type=str, required=True,
                        help="Dev set ground truth (JSON)")
    parser.add_argument("--output", type=str, default="./grid_search_results",
                        help="Output directory")

    args = parser.parse_args()

    # Load data
    print("Loading data...")
    with open(args.dev_data, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    with open(args.dev_gt, 'r', encoding='utf-8') as f:
        dev_gt = json.load(f)

    # Run Two-Stage Grid Search
    result = two_stage_grid_search(
        baseline=args.baseline,
        dev_data=dev_data,
        dev_gt=dev_gt,
        output_dir=args.output
    )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Baseline: {args.baseline}")

    if "stage1" in result and not result["stage1"].get("skipped"):
        print(f"\nStage 1 (EM):")
        print(f"  Best params: {result['stage1']['best_params']}")
        print(f"  Best EM.F1:  {result['stage1']['best_em_f1']:.4f}")

    print(f"\nStage 2 (AM Oracle):")
    print(f"  Best params:    {result['stage2']['best_params']}")
    print(f"  Best AM.Recall: {result['stage2']['best_am_recall']:.4f}")
    print(f"  Best AM.F1:     {result['stage2']['best_am_f1']:.4f}")

    print(f"\nStage 3 (Full Pipeline):")
    print(f"  Final params: {result['stage3']['final_params']}")
    fm = result['stage3']['final_metrics']
    print(f"  EM.F1:        {fm['EM']['F1']:.4f}")
    print(f"  AM.Recall:    {fm['AM_Conditional']['Recall']:.4f}")
    print(f"  AM.F1:        {fm['AM_Conditional']['F1']:.4f}")
    print(f"  Joint Acc:    {fm['Joint_Exact_Match']['Accuracy']:.4f}")


if __name__ == "__main__":
    main()
