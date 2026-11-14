"""
Baseline: Embedding Hybrid (EM from frozen_sentence_encoder + AM from embedding_threshold)

Pipeline:
  EM: Record embedding + cosine similarity (from frozen_sentence_encoder_baseline)
  AM: Attribute embedding + similarity matrix + greedy merge (from embedding_threshold_baseline)

Model: sentence-transformers/all-mpnet-base-v2
Supports: 1-1, N-1, 1-N, N-N correspondences
"""

import json
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, FrozenSet
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer


@dataclass
class Config:
    """Configuration parameters"""
    model_name: str = 'all-mpnet-base-v2'
    # EM parameters
    em_threshold: float = 0.7  # EM threshold
    max_sequence_length: int = 384  # Model max input length
    # AM parameters
    similarity_threshold: float = 0.5  # Single attribute similarity threshold
    top_k: int = 3  # Top-k candidates per attribute
    group_threshold: float = 0.6  # Group construction threshold
    # N-N related parameters
    enable_nn: bool = True  # Whether to enable N-N candidates
    nn_max_size: int = 4  # Max attributes per side for N-N groups


# =============================================================================
# Utility Functions
# =============================================================================

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
    return "\n".join(lines)


def canonicalize_correspondence(corr) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Convert correspondence to unordered form"""
    if isinstance(corr, str):
        corr = json.loads(corr)
    left = frozenset(x.lower().strip() for x in corr[0])
    right = frozenset(x.lower().strip() for x in corr[1])
    return (left, right)


# =============================================================================
# Embedding Hybrid Matcher
# =============================================================================

class EmbeddingHybridMatcher:
    """
    Embedding Hybrid Matcher

    EM: Record embedding + cosine similarity (from frozen_sentence_encoder)
    AM: Similarity matrix + greedy merge (from embedding_threshold)
    """

    def __init__(self, config: Config = None):
        self.config = config or Config()

        print(f"Loading model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)
        self.model.eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode text list"""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))

    # =========================================================================
    # Entity Matching (from frozen_sentence_encoder_baseline)
    # =========================================================================

    def entity_matching(self, left: dict, right: dict) -> Tuple[int, float]:
        """
        Entity Matching: Determine if two records match

        s_EM = cos(E(r_L), E(r_R))
        y_hat = 1[s_EM >= tau_EM]
        """
        # Serialize records
        left_text = serialize_record(left)
        right_text = serialize_record(right)

        # Check input length
        left_tokens = len(left_text.split())
        right_tokens = len(right_text.split())
        if left_tokens > self.config.max_sequence_length or \
           right_tokens > self.config.max_sequence_length:
            print(f"Warning: Long input detected. Left: {left_tokens}, Right: {right_tokens} tokens")

        # Encode
        embeddings = self.encode([left_text, right_text])

        # Compute similarity
        similarity = self.cosine_similarity(embeddings[0], embeddings[1])

        # Threshold decision
        pred_label = 1 if similarity >= self.config.em_threshold else 0

        return pred_label, similarity

    # =========================================================================
    # Attribute Matching (from embedding_threshold_baseline)
    # =========================================================================

    def compute_similarity_matrix(
        self,
        left_texts: List[str],
        right_texts: List[str]
    ) -> np.ndarray:
        """Compute L x R cosine similarity matrix for single attributes"""
        if not left_texts or not right_texts:
            return np.array([])

        left_embs = self.encode(left_texts)
        right_embs = self.encode(right_texts)

        # Normalize
        left_norm = left_embs / (np.linalg.norm(left_embs, axis=1, keepdims=True) + 1e-8)
        right_norm = right_embs / (np.linalg.norm(right_embs, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity matrix
        sim_matrix = np.dot(left_norm, right_norm.T)

        return sim_matrix

    def _compute_nn_score_from_edges(self, edge_scores: List[float]) -> float:
        """Compute N-N group score (based on intersection edges only)"""
        if not edge_scores:
            return 0.0
        return float(np.min(edge_scores))

    def build_candidates_from_matrix(
        self,
        sim_matrix: np.ndarray,
        left_attrs: List[str],
        right_attrs: List[str]
    ) -> List[Dict]:
        """Heuristically generate candidate groups based on single attribute similarity matrix"""
        n_left, n_right = sim_matrix.shape
        top_k = self.config.top_k
        threshold = self.config.similarity_threshold

        # Collect all candidate 1-1 pairs
        candidate_pairs = set()
        pair_scores = {}

        # Direction 1: Find top-k right for each left
        for i in range(n_left):
            valid_indices = np.where(sim_matrix[i] >= threshold)[0]
            if len(valid_indices) > 0:
                sorted_valid = sorted(valid_indices, key=lambda j: -sim_matrix[i, j])[:top_k]
                for j in sorted_valid:
                    candidate_pairs.add((i, j))
                    pair_scores[(i, j)] = float(sim_matrix[i, j])

        # Direction 2: Find top-k left for each right
        for j in range(n_right):
            valid_indices = np.where(sim_matrix[:, j] >= threshold)[0]
            if len(valid_indices) > 0:
                sorted_valid = sorted(valid_indices, key=lambda i: -sim_matrix[i, j])[:top_k]
                for i in sorted_valid:
                    candidate_pairs.add((i, j))
                    pair_scores[(i, j)] = float(sim_matrix[i, j])

        # Group by right -> N-1 candidates
        right_to_lefts = defaultdict(list)
        for (i, j) in candidate_pairs:
            right_to_lefts[j].append((i, pair_scores[(i, j)]))

        # Group by left -> 1-N candidates
        left_to_rights = defaultdict(list)
        for (i, j) in candidate_pairs:
            left_to_rights[i].append((j, pair_scores[(i, j)]))

        # Generate candidates
        candidates = []

        # ===== 1-1 candidates: Each high-score edge is an independent candidate =====
        for (i, j), score in pair_scores.items():
            candidates.append({
                "left": frozenset([left_attrs[i]]),
                "right": frozenset([right_attrs[j]]),
                "score": score,
                "type": "1-1"
            })

        # ===== N-1 candidates: Multiple left pointing to same right =====
        for right_idx, left_list in right_to_lefts.items():
            if len(left_list) > 1:
                left_indices = [l[0] for l in left_list]
                scores = [l[1] for l in left_list]
                min_score = np.min(scores)

                candidates.append({
                    "left": frozenset(left_attrs[i] for i in left_indices),
                    "right": frozenset([right_attrs[right_idx]]),
                    "score": float(min_score),
                    "type": "N-1"
                })

        # ===== 1-N candidates: One left pointing to multiple right =====
        for left_idx, right_list in left_to_rights.items():
            if len(right_list) > 1:
                right_indices = [r[0] for r in right_list]
                scores = [r[1] for r in right_list]
                min_score = np.min(scores)

                candidates.append({
                    "left": frozenset([left_attrs[left_idx]]),
                    "right": frozenset(right_attrs[j] for j in right_indices),
                    "score": float(min_score),
                    "type": "1-N"
                })

        # ===== N-N candidates: Merge from N-1 and 1-N intersections =====
        if self.config.enable_nn:
            n1_candidates = []
            for right_idx, left_list in right_to_lefts.items():
                if len(left_list) > 1:
                    left_indices = frozenset(l[0] for l in left_list)
                    edges = {(l[0], right_idx): l[1] for l in left_list}
                    n1_candidates.append({
                        "left_indices": left_indices,
                        "right_indices": frozenset([right_idx]),
                        "edges": edges
                    })

            one_n_candidates = []
            for left_idx, right_list in left_to_rights.items():
                if len(right_list) > 1:
                    right_indices = frozenset(r[0] for r in right_list)
                    edges = {(left_idx, r[0]): r[1] for r in right_list}
                    one_n_candidates.append({
                        "left_indices": frozenset([left_idx]),
                        "right_indices": right_indices,
                        "edges": edges
                    })

            seen_nn = set()
            for n1 in n1_candidates:
                for one_n in one_n_candidates:
                    left_overlap = n1["left_indices"] & one_n["left_indices"]
                    right_overlap = n1["right_indices"] & one_n["right_indices"]

                    if left_overlap and right_overlap:
                        merged_left = n1["left_indices"] | one_n["left_indices"]
                        merged_right = n1["right_indices"] | one_n["right_indices"]

                        if len(merged_left) > 1 and len(merged_right) > 1:
                            if (len(merged_left) > self.config.nn_max_size or
                                len(merged_right) > self.config.nn_max_size):
                                continue

                            nn_key = (merged_left, merged_right)
                            if nn_key in seen_nn:
                                continue
                            seen_nn.add(nn_key)

                            merged_edges = {**n1["edges"], **one_n["edges"]}
                            edge_scores = list(merged_edges.values())
                            nn_score = self._compute_nn_score_from_edges(edge_scores)

                            candidates.append({
                                "left": frozenset(left_attrs[i] for i in merged_left),
                                "right": frozenset(right_attrs[j] for j in merged_right),
                                "score": nn_score,
                                "type": "N-N"
                            })

        return candidates

    def construct_groups(self, candidates: List[Dict]) -> List[List[List[str]]]:
        """Greedy construction of attribute groups"""
        candidates = sorted(candidates, key=lambda x: -x["score"])

        used_left = set()
        used_right = set()
        correspondences = []

        for cand in candidates:
            if cand["score"] < self.config.group_threshold:
                continue
            if cand["left"] & used_left or cand["right"] & used_right:
                continue

            correspondences.append([list(cand["left"]), list(cand["right"])])
            used_left |= cand["left"]
            used_right |= cand["right"]

        return correspondences

    def attribute_matching(
        self,
        left: dict,
        right: dict
    ) -> Tuple[List[List[List[str]]], Dict]:
        """Attribute Matching Pipeline"""
        left_flat = flatten_record(left)
        right_flat = flatten_record(right)

        if not left_flat or not right_flat:
            return [], {"left_attrs": 0, "right_attrs": 0}

        left_attrs = list(left_flat.keys())
        right_attrs = list(right_flat.keys())

        left_texts = [f"{attr}: {left_flat[attr]}" for attr in left_attrs]
        right_texts = [f"{attr}: {right_flat[attr]}" for attr in right_attrs]

        sim_matrix = self.compute_similarity_matrix(left_texts, right_texts)

        if sim_matrix.size == 0:
            return [], {"left_attrs": len(left_flat), "right_attrs": len(right_flat)}

        candidates = self.build_candidates_from_matrix(sim_matrix, left_attrs, right_attrs)
        correspondences = self.construct_groups(candidates)

        stats = {
            "left_attrs": len(left_flat),
            "right_attrs": len(right_flat),
            "matrix_shape": f"{sim_matrix.shape[0]}x{sim_matrix.shape[1]}",
            "candidates_generated": len(candidates),
            "output_correspondences": len(correspondences)
        }

        return correspondences, stats

    # =========================================================================
    # Full Pipeline
    # =========================================================================

    def predict(self, pair: dict) -> dict:
        """Full Pipeline: EM -> AM (if EM predicts match)"""
        left = pair['left']
        right = pair['right']

        left_id = str(left.get('_id') or left.get('id') or left.get('ID') or '')
        right_id = str(right.get('_id') or right.get('id') or right.get('ID') or '')
        pair_id = f"{left_id}_{right_id}"

        # Stage 1: Entity Matching
        em_pred, em_score = self.entity_matching(left, right)

        result = {
            "pair_id": pair_id,
            "em_pred": em_pred,
            "em_score": round(em_score, 4),
            "am_pred": [],
            "am_stats": {}
        }

        # Stage 2: Attribute Matching (only if EM predicts match)
        if em_pred == 1:
            am_pred, am_stats = self.attribute_matching(left, right)
            result["am_pred"] = am_pred
            result["am_stats"] = am_stats

        return result


# =============================================================================
# Evaluation
# =============================================================================

class Evaluator:
    """Evaluator"""

    @staticmethod
    def _build_gt_index(ground_truth: List[dict]) -> Dict[str, dict]:
        """Build GT index"""
        gt_by_id = {}
        for gt in ground_truth:
            if 'pair_id' in gt:
                pair_id = gt['pair_id']
            elif 'left_right_pair' in gt:
                pair_id = gt['left_right_pair']
            else:
                left_id = gt.get('left_id', gt.get('left_ID', ''))
                right_id = gt.get('right_id', gt.get('right_ID', ''))
                pair_id = f"{left_id}_{right_id}"
            gt_by_id[pair_id] = gt
        return gt_by_id

    @staticmethod
    def _build_label_index(test_data: List[dict]) -> Dict[str, int]:
        """Build label index from test_data"""
        label_by_id = {}
        for item in test_data:
            pair_id = item.get('left_right_pair', '')
            if not pair_id:
                left = item.get('left', {})
                right = item.get('right', {})
                left_id = str(left.get('_id') or left.get('id') or left.get('ID') or '')
                right_id = str(right.get('_id') or right.get('id') or right.get('ID') or '')
                pair_id = f"{left_id}_{right_id}"
            label_by_id[pair_id] = item.get('label', 1)
        return label_by_id

    @staticmethod
    def evaluate_end_to_end(
        predictions: List[dict],
        ground_truth: List[dict],
        test_data: List[dict] = None
    ) -> dict:
        """End-to-end evaluation"""
        gt_by_id = Evaluator._build_gt_index(ground_truth)
        total_gt_correspondences = sum(len(gt.get('output', [])) for gt in ground_truth)

        label_by_id = {}
        if test_data:
            label_by_id = Evaluator._build_label_index(test_data)

        em_tp, em_fp, em_fn, em_tn = 0, 0, 0, 0
        e2e_am_tp, e2e_am_fp, e2e_am_fn = 0, 0, 0
        e2e_total_pairs = 0
        e2e_exact_matches = 0
        cond_am_tp, cond_am_fp, cond_am_fn = 0, 0, 0
        cond_total_pairs = 0
        cond_exact_matches = 0
        joint_correct = 0
        total_pairs = 0

        for pred in predictions:
            pair_id = pred['pair_id']
            total_pairs += 1

            if pair_id in label_by_id:
                true_label = label_by_id[pair_id]
            elif pair_id in gt_by_id:
                true_label = 1
            else:
                true_label = 0

            pred_label = pred['em_pred']

            true_corrs = set()
            if true_label == 1 and pair_id in gt_by_id:
                gt = gt_by_id[pair_id]
                if 'output' in gt:
                    true_corrs = {canonicalize_correspondence(c) for c in gt['output']}

            pred_corrs = set()
            if pred_label == 1:
                pred_corrs = {canonicalize_correspondence(c) for c in pred['am_pred']}

            # EM statistics
            if pred_label == 1 and true_label == 1:
                em_tp += 1
            elif pred_label == 1 and true_label == 0:
                em_fp += 1
            elif pred_label == 0 and true_label == 1:
                em_fn += 1
            else:
                em_tn += 1

            # AM statistics
            pair_tp = len(pred_corrs & true_corrs)
            pair_fp = len(pred_corrs - true_corrs)
            pair_fn = len(true_corrs - pred_corrs)
            pair_exact = (pair_fp == 0 and pair_fn == 0)

            if pred_label == 1:
                e2e_am_tp += pair_tp
                e2e_am_fp += pair_fp
                e2e_am_fn += pair_fn
                e2e_total_pairs += 1
                if pair_exact:
                    e2e_exact_matches += 1

            if pred_label == 1 and true_label == 1:
                cond_am_tp += pair_tp
                cond_am_fp += pair_fp
                cond_am_fn += pair_fn
                cond_total_pairs += 1
                if pair_exact:
                    cond_exact_matches += 1

            em_correct = (pred_label == true_label)
            if em_correct:
                if true_label == 0:
                    joint_correct += 1
                elif pair_exact:
                    joint_correct += 1

        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        em_precision = safe_div(em_tp, em_tp + em_fp)
        em_recall = safe_div(em_tp, em_tp + em_fn)
        em_f1 = safe_div(2 * em_precision * em_recall, em_precision + em_recall)
        em_accuracy = safe_div(em_tp + em_tn, total_pairs)

        e2e_precision = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fp)
        e2e_recall = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fn)
        e2e_f1 = safe_div(2 * e2e_precision * e2e_recall, e2e_precision + e2e_recall)
        e2e_accuracy = safe_div(e2e_am_tp, e2e_am_tp + e2e_am_fp + e2e_am_fn)
        e2e_pair_accuracy = safe_div(e2e_exact_matches, e2e_total_pairs)

        cond_precision = safe_div(cond_am_tp, cond_am_tp + cond_am_fp)
        cond_recall = safe_div(cond_am_tp, cond_am_tp + cond_am_fn)
        cond_f1 = safe_div(2 * cond_precision * cond_recall, cond_precision + cond_recall)
        cond_accuracy = safe_div(cond_am_tp, cond_am_tp + cond_am_fp + cond_am_fn)
        cond_pair_accuracy = safe_div(cond_exact_matches, cond_total_pairs)

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
# Run
# =============================================================================

def run_baseline(
    test_data_path: str,
    gt_path: str,
    output_path: str,
    em_threshold: float = 0.7,
    similarity_threshold: float = 0.5,
    top_k: int = 3,
    group_threshold: float = 0.6,
    enable_nn: bool = True,
    nn_max_size: int = 4
):
    """Run Embedding Hybrid baseline"""
    print("Loading data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    print(f"  Test pairs: {len(test_data)}")
    print(f"  Ground truth: {len(ground_truth)}")

    config = Config(
        em_threshold=em_threshold,
        similarity_threshold=similarity_threshold,
        top_k=top_k,
        group_threshold=group_threshold,
        enable_nn=enable_nn,
        nn_max_size=nn_max_size
    )

    matcher = EmbeddingHybridMatcher(config)

    print("\nRunning predictions...")
    predictions = []
    for i, pair in enumerate(test_data):
        result = matcher.predict(pair)
        predictions.append(result)

        label = pair.get('label', 1)
        pred = result['em_pred']
        status = "✓" if pred == label else "✗"
        if pred == 1:
            num_corr = len(result['am_pred'])
            print(f"  [{i+1:3d}/{len(test_data)}] {result['pair_id']} -> pred={pred} {status}, {num_corr} correspondences")
        else:
            print(f"  [{i+1:3d}/{len(test_data)}] {result['pair_id']} -> pred={pred} {status}")

    print("\nEvaluating...")
    metrics = Evaluator.evaluate_end_to_end(predictions, ground_truth, test_data)

    print("\n" + "=" * 60)
    print("Results: Embedding Hybrid Baseline")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Model: {config.model_name}")
    print(f"  EM Threshold: {em_threshold}")
    print(f"  AM Similarity Threshold: {similarity_threshold}")
    print(f"  Top-k: {top_k}")
    print(f"  Group Threshold: {group_threshold}")
    print(f"  N-N Enabled: {enable_nn}")
    if enable_nn:
        print(f"  N-N Max Size: {nn_max_size}")

    print(f"\nEntity Matching:")
    print(f"  TP={metrics['EM']['TP']}, FP={metrics['EM']['FP']}, FN={metrics['EM']['FN']}, TN={metrics['EM']['TN']}")
    print(f"  Precision: {metrics['EM']['Precision']}")
    print(f"  Recall:    {metrics['EM']['Recall']}")
    print(f"  F1:        {metrics['EM']['F1']}")
    print(f"  Accuracy:  {metrics['EM']['Accuracy']}")

    print(f"\nAttribute Matching (End-to-End):")
    print(f"  TP={metrics['AM_EndToEnd']['TP']}, FP={metrics['AM_EndToEnd']['FP']}, FN={metrics['AM_EndToEnd']['FN']}")
    print(f"  Precision:     {metrics['AM_EndToEnd']['Precision']}")
    print(f"  Recall:        {metrics['AM_EndToEnd']['Recall']}")
    print(f"  F1:            {metrics['AM_EndToEnd']['F1']}")
    print(f"  Accuracy:      {metrics['AM_EndToEnd']['Accuracy']}")
    print(f"  Pair Accuracy: {metrics['AM_EndToEnd']['Pair_Accuracy']} [{metrics['AM_EndToEnd']['Exact_Matches']}/{metrics['AM_EndToEnd']['Total_Pairs']}]")

    print(f"\nAttribute Matching (Conditional - TP only):")
    print(f"  TP={metrics['AM_Conditional']['TP']}, FP={metrics['AM_Conditional']['FP']}, FN={metrics['AM_Conditional']['FN']}")
    print(f"  Precision:     {metrics['AM_Conditional']['Precision']}")
    print(f"  Recall:        {metrics['AM_Conditional']['Recall']}")
    print(f"  F1:            {metrics['AM_Conditional']['F1']}")
    print(f"  Accuracy:      {metrics['AM_Conditional']['Accuracy']}")
    print(f"  Pair Accuracy: {metrics['AM_Conditional']['Pair_Accuracy']} [{metrics['AM_Conditional']['Exact_Matches']}/{metrics['AM_Conditional']['Total_Pairs']}]")
    print(f"  Total GT Correspondences: {metrics['AM_Conditional']['Total_GT_Correspondences']}")

    print(f"\nJoint Exact-Match:")
    print(f"  Accuracy: {metrics['Joint_Exact_Match']['Accuracy']} [{metrics['Joint_Exact_Match']['Correct']}/{metrics['Joint_Exact_Match']['Total']}]")

    output = {
        "config": {
            "method": "Embedding Hybrid (EM: record similarity, AM: matrix + greedy)",
            "model": config.model_name,
            "em_threshold": em_threshold,
            "similarity_threshold": similarity_threshold,
            "top_k": top_k,
            "group_threshold": group_threshold,
            "enable_nn": enable_nn,
            "nn_max_size": nn_max_size
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
        description="Baseline: Embedding Hybrid (EM from frozen_sentence_encoder + AM from embedding_threshold)"
    )
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    # EM parameters
    parser.add_argument("--em_threshold", type=float, default=0.7,
                        help="EM threshold (default: 0.7)")
    # AM parameters
    parser.add_argument("--similarity_threshold", type=float, default=0.5,
                        help="AM similarity threshold (default: 0.5)")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k candidates per attribute (default: 3)")
    parser.add_argument("--group_threshold", type=float, default=0.6,
                        help="Group selection threshold (default: 0.6)")
    # N-N parameters
    parser.add_argument("--enable_nn", action="store_true", default=True,
                        help="Enable N-N candidate generation (default: True)")
    parser.add_argument("--disable_nn", action="store_true",
                        help="Disable N-N candidate generation")
    parser.add_argument("--nn_max_size", type=int, default=4,
                        help="Maximum size per side for N-N groups (default: 4)")

    args = parser.parse_args()

    enable_nn = not args.disable_nn

    run_baseline(
        test_data_path=args.test_data,
        gt_path=args.ground_truth,
        output_path=args.output,
        em_threshold=args.em_threshold,
        similarity_threshold=args.similarity_threshold,
        top_k=args.top_k,
        group_threshold=args.group_threshold,
        enable_nn=enable_nn,
        nn_max_size=args.nn_max_size
    )
