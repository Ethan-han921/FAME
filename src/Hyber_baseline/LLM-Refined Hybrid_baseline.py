"""
Baseline: Embedding + LLM

Pipeline:
  1. Single attribute embedding
  2. Compute L x R cosine similarity matrix
  3. Generate candidates (with scores)
  4. Send candidates + scores to LLM
  5. LLM selects correct matches

Model: sentence-transformers/all-mpnet-base-v2
Supports: 1-1, 1-N, N-1, N-N correspondences
"""

import json
import logging
import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, FrozenSet
from dataclasses import dataclass
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# Load .env file
load_dotenv()


@dataclass
class Config:
    """Configuration parameters"""
    model_name: str = 'all-mpnet-base-v2'
    llm_model: str = 'gpt-5.6-luna'
    similarity_threshold: float = 0.5
    llm_max_tokens: int = 2000
    llm_temperature: float = 0.0


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


def canonicalize_correspondence(corr) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """Convert correspondence to unordered form"""
    if isinstance(corr, str):
        corr = json.loads(corr)
    left = frozenset(x.lower().strip() for x in corr[0])
    right = frozenset(x.lower().strip() for x in corr[1])
    return (left, right)


class PipelineLogger:

    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Main logger
        self.main_logger = logging.getLogger('main')
        self.main_logger.setLevel(logging.INFO)
        self.main_logger.handlers = []

        main_handler = logging.FileHandler(
            os.path.join(log_dir, 'run.log'), encoding='utf-8')
        main_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.main_logger.addHandler(main_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.main_logger.addHandler(console_handler)

        # Prompt log
        self.prompt_file = open(
            os.path.join(log_dir, 'prompt.log'), 'w', encoding='utf-8')

        # Response log
        self.response_file = open(
            os.path.join(log_dir, 'response.log'), 'w', encoding='utf-8')

        # Detailed step log
        self.step_file = open(
            os.path.join(log_dir, 'steps.log'), 'w', encoding='utf-8')

    def info(self, msg: str):
        self.main_logger.info(msg)

    def log_header(self, config: Config, test_data_path: str, gt_path: str):
        """Log run header information"""
        self.info("=" * 60)
        self.info("EMBEDDING + LLM BASELINE")
        self.info("=" * 60)
        self.info(f"Embedding Model: {config.model_name}")
        self.info(f"LLM Model:       {config.llm_model}")
        self.info(f"Threshold:       {config.similarity_threshold}")
        self.info(f"Test Data:       {test_data_path}")
        self.info(f"Ground Truth:    {gt_path}")
        self.info(f"Log Dir:         {self.log_dir}")

    def log_step(self, pair_id: str, step_name: str, content: str):
        """Log detailed step"""
        self.step_file.write(f"\n{'=' * 60}\n")
        self.step_file.write(f"Pair: {pair_id}\n")
        self.step_file.write(f"Step: {step_name}\n")
        self.step_file.write(f"{'=' * 60}\n")
        self.step_file.write(content)
        self.step_file.write("\n")
        self.step_file.flush()

    def log_prompt(self, pair_id: str, system_prompt: str, user_prompt: str):
        """Log LLM prompt"""
        self.prompt_file.write(f"\n{'=' * 60}\n")
        self.prompt_file.write(f"Pair: {pair_id}\n")
        self.prompt_file.write(f"{'=' * 60}\n")
        self.prompt_file.write("[SYSTEM PROMPT]\n\n")
        self.prompt_file.write(system_prompt)
        self.prompt_file.write("\n\n[USER PROMPT]\n\n")
        self.prompt_file.write(user_prompt)
        self.prompt_file.write("\n")
        self.prompt_file.flush()

    def log_response(self, pair_id: str, response: str):
        """Log LLM response"""
        self.response_file.write(f"[{pair_id}]\n")
        self.response_file.write(response)
        self.response_file.write("\n\n")
        self.response_file.flush()

    def close(self):
        """Close files"""
        self.prompt_file.close()
        self.response_file.close()
        self.step_file.close()


# Embedding + LLM Selector

class EmbeddingLLMMatcher:
    """
    Embedding + LLM for Attribute Matching

    Pipeline:
    1. Single attribute embeddings
    2. Compute L x R cosine similarity matrix
    3. Generate candidates (with scores)
    4. LLM selection
    """

    def __init__(self, config: Config = None, logger: PipelineLogger = None):
        self.config = config or Config()
        self.logger = logger

        print(f"Loading embedding model: {self.config.model_name}")
        self.model = SentenceTransformer(self.config.model_name)

        print(f"Using LLM: {self.config.llm_model}")
        self.client = OpenAI()

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode text list"""
        return self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

    def compute_similarity_matrix(
        self,
        left_texts: List[str],
        right_texts: List[str]
    ) -> np.ndarray:
        """
        Compute L x R cosine similarity matrix for single attributes

        Returns:
            sim_matrix: shape (len(left), len(right))
        """
        if not left_texts or not right_texts:
            return np.array([])

        left_embs = self.encode(left_texts)
        right_embs = self.encode(right_texts)

        # Normalize
        left_norm = left_embs / \
            (np.linalg.norm(left_embs, axis=1, keepdims=True) + 1e-8)
        right_norm = right_embs / \
            (np.linalg.norm(right_embs, axis=1, keepdims=True) + 1e-8)

        # Cosine similarity matrix
        sim_matrix = np.dot(left_norm, right_norm.T)

        return sim_matrix

    def generate_candidates(
        self,
        sim_matrix: np.ndarray,
        left_attrs: List[str],
        right_attrs: List[str],
        left_flat: Dict[str, str],
        right_flat: Dict[str, str]
    ) -> List[Dict]:
        """
        Generate candidates based on similarity matrix

        Take all (left, right) pairs above threshold
        """
        n_left, n_right = sim_matrix.shape
        candidates = []

        for i in range(n_left):
            for j in range(n_right):
                score = sim_matrix[i, j]
                if score >= self.config.similarity_threshold:
                    candidates.append({
                        "left_attr": left_attrs[i],
                        "left_value": left_flat[left_attrs[i]],
                        "right_attr": right_attrs[j],
                        "right_value": right_flat[right_attrs[j]],
                        "score": float(score)
                    })

        # Sort by score descending
        candidates = sorted(candidates, key=lambda x: -x["score"])

        return candidates

    def _get_system_prompt(self) -> str:
        """LLM system prompt"""
        return """You are an expert in schema matching and data integration.

Your task is to select correct attribute correspondences from the candidates provided.
Each candidate shows a potential match between a Left attribute and a Right attribute.

Rules:
1. Select correspondences by considering both attribute names and values:
    - Attribute names: The names, individually or in combination, should convey equivalent meanings.
    - Attribute values: Numeric values should be equal after normalization; string values should be semantically equivalent.

2. You should select the most valid candidate, and combine multiple candidates into 1-to-1, 1-to-many, many-to-1, or many-to-many groups.
3. You may reject incorrect candidates.
4. Correspondences can be:
   - 1-1: one left attr <-> one right attr
   - N-1: multiple left attrs <-> one right attr (combine multiple left candidates pointing to same right)
   - 1-N: one left attr <-> multiple right attrs (combine multiple right candidates pointing to same left)
   - N-N: multiple left attrs <-> multiple right attrs (multiple attributes on both sides jointly represent equivalent information)

5. Only output correspondences you are confident about

Output format:
Return ONLY a JSON array of correspondences. Each correspondence is [[left_attrs], [right_attrs]].
Example: [[["Title"], ["title"]], [["Author"], ["authors"]], [["first_name", "last_name"], ["full_name"]]]

If no valid correspondences exist, return: []"""

    def _build_prompt(
        self,
        candidates: List[Dict],
        left_flat: Dict[str, str],
        right_flat: Dict[str, str]
    ) -> str:
        """Build LLM prompt"""
        prompt = "## Left Record Attributes:\n"
        for attr, value in left_flat.items():
            prompt += f"- {attr}: {value}\n"

        prompt += "\n## Right Record Attributes:\n"
        for attr, value in right_flat.items():
            prompt += f"- {attr}: {value}\n"

        prompt += "\n## Candidates:\n"
        for i, cand in enumerate(candidates):
            prompt += f"{i+1}. [{cand['left_attr']}] <-> [{cand['right_attr']}]\n"
            prompt += f"    Left value: {cand['left_value']}\n"
            prompt += f"    Right value: {cand['right_value']}\n"

        prompt += "\n## Task:\n"
        prompt += "Select the correct correspondences from the candidates above.\n"
        prompt += "You can combine multiple candidates into N-1 or 1-N correspondences if appropriate.\n"
        prompt += "Output ONLY a JSON array of correspondences.\n"

        return prompt

    def _parse_llm_response(self, response: str) -> List[List[List[str]]]:
        """Parse LLM response"""
        import re

        # Remove markdown code blocks
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        response = response.strip()

        # Find JSON array
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if not json_match:
            return []

        json_str = json_match.group()

        # Try to parse
        correspondences = None
        try:
            correspondences = json.loads(json_str)
        except json.JSONDecodeError as e:
            # If "Extra data" error, may be missing outer brackets
            if "Extra data" in str(e):
                try:
                    # Try adding outer brackets
                    fixed_str = "[" + json_str + "]"
                    correspondences = json.loads(fixed_str)
                except:
                    pass

            # If still failed, try parsing by comma separation
            if correspondences is None:
                try:
                    # Match all [[...], [...]] patterns
                    pattern = r'\[\s*\[([^\[\]]*)\]\s*,\s*\[([^\[\]]*)\]\s*\]'
                    matches = re.findall(pattern, json_str)
                    if matches:
                        correspondences = []
                        for left, right in matches:
                            left_attrs = [s.strip().strip('"\'')
                                          for s in left.split(',')]
                            right_attrs = [s.strip().strip('"\'')
                                           for s in right.split(',')]
                            correspondences.append([left_attrs, right_attrs])
                except:
                    pass

        if correspondences is None:
            print(f"Parse error: Could not parse response")
            return []

        # Validate format
        valid_corrs = []
        for corr in correspondences:
            if isinstance(corr, list) and len(corr) == 2:
                if isinstance(corr[0], list) and isinstance(corr[1], list):
                    valid_corrs.append(corr)

        return valid_corrs

    def attribute_matching(
        self,
        left: dict,
        right: dict,
        pair_id: str = None
    ) -> Tuple[List[List[List[str]]], Dict]:
        """
        Attribute Matching Pipeline

        Returns:
            (correspondences, stats)
        """
        # Step 1: Flatten records
        left_flat = flatten_record(left)
        right_flat = flatten_record(right)

        if self.logger and pair_id:
            step1_content = "Left Record (flattened):\n"
            for attr, value in left_flat.items():
                step1_content += f"  {attr}: {value}\n"
            step1_content += "\nRight Record (flattened):\n"
            for attr, value in right_flat.items():
                step1_content += f"  {attr}: {value}\n"
            self.logger.log_step(
                pair_id, "Step 1: Flatten Records", step1_content)

        if not left_flat or not right_flat:
            return [], {"left_attrs": 0, "right_attrs": 0}

        # Step 2: Single attribute list
        left_attrs = list(left_flat.keys())
        right_attrs = list(right_flat.keys())

        # Step 3: Serialize single attributes "attr_name: value"
        left_texts = [f"{attr}: {left_flat[attr]}" for attr in left_attrs]
        right_texts = [f"{attr}: {right_flat[attr]}" for attr in right_attrs]

        if self.logger and pair_id:
            step2_content = "Left texts for embedding:\n"
            for i, text in enumerate(left_texts):
                step2_content += f"  [{i}] {text}\n"
            step2_content += "\nRight texts for embedding:\n"
            for i, text in enumerate(right_texts):
                step2_content += f"  [{i}] {text}\n"
            self.logger.log_step(
                pair_id, "Step 2: Prepare Embedding Texts", step2_content)

        # Step 4: Compute L x R similarity matrix
        sim_matrix = self.compute_similarity_matrix(left_texts, right_texts)

        if sim_matrix.size == 0:
            return [], {"left_attrs": len(left_flat), "right_attrs": len(right_flat)}

        if self.logger and pair_id:
            step3_content = f"Similarity Matrix Shape: {sim_matrix.shape}\n\n"
            step3_content += "Matrix (rows=left, cols=right):\n"
            step3_content += f"{'':>20}"
            for attr in right_attrs:
                step3_content += f"{attr[:10]:>12}"
            step3_content += "\n"
            for i, left_attr in enumerate(left_attrs):
                step3_content += f"{left_attr[:20]:>20}"
                for j in range(len(right_attrs)):
                    step3_content += f"{sim_matrix[i, j]:>12.4f}"
                step3_content += "\n"
            self.logger.log_step(
                pair_id, "Step 3: Compute Similarity Matrix", step3_content)

        # Step 5: Generate candidates
        candidates = self.generate_candidates(
            sim_matrix, left_attrs, right_attrs, left_flat, right_flat
        )

        if self.logger and pair_id:
            step4_content = f"Threshold: {self.config.similarity_threshold}\n"
            step4_content += f"Total candidates above threshold: {len(candidates)}\n\n"
            step4_content += "Candidates (sorted by score):\n"
            for i, cand in enumerate(candidates):
                step4_content += f"  {i+1}. [{cand['left_attr']}] <-> [{cand['right_attr']}] "
                step4_content += f"(score={cand['score']:.4f})\n"
                step4_content += f"      Left:  {cand['left_value']}\n"
                step4_content += f"      Right: {cand['right_value']}\n"
            self.logger.log_step(
                pair_id, "Step 4: Generate Candidates", step4_content)

        if not candidates:
            return [], {
                "left_attrs": len(left_flat),
                "right_attrs": len(right_flat),
                "candidates": 0
            }

        # Step 6: Build prompt and call LLM
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_prompt(candidates, left_flat, right_flat)

        if self.logger and pair_id:
            self.logger.log_prompt(pair_id, system_prompt, user_prompt)
            self.logger.log_step(pair_id, "Step 5: Build LLM Prompt",
                                 f"System prompt length: {len(system_prompt)} chars\n"
                                 f"User prompt length: {len(user_prompt)} chars\n"
                                 f"Total candidates in prompt: {len(candidates)}")

        try:
            response = self.client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.config.llm_max_tokens
            )

            result_text = response.choices[0].message.content.strip()

            if self.logger and pair_id:
                self.logger.log_response(pair_id, result_text)
                self.logger.log_step(pair_id, "Step 6: LLM Response",
                                     f"Raw response:\n{result_text}")

            correspondences = self._parse_llm_response(result_text)

            if self.logger and pair_id:
                step7_content = f"Parsed correspondences: {len(correspondences)}\n\n"
                for i, corr in enumerate(correspondences):
                    step7_content += f"  {i+1}. {corr[0]} <-> {corr[1]}\n"
                self.logger.log_step(
                    pair_id, "Step 7: Parse Response", step7_content)

        except Exception as e:
            print(f"LLM error: {e}")
            if self.logger and pair_id:
                self.logger.log_step(pair_id, "Step 6: LLM Error", str(e))
            correspondences = []

        stats = {
            "left_attrs": len(left_flat),
            "right_attrs": len(right_flat),
            "matrix_shape": f"{sim_matrix.shape[0]}x{sim_matrix.shape[1]}",
            "candidates": len(candidates),
            "output_correspondences": len(correspondences)
        }

        return correspondences, stats

    def predict(self, pair: dict) -> dict:
        """Full prediction pipeline"""
        left = pair['left']
        right = pair['right']

        # Support multiple ID field names
        left_id = str(left.get('_id') or left.get(
            'id') or left.get('ID') or '')
        right_id = str(right.get('_id') or right.get(
            'id') or right.get('ID') or '')
        pair_id = f"{left_id}_{right_id}"

        am_pred, am_stats = self.attribute_matching(left, right, pair_id)

        return {
            "pair_id": pair_id,
            "am_pred": am_pred,
            "am_stats": am_stats
        }


# Evaluation
class Evaluator:
    """Evaluator - unified with tfidf/frozen_sentence_encoder baseline"""

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
                left_id = str(left.get('_id') or left.get(
                    'id') or left.get('ID') or '')
                right_id = str(right.get('_id') or right.get(
                    'id') or right.get('ID') or '')
                pair_id = f"{left_id}_{right_id}"
            label_by_id[pair_id] = item.get('label', 1)
        return label_by_id

    @staticmethod
    def evaluate(predictions: List[dict], ground_truth: List[dict]) -> dict:
        """Evaluate AM performance (compatible with old interface, returns End-to-End results)"""
        gt_by_id = Evaluator._build_gt_index(ground_truth)

        am_tp, am_fp, am_fn = 0, 0, 0
        exact_matches = 0

        for pred in predictions:
            pair_id = pred['pair_id']

            true_corrs = set()
            if pair_id in gt_by_id:
                gt = gt_by_id[pair_id]
                if 'output' in gt:
                    true_corrs = {canonicalize_correspondence(
                        c) for c in gt['output']}

            pred_corrs = {canonicalize_correspondence(
                c) for c in pred['am_pred']}

            pair_tp = len(pred_corrs & true_corrs)
            pair_fp = len(pred_corrs - true_corrs)
            pair_fn = len(true_corrs - pred_corrs)

            am_tp += pair_tp
            am_fp += pair_fp
            am_fn += pair_fn

            if pair_fp == 0 and pair_fn == 0:
                exact_matches += 1

        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        precision = safe_div(am_tp, am_tp + am_fp)
        recall = safe_div(am_tp, am_tp + am_fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        accuracy = safe_div(am_tp, am_tp + am_fp + am_fn)
        pair_accuracy = safe_div(exact_matches, len(predictions))

        return {
            "TP": am_tp, "FP": am_fp, "FN": am_fn,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
            "Accuracy": round(accuracy, 4),
            "Pair_Accuracy": round(pair_accuracy, 4),
            "Exact_Matches": exact_matches,
            "Total_Pairs": len(predictions)
        }

    @staticmethod
    def evaluate_end_to_end(
        predictions: List[dict],
        ground_truth: List[dict],
        test_data: List[dict] = None
    ) -> dict:
        """
        End-to-end evaluation - unified with tfidf/frozen_sentence_encoder baseline

        Three AM metrics:
        - AM_EndToEnd: All pairs (this baseline runs AM on all pairs)
        - AM_Conditional: Only pairs with label=1 (pred=1 AND label=1)
        - Joint_Exact_Match: EM correct AND AM exact match

        Note: This baseline has no EM, assumes pred=1 for all pairs
        """
        gt_by_id = Evaluator._build_gt_index(ground_truth)

        # Calculate total GT correspondences
        total_gt_correspondences = sum(
            len(gt.get('output', [])) for gt in ground_truth)

        # Infer label from test_data or GT
        label_by_id = {}
        if test_data:
            label_by_id = Evaluator._build_label_index(test_data)

        # This baseline has no EM, assumes pred=1 for all
        em_tp, em_fp, em_fn, em_tn = 0, 0, 0, 0

        # End-to-End AM: All pairs
        e2e_am_tp, e2e_am_fp, e2e_am_fn = 0, 0, 0
        e2e_total_pairs = 0
        e2e_exact_matches = 0

        # Conditional AM: Only pairs with label=1 (pred=1 AND label=1)
        cond_am_tp, cond_am_fp, cond_am_fn = 0, 0, 0
        cond_total_pairs = 0
        cond_exact_matches = 0

        # Joint Exact Match
        joint_correct = 0
        total_pairs = 0

        for pred in predictions:
            pair_id = pred['pair_id']
            total_pairs += 1

            # Get true label
            if pair_id in label_by_id:
                true_label = label_by_id[pair_id]
            elif pair_id in gt_by_id:
                true_label = 1
            else:
                true_label = 0

            # This baseline has no EM, assumes pred=1
            pred_label = 1

            # Get AM ground truth
            true_corrs = set()
            if true_label == 1 and pair_id in gt_by_id:
                gt = gt_by_id[pair_id]
                if 'output' in gt:
                    true_corrs = {canonicalize_correspondence(
                        c) for c in gt['output']}

            pred_corrs = {canonicalize_correspondence(
                c) for c in pred['am_pred']}

            # EM statistics (assumes pred=1)
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

            # End-to-End AM: All pairs
            e2e_am_tp += pair_tp
            e2e_am_fp += pair_fp
            e2e_am_fn += pair_fn
            e2e_total_pairs += 1
            if pair_exact:
                e2e_exact_matches += 1

            # Conditional AM: Only pairs with label=1 (pred=1 AND label=1)
            if true_label == 1:
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
                    joint_correct += 1
                elif pair_exact:
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
                "Accuracy": round(em_accuracy, 4),
                "Note": "No EM in this baseline, assuming pred=1 for all pairs"
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


# Run
def run_baseline(
    test_data_path: str,
    gt_path: str,
    output_path: str,
    similarity_threshold: float = 0.5,
    llm_model: str = 'gpt-5.6-luna',
    log_dir: str = None
):
    """Run Embedding + LLM baseline"""
    # Create log directory
    if log_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/embedding_llm_{timestamp}"

    logger = PipelineLogger(log_dir)

    config = Config(
        similarity_threshold=similarity_threshold,
        llm_model=llm_model
    )

    logger.log_header(config, test_data_path, gt_path)

    logger.info("Loading data...")
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    with open(gt_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    logger.info(f"  Test pairs: {len(test_data)}")
    logger.info(f"  Ground truth: {len(ground_truth)}")

    matcher = EmbeddingLLMMatcher(config, logger)

    logger.info("")
    logger.info("=" * 60)
    logger.info("RUNNING PREDICTIONS")
    logger.info("=" * 60)

    predictions = []
    for i, pair in enumerate(test_data):
        result = matcher.predict(pair)
        predictions.append(result)

        num_corr = len(result['am_pred'])
        logger.info(
            f"  [{i+1:3d}/{len(test_data)}] {result['pair_id']} -> {num_corr} correspondences")

    logger.info("")
    logger.info("=" * 60)
    logger.info("EVALUATION")
    logger.info("=" * 60)

    # Unified evaluation (consistent with tfidf/frozen_sentence_encoder baseline)
    metrics = Evaluator.evaluate_end_to_end(
        predictions, ground_truth, test_data)

    # EM results (this baseline has no EM, assumes pred=1)
    logger.info(f"\nEntity Matching (assumed pred=1 for all):")
    logger.info(
        f"  TP={metrics['EM']['TP']}, FP={metrics['EM']['FP']}, FN={metrics['EM']['FN']}, TN={metrics['EM']['TN']}")
    logger.info(f"  Precision: {metrics['EM']['Precision']}")
    logger.info(f"  Recall:    {metrics['EM']['Recall']}")
    logger.info(f"  F1:        {metrics['EM']['F1']}")
    logger.info(f"  Accuracy:  {metrics['EM']['Accuracy']}")

    # End-to-End AM
    logger.info(f"\nAttribute Matching (End-to-End):")
    logger.info(
        f"  TP={metrics['AM_EndToEnd']['TP']}, FP={metrics['AM_EndToEnd']['FP']}, FN={metrics['AM_EndToEnd']['FN']}")
    logger.info(f"  Precision:     {metrics['AM_EndToEnd']['Precision']}")
    logger.info(f"  Recall:        {metrics['AM_EndToEnd']['Recall']}")
    logger.info(f"  F1:            {metrics['AM_EndToEnd']['F1']}")
    logger.info(f"  Accuracy:      {metrics['AM_EndToEnd']['Accuracy']}")
    logger.info(
        f"  Pair Accuracy: {metrics['AM_EndToEnd']['Pair_Accuracy']} [{metrics['AM_EndToEnd']['Exact_Matches']}/{metrics['AM_EndToEnd']['Total_Pairs']}]")

    # Conditional AM (pred=1 AND label=1)
    logger.info(f"\nAttribute Matching (Conditional - label=1 only):")
    logger.info(
        f"  TP={metrics['AM_Conditional']['TP']}, FP={metrics['AM_Conditional']['FP']}, FN={metrics['AM_Conditional']['FN']}")
    logger.info(f"  Precision:     {metrics['AM_Conditional']['Precision']}")
    logger.info(f"  Recall:        {metrics['AM_Conditional']['Recall']}")
    logger.info(f"  F1:            {metrics['AM_Conditional']['F1']}")
    logger.info(f"  Accuracy:      {metrics['AM_Conditional']['Accuracy']}")
    logger.info(
        f"  Pair Accuracy: {metrics['AM_Conditional']['Pair_Accuracy']} [{metrics['AM_Conditional']['Exact_Matches']}/{metrics['AM_Conditional']['Total_Pairs']}]")
    logger.info(
        f"  Total GT Correspondences: {metrics['AM_Conditional']['Total_GT_Correspondences']}")

    # Joint Exact Match
    logger.info(f"\nJoint Exact-Match:")
    logger.info(
        f"  Accuracy: {metrics['Joint_Exact_Match']['Accuracy']} [{metrics['Joint_Exact_Match']['Correct']}/{metrics['Joint_Exact_Match']['Total']}]")

    output = {
        "config": {
            "method": "Embedding + LLM",
            "embedding_model": config.model_name,
            "llm_model": config.llm_model,
            "similarity_threshold": similarity_threshold
        },
        "metrics": metrics,
        "predictions": predictions
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # Save metrics to log directory
    with open(os.path.join(log_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Save detailed diagnostics
    diagnostics = []
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

    for pred in predictions:
        pair_id = pred['pair_id']
        true_corrs = set()
        if pair_id in gt_by_id:
            gt = gt_by_id[pair_id]
            if 'output' in gt:
                true_corrs = {canonicalize_correspondence(
                    c) for c in gt['output']}

        pred_corrs = {canonicalize_correspondence(c) for c in pred['am_pred']}

        tp = len(pred_corrs & true_corrs)
        fp = len(pred_corrs - true_corrs)
        fn = len(true_corrs - pred_corrs)

        diagnostics.append({
            "pair_id": pair_id,
            "pred_corrs": pred['am_pred'],
            "true_corrs": gt_by_id.get(pair_id, {}).get('output', []),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "exact_match": fp == 0 and fn == 0,
            "stats": pred['am_stats']
        })

    with open(os.path.join(log_dir, 'diagnostics.json'), 'w', encoding='utf-8') as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to: {output_path}")
    logger.info(f"Logs saved to: {log_dir}")

    logger.close()

    return output


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Baseline: Embedding + LLM"
    )
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.5)
    parser.add_argument("--llm_model", type=str, default='gpt-5.6-luna')
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Log directory (default: logs/embedding_llm_TIMESTAMP)")

    args = parser.parse_args()

    run_baseline(
        test_data_path=args.test_data,
        gt_path=args.ground_truth,
        output_path=args.output,
        similarity_threshold=args.similarity_threshold,
        llm_model=args.llm_model,
        log_dir=args.log_dir
    )
