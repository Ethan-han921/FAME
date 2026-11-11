#!/usr/bin/env python
"""
Test cases for Joint Exact-Match Accuracy.

Covers:
1. EM and AM both correct
2. EM correct, but AM missing one group
3. EM correct, but AM has extra group
4. AM matches, but EM wrong
5. Different order of groups and attributes within groups (should still match)
6. Different group structure (should NOT match even if pairwise edges are similar)
7. Correct negative case with empty AM
8. Missing prediction or AM parse failure

Run:
    python two_stage/test_joint_exact_match.py
"""

import sys
import json
import logging
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import (
    normalize_correspondence_strict,
    normalize_am_set,
    evaluate_joint_exact_match
)


def setup_logger():
    """Setup a simple logger for tests."""
    logger = logging.getLogger('test_joint')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    return logger


def test_normalize_correspondence_strict():
    """Test the strict normalization function."""
    print("\n=== Test normalize_correspondence_strict ===")

    # Test 1: Basic case
    corr = '[["Title"], ["title"]]'
    result = normalize_correspondence_strict(corr)
    assert result == (("Title",), ("title",)), f"Failed: {result}"
    print("  [PASS] Basic case")

    # Test 2: Multiple attributes (order should be sorted)
    corr = '[["b", "a"], ["c"]]'
    result = normalize_correspondence_strict(corr)
    assert result == (("a", "b"), ("c",)), f"Failed: {result}"
    print("  [PASS] Multiple attributes sorted")

    # Test 3: Preserve case
    corr = '[["Title"], ["TITLE"]]'
    result = normalize_correspondence_strict(corr)
    assert result == (("Title",), ("TITLE",)), f"Failed: {result}"
    print("  [PASS] Case preserved")

    # Test 4: Preserve spaces and nested paths
    corr = '[["publication_info.date_year"], ["Date Year"]]'
    result = normalize_correspondence_strict(corr)
    assert result == (("publication_info.date_year",), ("Date Year",)), f"Failed: {result}"
    print("  [PASS] Spaces and paths preserved")

    # Test 5: Invalid JSON
    corr = 'not valid json'
    result = normalize_correspondence_strict(corr)
    assert result is None, f"Should be None for invalid JSON: {result}"
    print("  [PASS] Invalid JSON returns None")

    # Test 6: Wrong structure (not a list of 2)
    corr = '[["a"]]'
    result = normalize_correspondence_strict(corr)
    assert result is None, f"Should be None for wrong structure: {result}"
    print("  [PASS] Wrong structure returns None")

    print("  All normalize_correspondence_strict tests passed!")


def test_normalize_am_set():
    """Test the AM set normalization function."""
    print("\n=== Test normalize_am_set ===")

    # Test 1: Empty list
    result = normalize_am_set([])
    assert result == frozenset(), f"Failed: {result}"
    print("  [PASS] Empty list")

    # Test 2: Single correspondence
    am_list = ['[["Title"], ["title"]]']
    result = normalize_am_set(am_list)
    expected = frozenset({(("Title",), ("title",))})
    assert result == expected, f"Failed: {result}"
    print("  [PASS] Single correspondence")

    # Test 3: Multiple correspondences
    am_list = [
        '[["Title"], ["title"]]',
        '[["Author"], ["author"]]'
    ]
    result = normalize_am_set(am_list)
    expected = frozenset({
        (("Title",), ("title",)),
        (("Author",), ("author",))
    })
    assert result == expected, f"Failed: {result}"
    print("  [PASS] Multiple correspondences")

    # Test 4: With parse error
    am_list = ['[["Title"], ["title"]]', 'invalid json']
    result = normalize_am_set(am_list)
    assert result is None, f"Should be None when parse fails: {result}"
    print("  [PASS] Parse error returns None")

    # Test 5: None input
    result = normalize_am_set(None)
    assert result == frozenset(), f"Failed: {result}"
    print("  [PASS] None input returns empty frozenset")

    print("  All normalize_am_set tests passed!")


def test_joint_exact_match():
    """Test the Joint Exact-Match evaluation."""
    print("\n=== Test evaluate_joint_exact_match ===")
    logger = setup_logger()

    # Test Case 1: EM and AM both correct
    print("\n  --- Case 1: EM and AM both correct ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    am_preds = {('L1', 'R1'): ['[["Title"], ["title"]]']}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 1
    assert result['metrics']['joint_exact_match_accuracy'] == 1.0
    print("  [PASS] Case 1")

    # Test Case 2: EM correct, AM missing one group
    print("\n  --- Case 2: EM correct, AM missing one group ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    am_preds = {('L1', 'R1'): ['[["Title"], ["title"]]']}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]', '[["Author"], ["author"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0
    assert result['diagnostics'][0]['reason'] == 'AM missing 1 groups'
    print("  [PASS] Case 2")

    # Test Case 3: EM correct, AM has extra group
    print("\n  --- Case 3: EM correct, AM has extra group ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    am_preds = {('L1', 'R1'): ['[["Title"], ["title"]]', '[["Extra"], ["extra"]]']}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0
    assert result['diagnostics'][0]['reason'] == 'AM extra 1 groups'
    print("  [PASS] Case 3")

    # Test Case 4: AM matches, but EM wrong (FN)
    print("\n  --- Case 4: AM matches, but EM wrong (FN) ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 0}  # Wrong!
    am_preds = {('L1', 'R1'): []}  # Not run because em_pred=0
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0
    assert result['diagnostics'][0]['em_correct'] == False
    print("  [PASS] Case 4")

    # Test Case 5: Different order of groups and attributes (should still match)
    print("\n  --- Case 5: Different order (should match) ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    # Prediction has different order: ["b", "a"] instead of ["a", "b"]
    am_preds = {('L1', 'R1'): [
        '[["b", "a"], ["c"]]',
        '[["Title"], ["title"]]'
    ]}
    # GT has original order
    am_gt = {('L1', 'R1'): [
        '[["Title"], ["title"]]',
        '[["a", "b"], ["c"]]'
    ]}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 1, f"Should match: {result['diagnostics']}"
    print("  [PASS] Case 5")

    # Test Case 6: Different group structure (should NOT match)
    print("\n  --- Case 6: Different group structure (should NOT match) ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    # Prediction has group [["a", "b"], ["c"]]
    am_preds = {('L1', 'R1'): ['[["a", "b"], ["c"]]']}
    # GT has it split into two groups: [["a"], ["c"]] and [["b"], ["c"]]
    am_gt = {('L1', 'R1'): ['[["a"], ["c"]]', '[["b"], ["c"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0, "Different structures should NOT match"
    print("  [PASS] Case 6")

    # Test Case 7: Correct negative case with empty AM
    print("\n  --- Case 7: Correct negative case (TN) ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 0}]
    em_preds = {('L1', 'R1'): 0}
    am_preds = {('L1', 'R1'): []}
    am_gt = {('L1', 'R1'): []}  # Empty for negative

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 1
    assert result['diagnostics'][0]['joint_exact_match'] == True
    print("  [PASS] Case 7")

    # Test Case 8: Missing prediction
    print("\n  --- Case 8: Missing prediction ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {}  # Missing!
    am_preds = {}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0
    assert result['metrics']['missing_predictions'] == 1
    assert result['diagnostics'][0]['reason'] == 'EM prediction missing'
    print("  [PASS] Case 8")

    # Test Case 9: AM parse failure
    print("\n  --- Case 9: AM parse failure ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 1}]
    em_preds = {('L1', 'R1'): 1}
    am_preds = {('L1', 'R1'): ['invalid json', '[["Title"], ["title"]]']}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    assert result['metrics']['joint_exact_match_correct'] == 0
    assert result['metrics']['invalid_am_outputs'] == 1
    print("  [PASS] Case 9")

    # Test Case 10: Invalid GT - negative with non-empty AM
    print("\n  --- Case 10: Invalid GT - negative with non-empty AM ---")
    gt_pairs = [{'left_id': 'L1', 'right_id': 'R1', 'label': 0}]
    em_preds = {('L1', 'R1'): 0}
    am_preds = {('L1', 'R1'): []}
    am_gt = {('L1', 'R1'): ['[["Title"], ["title"]]']}  # Invalid! Should be empty

    try:
        evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "label=0 but non-empty AM ground truth" in str(e)
        print("  [PASS] Case 10 - Correctly raised error")

    # Test Case 11: N=0 should raise error
    print("\n  --- Case 11: N=0 should raise error ---")
    try:
        evaluate_joint_exact_match([], {}, {}, {}, logger)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "N=0" in str(e)
        print("  [PASS] Case 11 - Correctly raised error for N=0")

    # Test Case 12: Multiple pairs mixed results
    print("\n  --- Case 12: Multiple pairs with mixed results ---")
    gt_pairs = [
        {'left_id': 'L1', 'right_id': 'R1', 'label': 1},  # Will be correct
        {'left_id': 'L2', 'right_id': 'R2', 'label': 1},  # EM wrong
        {'left_id': 'L3', 'right_id': 'R3', 'label': 0},  # TN correct
        {'left_id': 'L4', 'right_id': 'R4', 'label': 0},  # FP
    ]
    em_preds = {
        ('L1', 'R1'): 1,
        ('L2', 'R2'): 0,  # Wrong
        ('L3', 'R3'): 0,
        ('L4', 'R4'): 1,  # Wrong
    }
    am_preds = {
        ('L1', 'R1'): ['[["Title"], ["title"]]'],
        ('L2', 'R2'): [],
        ('L3', 'R3'): [],
        ('L4', 'R4'): ['[["Title"], ["title"]]'],
    }
    am_gt = {
        ('L1', 'R1'): ['[["Title"], ["title"]]'],
        ('L2', 'R2'): ['[["Title"], ["title"]]'],
        ('L3', 'R3'): [],
        ('L4', 'R4'): [],
    }

    result = evaluate_joint_exact_match(gt_pairs, em_preds, am_preds, am_gt, logger)
    # Only L1 and L3 should be correct
    assert result['metrics']['joint_exact_match_correct'] == 2
    assert result['metrics']['joint_exact_match_total'] == 4
    assert result['metrics']['joint_exact_match_accuracy'] == 0.5
    print("  [PASS] Case 12")

    print("\n  All evaluate_joint_exact_match tests passed!")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Joint Exact-Match Accuracy Tests")
    print("=" * 60)

    test_normalize_correspondence_strict()
    test_normalize_am_set()
    test_joint_exact_match()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == '__main__':
    main()
