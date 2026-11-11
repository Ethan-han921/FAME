#!/usr/bin/env python
"""
Benchmark script to test multiple datasets with both two-stage and whole-stage modes.

Usage:
    python two_stage/run_benchmark.py
    python two_stage/run_benchmark.py --model gpt-5.6-luna
    python two_stage/run_benchmark.py --limit 10  # for quick test
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Test configurations
TESTS = [
    {
        'name': 'semi_rel_50_100',
        'dataset': 'semi_rel',
        'data_file': 'two_stage/data/Test_dataset/transoformd_dataset/semi_rel_test_50_100.json',
    },
    {
        'name': 'semi_heter_50_100',
        'dataset': 'semi_heter',
        'data_file': 'two_stage/data/Test_dataset/transoformd_dataset/semi_heter_test_50_100.json',
    },
]

MODES = ['two_stage', 'whole']


def run_test(dataset: str, data_file: str, mode: str, model: str, limit: int = None):
    """Run a single test."""
    cmd = [
        sys.executable,
        'two_stage/pipeline.py',
        '--dataset', dataset,
        '--data-file', data_file,
        '--mode', mode,
        '--model', model,
    ]

    if limit:
        cmd.extend(['--limit', str(limit)])

    print(f"\n{'=' * 70}")
    print(f"Running: {dataset} | Mode: {mode} | Model: {model}")
    print(f"Data: {data_file}")
    if limit:
        print(f"Limit: {limit}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='Run benchmark tests')
    parser.add_argument('--model', '-m', default='gpt-5.6-luna', help='Model to use')
    parser.add_argument('--limit', '-l', type=int, default=None, help='Limit pairs (for quick test)')
    parser.add_argument('--mode', choices=['two_stage', 'whole', 'both'], default='both',
                        help='Which mode(s) to run')
    args = parser.parse_args()

    modes = MODES if args.mode == 'both' else [args.mode]

    print("=" * 70)
    print("BENCHMARK TEST")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Modes: {modes}")
    print(f"Limit: {args.limit or 'None (full dataset)'}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = []

    for test in TESTS:
        for mode in modes:
            success = run_test(
                dataset=test['dataset'],
                data_file=test['data_file'],
                mode=mode,
                model=args.model,
                limit=args.limit
            )
            results.append({
                'name': test['name'],
                'mode': mode,
                'success': success
            })

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Test':<25} {'Mode':<12} {'Status':<10}")
    print("-" * 70)
    for r in results:
        status = "✓ PASS" if r['success'] else "✗ FAIL"
        print(f"{r['name']:<25} {r['mode']:<12} {status:<10}")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Return exit code
    all_success = all(r['success'] for r in results)
    sys.exit(0 if all_success else 1)


if __name__ == '__main__':
    main()
