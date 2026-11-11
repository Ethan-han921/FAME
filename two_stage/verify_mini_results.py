"""
Verify and save gpt-5.4-mini experiment results (3 runs average)
"""
import json
from pathlib import Path

LOG_DIR = Path(__file__).parent / 'logs'
OUTPUT_FILE = LOG_DIR / 'mini_verified_results.txt'


def load_mini_results():
    """Load all mini results from logs."""
    folders = {
        'nyc_two_stage': ('NYC — Two-stage', 'stage1_em', 'stage2_attr'),
        'nyc_whole': ('NYC — Whole-stage', 'em_metrics', 'attr_metrics'),
        'semi_rel_two_stage': ('Semi-rel — Two-stage', 'stage1_em', 'stage2_attr'),
        'semi_rel_whole': ('Semi-rel — Whole-stage', 'em_metrics', 'attr_metrics'),
        'semi_heter_two_stage': ('Semi-heter — Two-stage', 'stage1_em', 'stage2_attr'),
        'semi_heter_whole': ('Semi-heter — Whole-stage', 'em_metrics', 'attr_metrics'),
    }

    all_results = {}

    for folder, (label, em_key, am_key) in folders.items():
        folder_path = LOG_DIR / folder
        results = []

        for subdir in folder_path.iterdir():
            if 'mini' in subdir.name and subdir.is_dir():
                result_file = subdir / 'final_results.json'
                if result_file.exists():
                    with open(result_file) as f:
                        data = json.load(f)
                    results.append({
                        'em': data[em_key],
                        'am': data[am_key]
                    })

        all_results[label] = results

    return all_results


def calculate_average(results):
    """Calculate average metrics from multiple runs."""
    n = len(results)
    if n == 0:
        return None, None

    # EM average
    em_avg = {
        'tp': sum(r['em']['tp'] for r in results) / n,
        'fp': sum(r['em']['fp'] for r in results) / n,
        'tn': sum(r['em']['tn'] for r in results) / n,
        'fn': sum(r['em']['fn'] for r in results) / n,
    }

    # Recalculate metrics from TP/FP/TN/FN
    tp, fp, tn, fn = em_avg['tp'], em_avg['fp'], em_avg['tn'], em_avg['fn']
    em_avg['accuracy'] = (tp + tn) / (tp + fp + tn + fn)
    em_avg['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    em_avg['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    em_avg['f1'] = 2 * em_avg['precision'] * em_avg['recall'] / (em_avg['precision'] + em_avg['recall']) if (em_avg['precision'] + em_avg['recall']) > 0 else 0

    # AM average
    am_avg = {
        'tp': sum(r['am']['tp'] for r in results) / n,
        'fp': sum(r['am']['fp'] for r in results) / n,
        'fn': sum(r['am']['fn'] for r in results) / n,
        'total_pairs': sum(r['am']['total_pairs'] for r in results) / n,
        'exact_matches': sum(r['am']['exact_matches'] for r in results) / n,
    }

    # Recalculate metrics
    tp, fp, fn = am_avg['tp'], am_avg['fp'], am_avg['fn']
    am_avg['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    am_avg['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    am_avg['f1'] = 2 * am_avg['precision'] * am_avg['recall'] / (am_avg['precision'] + am_avg['recall']) if (am_avg['precision'] + am_avg['recall']) > 0 else 0
    am_avg['accuracy'] = am_avg['exact_matches'] / am_avg['total_pairs'] if am_avg['total_pairs'] > 0 else 0

    return em_avg, am_avg


def verify_metrics(em, am):
    """Verify that metrics are calculated correctly."""
    # EM verification
    tp, fp, tn, fn = em['tp'], em['fp'], em['tn'], em['fn']

    calc_accuracy = (tp + tn) / (tp + fp + tn + fn)
    calc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    calc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    calc_f1 = 2 * calc_precision * calc_recall / (calc_precision + calc_recall) if (calc_precision + calc_recall) > 0 else 0

    em_ok = (
        abs(em['accuracy'] - calc_accuracy) < 0.0001 and
        abs(em['precision'] - calc_precision) < 0.0001 and
        abs(em['recall'] - calc_recall) < 0.0001 and
        abs(em['f1'] - calc_f1) < 0.0001
    )

    # AM verification
    tp, fp, fn = am['tp'], am['fp'], am['fn']

    calc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    calc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    calc_f1 = 2 * calc_precision * calc_recall / (calc_precision + calc_recall) if (calc_precision + calc_recall) > 0 else 0
    calc_accuracy = am['exact_matches'] / am['total_pairs'] if am['total_pairs'] > 0 else 0

    am_ok = (
        abs(am['accuracy'] - calc_accuracy) < 0.0001 and
        abs(am['precision'] - calc_precision) < 0.0001 and
        abs(am['recall'] - calc_recall) < 0.0001 and
        abs(am['f1'] - calc_f1) < 0.0001
    )

    return em_ok, am_ok


def main():
    all_results = load_mini_results()

    output = []
    output.append("=" * 90)
    output.append("gpt-5.4-mini Verified Results (3 runs average)")
    output.append("=" * 90)

    # Calculate and verify all
    computed = {}
    all_verified = True

    for label, results in all_results.items():
        em_avg, am_avg = calculate_average(results)
        em_ok, am_ok = verify_metrics(em_avg, am_avg)

        if not (em_ok and am_ok):
            all_verified = False

        computed[label] = {
            'em': em_avg,
            'am': am_avg,
            'n_runs': len(results),
            'verified': em_ok and am_ok
        }

    # Entity Matching Table
    output.append("")
    output.append("Entity Matching (EM)")
    output.append("")
    output.append(f"| {'Method':<26} | {'Accuracy':>8} | {'F1':>6} | {'Recall':>6} | {'Precision':>9} |")
    output.append(f"|{'-'*28}|{'-'*10}|{'-'*8}|{'-'*8}|{'-'*11}|")

    order = [
        'NYC — Two-stage', 'NYC — Whole-stage',
        'Semi-rel — Two-stage', 'Semi-rel — Whole-stage',
        'Semi-heter — Two-stage', 'Semi-heter — Whole-stage'
    ]

    for label in order:
        em = computed[label]['em']
        output.append(f"| {label:<26} | {em['accuracy']*100:>7.2f}% | {em['f1']*100:>5.2f}% | {em['recall']*100:>5.2f}% | {em['precision']*100:>8.2f}% |")

    # Attribute Matching Table
    output.append("")
    output.append("Attribute Matching (AM)")
    output.append("")
    output.append(f"| {'Method':<26} | {'Accuracy':>8} | {'F1':>6} | {'Recall':>6} | {'Precision':>9} |")
    output.append(f"|{'-'*28}|{'-'*10}|{'-'*8}|{'-'*8}|{'-'*11}|")

    for label in order:
        am = computed[label]['am']
        output.append(f"| {label:<26} | {am['accuracy']*100:>7.2f}% | {am['f1']*100:>5.2f}% | {am['recall']*100:>5.2f}% | {am['precision']*100:>8.2f}% |")

    # Verification status
    output.append("")
    output.append("=" * 90)
    output.append(f"Verification: {'All PASSED ✓' if all_verified else 'FAILED ✗'}")
    output.append("=" * 90)

    # Detailed verification
    output.append("")
    output.append("Detailed Verification:")
    output.append("-" * 90)

    for label in order:
        c = computed[label]
        em, am = c['em'], c['am']
        output.append(f"\n{label} ({c['n_runs']} runs):")
        output.append(f"  EM: TP={em['tp']:.1f}, FP={em['fp']:.1f}, TN={em['tn']:.1f}, FN={em['fn']:.1f}")
        output.append(f"      Accuracy={(em['tp']+em['tn'])/(em['tp']+em['fp']+em['tn']+em['fn'])*100:.2f}% ✓")
        output.append(f"      Precision={em['tp']/(em['tp']+em['fp'])*100:.2f}% ✓")
        output.append(f"      Recall={em['tp']/(em['tp']+em['fn'])*100:.2f}% ✓")
        output.append(f"  AM: TP={am['tp']:.1f}, FP={am['fp']:.1f}, FN={am['fn']:.1f}")
        output.append(f"      Exact={am['exact_matches']:.1f}/{am['total_pairs']:.1f} = {am['accuracy']*100:.2f}% ✓")
        output.append(f"      F1={am['f1']*100:.2f}% ✓")

    output.append("")
    output.append("=" * 90)

    # Write to file
    content = '\n'.join(output)
    with open(OUTPUT_FILE, 'w') as f:
        f.write(content)

    print(f"Results saved to: {OUTPUT_FILE}")
    print("")
    print(content)


if __name__ == '__main__':
    main()
