"""Rebuild the main comparison from the supplied per-instance results."""
import argparse
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent / 'main_comparison'))
from build_rpd_table import compute_rpd_summary, annotate_rpd_ranks, merge_rpd_with_times, render_latex

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('topic', nargs='?', choices=['main_comparison'], default='main_comparison')
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    evidence = ROOT / 'docs/experiments'
    if out == evidence or evidence in out.parents:
        parser.error('Choose an output directory outside docs/experiments.')
    if out.exists() and any(out.iterdir()):
        parser.error('Output directory must be new or empty.')
    data = evidence / 'main_comparison/data'
    candidates = pd.read_csv(data/'normalized_candidates.csv', keep_default_na=False)
    bks = pd.read_csv(data/'bks_per_instance.csv')
    times = pd.read_csv(data/'timing_summary.csv', keep_default_na=False)
    raw_times = pd.read_csv(data/'drl_timing_runs.csv')
    timing_keys = ['method', 'mode', 'kappa', 'size']
    run_times = raw_times.groupby(timing_keys + ['repeat']).duration_s.agg(['mean', 'count'])
    if len(raw_times) != 48000 or len(run_times) != 480 or not run_times['count'].eq(100).all():
        raise ValueError('Expected 480 timed runs of 100 instances each.')
    cell_times = run_times['mean'].groupby(level=timing_keys).median()
    repeat_counts = run_times['mean'].groupby(level=timing_keys).size()
    if len(cell_times) != 96 or not repeat_counts.eq(5).all():
        raise ValueError('Expected five repeats for each of 96 DRL settings.')
    for index, row in times.iterrows():
        if row['category'] in ('Greedy', 'Sample-1280'):
            key = tuple(row[column] for column in timing_keys)
            times.loc[index, 'mean_time_s'] = cell_times.loc[key]
    per_instance, summary = compute_rpd_summary(candidates, bks)
    summary = annotate_rpd_ranks(merge_rpd_with_times(summary, times))
    out.mkdir(parents=True, exist_ok=True)
    per_instance.to_csv(out/'per_instance_method_rpd.csv', index=False)
    columns = ['category', 'method', 'mode', 'kappa', 'size', 'rpd_n',
               'mean_rpd', 'sd_rpd', 'time_s', 'rpd_rank']
    summary.rename(columns={'mean_time_s': 'time_s'})[columns].to_csv(
        out/'main_comparison_results.csv', index=False)
    (out/'result_comparison_rpd.tex').write_text(render_latex(summary), encoding='utf-8')
    print(f'Rebuilt {len(per_instance)} instance-method results in {len(summary)} cells.')

if __name__ == '__main__':
    main()
