"""Export every result: CSV tables, statistical tests, PNG figures and RESULTS.md."""
from pathlib import Path
from benchmark.config import PRIMARY
from benchmark.analysis import comparison
from benchmark.statistics import summarize_repeated_runs, friedman_nemenyi_tests, wilcoxon_holm_tests
from benchmark.results_markdown import write_results
from algo._constants import BENCHMARK_METRICS
from benchmark.checkpoints import atomic_csv
from figures import export_figures


def export_report(frame, output):
    """Write all derived files into output and return the number of figures."""
    output = Path(output)
    summary = summarize_repeated_runs(frame)
    ranking, _ = comparison(frame, PRIMARY)
    native_frame = frame.copy()
    native_frame['total_time_s'] = native_frame.native_reported_time_s
    native, _ = comparison(native_frame, PRIMARY)
    friedman, ranks, nemenyi = friedman_nemenyi_tests(frame, PRIMARY)
    wilcoxon = wilcoxon_holm_tests(frame, PRIMARY)
    macro = (frame.groupby(['dataset', 'method'])[[*BENCHMARK_METRICS, 'total_time_s']].mean()
             .groupby('method').mean().reset_index())
    for name, value in [('summary', summary), ('ranking', ranking), ('native_timing_sensitivity', native),
                        ('friedman', friedman), ('mean_ranks', ranks), ('nemenyi', nemenyi),
                        ('wilcoxon_holm', wilcoxon), ('macro_means', macro)]:
        atomic_csv(value, output / (name + '.csv'))
    figures = export_figures(output, summary, PRIMARY, ranking, frame)
    significance = write_results(frame, output, figures)
    atomic_csv(significance, output / 'lc_significance.csv')
    return figures
