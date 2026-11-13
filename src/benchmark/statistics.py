"""Seed aggregation and dataset-block statistical comparisons."""
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range
from algo._constants import EPS, BENCHMARK_METRICS, METRIC_DIRECTIONS, STATISTICAL_TEST_ALPHA
from .config import metric_methods

def sample_std(series):
    series = pd.Series(series, dtype=np.float64).dropna()
    if series.size <= 1:
        return np.nan
    return float(series.std(ddof=1))


def mean_std_text(mean, std, precision=4):
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.{precision}f} ± undefined"
    std = float(std)
    return f"{float(mean):.{precision}f} ± {std:.{precision}f}"


def summarize_repeated_runs(all_runs_df):
    group_cols = ["dataset", "samples", "features", "classes", "method"]
    group_cols += [col for col in ('configuration', 'run_group') if col in all_runs_df]
    value_cols = [
        "objective",
        "pc",
        "pe",
        "sil",
        "pur",
        "nmi",
        "ari",
        "iterations",
        "total_time_s",
    ]
    value_cols = [col for col in value_cols if col in all_runs_df.columns]

    aggregations = {
        "n_seeds": ("seed", "nunique"),
        "seed_min": ("seed", "min"),
        "seed_max": ("seed", "max"),
        "converged_runs": ("converged", "sum"),
    }
    for col in value_cols:
        aggregations[f"{col}_mean"] = (col, "mean")
        aggregations[f"{col}_std"] = (col, sample_std)
        aggregations[f"{col}_valid_runs"] = (col, "count")
    for col in ("numerically_collapsed", "silhouette_defined"):
        if col in all_runs_df:
            aggregations[f"{col}_runs"] = (col, "sum")

    summary = (
        all_runs_df
        .groupby(group_cols, dropna=False)
        .agg(**aggregations)
        .reset_index()
    )

    for col in value_cols:
        summary[f"{col}_mean_std"] = [
            mean_std_text(mean, std)
            for mean, std in zip(summary[f"{col}_mean"], summary[f"{col}_std"])
        ]

    formatted_cols = [f"{col}_mean_std" for col in value_cols]
    numeric_cols = []
    for col in value_cols:
        numeric_cols.extend([f"{col}_mean", f"{col}_std"])

    return summary[
        group_cols
        + ["n_seeds", "seed_min", "seed_max", "converged_runs"]
        + formatted_cols
        + numeric_cols
        + [f"{col}_valid_runs" for col in value_cols]
        + [col for col in ("numerically_collapsed_runs", "silhouette_defined_runs") if col in summary]
    ]


def metric_blocks(all_runs_df, metric, methods):
    dataset_method_means = (
        all_runs_df
        .groupby(["dataset", "method"], dropna=False)[metric]
        .mean()
        .reset_index()
    )
    blocks = dataset_method_means.pivot_table(
        index="dataset",
        columns="method",
        values=metric,
        aggfunc="mean",
    )
    complete = all_runs_df.groupby(["dataset", "method"])[metric].agg(["count", "size"])
    for (dataset, method), row in complete.iterrows():
        if row['count'] != row['size'] and method in blocks.columns:
            blocks.loc[dataset, method] = np.nan
    blocks = blocks.reindex(columns=list(methods)).round(10)
    # Drop a method that is undefined on some dataset (e.g. PCM Silhouette), never the datasets:
    # dropping rows would shrink every method's test to a few blocks.
    return blocks.loc[:, blocks.notna().all()].dropna()


def ranked_metric_blocks(all_runs_df, metric, methods):
    higher_is_better = METRIC_DIRECTIONS[metric]
    blocks = metric_blocks(all_runs_df, metric, methods)
    ranks = blocks.rank(axis=1, ascending=not higher_is_better, method="average")
    return blocks, ranks


def holm_adjust(p_values):
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., p[index] * (len(p) - rank)))
        adjusted[index] = running
    return adjusted


def wilcoxon_holm_tests(all_runs_df, methods, metrics=('ari', 'nmi', 'pur', 'sil'), alpha=STATISTICAL_TEST_ALPHA):
    """Pairwise Wilcoxon signed-rank tests over dataset means, Holm-corrected within each metric."""
    from scipy.stats import wilcoxon
    rows = []
    for metric in metrics:
        blocks = metric_blocks(all_runs_df, metric, metric_methods(metric, methods))
        higher = METRIC_DIRECTIONS[metric]
        pairs = []
        for a, b in combinations(blocks.columns, 2):
            diff = (blocks[a] - blocks[b]).to_numpy(dtype=np.float64)
            if not higher:
                diff = -diff
            p = 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method='zsplit').pvalue)
            pairs.append({'metric': metric, 'method_a': a, 'method_b': b, 'n_datasets': len(diff),
                          'a_better': int((diff > 0).sum()), 'b_better': int((diff < 0).sum()),
                          'ties': int((diff == 0).sum()), 'median_difference_a_minus_b': float(np.median(diff)),
                          'p_value': p})
        for row, adjusted in zip(pairs, holm_adjust([r['p_value'] for r in pairs]) if pairs else []):
            row['p_holm'] = float(adjusted)
            row['significant_holm_0_05'] = bool(adjusted < alpha)
        rows += pairs
    return pd.DataFrame(rows)


def proposed_versus_others_tests(all_runs_df, proposed, others, datasets, metrics=('ari', 'nmi', 'pur'),
                                 alpha=STATISTICAL_TEST_ALPHA):
    """Each proposed method vs each other method: Wilcoxon signed-rank over dataset means.

    Holm correction covers all len(proposed) x len(others) tests of one metric, so choosing the
    best proposed variant afterwards does not inflate significance. Each dataset counts once.
    """
    from scipy.stats import wilcoxon
    rows = []
    for metric in metrics:
        means = all_runs_df.groupby(['dataset', 'method'])[metric].mean().unstack().loc[list(datasets)].round(10)
        tests = []
        for p in proposed:
            for o in others:
                diff = (means[p] - means[o]).to_numpy(dtype=np.float64)
                if not METRIC_DIRECTIONS[metric]:
                    diff = -diff
                tests.append({'metric': metric, 'proposed': p, 'other': o, 'n_datasets': len(diff),
                              'proposed_better': int((diff > 0).sum()), 'other_better': int((diff < 0).sum()),
                              'median_difference': float(np.median(diff)),
                              'p_value': 1.0 if np.allclose(diff, 0) else float(wilcoxon(diff, zero_method='zsplit').pvalue)})
        for row, adjusted in zip(tests, holm_adjust([t['p_value'] for t in tests])):
            row['p_holm'] = float(adjusted)
            row['significant_holm_0_05'] = bool(adjusted < alpha)
        rows += tests
    return pd.DataFrame(rows)


def friedman_nemenyi_tests(all_runs_df, methods, metrics=BENCHMARK_METRICS, alpha=STATISTICAL_TEST_ALPHA):
    friedman_rows = []
    rank_rows = []
    nemenyi_rows = []

    for metric in metrics:
        blocks, ranks = ranked_metric_blocks(all_runs_df, metric, metric_methods(metric, methods))
        n_blocks = int(blocks.shape[0])
        n_methods = int(blocks.shape[1])
        if n_blocks == 0 or n_methods < 2:
            continue

        values = blocks.to_numpy(dtype=np.float64)
        if np.all(np.nanmax(values, axis=1) - np.nanmin(values, axis=1) <= EPS):
            statistic = 0.0
            p_value = 1.0
        else:
            statistic, p_value = friedmanchisquare(
                *[blocks[method].to_numpy(dtype=np.float64) for method in blocks.columns]
            )
            statistic = float(statistic)
            p_value = float(p_value)

        friedman_rows.append({
            "metric": metric,
            "higher_is_better": METRIC_DIRECTIONS[metric],
            "block_unit": "dataset_mean_over_seeds",
            "n_blocks": n_blocks,
            "n_methods": n_methods,
            "statistic": statistic,
            "p_value": p_value,
            "significant_alpha_0_05": bool(p_value < alpha),
        })

        mean_ranks = ranks.mean(axis=0)
        for method, mean_rank in mean_ranks.items():
            rank_rows.append({
                "metric": metric,
                "method": method,
                "block_unit": "dataset_mean_over_seeds",
                "mean_rank": float(mean_rank),
                "n_blocks": n_blocks,
            })

        rank_standard_error = np.sqrt(n_methods * (n_methods + 1) / (6.0 * n_blocks))
        critical_distance = (
            studentized_range.ppf(1.0 - alpha, n_methods, np.inf)
            / np.sqrt(2.0)
            * rank_standard_error
        )

        for method_a, method_b in combinations(blocks.columns, 2):
            rank_a = float(mean_ranks[method_a])
            rank_b = float(mean_ranks[method_b])
            rank_diff = abs(rank_a - rank_b)
            q_statistic = rank_diff / rank_standard_error
            p_value = float(studentized_range.sf(q_statistic * np.sqrt(2.0), n_methods, np.inf))
            nemenyi_rows.append({
                "metric": metric,
                "method_a": method_a,
                "method_b": method_b,
                "block_unit": "dataset_mean_over_seeds",
                "mean_rank_a": rank_a,
                "mean_rank_b": rank_b,
                "rank_diff": rank_diff,
                "critical_distance_alpha_0_05": float(critical_distance),
                "q_statistic": float(q_statistic),
                "p_value": min(max(p_value, 0.0), 1.0),
                "significant_alpha_0_05": bool(rank_diff > critical_distance),
            })

    friedman_df = pd.DataFrame(friedman_rows)
    rank_df = pd.DataFrame(rank_rows)
    nemenyi_df = pd.DataFrame(nemenyi_rows)
    return friedman_df, rank_df, nemenyi_df
