"""Derived plotting values; no model fitting, file access or plotting imports."""
import numpy as np
from .config import DATASETS
from algo._constants import BENCHMARK_METRICS

FOCUS_METHOD = 'LC-FixedHalfT'
BASELINE_METHOD = 'FCM'
HEATMAP_NAMES = tuple('heatmap_'+name for name in ('PC','PE','Sil','Pur','NMI','ARI'))
FIGURE_NAMES = HEATMAP_NAMES + ('PC','PE','Sil','Pur','NMI','ARI',
    'iteration','time','convergence_and_runtime',
    'win_rank_summary','effect_size_lc_fixed_half_t_vs_fcm',
    'iteration_reduction_lc_fixed_half_t_vs_fcm','per_iteration_cost_profile',
    'mcfcm_fuzzifier_statistics','lc_fixed_half_t_pipeline','temperature_controls')


def relative_percent(values, reference, lower_is_better=False):
    """Direction-adjusted relative change; a zero reference stays undefined."""
    values, reference = np.broadcast_arrays(np.asarray(values,float), np.asarray(reference,float))
    result = np.full(values.shape, np.nan)
    valid = np.isfinite(values) & np.isfinite(reference) & (np.abs(reference)>np.finfo(float).eps)
    sign = -1. if lower_is_better else 1.
    np.divide(sign*100.*(values-reference),np.abs(reference),out=result,where=valid)
    return result


def relative_metric_changes(summary, focus=FOCUS_METHOD, baseline=BASELINE_METHOD):
    import pandas as pd
    indexed = summary.set_index(['method','dataset'])
    records = []
    for metric in BENCHMARK_METRICS:
        values = indexed.loc[focus, metric+'_mean'].reindex(DATASETS).to_numpy(float)
        reference = indexed.loc[baseline, metric+'_mean'].reindex(DATASETS).to_numpy(float)
        changes = relative_percent(values,reference,lower_is_better=metric=='pe')
        finite = changes[np.isfinite(changes)]
        records.append({'metric':metric,'percent':float(finite.mean()) if len(finite) else np.nan,
                        'defined_datasets':len(finite)})
    return pd.DataFrame(records).set_index('metric')


def iteration_reductions(summary, focus=FOCUS_METHOD, baseline=BASELINE_METHOD):
    indexed = summary.set_index(['method','dataset'])
    reference = indexed.loc[baseline,'iterations_mean'].reindex(DATASETS)
    values = indexed.loc[focus,'iterations_mean'].reindex(DATASETS)
    return relative_percent(values,reference,lower_is_better=True)


def amortized_iteration_costs(frame, methods):
    """Mean of each converged fit's whole-fit milliseconds / outer cycles.

    Preprocessing and PCM/PFCM's preliminary FCM initialization remain included;
    this is not a direct timer of an individual optimization iteration.
    """
    eligible = frame[frame.method.isin(methods) & frame.method.ne('DPC')
                     & frame.converged & frame.iterations.gt(0)].copy()
    eligible['cost_ms'] = 1000.*eligible.total_time_s/eligible.iterations
    values = eligible.groupby(['dataset','method']).cost_ms.mean().unstack('method')
    return values.reindex(index=DATASETS,columns=[m for m in methods if m!='DPC'])


def fuzzifier_statistics(frame):
    columns = ['m_i_min','m_i_mean','m_i_max']
    return frame[frame.method.eq('MC-FCM')].groupby('dataset')[columns].mean().reindex(DATASETS)
