"""Complete-cell rankings and first-place credits.

Wins and the ranking score use only the label-agreement metrics (ARI, NMI, purity), which are
defined for every method on every dataset. Per-metric credits for PC, PE and Silhouette are still
reported, as descriptive columns only: PC/PE reward a lower temperature by construction, and
Silhouette uses no labels.
"""
import numpy as np
import pandas as pd
from .config import DATASETS, LABEL_METRICS
from algo._constants import BENCHMARK_METRICS
METRICS=list(BENCHMARK_METRICS)
FIELDS=METRICS+['total_time_s']

def complete_means(frame):
    grouped=frame.groupby(['dataset','method'])
    means=grouped[FIELDS].mean()
    sizes=grouped.size()
    for field in FIELDS:
        means.loc[grouped[field].count()!=sizes,field]=np.nan
    return means


def comparison(frame,methods):
    subset=frame[frame.method.isin(methods)]
    means=complete_means(subset).round(10)
    normalized,raw_ranks,wins,eligible={},{},{},{}
    for field in FIELDS:
        v=means[field].unstack('method').reindex(index=DATASETS,columns=methods)
        ascending=field in ('pe','total_time_s')
        rank=v.rank(axis=1,ascending=ascending,method='average')
        normalized[field]=(rank-1).div(v.count(axis=1)-1,axis=0).mean()
        raw_ranks[field]=rank.mean()
        extreme=v.min(axis=1) if ascending else v.max(axis=1)
        wins[field]=v.eq(extreme,axis=0).sum()
        eligible[field]=v.count()
    records=[]
    for method in methods:
        complete=all(eligible[f][method]==len(DATASETS) for f in LABEL_METRICS)
        row={'method':method,
             'overall_score':float(np.mean([normalized[f][method] for f in LABEL_METRICS])) if complete else np.nan,
             'external_normalized_rank':float(np.mean([normalized[f][method] for f in LABEL_METRICS])),
             'external_mean_rank':float(np.mean([raw_ranks[f][method] for f in LABEL_METRICS])),
             'wins':int(sum(wins[f][method] for f in LABEL_METRICS)),
             'eligible':int(sum(eligible[f][method] for f in LABEL_METRICS)),
             'total_time_s_mean':float(subset.loc[subset.method==method,'total_time_s'].mean()),
             'native_reported_time_s_mean':float(subset.loc[subset.method==method,'native_reported_time_s'].mean())
                                          if 'native_reported_time_s' in subset else np.nan}
        for f in METRICS:row[f+'_wins']=int(wins[f][method]) if eligible[f][method] else np.nan
        records.append(row)
    return pd.DataFrame(records),means
