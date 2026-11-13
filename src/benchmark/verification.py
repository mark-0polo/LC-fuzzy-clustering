"""Independent saved-prediction, CSV, SQLite and scientific-figure audit."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import math
import sqlite3
import statistics
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from threadpoolctl import threadpool_limits
from data_loader import load_uci_datasets
from .checkpoints import atomic_json
from .figure_data import FIGURE_NAMES
from .config import LABEL_METRICS

# The audited PNGs are this program's own 600-DPI output; wide figures exceed PIL's default
# decompression-bomb limit, which guards against untrusted images.
Image.MAX_IMAGE_PIXELS = None


def verify_plot_data(payload, frame, summary, ranking):
    """Check captured artist heights with independent scalar calculations."""
    datasets=payload['datasets']
    indexed=summary.set_index(['method','dataset'])
    checked=0
    def dataset_mean(dataset,method,field):
        rows=frame[(frame.dataset==dataset)&(frame.method==method)]
        values=[float(v) for v in rows[field] if pd.notna(v)]
        return statistics.fmean(values) if values and len(values)==len(rows) else math.nan
    for axis in payload['axes']:
        source=axis['source'];kind=source['kind']
        datasets=axis.get('datasets',payload['datasets'])   # a bar row covers one dataset group
        for series in axis['series']:
            label=series['label'];target=[]
            if kind=='summary':
                target=[indexed.loc[(label,d),source['field']]*source['scale'] for d in datasets]
            elif kind=='ranking':
                target=[ranking.loc[m,source['field']] for m in source['methods']]
            elif kind=='macro':
                target=[statistics.fmean(indexed.loc[m,source['field']].dropna()) for m in source['methods']]
            elif kind=='relative_change':
                for metric in ['pc','pe','sil','pur','nmi','ari']:
                    values=[]
                    for dataset in datasets:
                        a=dataset_mean(dataset,source['focus'],metric)
                        b=dataset_mean(dataset,source['baseline'],metric)
                        if math.isfinite(a) and math.isfinite(b) and abs(b)>np.finfo(float).eps:
                            values.append(100*((b-a) if metric=='pe' else (a-b))/abs(b))
                    target.append(statistics.fmean(values) if values else math.nan)
            elif kind=='iteration_reduction':
                for dataset in datasets:
                    a=dataset_mean(dataset,source['focus'],'iterations')
                    b=dataset_mean(dataset,source['baseline'],'iterations')
                    target.append(100*(b-a)/abs(b) if math.isfinite(a) and math.isfinite(b) and b!=0 else math.nan)
            elif kind=='iteration_cost':
                assert label!='DPC'
                for dataset in datasets:
                    rows=frame[(frame.dataset==dataset)&(frame.method==label)]
                    values=[1000.*r.total_time_s/r.iterations for r in rows.itertuples()
                            if r.converged and pd.notna(r.iterations) and r.iterations>0]
                    target.append(statistics.fmean(values) if values else math.nan)
            elif kind=='fuzzifier':
                target=[dataset_mean(d,source['method'],label) for d in datasets]
            else:
                raise AssertionError('Unknown plot source: '+kind)
            actual=[math.nan if v is None else float(v) for v in series['values']]
            np.testing.assert_allclose(actual,target,atol=1e-10,rtol=1e-10,equal_nan=True)
            checked+=len(target)
    return checked


def verify_experiment(output, data_dir):
    output = Path(output)
    manifest = json.loads((output/'experiment_manifest.json').read_text())['experiment_identity']
    methods, seeds = manifest['methods'], manifest['seeds']
    frame = pd.read_csv(output/'all_runs.csv', float_precision='round_trip')
    expected = len(methods)*len(seeds)*len(manifest['datasets'])
    assert len(frame) == expected and not frame.duplicated(['dataset','method','seed']).any()
    assert set(frame.method) == set(methods)
    assert all(set(g.seed) == set(seeds) for _,g in frame.groupby(['dataset','method']))
    summary = pd.read_csv(output/'summary.csv', float_precision='round_trip')
    metrics = ['pc','pe','sil','pur','nmi','ari']
    checked, cycles, maximum_error = 0, 0, 0.
    with threadpool_limits(limits=1):
        for data in load_uci_datasets(data_dir):
            distances = cdist(data['X'], data['X'])
            mc=frame[(frame.dataset==data['name'])&frame.method.eq('MC-FCM')]
            if len(mc):
                first=mc.iloc[0];count=max(1,len(data['X'])//data['n_clusters'])
                delta=np.sort(distances,axis=1)[:,:count].sum(axis=1)
                q=(delta-delta.min())/np.ptp(delta) if np.ptp(delta)>np.finfo(float).eps else np.full(len(delta),.5)
                powers=first.fuzzifier_lower+(first.fuzzifier_upper-first.fuzzifier_lower)*q**first.mc_power
                for field,expected_value in [('m_i_min',powers.min()),('m_i_mean',powers.mean()),('m_i_max',powers.max())]:
                    np.testing.assert_allclose(mc[field],expected_value,rtol=1e-8,atol=2e-8)
            for _,row in frame[frame.dataset == data['name']].iterrows():
                path = output/row.prediction_file
                assert hashlib.sha256(path.read_bytes()).hexdigest() == row.prediction_sha256
                with np.load(path, allow_pickle=False) as saved:
                    labels = saved['labels']
                    np.testing.assert_array_equal(saved['row_indices'], np.arange(len(data['X'])))
                    values = {'pc':np.nan, 'pe':np.nan,
                              'pur':sum(Counter(data['y'][labels==k]).most_common(1)[0][1] for k in np.unique(labels))/len(labels),
                              'ari':adjusted_rand_score(data['y'], labels),
                              'nmi':normalized_mutual_info_score(data['y'], labels, average_method='arithmetic'),
                              'sil':silhouette_score(distances, labels, metric='precomputed')
                                    if 1 < len(np.unique(labels)) < len(labels) else np.nan}
                    if row.membership_kind == 'fuzzy':
                        U = saved['U']
                        assert np.isfinite(U).all() and (U >= 0).all()
                        np.testing.assert_allclose(U.sum(1), 1., atol=1e-12, rtol=0)
                        np.testing.assert_array_equal(U.argmax(1), labels)
                        values['pc'] = float(np.sum(U*U)/len(U))
                        values['pe'] = float(-np.sum(U*np.log(np.maximum(U,np.finfo(float).tiny)))/len(U))
                    for field,value in values.items():
                        actual = float(row[field])
                        if math.isnan(value):
                            assert math.isnan(actual), (row.method, field)
                        else:
                            error = abs(value-actual)
                            maximum_error = max(maximum_error, error)
                            assert error < 1e-8, (data['name'],row.method,row.seed,field,value,actual)
                        checked += 1
                    history = saved['objective_history']
                    if len(history):
                        assert len(history) == max(1,int(row.iterations))
                        assert np.all(np.diff(history) <= 1e-9*np.maximum(1,np.abs(history[:-1])))
                        assert np.isclose(history[-1],row.objective,atol=1e-9,rtol=1e-10)
                        cycles += len(history)
            print('Independent predictions verified:', data['name'], flush=True)
    aggregation_cells = 0
    for _,row in summary.iterrows():
        group = frame[(frame.dataset==row.dataset)&(frame.method==row.method)]
        for field in metrics+['objective','iterations','total_time_s']:
            values = [float(v) for v in group[field] if pd.notna(v)]
            for suffix,value in [('mean',statistics.fmean(values) if values else math.nan),
                                 ('std',statistics.stdev(values) if len(values)>1 else math.nan)]:
                actual = row[field+'_'+suffix]
                assert (pd.isna(actual) and math.isnan(value)) or math.isclose(actual,value,rel_tol=1e-9,abs_tol=1e-10)
                aggregation_cells += 1
    ranking = pd.read_csv(output/'ranking.csv').set_index('method')
    for method in methods:
        count,eligible,label_count,label_eligible = 0,0,0,0
        for field in metrics:
            field_count,field_eligible=0,0
            for dataset in frame.dataset.unique():
                cells = {}
                for name,g in frame[frame.dataset==dataset].groupby('method'):
                    if g[field].notna().all():cells[name] = round(statistics.fmean(g[field]),10)
                if method in cells:
                    eligible += 1
                    field_eligible+=1
                    extreme = (min if field=='pe' else max)(cells.values())
                    credit=int(cells[method]==extreme)
                    count+=credit;field_count+=credit
            actual=ranking.loc[method,field+'_wins']
            assert (pd.isna(actual) and field_eligible==0) or actual==field_count
            if field in LABEL_METRICS:
                label_count+=field_count;label_eligible+=field_eligible
        assert int(ranking.loc[method,'wins']) == label_count
        assert int(ranking.loc[method,'eligible']) == label_eligible
    annotations, outlined = 0, 0
    for metric,label in [('pc','PC'),('pe','PE'),('sil','Sil'),('pur','Pur'),('nmi','NMI'),('ari','ARI')]:
        with Image.open(output/'Figures'/('heatmap_'+label+'.png')) as picture:
            actual = json.loads(picture.info['HeatmapAnnotations'])
            winners = json.loads(picture.info['HeatmapWinners'])
        expected_annotations = []
        means = summary.set_index(['method','dataset'])
        for r, method in enumerate(methods):
            for c, dataset in enumerate(manifest['datasets']):
                value = means.loc[(method,dataset),metric+'_mean']
                expected_annotations.append({'text':f'{value:.3f}' if pd.notna(value) else '--',
                                             'position':[c+.5,r+.5]})
        assert actual == expected_annotations
        annotations += len(actual)
        # The outlined cells must be exactly the best method(s) on each dataset.
        expected_winners = []
        for c, dataset in enumerate(manifest['datasets']):
            cells = {r: round(float(means.loc[(method,dataset),metric+'_mean']),10) for r, method in enumerate(methods)
                     if pd.notna(means.loc[(method,dataset),metric+'_mean'])}
            if cells:
                best = (min if metric=='pe' else max)(cells.values())
                expected_winners += [[r, c] for r in sorted(cells) if cells[r]==best]
        assert sorted(winners) == sorted(expected_winners), (metric, winners, expected_winners)
        outlined += len(winners)
    figure_manifest=json.loads((output/'Figures/manifest.json').read_text())
    sources=figure_manifest['rendering_source_sha256']
    assert set(sources)=={'figures.py','benchmark/figure_data.py','benchmark/figure_style.py'}
    source_root=Path(__file__).resolve().parents[1]
    for name,digest in sources.items():
        assert hashlib.sha256((source_root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
    files=figure_manifest['files']
    assert {Path(item['path']).stem for item in files}==set(FIGURE_NAMES)
    assert len(files)==len(FIGURE_NAMES)
    assert {p.name for p in (output/'Figures').iterdir()} == {
        Path(item['path']).name for item in files} | {'manifest.json'}
    plotted_values=0
    for item in files:
        path = output/item['path']
        assert path.suffix == '.png'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        with Image.open(path) as picture:
            assert picture.format == 'PNG' and min(picture.info['dpi']) > 599
            assert picture.width >= 4000 and picture.height >= 2500
            if not path.stem.startswith('heatmap_'):
                assert 'PlotData' in picture.info
                payload=json.loads(picture.info['PlotData'])
                plotted_values+=verify_plot_data(payload,frame,summary,ranking)
            picture.verify()
    with sqlite3.connect(output/'checkpoint.sqlite3') as db:
        assert db.execute('pragma integrity_check').fetchone()[0] == 'ok'
        assert db.execute('select count(*) from fits').fetchone()[0] == expected
    result = {'status':'PASS','fits':expected,'recomputed_metric_cells':checked,
              'maximum_metric_absolute_error':maximum_error,'checked_objective_cycles':cycles,
              'mean_and_sample_sd_cells':aggregation_cells,'verified_win_cells':len(methods)*6,
              'heatmap_annotations':annotations,'figure_files':len(files),'figure_format':'PNG only',
              'outlined_heatmap_winners':outlined,'independently_checked_plot_values':plotted_values,
              'png_dpi':600,'sqlite_integrity':'ok','capped_fits':int((~frame.converged).sum())}
    atomic_json(output/'verification.json',result)
    return result
