"""Outer workflow: durable fitting, prediction storage, reporting and audit."""
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
import hashlib
import json
import os
import platform
import subprocess
import sys
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from benchmark.registry import make_model
from benchmark.config import PRIMARY, DATASETS, PROTOCOL_VERSION
from metrics import evaluate_memberships, purity_score, normalized_mutual_info, adjusted_rand_index, silhouette_score_np
from algo._numerics import pairwise_distances
from benchmark.checkpoints import FitCheckpoint, WriterLock, atomic_json, atomic_csv
from data_loader import load_uci_datasets
from benchmark.verification import verify_experiment
from benchmark.reporting import export_report


def source_identity(root, seeds):
    files = sorted((root/'src/algo').glob('*.py')) + [
        root/'src/data_loader.py', root/'src/metrics.py',
        *[root/'src/benchmark'/name for name in
          ['config.py', 'registry.py', 'analysis.py', 'statistics.py', 'checkpoints.py']],
        Path(__file__)]
    hashes = {p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in files}
    data_hashes = json.loads((root/'data/checksums.json').read_text())
    for name,digest in data_hashes.items():
        if hashlib.sha256((root/'data/uci'/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Dataset checksum mismatch: '+name)
    return {'protocol':PROTOCOL_VERSION,'methods':PRIMARY,'datasets':DATASETS,'seeds':seeds,
            'source_hashes':hashes,'dataset_hashes':data_hashes,'max_iter':5000,'tol':1e-5,'blas_threads':1,
            'versions':{n:version(n) for n in ['numpy','scipy','pandas','scikit-learn','matplotlib','threadpoolctl']},
            'selection_is_independent_confirmation':False}


def state(output, status, completed, expected, **extra):
    atomic_json(output/'workflow_state.json',{'status':status,'pid':os.getpid(),
                'completed_fits':completed,'expected_fits':expected,
                'updated_utc':datetime.now(timezone.utc).isoformat(),
                'resume':'python reproduce.py --output "'+str(output)+'" --background',**extra})


def fit_one(data, method, seed, output):
    model = make_model(method,data['n_clusters'],seed)
    started = perf_counter()
    result = model.fit(data['X'])
    wall = perf_counter()-started
    labels = result.labels if result.labels is not None else result.U.argmax(1)
    if result.membership_kind == 'fuzzy':
        scores = evaluate_memberships(data['X'],data['y'],result.U,data['distances'])
    else:
        scores = {'pc':np.nan,'pe':np.nan,'sil':silhouette_score_np(data['X'],labels,data['distances']),
                  'pur':purity_score(data['y'],labels),'nmi':normalized_mutual_info(data['y'],labels),
                  'ari':adjusted_rand_index(data['y'],labels)}
    folder = output/'Predictions'/data['name'].replace(' ','_')
    folder.mkdir(parents=True,exist_ok=True)
    path = folder/f'{method}_{seed}.npz'
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    payload = {'labels':labels,'centers':result.centers,'row_indices':np.arange(len(data['X'])),
               'objective_history':getattr(model,'objective_history_',np.array([]))}
    if result.U is not None:payload['U'] = result.U
    if result.typicalities is not None:payload['typicalities'] = result.typicalities
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**payload)
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    return {'dataset':data['name'],'samples':data['samples'],'features':data['features'],'classes':data['classes'],
            'method':method,'configuration':method,'seed':seed,'converged':result.converged,
            'iterations':result.iterations if method!='DPC' else np.nan,'objective':result.objective,
            'membership_kind':result.membership_kind,'hard_clusters':len(np.unique(labels)),
            'silhouette_defined':bool(np.isfinite(scores['sil'])),**scores,**result.extra,
            'total_time_s':wall,'native_reported_time_s':result.total_time_s,
            'timing_kind':'fresh_whole_fit_wall_time','prediction_file':path.relative_to(output).as_posix(),
            'prediction_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args, root):
    root = Path(root).resolve()
    output = Path(args.output)
    if not output.is_absolute():output = root/output
    output = output.resolve();output.mkdir(parents=True,exist_ok=True)
    expected = len(PRIMARY)*len(DATASETS)*args.n_init
    completed = 0
    with WriterLock(output/'.pipeline.lock'):
        try:
            if args.figures_only:
                saved = json.loads((output/'experiment_manifest.json').read_text())['experiment_identity']
                if source_identity(root,saved['seeds']) != saved:
                    raise RuntimeError('Numerical code/data/settings changed; use a new output directory.')
                frame = pd.read_csv(output/'all_runs.csv',float_precision='round_trip')
                expected = len(PRIMARY)*len(DATASETS)*len(saved['seeds'])
            else:
                state(output,'testing',0,expected)
                environment = dict(os.environ,PYTHONPATH=str(root/'src'))
                subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-t','.','-v'],
                               cwd=root,env=environment,check=True)
                identity = source_identity(root,list(range(42,42+args.n_init)))
                with threadpool_limits(limits=1),FitCheckpoint(output,identity) as checkpoint:
                    atomic_json(output/'experiment_manifest.json',{'experiment_identity':identity,
                        'runtime':{'python':sys.version,'platform':platform.platform(),'processor':platform.processor(),
                                   'timing':'Fresh fit wall time including required preprocessing; evaluation and I/O excluded.'}})
                    rows = checkpoint.rows()
                    keys = {(r['dataset'],r['method'],r['seed']) for r in rows}
                    completed = len(rows)
                    for data in load_uci_datasets(root/'data/uci'):
                        data['distances'] = pairwise_distances(data['X'])
                        for seed in identity['seeds']:
                            offset = (seed-42)%len(PRIMARY)
                            for method in PRIMARY[offset:]+PRIMARY[:offset]:
                                key = data['name'],method,seed
                                if key in keys:continue
                                row = fit_one(data,method,seed,output)
                                checkpoint.save(row);rows.append(row);keys.add(key)
                                completed = len(rows)
                                state(output,'fitting',completed,expected,last_fit=key)
                            print(f'{completed}/{expected}: {data["name"]}, seed {seed}',flush=True)
                        atomic_csv(pd.DataFrame(rows),output/'all_runs.csv')
                    frame = pd.DataFrame(rows)
                    atomic_csv(frame,output/'all_runs.csv')
            if len(frame)!=expected or frame.duplicated(['dataset','method','seed']).any():
                raise RuntimeError('Incomplete or duplicated fit records; resume fitting first.')
            completed = len(frame)
            state(output,'reporting',completed,expected)
            figures = export_report(frame,output)
            state(output,'verifying',completed,expected)
            checks = verify_experiment(output,root/'data/uci')
            state(output,'complete',completed,expected,figures=figures,verification=checks)
            print(f'Complete: {output}/RESULTS.md; {completed} fits; {figures} figure files; independent verification PASS.',flush=True)
        except BaseException as error:
            state(output,'failed_or_interrupted',completed,expected,error=str(error))
            raise
