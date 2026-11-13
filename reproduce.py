"""One command: install the pinned environment, test, benchmark, report and plot."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import venv
import json
import os

ROOT = Path(__file__).resolve().parent


def prepare_environment(root):
    """Install requirements into the project venv, including an activated empty one."""
    environment = root / '.venv'
    executable = environment / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    if not executable.exists():
        print('Creating the project Python environment...', flush=True)
        venv.EnvBuilder(with_pip=False).create(environment)
    pip_ready = subprocess.run([str(executable), '-m', 'pip', '--version'],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not pip_ready:
        subprocess.run([str(executable), '-m', 'ensurepip', '--upgrade'], cwd=root, check=True)
    requirements = root / 'requirements.txt'
    fingerprint = hashlib.sha256(requirements.read_bytes()).hexdigest()
    stamp = environment / '.requirements.sha256'
    if not stamp.exists() or stamp.read_text().strip() != fingerprint:
        subprocess.run([str(executable), '-m', 'pip', 'install', '--disable-pip-version-check',
                        '--no-cache-dir', '-r', str(requirements)], cwd=root, check=True)
        stamp.write_text(fingerprint + '\n', encoding='utf-8')
    return executable


def main():
    # Windows detached Python may start without usable inherited console streams.
    # Open the log inside the child as well, before launching any workflow steps.
    background_log = os.environ.get('CLUSTERING_BACKGROUND_LOG')
    if background_log:
        log_stream = Path(background_log).open('a', encoding='utf-8', buffering=1)
        sys.stdout = sys.stderr = log_stream
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=None, help='Output directory, relative to the repository.')
    parser.add_argument('--n-init', type=int, default=30, help='Seeds per dataset/method (30 for the full study).')
    parser.add_argument('--figures-only', action='store_true', help='Redraw existing results without fitting models.')
    parser.add_argument('--skip-bootstrap', action='store_true', help='Use the current, already installed Python environment.')
    parser.add_argument('--background', action='store_true', help='Run independently of this terminal/session; progress and log stay in the output folder.')
    args = parser.parse_args()
    if args.output is None:
        args.output = 'results/benchmark'
    if args.n_init < 1:
        parser.error('n-init must be positive.')
    if not (3, 11) <= sys.version_info[:2] < (3, 14):
        parser.error('Use Python 3.11, 3.12 or 3.13 with these pinned dependencies.')

    if not args.skip_bootstrap:
        executable = prepare_environment(ROOT)
        if Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
            subprocess.run([str(executable), str(Path(__file__).resolve()), *sys.argv[1:], '--skip-bootstrap'],
                           cwd=ROOT, check=True)
            return

    if args.background:
        output = (ROOT / args.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(ROOT / 'src'))
        from benchmark.checkpoints import WriterLock
        try:
            with WriterLock(output / '.pipeline.lock'):
                pass
        except RuntimeError:
            print(f'A workflow already owns {output}; read workflow_state.json and run.log.')
            return
        environment_variables = os.environ.copy()
        environment_variables['PYTHONUTF8'] = '1'
        environment_variables['CLUSTERING_BACKGROUND_LOG'] = str(output / 'run.log')
        arguments = [arg for arg in sys.argv[1:] if arg != '--background']
        options = {'creationflags': subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == 'win32' else {'start_new_session': True}
        with (output / 'run.log').open('ab') as log:
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'reproduce.py'), *arguments, '--skip-bootstrap'],
                                       cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       env=environment_variables, **options)
        (output / 'background_process.json').write_text(json.dumps({'pid': process.pid, 'log': 'run.log'}, indent=2) + '\n', encoding='utf-8')
        print(f'Background workflow started: PID {process.pid}; log={output / "run.log"}', flush=True)
        return

    sys.path.insert(0, str(ROOT / 'src'))
    from benchmark.runner import run
    run(args, ROOT)


if __name__ == '__main__':
    try:
        main()
    except subprocess.CalledProcessError as error:
        print(f'Step failed (exit {error.returncode}): {error.cmd}', file=sys.stderr)
        raise SystemExit(error.returncode) from None
