"""Transactional per-fit checkpoints and process-released writer locks."""
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sqlite3


def portable(value):
    if isinstance(value, dict):
        return {str(k): portable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(v) for v in value]
    if hasattr(value, 'item'):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(portable(value), stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_csv(frame, path):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    frame.to_csv(temporary, index=False, lineterminator='\n')
    os.replace(temporary, path)


class WriterLock(AbstractContextManager):
    def __init__(self, path):
        self.path, self.stream = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            self.stream.close()
            self.stream = None
            raise RuntimeError('Another process owns this output: ' + str(self.path.parent)) from error
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class FitCheckpoint(AbstractContextManager):
    def __init__(self, output, identity):
        self.output = Path(output)
        self.identity = json.dumps(portable(identity), sort_keys=True, allow_nan=False)
        self.lock = WriterLock(self.output / '.fit.lock')

    def __enter__(self):
        self.lock.__enter__()
        try:
            self.connection = sqlite3.connect(self.output / 'checkpoint.sqlite3', timeout=5)
            self.connection.execute('PRAGMA journal_mode=WAL')
            self.connection.execute('PRAGMA synchronous=FULL')
            self.connection.execute('CREATE TABLE IF NOT EXISTS metadata (identity TEXT NOT NULL)')
            self.connection.execute('CREATE TABLE IF NOT EXISTS fits (dataset TEXT, configuration TEXT, seed INTEGER, '
                                    'payload TEXT NOT NULL, PRIMARY KEY(dataset,configuration,seed))')
            previous = self.connection.execute('SELECT identity FROM metadata').fetchall()
            if previous and previous != [(self.identity,)]:
                raise RuntimeError('Checkpoint code/data/settings differ. Use a new output directory.')
            if not previous:
                self.connection.execute('INSERT INTO metadata VALUES (?)', (self.identity,))
            self.connection.commit()
            return self
        except BaseException:
            if hasattr(self, 'connection'):
                self.connection.close()
            self.lock.__exit__()
            raise

    def rows(self):
        return [json.loads(row[0]) for row in self.connection.execute('SELECT payload FROM fits ORDER BY rowid')]

    def save(self, row):
        payload = json.dumps(portable(row), allow_nan=False)
        with self.connection:
            self.connection.execute('INSERT INTO fits VALUES (?,?,?,?)',
                                    (row['dataset'], row['configuration'], int(row['seed']), payload))

    def progress(self, completed, expected, last=None, status='running'):
        atomic_json(self.output / 'progress.json', {'status': status, 'pid': os.getpid(),
                    'completed_fits': completed, 'expected_fits': expected, 'last_fit': last,
                    'updated_utc': datetime.now(timezone.utc).isoformat(),
                    'resume': 'python reproduce.py --skip-bootstrap --output ' + str(self.output)})

    def __exit__(self, *args):
        self.connection.close()
        self.lock.__exit__(*args)
