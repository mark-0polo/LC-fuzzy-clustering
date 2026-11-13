"""Environment isolation, activated-venv setup and failed-install recovery."""
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import reproduce


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / 'requirements.txt').write_text('numpy==2.2.6\n', encoding='utf-8')
        self.environment = self.root / '.venv'
        self.executable = self.environment / (
            'Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')

    def existing_environment(self):
        self.executable.parent.mkdir(parents=True)
        self.executable.touch()

    def test_new_environment_gets_pip_and_requirements_using_only_venv_python(self):
        def create_environment(path):
            self.assertEqual(path, self.environment)
            self.existing_environment()

        with patch.object(reproduce.venv, 'EnvBuilder') as builder, \
                patch.object(reproduce.subprocess, 'run') as command:
            builder.return_value.create.side_effect = create_environment
            command.side_effect = [subprocess.CompletedProcess([], 1),
                                   subprocess.CompletedProcess([], 0),
                                   subprocess.CompletedProcess([], 0)]
            self.assertEqual(reproduce.prepare_environment(self.root), self.executable)
        builder.assert_called_once_with(with_pip=False)
        commands = [item.args[0] for item in command.call_args_list]
        self.assertTrue(all(item[0] == str(self.executable) for item in commands))
        self.assertEqual(commands[1][1:], ['-m', 'ensurepip', '--upgrade'])
        self.assertEqual(commands[2][-2:], ['-r', str(self.root / 'requirements.txt')])
        expected = hashlib.sha256((self.root / 'requirements.txt').read_bytes()).hexdigest()
        self.assertEqual((self.environment / '.requirements.sha256').read_text().strip(), expected)

    def test_activated_empty_environment_installs_before_running_workflow(self):
        self.existing_environment()
        with patch.object(reproduce, 'ROOT', self.root), \
                patch.object(reproduce.sys, 'prefix', str(self.environment)), \
                patch.object(reproduce.sys, 'argv', ['reproduce.py']), \
                patch.object(reproduce.sys, 'path', sys.path.copy()), \
                patch.dict(reproduce.os.environ, {}, clear=True), \
                patch.object(reproduce.venv, 'EnvBuilder') as builder, \
                patch.object(reproduce.subprocess, 'run') as command, \
                patch('benchmark.runner.run') as workflow:
            command.return_value = subprocess.CompletedProcess([], 0)
            workflow.side_effect = lambda *_: self.assertTrue(
                (self.environment / '.requirements.sha256').exists())
            reproduce.main()
        builder.assert_not_called()
        self.assertEqual(command.call_count, 2)
        self.assertEqual(command.call_args.args[0][-2:], ['-r', str(self.root / 'requirements.txt')])
        workflow.assert_called_once()

    def test_matching_requirements_skip_reinstall_but_changed_requirements_install(self):
        self.existing_environment()
        with patch.object(reproduce.subprocess, 'run') as command:
            command.return_value = subprocess.CompletedProcess([], 0)
            reproduce.prepare_environment(self.root)
            command.reset_mock()
            reproduce.prepare_environment(self.root)
            self.assertEqual(command.call_count, 1)
            (self.root / 'requirements.txt').write_text('numpy==2.2.6\npandas==2.3.1\n', encoding='utf-8')
            command.reset_mock()
            reproduce.prepare_environment(self.root)
            self.assertEqual(command.call_count, 2)

    def test_failed_install_does_not_mark_environment_ready(self):
        self.existing_environment()
        with patch.object(reproduce.subprocess, 'run') as command:
            command.side_effect = [subprocess.CompletedProcess([], 0),
                                   subprocess.CalledProcessError(1, ['pip', 'install'])]
            with self.assertRaises(subprocess.CalledProcessError):
                reproduce.prepare_environment(self.root)
        self.assertFalse((self.environment / '.requirements.sha256').exists())

    def test_skip_bootstrap_uses_current_environment_without_installing(self):
        with patch.object(reproduce, 'ROOT', self.root), \
                patch.object(reproduce.sys, 'argv', ['reproduce.py', '--skip-bootstrap']), \
                patch.object(reproduce.sys, 'path', sys.path.copy()), \
                patch.dict(reproduce.os.environ, {}, clear=True), \
                patch.object(reproduce, 'prepare_environment') as prepare, \
                patch('benchmark.runner.run') as workflow:
            reproduce.main()
        prepare.assert_not_called()
        workflow.assert_called_once()


if __name__ == '__main__':
    unittest.main()
