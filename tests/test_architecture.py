"""Algorithm files have no dependency on data loading, plotting or workflow."""
import ast
import importlib.util
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]

class ArchitectureTests(unittest.TestCase):
    def test_algorithm_dependencies_stay_inside_the_numerical_layer(self):
        files = list((ROOT / 'src/algo').glob('*.py'))
        self.assertGreaterEqual(len(files), 9)
        forbidden = ('benchmark', 'data_loader', 'figures', 'metrics', 'dafcm')
        for path in files:
            with self.subTest(file=path.name):
                for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
                    if isinstance(node, ast.Import):
                        modules = [item.name for item in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [importlib.util.resolve_name('.'*node.level + (node.module or ''), 'algo')
                                   if node.level else node.module or '']
                    else:
                        continue
                    for module in modules:
                        self.assertNotIn(module.split('.')[0], forbidden)

    def test_old_package_and_removed_algorithm_files_are_absent(self):
        self.assertFalse((ROOT / 'src/dafcm').exists())
        self.assertFalse((ROOT / 'src/algo/dafcm.py').exists())
