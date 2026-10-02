"""Tests for the independent two-hour pilot, without model forward calls."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parent


class PolicyTests(unittest.TestCase):
    def policy(self):
        spec = importlib.util.spec_from_file_location('pilot_policy', ROOT / 'pilot_policy.py')
        self.assertTrue((ROOT / 'pilot_policy.py').is_file(), 'pilot policy is not implemented')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_only_approved_candidates_and_equal_training(self):
        p = self.policy()
        rows = p.candidates()
        self.assertEqual([(r['name'], r['weight_decay'], r['router_temperature']) for r in rows],
                         [('C0', .0001, 1.), ('C-Reg', .001, 1.), ('C-Temp', .0001, 1.25)])
        for row in rows:
            p.validate_candidate(row)
            self.assertEqual((row['seed'], row['epochs'], row['batch_size'], row['learning_rate']),
                             (13, 1, 64, .001))
        for key, value in [('seed', 29), ('epochs', 2), ('batch_size', 32), ('weight_decay', .2),
                           ('precision', 'float32'), ('new_field', 1)]:
            bad = dict(rows[0], **{key: value})
            with self.assertRaises(ValueError):
                p.validate_candidate(bad)

    def test_reject_test_scope_and_outside_paths(self):
        p = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(p.input_path(root, 'train', 'train/index.json'), root / 'train/index.json')
            for split, path in [('test', 'test/index.json'), ('train', '../train/index.json'),
                                ('validation', 'train/index.json'), ('train', '/tmp/index.json')]:
                with self.assertRaises(ValueError):
                    p.input_path(root, split, path)

    def test_supervisor_once_only_and_reports_real_exit(self):
        p = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = p.run_bounded([sys.executable, '-c', 'raise SystemExit(7)'], root, 3)
            self.assertEqual(report['returncode'], 7)
            self.assertFalse(report['timed_out'])
            self.assertEqual(json.loads((root / 'terminal.json').read_text())['returncode'], 7)
            with self.assertRaises(FileExistsError):
                p.run_bounded([sys.executable, '-c', 'raise SystemExit(0)'], root, 3)

    def test_deadline_stops_owned_child_and_retains_output(self):
        p = self.policy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = p.run_bounded([sys.executable, '-u', '-c',
                                   'import time; print("started", flush=True); time.sleep(5)'], root, .3)
            self.assertTrue(report['timed_out'])
            self.assertLess(report['whole_seconds'], 2.)
            self.assertIn('started', (root / 'worker.stdout').read_text())


if __name__ == '__main__':
    unittest.main(verbosity=2)
