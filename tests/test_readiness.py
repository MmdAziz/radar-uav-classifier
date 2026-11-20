import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from PIL import Image
import torch

from src import train
from src.check_environment import dependency_errors
from src.experiment import PROJECT_ROOT, RadarDataset, atomic_save, build_model, digest, environment, train_epoch
from src.recovery import dataset_identity, load_recovery


class ReadinessTests(unittest.TestCase):
    def test_training_image_identity_and_legacy_rejection(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'outputs') as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            image = root / 'data/0.png'
            Image.new('RGB', (8, 8)).save(image)
            pd.DataFrame([{'filename': '0.png', 'drone_1_else_0': 0, 'radar_freq': '24GHz'}]).to_csv(root / 'train_binary.csv', index=False)
            ds = RadarDataset(root, 'train')
            before = dataset_identity(ds)
            Image.new('RGB', (8, 8), 'red').save(image)
            after = dataset_identity(ds)
            self.assertEqual(before['csv_sha256'], after['csv_sha256'])
            self.assertNotEqual(before['images_sha256'], after['images_sha256'])
            atomic_save(root / 'legacy.pt', {'epoch': 1, 'model': {}})
            with self.assertRaisesRegex(ValueError, 'Legacy checkpoint'):
                load_recovery(root / 'legacy.pt', None, None, None, {}, {}, {})

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_nested_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_logging_without_git_and_outside_project(self):
        old = Path.cwd()
        try:
            with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'outputs') as tmp:
                os.chdir(tmp)
                for result in (FileNotFoundError(), subprocess.CompletedProcess([], 128, '', 'not a repo')):
                    with patch('src.experiment.subprocess.run', side_effect=result if isinstance(result, Exception) else None,
                               return_value=result):
                        info = environment()
                    self.assertIsNone(info['git_commit'])
                    self.assertEqual(info['source_hashes']['src/train.py'], digest(PROJECT_ROOT / 'src/train.py'))
                os.chdir(old)
        finally:
            os.chdir(old)

    def test_dependency_profiles(self):
        versions = {'torch': '2.6.0+cpu', 'torchvision': '0.21.0+cpu', 'numpy': '2.2.6',
                    'pandas': '2.2.3', 'Pillow': '11.1.0'}
        self.assertEqual(dependency_errors(versions), [])
        self.assertTrue(dependency_errors({**versions, 'numpy': '3.0.0'}))
        other = {**versions, 'torch': '2.9.0+cu126', 'torchvision': '0.24.0+cu126'}
        self.assertEqual(dependency_errors(other, colab=True), [])
        self.assertTrue(dependency_errors(other))
        self.assertIn('torch==2.6.0', (PROJECT_ROOT / 'requirements-train.txt').read_text())
        runtime = (PROJECT_ROOT / 'requirements-runtime.txt').read_text()
        self.assertIn('numpy>=1.26.4,<3', runtime)
        self.assertNotIn('torch==', runtime)

    def test_atomic_save_survives_serialization_failure(self):
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'outputs') as tmp:
            path = Path(tmp) / 'last.pt'
            atomic_save(path, {'old': True})
            before = path.read_bytes()
            with patch('src.experiment.torch.save', side_effect=RuntimeError('interruption')):
                with self.assertRaises(RuntimeError):
                    atomic_save(path, {'new': True})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(list(Path(tmp).glob('*.tmp')), [])

    def test_interrupted_resumed_training_equals_uninterrupted(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT / 'outputs') as tmp:
            base = Path(tmp)
            root = base / 'dataset'
            (root / 'data').mkdir(parents=True)
            rows = []
            for i in range(4):
                Image.new('RGB', (32, 32), (i * 60, 20, 100)).save(root / 'data' / f'{i}.png')
                rows.append({'filename': f'{i}.png', 'drone_1_else_0': i % 2, 'radar_freq': '24GHz'})
            pd.DataFrame(rows).to_csv(root / 'train_binary.csv', index=False)
            (base / 'configs').mkdir()
            config = {'epochs': 2, 'batch_size': 2, 'seed': 3407, 'lr': 1e-5,
                      'expected_train_rows': 4, 'train_csv_sha256': digest(root / 'train_binary.csv')}
            (base / 'configs/r001.json').write_text(json.dumps(config))
            before = {str(p): digest(p) for p in root.rglob('*') if p.is_file()}
            def fixture_epoch(*args):
                # Exercise all CPU random states in addition to the shuffled DataLoader.
                random.random(); np.random.random(); torch.rand(1)
                return train_epoch(*args)
            def invoke(out, resume=False, interrupt=False):
                argv = ['train', '--data-root', str(root), '--output-dir', str(out), '--device', 'cpu']
                if resume:
                    argv += ['--resume', str(out / 'last.pt')]
                calls = 0
                def epoch(*args):
                    nonlocal calls
                    calls += 1
                    if interrupt and calls == 2:
                        raise RuntimeError('simulated interruption')
                    return fixture_epoch(*args)
                with patch.object(sys, 'argv', argv), patch.object(train, 'PROJECT_ROOT', base), \
                     patch.object(train, 'build_model', side_effect=lambda **kwargs: build_model(False)), \
                     patch.object(train, 'digest', side_effect=lambda path: 'fixture-initialization' if Path(path).name == 'resnet18-f37072fd.pth' else digest(path)), \
                     patch.object(train, 'train_epoch', side_effect=epoch), \
                     patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('download attempted')):
                    train.main()
            full, resumed = base / 'full', base / 'resumed'
            invoke(full)
            with self.assertRaisesRegex(RuntimeError, 'simulated interruption'):
                invoke(resumed, interrupt=True)
            # Simulate a truncated/ahead log; checkpoint history must repair it.
            (resumed / 'training.jsonl').write_text('partial garbage')
            invoke(resumed, resume=True)
            a = torch.load(full / 'last.pt', weights_only=True)
            b = torch.load(resumed / 'last.pt', weights_only=True)
            for key in ('model', 'optimizer', 'rng', 'epoch'):
                self.assert_nested_equal(a[key], b[key])
            for record_a, record_b in zip(a['history'], b['history']):
                for key in ('epoch', 'rows', 'loss', 'accuracy'):
                    self.assertEqual(record_a[key], record_b[key])
            self.assertEqual(len((resumed / 'training.jsonl').read_text().splitlines()), 2)
            self.assertEqual(before, {str(p): digest(p) for p in root.rglob('*') if p.is_file()})
            self.assertFalse((root / 'eval_binary.csv').exists())
            for key in ('config', 'dataset_identity', 'signature'):
                changed = copy.deepcopy(a[key]); changed['changed'] = True
                inputs = {k: a[k] for k in ('config', 'dataset_identity', 'signature')}
                inputs[key] = changed
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    load_recovery(full / 'last.pt', None, None, None, inputs['config'], inputs['dataset_identity'], inputs['signature'])
            # Completion cannot silently become another training run.
            with self.assertRaisesRegex(ValueError, 'already completed'):
                invoke(resumed, resume=True)
            Path('outputs/r001_resume_smoke.json').write_text(json.dumps({
                'synthetic_only': True, 'epochs': 2, 'rows': 4,
                'model_optimizer_rng_exact_match': True, 'training_history_exact_except_timing': True,
                'evaluation_csv_present': False}, indent=2))
