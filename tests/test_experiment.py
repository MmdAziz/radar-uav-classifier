import json
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

import pandas as pd
from PIL import Image
import torch

from src.experiment import (RadarDataset, build_model, digest, loader, metrics,
                            predict, seed_all, summarize, train_epoch, transform)


class ExperimentTests(unittest.TestCase):
    def test_evaluation_requires_frozen_training(self):
        from src.evaluate import main
        with tempfile.TemporaryDirectory(dir='outputs') as tmp:
            run = Path(tmp)
            (run / 'frozen_config.json').write_text(json.dumps({'status': 'prepared', 'config': {'epochs': 20}, 'epochs_completed': 0}))
            with patch.object(sys, 'argv', ['evaluate', '--data-root', str(run / 'absent'), '--run-dir', str(run), '--output-dir', str(run / 'results')]):
                with self.assertRaisesRegex(ValueError, 'not completed/frozen'):
                    main()
            self.assertFalse((run / 'results').exists())

    def test_metrics(self):
        m = metrics([0, 0, 1, 1], [0, 1, 0, 1])
        self.assertEqual(m['confusion_matrix'], [[1, 1], [1, 1]])
        for key in ('accuracy', 'uav_precision', 'uav_recall', 'uav_f1', 'false_positive_rate'):
            self.assertEqual(m[key], 0.5)
        self.assertIsNone(metrics([0], [0])['uav_recall'])
        self.assertEqual(metrics([1], [0])['uav_f1'], 0)

    def test_transform(self):
        x = transform()(Image.new('RGB', (512, 512), (255, 0, 0)))
        self.assertEqual(tuple(x.shape), (3, 224, 224))
        self.assertEqual(float(x[0].min()), 1)
        self.assertEqual(float(x[1].max()), 0)

    def test_fixture_smoke(self):
        # No pretrained network, no real dataset, two examples and one optimizer step.
        torch.set_num_threads(2)
        seed_all(3407)
        with tempfile.TemporaryDirectory(dir='outputs') as tmp:
            root = Path(tmp)
            (root / 'data').mkdir()
            rows = []
            for i in range(2):
                Image.new('RGB', (32, 32), (i * 255, 20, 70)).save(root / 'data' / f'{i}.png')
                rows.append({'filename': f'{i}.png', 'drone_1_else_0': i, 'radar_freq': f'{24 + i}GHz'})
            pd.DataFrame(rows).to_csv(root / 'train_binary.csv', index=False)
            before = {p.name: digest(p) for p in (root / 'data').glob('*')}
            # eval_binary.csv deliberately does not exist: training must not need it.
            ds = RadarDataset(root, 'train', 2)
            with patch('torch.hub.load_state_dict_from_url', side_effect=AssertionError('download attempted')):
                net = build_model()
                self.assertEqual(net.fc.in_features, 512)
                self.assertEqual(net.fc.out_features, 2)
                original = net.fc.weight.detach().clone()
                opt = torch.optim.Adam(net.parameters(), lr=1e-5)
                log = train_epoch(net, loader(ds, 2, True, 3407), opt, 'cpu')
                self.assertEqual(log['rows'], 2)
                self.assertFalse(torch.equal(original, net.fc.weight))
                checkpoint = root / 'fixture.pt'
                torch.save({'model': net.state_dict()}, checkpoint)
                restored = build_model()
                restored.load_state_dict(torch.load(checkpoint, weights_only=True)['model'])
                predictions = predict(restored, loader(ds, 2, False, 3407), 'cpu')
                result = summarize(predictions)
                self.assertEqual(result['overall']['rows'], 2)
                self.assertEqual(len(result['per_frequency']), 2)
                for a, b in zip(net.parameters(), restored.parameters()):
                    self.assertTrue(torch.equal(a, b))
                Path('outputs/r001_smoke_result.json').write_text(json.dumps({'synthetic_only': True, 'training': log, 'metrics': result}, indent=2))
            self.assertEqual(before, {p.name: digest(p) for p in (root / 'data').glob('*')})

    def test_seed_and_labels(self):
        seed_all(3407)
        a = torch.rand(3)
        seed_all(3407)
        self.assertTrue(torch.equal(a, torch.rand(3)))
        with tempfile.TemporaryDirectory(dir='outputs') as tmp:
            pd.DataFrame([{'filename': 'x.png', 'drone_1_else_0': 2, 'radar_freq': '24GHz'}]).to_csv(Path(tmp) / 'train_binary.csv', index=False)
            with self.assertRaises(ValueError):
                RadarDataset(tmp, 'train')


if __name__ == '__main__':
    unittest.main()
