"""Small artificial fixtures; these are not evidence about the real dataset."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd
from PIL import Image

from src.audit_dataset import COLUMNS, audit, image_path, main


class AuditTests(unittest.TestCase):
    def test_integrity_overlap_and_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'source'
            data = root / 'data'
            folder = '24GHz/drone/interval_0.25_wd_256/2024-03-01_09-55-30'
            near = '24GHz/drone/interval_0.25_wd_256/2024-03-01_09-55-45'
            (data / folder).mkdir(parents=True)
            (data / near).mkdir(parents=True)
            good = folder + '/0.png'
            copy = near + '/0.png'
            Image.new('RGB', (8, 9)).save(data / good)
            (data / copy).write_bytes((data / good).read_bytes())
            (data / folder / 'bad.png').write_bytes(b'not a png')
            train = [[good, '1', 'fixture', '24GHz', '0.25', '256'],
                     [good.replace('/', '\\'), '1', 'fixture', '24GHz', '0.25', '256'],
                     [folder + '/missing.png', '0', '', '24GHz', '0.25', '256'],
                     [folder + '/bad.png', '', '', '24GHz', '0.25', '256'],
                     ['../escape.png', '2', '', '24GHz', '0.25', '256']]
            evaluation = [[good, '1', 'fixture', '24GHz', '0.25', '256'],
                          [copy, '1', 'fixture', '24GHz', '0.25', '256']]
            for split, rows in [('train', train), ('eval', evaluation)]:
                pd.DataFrame(rows, columns=COLUMNS).to_csv(root / f'{split}_binary.csv', index=False)
            def snapshot():
                return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
            before = snapshot()
            result, manifest = audit(root)
            self.assertEqual(before, snapshot())
            self.assertEqual(result['images']['status_counts_unique'], {'ok': 2, 'unreadable': 1, 'missing': 1})
            self.assertEqual(len(manifest), 6)
            self.assertEqual(result['partitions']['train']['duplicate_excess_rows'], 1)
            self.assertEqual(len(result['partitions']['train']['invalid_label_rows']), 2)
            self.assertEqual(len(result['overlap']['paths']), 1)
            self.assertEqual(len(result['overlap']['byte_hashes']), 1)
            self.assertEqual(result['temporal_screen']['eval_groups_with_same_class_frequency_neighbor'], 2)
            self.assertEqual(result['images']['dimensions_modes_unique'], [{'width': 8, 'height': 9, 'mode': 'RGB', 'images': 2}])

    def test_reject_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in ('../escape.png', '/absolute.png', 'C:\\file.png', '', 'file.jpg'):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    image_path(Path(tmp), path)

    def test_missing_inputs_produce_incomplete_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            code = main(['--data-root', str(root / 'missing'), '--output-dir', str(root / 'output')])
            self.assertEqual(code, 2)
            self.assertEqual(len(list((root / 'output').glob('*/summary.json'))), 1)

    def test_output_cannot_be_inside_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                main(['--data-root', tmp, '--output-dir', str(Path(tmp) / 'output')])

    def test_schema_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'train_binary.csv').write_text('filename\na.png\n', encoding='utf-8')
            result, _ = audit(root)
            self.assertEqual(result['status'], 'incomplete')
            self.assertIn('missing required columns', result['partitions']['train']['error'])


if __name__ == '__main__':
    unittest.main()
