"""Shared fixed-protocol components; importing this module never downloads weights."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

from .audit_dataset import image_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    atomic_write(path, lambda stream: stream.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')))


def atomic_write(path, writer):
    """Same-directory replacement; old checkpoint survives failed serialization."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_save(path, state):
    atomic_write(path, lambda stream: torch.save(state, stream))


def seed_all(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def environment():
    try:
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=10)
        commit = git.stdout.strip() if git.returncode == 0 else None
        git_status = 'available' if commit else 'uninitialized_or_unavailable'
    except (OSError, subprocess.TimeoutExpired):
        commit, git_status = None, 'missing_or_unavailable'
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'packages': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
            'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
            'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
            'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            'git_commit': commit, 'git_status': git_status,
            'source_hashes': {p.relative_to(PROJECT_ROOT).as_posix(): digest(p)
                              for p in sorted((PROJECT_ROOT / 'src').glob('*.py'))}}


def transform():
    return transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR,
                                                  antialias=True), transforms.ToTensor()])


class RadarDataset(Dataset):
    def __init__(self, root, partition, expected_rows=None):
        self.root = Path(root)
        self.csv = self.root / f'{partition}_binary.csv'
        self.frame = pd.read_csv(self.csv, dtype=str, keep_default_na=False)
        if not {'filename', 'drone_1_else_0', 'radar_freq'}.issubset(self.frame.columns):
            raise ValueError('Required columns absent')
        if expected_rows is not None and len(self.frame) != expected_rows:
            raise ValueError('Unexpected partition row count')
        if not len(self.frame) or not self.frame.drone_1_else_0.isin(['0', '1']).all():
            raise ValueError('Empty data or invalid binary labels')
        self.paths = [image_path(self.root / 'data', p) for p in self.frame.filename]
        if len(set(self.paths)) != len(self.paths):
            raise ValueError('Duplicate image paths')
        self.tf = transform()

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.paths[index]) as im:
            tensor = self.tf(im.convert('RGB'))
        return tensor, int(row.drone_1_else_0), row.radar_freq, row.filename


def loader(dataset, batch_size, shuffle, seed):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def build_model(pretrained=False, allow_download=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    if pretrained and not allow_download:
        cached = Path(torch.hub.get_dir()) / 'checkpoints' / weights.url.rsplit('/', 1)[-1]
        if not cached.is_file():
            raise RuntimeError('Pretrained weights are not cached. Explicitly pass --allow-weight-download at training time.')
    net = models.resnet18(weights=weights if pretrained else None)
    net.fc = nn.Linear(net.fc.in_features, 2, bias=True)
    return net


def train_epoch(net, batches, optimizer, device):
    net.train()
    total = correct = 0
    loss_sum = 0.0
    for images, labels, _, _ in batches:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = net(images)
        loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        loss.backward()
        optimizer.step()
        total += len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        loss_sum += float(loss.detach()) * len(labels)
    return {'rows': total, 'loss': loss_sum / total, 'accuracy': correct / total}


def metrics(truth, predicted):
    cm = [[0, 0], [0, 0]]
    for y, p in zip(truth, predicted, strict=True):
        cm[y][p] += 1
    tn, fp = cm[0]
    fn, tp = cm[1]
    def ratio(a, b):
        return a / b if b else None
    return {'rows': len(truth), 'class_counts': {'non_uav': tn + fp, 'uav': fn + tp},
            'confusion_matrix': cm, 'matrix_order': 'rows=true, columns=predicted; labels=[0,1]',
            'accuracy': ratio(tn + tp, len(truth)), 'uav_precision': ratio(tp, tp + fp),
            'uav_recall': ratio(tp, tp + fn), 'uav_f1': ratio(2 * tp, 2 * tp + fp + fn),
            'false_positive_rate': ratio(fp, fp + tn)}


@torch.inference_mode()
def predict(net, batches, device):
    net.eval()
    rows = []
    for images, labels, frequencies, filenames in batches:
        probabilities = net(images.to(device)).softmax(1).cpu()
        for label, prob, freq, name in zip(labels.tolist(), probabilities, frequencies, filenames):
            rows.append({'filename': name, 'radar_freq': freq, 'label': label,
                         'prediction': int(prob.argmax()), 'p_uav': float(prob[1])})
    return rows


def summarize(rows):
    def subset(values):
        return metrics([r['label'] for r in values], [r['prediction'] for r in values])
    return {'overall': subset(rows), 'per_frequency': {f: subset([r for r in rows if r['radar_freq'] == f])
            for f in sorted({r['radar_freq'] for r in rows})}, 'undefined_metric_policy': 'null when denominator is zero'}


def safe_output(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output and dataset must be separate non-nested directories')
    output.mkdir(parents=True, exist_ok=False)
    return output
