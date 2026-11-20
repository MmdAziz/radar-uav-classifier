"""Epoch-boundary recovery. Checkpoints use weights_only-compatible types."""
import random

import numpy as np
import torch

from .experiment import atomic_save, digest


def dataset_identity(dataset):
    # Only training references, in CSV order; portable across mount locations.
    import hashlib
    import json
    h = hashlib.sha256()
    for name, path in zip(dataset.frame.filename, dataset.paths, strict=True):
        h.update(json.dumps([name, digest(path)], separators=(',', ':')).encode())
        h.update(b'\n')
    return {'csv_sha256': digest(dataset.csv), 'rows': len(dataset), 'images_sha256': h.hexdigest()}


def capture_rng(batches):
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': [state[0], state[1].tolist(), state[2], state[3], state[4]],
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'loader': batches.generator.get_state()}


def restore_rng(state, batches):
    random.setstate(state['python'])
    n = state['numpy']
    np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda']:
        if len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('CUDA device count differs from recovery state')
        torch.cuda.set_rng_state_all(state['cuda'])
    batches.generator.set_state(state['loader'])


def recovery_signature(env, device):
    return {**{key: env[key] for key in ('python', 'torch', 'torchvision', 'cuda', 'cudnn', 'gpu', 'source_hashes')},
            'device': device, 'cuda_devices': torch.cuda.device_count(),
            'numeric_packages': {key: env['packages'].get(key) for key in ('numpy', 'pandas', 'Pillow')}}


def save_recovery(path, net, optimizer, batches, epoch, config, identity, signature, history):
    atomic_save(path, {'format_version': 1, 'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                      'epoch': epoch, 'config': config, 'dataset_identity': identity, 'signature': signature,
                      'rng': capture_rng(batches), 'history': history})


def load_recovery(path, net, optimizer, batches, config, identity, signature):
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state.get('format_version') != 1:
        raise ValueError('Legacy checkpoint lacks complete recovery state; cannot safely resume')
    for key, expected in (('config', config), ('dataset_identity', identity), ('signature', signature)):
        if state[key] != expected:
            raise ValueError(f'Resume {key} mismatch')
    if not 1 <= state['epoch'] <= config['epochs']:
        raise ValueError('Invalid completed epoch')
    if [row['epoch'] for row in state['history']] != list(range(1, state['epoch'] + 1)):
        raise ValueError('Recovery history does not match completed epochs')
    net.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    restore_rng(state['rng'], batches)
    return state['epoch'], state['history']
