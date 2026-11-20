"""Train R-001 using training data only; final epoch is predeclared."""
import argparse
import json
from pathlib import Path
import time
from datetime import datetime, timezone

import torch

from .experiment import (RadarDataset, build_model, digest, environment, loader,
                         safe_output, seed_all, train_epoch, write_json, atomic_save, atomic_write, PROJECT_ROOT)
from .recovery import dataset_identity, recovery_signature, save_recovery, load_recovery


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True, help='New run directory, or original directory with --resume')
    p.add_argument('--config', type=Path, default=PROJECT_ROOT / 'configs/r001.json')
    p.add_argument('--resume', type=Path, help='Original run last.pt; resume at next completed epoch')
    p.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    p.add_argument('--allow-weight-download', action='store_true')
    args = p.parse_args()
    config = json.loads(args.config.read_text())
    canonical = json.loads((PROJECT_ROOT / 'configs/r001.json').read_text())
    if config != canonical:
        raise ValueError('R-001 config differs from predeclared protocol; declare a separate experiment first')
    seed_all(config['seed'])
    dataset = RadarDataset(args.data_root, 'train', config['expected_train_rows'])
    if digest(dataset.csv) != config['train_csv_sha256']:
        raise ValueError('Training CSV differs from audited version')
    device = 'cuda' if args.device == 'auto' and torch.cuda.is_available() else ('cpu' if args.device == 'auto' else args.device)
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    identity = dataset_identity(dataset)
    env = environment()
    signature = recovery_signature(env, device)
    if args.resume:
        output = args.output_dir.resolve()
        root = args.data_root.resolve()
        if output.is_relative_to(root) or root.is_relative_to(output):
            raise ValueError('Output and dataset must be separate non-nested directories')
        if args.resume.resolve() != output / 'last.pt':
            raise ValueError('Resume must use last.pt in the original output directory')
        frozen = json.loads((output / 'frozen_config.json').read_text())
        if frozen['status'] == 'frozen_complete':
            raise ValueError('Run already completed and frozen')
        if frozen['config'] != config or frozen.get('dataset_identity') != identity:
            raise ValueError('Frozen configuration or dataset identity mismatch')
        net = build_model(pretrained=False).to(device)
    else:
        output = safe_output(args.data_root, args.output_dir)
        net = None
    if not args.resume:
        frozen = {'config': config, 'train_csv_sha256': digest(dataset.csv), 'data_root': str(args.data_root),
              'reference_commit': '78c899b1e36df3b0e8e19a390f34b9e209a5ccdb',
              'device': device, 'environment': env, 'dataset_identity': identity, 'status': 'prepared',
              'transform': 'PIL RGB; bilinear 224x224 antialias=True; ToTensor; no normalization',
              'optimizer': {'name': 'Adam', 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0, 'amsgrad': False},
              'scheduler': None, 'drop_last': False, 'workers': 0, 'loss': 'unweighted cross entropy',
              'checkpoint_selection': 'final_epoch', 'epochs_completed': 0}
        write_json(output / 'frozen_config.json', frozen)
        net = build_model(pretrained=True, allow_download=args.allow_weight_download).to(device)
        weight_path = Path(torch.hub.get_dir()) / 'checkpoints' / 'resnet18-f37072fd.pth'
        frozen['pretrained_file_sha256'] = digest(weight_path)
        write_json(output / 'frozen_config.json', frozen)
    optimizer = torch.optim.Adam(net.parameters(), lr=config['lr'], betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=0, amsgrad=False)
    batches = loader(dataset, config['batch_size'], True, config['seed'])
    completed, history = 0, []
    if args.resume:
        completed, history = load_recovery(args.resume, net, optimizer, batches, config, identity, signature)
        events = frozen.setdefault('resume_events', [])
        events.append({'utc': datetime.now(timezone.utc).isoformat(), 'completed_epoch': completed,
                       'checkpoint_sha256': digest(args.resume), 'environment': env})
        write_json(output / 'frozen_config.json', frozen)
    # Checkpoint history is authoritative if a crash left the text log ahead/behind.
    def save_log():
        atomic_write(output / 'training.jsonl', lambda stream: stream.write(
            ''.join(json.dumps(row) + '\n' for row in history).encode('utf-8')))
    save_log()
    for epoch in range(completed + 1, config['epochs'] + 1):
        start = time.perf_counter()
        record = {'epoch': epoch, **train_epoch(net, batches, optimizer, device),
                  'seconds': time.perf_counter() - start}
        history.append(record)
        save_recovery(output / 'last.pt', net, optimizer, batches, epoch, config, identity, signature, history)
        save_log()
        print(record, flush=True)
    checkpoint = output / 'final.pt'
    atomic_save(checkpoint, {'model': net.state_dict(), 'epoch': config['epochs'], 'config': config})
    frozen.update(status='frozen_complete', epochs_completed=config['epochs'], checkpoint_sha256=digest(checkpoint))
    write_json(output / 'frozen_config.json', frozen)


if __name__ == '__main__':
    main()
