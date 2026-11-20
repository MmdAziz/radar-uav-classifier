"""Explicit final evaluation of a frozen R-001 checkpoint; never called by training."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .experiment import (RadarDataset, build_model, digest, environment, loader,
                         predict, safe_output, seed_all, summarize, write_json)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    frozen = json.loads((args.run_dir / 'frozen_config.json').read_text())
    config = frozen['config']
    checkpoint = args.run_dir / 'final.pt'
    if frozen['status'] != 'frozen_complete' or frozen['epochs_completed'] != config['epochs']:
        raise ValueError('Training has not completed/frozen')
    if digest(checkpoint) != frozen['checkpoint_sha256']:
        raise ValueError('Checkpoint has changed since freeze')
    seed_all(config['seed'])
    dataset = RadarDataset(args.data_root, 'eval', config['expected_eval_rows'])
    if digest(dataset.csv) != config['eval_csv_sha256']:
        raise ValueError('Evaluation CSV differs from audited version')
    output = safe_output(args.data_root, args.output_dir)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if state['epoch'] != config['epochs'] or state['config'] != config:
        raise ValueError('Checkpoint/config mismatch')
    net = build_model().to(args.device)
    net.load_state_dict(state['model'])
    rows = predict(net, loader(dataset, config['batch_size'], False, config['seed']), args.device)
    pd.DataFrame(rows).to_csv(output / 'predictions.csv', index=False)
    write_json(output / 'results.json', {**summarize(rows), 'checkpoint_sha256': digest(checkpoint),
               'eval_csv_sha256': digest(dataset.csv), 'frozen_run': frozen, 'environment': environment()})


if __name__ == '__main__':
    main()
