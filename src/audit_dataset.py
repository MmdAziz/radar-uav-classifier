"""Audit published CSVs and PNGs without writing to the source dataset."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import io
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import platform
import sys
import warnings

import pandas as pd
import PIL
from PIL import Image

COLUMNS = ['filename', 'drone_1_else_0', 'drone_model', 'radar_freq', 'interval', 'wd']
LABEL = 'drone_1_else_0'
EXPECTED = {'train': {'rows': 23504, '0': 11752, '1': 11752},
            'eval': {'rows': 2146, '0': 1229, '1': 917}}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def counts(frame, columns):
    return frame.groupby(columns, dropna=False).size().reset_index(name='rows').to_dict('records')


def image_path(root, value):
    """Reject absolute paths, traversal and symlinks escaping data/."""
    value = value.replace('\\', '/')
    rel = PurePosixPath(value)
    if not value or rel.is_absolute() or PureWindowsPath(value).drive or '..' in rel.parts:
        raise ValueError('empty, absolute, or traversal path')
    target = (root / rel).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('path escapes data directory')
    if target.suffix.lower() != '.png':
        raise ValueError('reference is not a PNG path')
    return target


def timestamp(parent):
    try:
        return datetime.strptime(PurePosixPath(parent).name, '%Y-%m-%d_%H-%M-%S')
    except ValueError:
        return None


def audit(root):
    result = {'partitions': {}, 'errors': [], 'images': {}, 'overlap': {},
              'limitations': [
                  'Only the two binary CSVs and their referenced PNGs are audited; simulation and unreferenced files are excluded.',
                  'Timestamp folders are recording proxies, not verified original trial or flight IDs. Clock synchronization and recording duration are unknown.',
                  'Close timestamps (<=30 seconds, fixed screening threshold) and cross-sensor matches do not prove shared trials or leakage.',
                  'Date/frequency associations can suggest acquisition confounding; no model shortcut or generalization failure is demonstrated.',
                  'PNG byte hashes detect exact encoded duplicates, not visually similar or differently encoded identical pixels.',
                  'Ground-truth label accuracy, non-UAV subtype composition, archive provenance/checksum and the evaluation partition role are not established by these checks.',
                  'No split, preprocessing, model, training, or evaluation-based tuning is performed.']}
    frames, paths, groups = {}, {}, {}
    manifest = []
    for split in ('train', 'eval'):
        csv = root / f'{split}_binary.csv'
        info = {'status': 'not_checked'}
        result['partitions'][split] = info
        try:
            raw = csv.read_bytes()
            frame = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False, skip_blank_lines=False)
            info.update(csv_sha256=sha256(raw), columns=list(frame.columns), rows=len(frame))
            absent = sorted(set(COLUMNS) - set(frame.columns))
            if absent:
                raise ValueError(f'missing required columns: {absent}')
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            info['error'] = str(exc)
            result['errors'].append(f'{split}: {exc}')
            continue
        frame = frame.map(lambda x: x.strip() if isinstance(x, str) else '')
        info.update(status='checked', missing_values={c: int((frame[c] == '').sum()) for c in COLUMNS},
                    invalid_label_rows=(frame.index[~frame[LABEL].isin(['0', '1'])] + 2).tolist(),
                    distributions={c: counts(frame, [c]) for c in COLUMNS if c != 'filename'},
                    class_frequency=counts(frame, [LABEL, 'radar_freq']),
                    class_model=counts(frame, [LABEL, 'drone_model']))
        observed = {'rows': len(frame), **{k: int((frame[LABEL] == k).sum()) for k in ('0', '1')}}
        info['reference_counts'] = {'expected_from_project_docs': EXPECTED[split],
                                    'observed': observed, 'match': observed == EXPECTED[split]}
        paths[split], groups[split] = Counter(), {}
        dates = []
        invalid = []
        for index, row in frame.iterrows():
            value = row['filename'].replace('\\', '/')
            parent = str(PurePosixPath(value).parent)
            dt = timestamp(parent)
            dates.append(dt.date().isoformat() if dt else '<unparsed>')
            group = groups[split].setdefault(parent, {'folder': parent, 'timestamp': dt.isoformat() if dt else None,
                                                     'labels': set(), 'frequencies': set(), 'rows': 0})
            group['labels'].add(row[LABEL])
            group['frequencies'].add(row['radar_freq'])
            group['rows'] += 1
            try:
                target = image_path(root / 'data', value)
                key = os.path.normcase(str(target))
                paths[split][key] += 1
                manifest.append({'partition': split, 'csv_row': index + 2, 'filename': value,
                                 'label': row[LABEL], 'key': key})
            except (ValueError, OSError) as exc:
                invalid.append({'csv_row': index + 2, 'filename': value, 'error': str(exc)})
        frame['date'] = dates
        frames[split] = frame
        info['invalid_paths'] = invalid
        info['duplicate_paths'] = [{'path': str(Path(k).relative_to((root / 'data').resolve())), 'rows': v}
                                   for k, v in sorted(paths[split].items()) if v > 1]
        info['duplicate_excess_rows'] = sum(v - 1 for v in paths[split].values())
        info['recording_folder_count'] = len(groups[split])
        info['unparsed_timestamp_rows'] = dates.count('<unparsed>')
        info['class_date_frequency'] = counts(frame, [LABEL, 'date', 'radar_freq'])
        info['recordings'] = [{**g, 'labels': sorted(g['labels']), 'frequencies': sorted(g['frequencies'])}
                              for _, g in sorted(groups[split].items())]
    cache, hashes = {}, defaultdict(list)
    for i, key in enumerate(sorted(set().union(*(set(p) for p in paths.values())))):
        entry = {'filename': Path(key).relative_to((root / 'data').resolve()).as_posix()}
        try:
            raw = Path(key).read_bytes()
            entry['sha256'] = sha256(raw)
            hashes[entry['sha256']].append(entry['filename'])
            with warnings.catch_warnings():
                warnings.simplefilter('error', Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(raw)) as im:
                    if im.format != 'PNG':
                        raise ValueError(f'actual image format: {im.format}')
                    im.verify()
                with Image.open(io.BytesIO(raw)) as im:
                    im.load()
                    entry.update(width=im.width, height=im.height, mode=im.mode, status='ok')
        except FileNotFoundError as exc:
            entry.update(status='missing', error=str(exc))
        except Exception as exc:
            entry.update(status='unreadable', error=f'{type(exc).__name__}: {exc}')
        cache[key] = entry
        if (i + 1) % 1000 == 0:
            print(f'Checked {i + 1} unique PNG references', flush=True)
    for row in manifest:
        row.update(cache[row.pop('key')])
    result['images'] = {
        'referenced_rows_checked': len(manifest), 'unique_paths_checked': len(cache),
        'status_counts_unique': dict(Counter(e['status'] for e in cache.values())),
        'dimensions_modes_unique': [dict(width=k[0], height=k[1], mode=k[2], images=v)
            for k, v in sorted(Counter((e['width'], e['height'], e['mode']) for e in cache.values() if e['status'] == 'ok').items())],
        'problems': [e for e in cache.values() if e['status'] != 'ok'],
        'exact_byte_duplicate_groups': [v for _, v in sorted(hashes.items()) if len(v) > 1]}
    for split, info in result['partitions'].items():
        if split in frames:
            info['image_status_rows'] = dict(Counter(r['status'] for r in manifest if r['partition'] == split))
    if len(frames) == 2:
        train, evaluation = groups['train'], groups['eval']
        result['overlap'] = {
            'paths': [Path(p).relative_to((root / 'data').resolve()).as_posix() for p in sorted(paths['train'].keys() & paths['eval'].keys())],
            'parent_folders': sorted(train.keys() & evaluation.keys()),
            'timestamp_identifiers': sorted({g['timestamp'] for g in train.values() if g['timestamp']} &
                                            {g['timestamp'] for g in evaluation.values() if g['timestamp']})}
        by_hash = defaultdict(set)
        for row in manifest:
            if 'sha256' in row:
                by_hash[row['sha256']].add(row['partition'])
        result['overlap']['byte_hashes'] = sorted(h for h, splits in by_hash.items() if len(splits) == 2)
        near = []
        for eg in evaluation.values():
            if not eg['timestamp']:
                continue
            for tg in train.values():
                if not tg['timestamp']:
                    continue
                delta = abs((datetime.fromisoformat(eg['timestamp']) - datetime.fromisoformat(tg['timestamp'])).total_seconds())
                if delta <= 30:
                    near.append({'eval_folder': eg['folder'], 'train_folder': tg['folder'], 'seconds': delta,
                                 'same_label': bool(eg['labels'] & tg['labels']),
                                 'same_frequency': bool(eg['frequencies'] & tg['frequencies'])})
        result['temporal_screen'] = {'threshold_seconds': 30, 'pairs': near,
            'eval_groups_with_same_class_frequency_neighbor': len({p['eval_folder'] for p in near if p['same_label'] and p['same_frequency']}),
            'eval_groups_with_cross_frequency_neighbor': len({p['eval_folder'] for p in near if not p['same_frequency']})}
        combined = pd.concat(frames.values(), ignore_index=True)
        valid = combined[combined[LABEL].isin(['0', '1']) & (combined['date'] != '<unparsed>')]
        date_sets = {label: sorted(set(valid.loc[valid[LABEL] == label, 'date'])) for label in ('0', '1')}
        result['acquisition'] = {'class_dates': date_sets, 'rows_with_valid_label_and_date': len(valid),
            'class_dates_disjoint_on_parsed_rows': bool(date_sets['0'] and date_sets['1']) and not bool(set(date_sets['0']) & set(date_sets['1'])),
            'class_frequency': counts(combined, [LABEL, 'radar_freq']),
            'class_date': counts(combined, [LABEL, 'date'])}
    result['status'] = 'incomplete' if result['errors'] else 'completed'
    return result, manifest


def markdown(result):
    lines = ['# Read-only dataset audit', '', f"Status: **{result['status']}**. Generated {result['run']['utc']}.", '',
             '## Reproduction', '', '```powershell', result['run']['command'], '```', '',
             f"Dependencies: Python {result['run']['python']}; pandas {pd.__version__}; Pillow {PIL.__version__}.", '',
             'CSV and script SHA-256 hashes, runtime configuration, full distributions, overlap lists and temporal pairs are in summary.json.',
             'manifest.csv records each checked reference, label, image status, dimensions/mode and PNG byte hash.', '',
             '## Verified observations', '', '| Partition | Rows | Non-UAV | UAV | Source folders | Reference counts match |',
             '|---|---:|---:|---:|---:|---|']
    for split, info in result['partitions'].items():
        if info['status'] == 'checked':
            c = info['reference_counts']['observed']
            lines.append(f"| {split} | {c['rows']} | {c['0']} | {c['1']} | {info['recording_folder_count']} | {info['reference_counts']['match']} |")
    for split, info in result['partitions'].items():
        lines += ['', f'### {split}', '']
        if info['status'] != 'checked':
            lines.append(f"Not checked: {info.get('error')}")
            continue
        for name in ('missing_values', 'invalid_label_rows', 'duplicate_excess_rows', 'invalid_paths', 'unparsed_timestamp_rows', 'image_status_rows'):
            lines.append(f'- {name}: {json.dumps(info[name])}')
        lines += ['', '| Class (1=UAV) | Frequency | Rows |', '|---|---|---:|']
        for row in info['class_frequency']:
            lines.append(f"| {row[LABEL]} | {row['radar_freq']} | {row['rows']} |")
        lines += ['', 'Other metadata distributions:', '', '```json', json.dumps({c: info['distributions'][c] for c in ('drone_model', 'interval', 'wd')}, indent=2), '```']
    lines += ['', '### Images and partition overlap', '',
              f"- Unique paths checked: {result['images']['unique_paths_checked']}; referenced rows checked: {result['images']['referenced_rows_checked']}.",
              f"- Unique image statuses: {result['images']['status_counts_unique']}.",
              f"- Dimensions/modes: {result['images']['dimensions_modes_unique']}.",
              f"- Exact-byte duplicate groups: {len(result['images']['exact_byte_duplicate_groups'])}."]
    for kind, items in result['overlap'].items():
        lines.append(f'- Cross-partition {kind} overlap: {len(items)}.')
    if 'temporal_screen' in result:
        t = result['temporal_screen']
        lines += [f"- Evaluation folders with a same-class/same-frequency training neighbor within 30 seconds: {t['eval_groups_with_same_class_frequency_neighbor']}.",
                  f"- Evaluation folders with a different-frequency training neighbor within 30 seconds: {t['eval_groups_with_cross_frequency_neighbor']}.",
                  '', '### Acquisition associations', '', '```json', json.dumps(result['acquisition'], indent=2), '```', '',
                  'Disjoint class dates, if reported above, establish a metadata association only. Frequency counts show acquisition balance; neither demonstrates that a model uses a shortcut.']
    lines += ['', '## Limitations and possible risks', ''] + ['- ' + s for s in result['limitations']]
    if result['errors']:
        lines += ['', '## Checks not completed', ''] + ['- ' + s for s in result['errors']]
    lines += ['', '## Next action', '', 'Resolve missing/corrupt files or schema issues if present. Obtain original trial boundaries and cross-sensor correspondence before claiming trial independence. Clarify final-test status; plan validation using training groups only in a separately authorized task.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path, help='Directory containing the binary CSVs and data/')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/audits'))
    args = parser.parse_args(argv)
    root, output = args.data_root.resolve(), args.output_dir.resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        parser.error('output directory and dataset root must be separate, non-nested directories')
    result, manifest = audit(root)
    result['run'] = {'utc': datetime.now(timezone.utc).isoformat(), 'data_root_argument': str(args.data_root),
                     'output_dir_argument': str(args.output_dir), 'python': platform.python_version(),
                     'platform': platform.platform(), 'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                     'script_sha256': sha256(Path(__file__).read_bytes()),
                     'command': '.\\.venv\\Scripts\\python.exe -m src.audit_dataset --data-root ' +
                     "'" + str(args.data_root).replace("'", "''") + "' --output-dir '" + str(args.output_dir).replace("'", "''") + "'"}
    output.mkdir(parents=True, exist_ok=True)
    run_dir = output / datetime.now(timezone.utc).strftime('audit_%Y%m%dT%H%M%S_%fZ')
    run_dir.mkdir(exist_ok=False)
    pd.DataFrame(manifest).to_csv(run_dir / 'manifest.csv', index=False)
    (run_dir / 'summary.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    (run_dir / 'audit_report.md').write_text(markdown(result), encoding='utf-8')
    print(f"Audit {result['status']}: {run_dir}")
    return 2 if result['status'] == 'incomplete' else 0


if __name__ == '__main__':
    sys.exit(main())
