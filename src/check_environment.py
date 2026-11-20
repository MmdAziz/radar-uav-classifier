"""Check installed training dependencies without installing or downloading anything."""
import argparse
import json
import importlib.metadata


def dependency_errors(versions, colab=False):
    errors = []
    for name, expected in (('pandas', '2.2.3'), ('Pillow', '11.1.0')):
        if versions.get(name) != expected:
            errors.append(f'{name} must be {expected}; got {versions.get(name)}')
    try:
        numpy = tuple(int(x) for x in versions['numpy'].split('.')[:3])
        if not (1, 26, 4) <= numpy < (3, 0, 0):
            errors.append('NumPy must satisfy >=1.26.4,<3')
    except (KeyError, ValueError):
        errors.append('NumPy absent or version unreadable')
    if not colab:
        for name, expected in (('torch', '2.6.0'), ('torchvision', '0.21.0')):
            if versions.get(name, '').split('+')[0] != expected:
                errors.append(f'{name} must be {expected} (CPU/CUDA suffix permitted)')
    else:
        # Paired imports and an actual forward pass below check runtime compatibility.
        for name in ('torch', 'torchvision'):
            if name not in versions:
                errors.append(f'{name} is missing')
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--colab', action='store_true', help='Retain preinstalled framework versions; smoke-check their APIs')
    p.add_argument('--require-cuda', action='store_true')
    args = p.parse_args()
    versions = {}
    for name in ('numpy', 'pandas', 'Pillow', 'torch', 'torchvision'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    errors = dependency_errors(versions, args.colab)
    if errors:
        raise SystemExit('\n'.join(errors))
    import torch
    from torchvision.models import resnet18
    device = 'cuda' if args.require_cuda else 'cpu'
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: select a GPU runtime')
    with torch.inference_mode():
        model = resnet18(weights=None).eval().to(device)
        assert model(torch.zeros(1, 3, 224, 224, device=device)).shape == (1, 1000)
    print(json.dumps({'versions': versions, 'device_checked': device,
                      'baseline_framework_versions': versions['torch'].split('+')[0] == '2.6.0' and
                      versions['torchvision'].split('+')[0] == '0.21.0',
                      'pretrained_weights_downloaded': False}, indent=2))


if __name__ == '__main__':
    main()
