#!/usr/bin/env python3
"""Verify the replacement bundle with Python's standard library only."""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = 'deepseek-ai/DeepSeek-V4.1-Flash'
BASE_REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle / 'bundle.json').read_text())
    if (manifest['base_model'], manifest['base_revision']) != (BASE_MODEL, BASE_REVISION):
        raise ValueError('This repository supports only the pinned DeepSeek V4.1 Flash revision')
    spec = manifest['hf_patch']
    filename = spec['filename']
    if Path(filename).name != filename:
        raise ValueError('Patch filename must be a simple filename inside the bundle')
    patch = bundle / filename
    with patch.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/'):
            raise ValueError('Found a Git LFS pointer, not weights. Run: git lfs pull')
    if patch.stat().st_size != spec['bytes']:
        raise ValueError('Patch size does not match bundle.json')
    if sha256_file(patch) != spec['sha256']:
        raise ValueError('Patch SHA256 does not match bundle.json')
    with patch.open('rb') as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError('Invalid safetensors header')
        header_size = struct.unpack('<Q', prefix)[0]
        if not 2 <= header_size <= min(16 * 1024 * 1024, spec['bytes'] - 8):
            raise ValueError('Invalid safetensors header length')
        header = json.loads(stream.read(header_size))
    tensors = {name: entry for name, entry in header.items() if name != '__metadata__'}
    if len(tensors) != spec['tensor_count'] or set(tensors) != set(manifest['base_edited_tensor_sha256']):
        raise ValueError('Tensor names/count do not match the replacement manifest')
    return manifest, patch, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'weights')
    args = parser.parse_args()
    manifest, patch, tensors = verify_bundle(args.bundle)
    print(json.dumps({'verified': True, 'file': str(patch), 'bytes': patch.stat().st_size,
                      'sha256': manifest['hf_patch']['sha256'], 'tensors': len(tensors),
                      'base_model': BASE_MODEL, 'base_revision': BASE_REVISION}, indent=2))


if __name__ == '__main__':
    main()
