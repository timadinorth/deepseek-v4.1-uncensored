#!/usr/bin/env python3
"""Build a separate checkpoint from original weights and replacement tensors."""
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from verify_patch import ROOT, verify_bundle


def tensor_hash(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def apply(base, bundle, output, copy_unchanged=False):
    base, output = Path(base).resolve(), Path(output).resolve()
    if not base.is_dir():
        raise ValueError('Base checkpoint directory does not exist')
    if output.exists() or output.is_relative_to(base):
        raise ValueError('Output must be a new directory outside the original checkpoint')
    manifest, patch_file, _ = verify_bundle(bundle)
    patch = load_file(str(patch_file), device='cpu')
    index = json.loads((base / 'model.safetensors.index.json').read_text())['weight_map']
    by_shard = defaultdict(list)
    for name in patch:
        shard = Path(index[name])
        if shard.is_absolute() or '..' in shard.parts:
            raise ValueError('Unexpected shard path in source index')
        by_shard[str(shard)].append(name)

    # Validate every replaced original before creating any output shard.
    for shard, names in by_shard.items():
        with safe_open(str(base / shard), framework='pt', device='cpu') as reader:
            for name in names:
                original = reader.get_tensor(name)
                if original.shape != patch[name].shape or original.dtype != patch[name].dtype:
                    raise ValueError(f'Shape/dtype mismatch: {name}')
                if tensor_hash(original) != manifest['base_edited_tensor_sha256'][name]:
                    raise ValueError(f'Original tensor does not match the measured revision: {name}')
    files = [p for p in base.rglob('*') if p.is_file() and '.cache' not in p.relative_to(base).parts]
    required = sum(p.stat().st_size for p in files
                   if copy_unchanged or str(p.relative_to(base)) in by_shard or p.suffix != '.safetensors')
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < required + 64 * 1024 * 1024:
        raise ValueError(f'Insufficient output disk space: need approximately {required} bytes plus margin')
    if not copy_unchanged and base.stat().st_dev != output.parent.stat().st_dev:
        raise ValueError('Hardlinks require the same filesystem; use --copy-unchanged for another filesystem')
    output.mkdir()
    for path in sorted(files):
        relative = path.relative_to(base)
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if str(relative) in by_shard:
            with safe_open(str(path), framework='pt', device='cpu') as reader:
                tensors = {name: patch[name] if name in patch else reader.get_tensor(name)
                           for name in reader.keys()}
                temporary = target.with_suffix('.tmp')
                save_file(tensors, str(temporary), metadata=reader.metadata())
                temporary.replace(target)
            # Check the actual serialized replacements, not only in-memory inputs.
            with safe_open(str(target), framework='pt', device='cpu') as reader:
                for name in by_shard[str(relative)]:
                    if tensor_hash(reader.get_tensor(name)) != tensor_hash(patch[name]):
                        raise RuntimeError(f'Written replacement failed verification: {name}')
            print(f'Rewrote and verified {relative}', flush=True)
        elif path.suffix == '.safetensors' and not copy_unchanged:
            os.link(path, target)
        else:
            shutil.copy2(path, target)
    (output / 'abliteration-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output / 'patch-application.json').write_text(json.dumps({
        'base_model': manifest['base_model'], 'base_revision': manifest['base_revision'],
        'patch_sha256': manifest['hf_patch']['sha256'], 'verified_source_tensors': len(patch),
        'verified_written_tensors': len(patch), 'rewritten_shards': len(by_shard),
        'unchanged_shards': 'copies' if copy_unchanged else 'hardlinks'}, indent=2) + '\n')
    (output / 'README.md').write_text(
        '# DeepSeek V4.1 Flash — experimental attention weight edit\n\n'
        'Base: deepseek-ai/DeepSeek-V4.1-Flash, revision ' + manifest['base_revision'] + '.\n\n'
        'This is a modified checkpoint. Original-model benchmark claims do not automatically apply.\n'
        'See abliteration-manifest.json, patch-application.json and the repository evaluation notes.\n')
    print(f'Checkpoint exported: {output}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'weights')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--copy-unchanged', action='store_true', help='Copy rather than hardlink unchanged shards')
    args = parser.parse_args()
    apply(args.base, args.bundle, args.output, args.copy_unchanged)


if __name__ == '__main__':
    main()
