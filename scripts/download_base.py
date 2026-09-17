#!/usr/bin/env python3
"""Download exactly the original revision expected by this patch."""
import argparse
import json
import shutil
from pathlib import Path

from verify_patch import BASE_MODEL, BASE_REVISION, ROOT, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--verify-shards', action='store_true', help='Also read and hash all 510 GB after download')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not list(args.output.glob('*.safetensors')) and shutil.disk_usage(args.output).free < 550_000_000_000:
        raise SystemExit('Need at least 550 GB free before the first download')
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=BASE_MODEL, revision=BASE_REVISION, local_dir=str(args.output), max_workers=8)
    inventory = json.loads((ROOT / 'base-model-files.json').read_text())
    assert inventory['repo_id'] == BASE_MODEL and inventory['revision'] == BASE_REVISION
    shards = [row for row in inventory['files'] if row['path'].endswith('.safetensors')]
    for row in shards:
        path = args.output / row['path']
        if path.stat().st_size != row['bytes']:
            raise RuntimeError(f'Shard size mismatch: {row["path"]}')
        if args.verify_shards and sha256_file(path) != row['sha256']:
            raise RuntimeError(f'Shard SHA256 mismatch: {row["path"]}')
    print(f'Download complete: {BASE_MODEL}@{BASE_REVISION}; {len(shards)} shard sizes verified')
    if args.verify_shards:
        print('All original shard SHA256 hashes verified')


if __name__ == '__main__':
    main()
