"""Exercise real safetensors replacements without downloading model weights."""
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from apply_patch import apply, tensor_hash
from verify_patch import BASE_MODEL, BASE_REVISION, sha256_file, verify_bundle


class PatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.base, self.bundle = self.root / 'base', self.root / 'bundle'
        self.base.mkdir()
        self.bundle.mkdir()
        self.old = {'selected.weight': torch.arange(32, dtype=torch.float32).reshape(4, 8),
                    'unchanged.in_same_shard': torch.tensor([7., 9.])}
        self.new = {'selected.weight': self.old['selected.weight'] + 100}
        save_file(self.old, str(self.base / 'one.safetensors'))
        save_file({'unaffected': torch.tensor([3., 4.])}, str(self.base / 'two.safetensors'))
        (self.base / 'config.json').write_text('{"fixture":true}')
        index = {'weight_map': {'selected.weight': 'one.safetensors',
                               'unchanged.in_same_shard': 'one.safetensors', 'unaffected': 'two.safetensors'}}
        (self.base / 'model.safetensors.index.json').write_text(json.dumps(index))
        save_file(self.new, str(self.bundle / 'hf-weight-patch.safetensors'))
        patch = self.bundle / 'hf-weight-patch.safetensors'
        self.manifest = {'base_model': BASE_MODEL, 'base_revision': BASE_REVISION,
                         'base_edited_tensor_sha256': {'selected.weight': tensor_hash(self.old['selected.weight'])},
                         'hf_patch': {'filename': patch.name, 'bytes': patch.stat().st_size,
                                      'sha256': sha256_file(patch), 'tensor_count': 1}}
        self.write_manifest()

    def write_manifest(self):
        (self.bundle / 'bundle.json').write_text(json.dumps(self.manifest))

    def test_replacement_preserves_original_and_unselected_values(self):
        before = sha256_file(self.base / 'one.safetensors')
        output = self.root / 'edited'
        apply(self.base, self.bundle, output)
        self.assertEqual(sha256_file(self.base / 'one.safetensors'), before)
        actual = load_file(str(output / 'one.safetensors'))
        self.assertTrue(torch.equal(actual['selected.weight'], self.new['selected.weight']))
        self.assertTrue(torch.equal(actual['unchanged.in_same_shard'], self.old['unchanged.in_same_shard']))
        self.assertEqual((self.base / 'two.safetensors').stat().st_ino, (output / 'two.safetensors').stat().st_ino)
        self.assertNotEqual((self.base / 'one.safetensors').stat().st_ino, (output / 'one.safetensors').stat().st_ino)
        self.assertTrue((output / 'patch-application.json').exists())

    def test_wrong_source_is_rejected_before_output_creation(self):
        self.manifest['base_edited_tensor_sha256']['selected.weight'] = '0' * 64
        self.write_manifest()
        output = self.root / 'edited'
        with self.assertRaisesRegex(ValueError, 'Original tensor'):
            apply(self.base, self.bundle, output)
        self.assertFalse(output.exists())

    def test_corrupt_patch_is_rejected(self):
        patch = self.bundle / 'hf-weight-patch.safetensors'
        raw = bytearray(patch.read_bytes())
        raw[-1] ^= 1
        patch.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            verify_bundle(self.bundle)

    def test_nested_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            apply(self.base, self.bundle, self.base / 'edited')

    def test_copy_mode_does_not_share_unchanged_shard(self):
        output = self.root / 'edited'
        apply(self.base, self.bundle, output, copy_unchanged=True)
        self.assertNotEqual((self.base / 'two.safetensors').stat().st_ino, (output / 'two.safetensors').stat().st_ino)

    def test_lfs_pointer_has_actionable_error(self):
        (self.bundle / 'hf-weight-patch.safetensors').write_text('version https://git-lfs.github.com/spec/v1\n')
        with self.assertRaisesRegex(ValueError, 'git lfs pull'):
            verify_bundle(self.bundle)


if __name__ == '__main__':
    unittest.main()
