from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mun.models import InstalledModel, _download_plan, _write_metadata, installed_models, remove_model


class Sibling:
    def __init__(self, name: str, size: int) -> None:
        self.rfilename = name
        self.size = size


class ModelTests(unittest.TestCase):
    def test_download_plan_prefers_safetensors(self) -> None:
        size, ignored = _download_plan(
            [Sibling("model.safetensors", 10), Sibling("pytorch_model.bin", 10), Sibling("config.json", 1)]
        )
        self.assertEqual(size, 11)
        self.assertIn("*.bin", ignored)

    def test_only_managed_model_directory_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "models"
            directory = root / "owner--model--abc"
            directory.mkdir(parents=True)
            (directory / "weights").write_bytes(b"1234")
            model = InstalledModel("owner/model", "abc", str(directory), "2026-08-09T00:00:00+00:00")
            _write_metadata(directory, model)

            self.assertEqual(installed_models(root), [model])
            removed, reclaimed = remove_model(root, "owner/model")

            self.assertEqual(removed, model)
            self.assertGreaterEqual(reclaimed, 4)
            self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()
