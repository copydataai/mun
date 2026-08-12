from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mun.models import (
    InstalledModel,
    _download_plan,
    _search_record,
    _write_metadata,
    installed_models,
    remove_model,
    search_models,
)


class Sibling:
    def __init__(self, name: str, size: int) -> None:
        self.rfilename = name
        self.size = size


class ModelTests(unittest.TestCase):
    def test_offline_search_reports_discovery_facts_without_qualification_claims(self) -> None:
        catalog = {
            "models": [
                {
                    "id": "owner/model",
                    "revision": "abc",
                    "library": "transformers",
                    "pipeline": "automatic-speech-recognition",
                    "gated": False,
                    "quality": "balanced",
                    "license": "apache-2.0",
                },
            ]
        }

        with patch("mun.models.load_catalog", return_value=catalog):
            records = search_models(None, 20, offline=True)

        self.assertEqual(records[0]["library"], "transformers")
        self.assertIs(records[0]["gated"], False)
        self.assertNotIn("evidence", records[0])
        self.assertNotIn("tested", records[0])

    def test_remote_search_reports_discovery_facts_without_qualification_claims(self) -> None:
        model = SimpleNamespace(
            id="owner/model",
            library_name="transformers",
            pipeline_tag="automatic-speech-recognition",
            sha="abc",
            downloads=10,
            likes=2,
            gated=True,
        )

        record = _search_record(model)

        self.assertEqual(record["library"], "transformers")
        self.assertIs(record["gated"], True)
        self.assertEqual(record["downloads"], 10)
        self.assertNotIn("evidence", record)
        self.assertNotIn("quality", record)

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
