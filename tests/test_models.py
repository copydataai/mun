from __future__ import annotations

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from mun.models import (
    MANIFEST_FILE,
    InstalledModel,
    VerificationResult,
    _download_plan,
    _search_record,
    _write_manifest,
    _write_metadata,
    installed_models,
    remove_model,
    search_models,
    verify_installed_model,
)
from mun.errors import MunError
from mun.runtime import TransformersRuntime


class Sibling:
    def __init__(self, name: str, size: int) -> None:
        self.rfilename = name
        self.size = size


class ModelTests(unittest.TestCase):
    def _installed_model(self, directory: Path, trust_remote_code: bool = False) -> InstalledModel:
        return InstalledModel(
            "owner/model",
            "abc",
            str(directory),
            "2026-08-09T00:00:00+00:00",
            trust_remote_code=trust_remote_code,
        )

    def test_manifest_records_artifacts_and_aggregate_digest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_bytes(b"{}")
            weights = directory / "weights" / "model.safetensors"
            weights.parent.mkdir()
            weights.write_bytes(b"weights")
            model = self._installed_model(directory)

            manifest = _write_manifest(directory, model)

            self.assertEqual(manifest["version"], 1)
            self.assertEqual(manifest["source_repository"], "owner/model")
            self.assertEqual(manifest["source_revision"], "abc")
            self.assertEqual(manifest["installed_at"], model.installed_at)
            self.assertIs(manifest["trust_remote_code"], False)
            self.assertEqual(
                manifest["files"],
                [
                    {"path": "config.json", "bytes": 2, "sha256": sha256(b"{}").hexdigest()},
                    {
                        "path": "weights/model.safetensors",
                        "bytes": 7,
                        "sha256": sha256(b"weights").hexdigest(),
                    },
                ],
            )
            self.assertRegex(manifest["artifact_digest"], r"^[0-9a-f]{64}$")
            self.assertTrue((directory / MANIFEST_FILE).is_file())

    def test_verification_detects_modified_missing_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "config.json"
            artifact.write_bytes(b"original")
            model = self._installed_model(directory)
            _write_manifest(directory, model)

            self.assertEqual(verify_installed_model(model).status, "verified")

            artifact.write_bytes(b"changed")
            self.assertEqual(verify_installed_model(model).status, "modified")
            artifact.unlink()
            self.assertEqual(verify_installed_model(model).status, "missing")
            artifact.write_bytes(b"original")
            (directory / "untracked.bin").write_bytes(b"extra")
            self.assertEqual(verify_installed_model(model).status, "unexpected_file")

    def test_metadata_is_allowed_but_untracked_cache_files_are_denied(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_bytes(b"{}")
            model = self._installed_model(directory)
            _write_manifest(directory, model)
            _write_metadata(directory, model)

            self.assertEqual(verify_installed_model(model).status, "verified")
            cache = directory / ".cache" / "download"
            cache.mkdir(parents=True)
            (cache / "state.json").write_bytes(b"{}")
            self.assertEqual(verify_installed_model(model).status, "unexpected_file")

    def test_legacy_installation_returns_typed_manifest_missing_guidance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            model = self._installed_model(Path(temporary))

            result = verify_installed_model(model)

            self.assertIsInstance(result, VerificationResult)
            self.assertEqual(result.status, "manifest_missing")
            self.assertIn("reinstall", result.guidance.lower())

    def test_remote_code_installation_is_classified_unsafe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "config.json").write_bytes(b"{}")
            model = self._installed_model(directory, trust_remote_code=True)
            _write_manifest(directory, model)

            self.assertEqual(verify_installed_model(model).status, "unsafe_remote_code")

    def test_remote_code_download_requires_explicit_acknowledgement(self) -> None:
        from mun.models import download_model

        with self.assertRaisesRegex(MunError, "acknowledge"):
            download_model("owner/model", Path("/models"), "abc", True, False)

    def test_transformers_runtime_verifies_before_importing_transformers(self) -> None:
        model = self._installed_model(Path("/missing/model"))

        with patch("mun.runtime.verify_installed_model") as verify:
            verify.return_value = VerificationResult("modified", guidance="reinstall the model")
            with patch.dict("sys.modules", {"transformers": None}):
                with self.assertRaisesRegex(Exception, "integrity verification failed: modified"):
                    TransformersRuntime(model)

        verify.assert_called_once_with(model)

    def test_transformers_runtime_reverifies_after_pipeline_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            artifact = directory / "config.json"
            artifact.write_bytes(b"original")
            model = self._installed_model(directory)
            _write_manifest(directory, model)

            torch = SimpleNamespace(
                float16=object(),
                float32=object(),
                cuda=SimpleNamespace(is_available=lambda: False),
                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
            )
            config = SimpleNamespace(model_type="whisper")

            def mutate_during_load(*args, **kwargs):
                artifact.write_bytes(b"changed")
                return object()

            transformers = SimpleNamespace(
                AutoConfig=SimpleNamespace(from_pretrained=lambda *args, **kwargs: config),
                pipeline=mutate_during_load,
            )
            with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
                with self.assertRaisesRegex(MunError, "changed while the runtime was loading"):
                    TransformersRuntime(model, "cpu")

    def test_transformers_runtime_refuses_unacknowledged_remote_code(self) -> None:
        model = self._installed_model(Path("/models/unsafe"), trust_remote_code=True)

        with self.assertRaisesRegex(Exception, "acknowledged"):
            TransformersRuntime(model)

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
