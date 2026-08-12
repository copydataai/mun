from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mun.config import load_config, reset_config, set_config
from mun.errors import MunError


class ConfigTests(unittest.TestCase):
    def test_set_load_and_reset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            with patch("mun.config.config_path", return_value=path):
                set_config("model", 'owner/model"name')
                set_config("offline", "true")
                self.assertEqual(load_config(), {"model": 'owner/model"name', "offline": True})
                self.assertTrue(reset_config())
                self.assertEqual(load_config(), {})

    def test_unknown_field_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text('mystery = "value"\n')
            with patch("mun.config.config_path", return_value=path):
                with self.assertRaisesRegex(MunError, "Unknown configuration field"):
                    load_config()

    def test_string_values_with_control_characters_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            value = "folder/" + "".join(chr(codepoint) for codepoint in range(32)) + "\x7f/café/🌙"
            with patch("mun.config.config_path", return_value=path):
                set_config("output_dir", value)
                self.assertEqual(load_config()["output_dir"], value)
                self.assertNotIn("\x7f", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
