import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from nsfw_scanner.plugins import Plugin, load_plugins
from nsfw_scanner.scanner import main


class FakeSession:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def get_inputs(self):
        return [type("Input", (), {"name": "image"})()]

    def run(self, _outputs, _inputs):
        return [self.values]


class PluginTests(unittest.TestCase):
    def test_named_class_map_maps_sexy_to_six(self):
        plugin = Plugin(
            "example",
            FakeSession([0.1, 0.8, 0.1]),
            {"sexy": 6, "porn": 10},
            class_labels=["neutral", "sexy", "porn"],
        )

        self.assertEqual(plugin.rating_for(Image.new("RGB", (4, 4))), 6)

    def test_unmapped_top_class_is_not_given_an_unsafe_rating(self):
        plugin = Plugin(
            "example",
            FakeSession([0.9, 0.05, 0.05]),
            {"sexy": 6, "porn": 10},
            class_labels=["neutral", "sexy", "porn"],
        )

        self.assertIsNone(plugin.rating_for(Image.new("RGB", (4, 4))))

    def test_legacy_index_map_remains_supported(self):
        plugin = Plugin(
            "example",
            FakeSession([0.1, 0.8, 0.1]),
            {1: 6, 2: 10},
        )

        self.assertEqual(plugin.rating_for(Image.new("RGB", (4, 4))), 6)

    def test_index_map_can_use_explicit_class_order(self):
        plugin = Plugin(
            "example",
            FakeSession([0.1, 0.9]),
            {1: 6},
            class_labels=["neutral", "sexy"],
        )

        self.assertEqual(plugin.rating_for(Image.new("RGB", (4, 4))), 6)

    def test_loader_accepts_named_labels_and_class_order(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "plugins.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump({
                    "plugins": [{
                        "name": "example",
                        "model": "example.onnx",
                        "class_labels": ["neutral", "sexy"],
                        "labels": {"sexy": 6},
                    }]
                }, handle)
            session = FakeSession([0.1, 0.9])
            with patch("nsfw_scanner.plugins._session_for",
                       return_value=session):
                plugins = load_plugins(path)

        self.assertEqual(len(plugins), 1)
        self.assertEqual(
            plugins[0].rating_for(Image.new("RGB", (4, 4))),
            6,
        )

    def test_list_plugins_includes_builtins_and_configured_section(self):
        from contextlib import redirect_stdout
        from io import StringIO

        output = StringIO()
        with redirect_stdout(output):
            main(["--list-plugins"])

        self.assertIn("Built-in classifiers:", output.getvalue())
        self.assertIn("nsfw-joint:", output.getvalue())
        self.assertIn("Configured plugins:", output.getvalue())


if __name__ == "__main__":
    unittest.main()
