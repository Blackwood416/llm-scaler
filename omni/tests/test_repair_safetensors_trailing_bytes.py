import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "tools" / "repair_safetensors_trailing_bytes.py"
SPEC = importlib.util.spec_from_file_location("repair_safetensors_trailing_bytes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_safetensors(path: Path, *, trailing=b"", gap=False, truncate=False):
    second_start = 8 if gap else 4
    header = {
        "first": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]},
        "second": {
            "dtype": "F32",
            "shape": [1],
            "data_offsets": [second_start, second_start + 4],
        },
        "__metadata__": {"test": "true"},
    }
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((8 - len(encoded) % 8) % 8)
    data = b"\x00\x00\x80?" + (b"GAP!" if gap else b"") + b"\x00\x00\x00@"
    payload = struct.pack("<Q", len(encoded)) + encoded + data + trailing
    if truncate:
        payload = payload[:-2]
    path.write_bytes(payload)


class RepairSafetensorsTrailingBytesTests(unittest.TestCase):
    def test_inspect_and_repair_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            output = Path(temp_dir) / "model_fixed.safetensors"
            _write_safetensors(source, trailing=b"x" * 64)

            inspection = MODULE.inspect_file(source)
            self.assertEqual(inspection.tensor_count, 2)
            self.assertEqual(inspection.trailing_bytes, 64)

            repaired, output_hash = MODULE.repair_file(
                source, output, expected_trailing_bytes=64
            )
            self.assertEqual(repaired.trailing_bytes, 64)
            self.assertEqual(output.stat().st_size, repaired.expected_file_size)
            self.assertEqual(MODULE.inspect_file(output).trailing_bytes, 0)
            self.assertEqual(len(output_hash), 64)
            self.assertTrue(source.read_bytes().endswith(b"x" * 64))

    def test_rejects_internal_gap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "gap.safetensors"
            _write_safetensors(source, gap=True)
            with self.assertRaisesRegex(MODULE.RepairError, "gap"):
                MODULE.inspect_file(source)

    def test_rejects_truncated_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "truncated.safetensors"
            _write_safetensors(source, truncate=True)
            with self.assertRaisesRegex(MODULE.RepairError, "truncated tensor data"):
                MODULE.inspect_file(source)

    def test_does_not_overwrite_output_without_force(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "model.safetensors"
            output = Path(temp_dir) / "model_fixed.safetensors"
            _write_safetensors(source, trailing=b"x" * 64)
            output.write_bytes(b"keep")
            with self.assertRaisesRegex(MODULE.RepairError, "already exists"):
                MODULE.repair_file(source, output)
            self.assertEqual(output.read_bytes(), b"keep")


if __name__ == "__main__":
    unittest.main()
