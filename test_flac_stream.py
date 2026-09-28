"""Regression checks for FLAC streams without a total-sample count."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf

from v8markers import UNKNOWN_FRAME_LIMIT, read_window


class FlacStreamTests(unittest.TestCase):
    def test_unknown_length_with_trailing_garbage(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "garbage.flac"
            samples = np.arange(10000, dtype=np.int16)
            sf.write(path, samples, 44100, subtype="PCM_16")
            with path.open("r+b") as stream:
                stream.seek(18)
                rate_and_total = int.from_bytes(stream.read(8), "big")
                stream.seek(18)
                stream.write((rate_and_total & ~((1 << 36) - 1)).to_bytes(8, "big"))
            with path.open("ab") as stream:
                stream.write(b"trailing invalid FLAC data" * 100)
            with sf.SoundFile(path) as source:
                recovered, error = read_window(source, 0, len(samples) + 100)
            self.assertIsNotNone(error)
            self.assertGreaterEqual(len(recovered), len(samples) - 2)

    def test_unknown_length_read_preserves_samples_before_eof(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unknown.flac"
            samples = np.arange(10000, dtype=np.int16)
            sf.write(path, samples, 44100, subtype="PCM_16")
            with path.open("r+b") as stream:
                header = stream.read(26)
                self.assertEqual(header[:8], b"fLaC\x00\x00\x00\x22")
                rate_and_total = int.from_bytes(header[18:26], "big")
                stream.seek(18)
                stream.write((rate_and_total & ~((1 << 36) - 1)).to_bytes(8, "big"))
            with sf.SoundFile(path) as source:
                self.assertGreaterEqual(source.frames, UNKNOWN_FRAME_LIMIT)
                recovered, error = read_window(source, 0, len(samples) + 100)
            self.assertIsNotNone(error)
            self.assertGreaterEqual(len(recovered), len(samples) - 2)
            np.testing.assert_allclose(recovered, samples[:len(recovered)] / 32768)


if __name__ == "__main__":
    unittest.main()
