#!/usr/bin/env python3
"""Convert timestamped NTSC Video8 fields to synchronized stereo WAV."""
import argparse
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

from timed import read_fields
from video8pcm import Block, _ntsc_field_audio, crc16_video8

RATE = 31_469


def decode(blocks, use_bad_crc, guard):
    parsed = []
    for address in range(132):
        raw = blocks[address * 13:(address + 1) * 13]
        recorded = int.from_bytes(raw[11:13], "little")
        calculated = crc16_video8(raw[:11])
        if recorded == calculated or use_bad_crc:
            parsed.append(Block(0, address, raw[1:11], recorded, calculated))
    audio = _ntsc_field_audio(parsed)
    for channel in range(2):
        values = audio[:, channel]
        missing = np.isnan(values)
        if guard:
            indices = np.flatnonzero(missing)
            for delta in range(1, guard + 1):
                missing[np.clip(indices - delta, 0, 524)] = True
                missing[np.clip(indices + delta, 0, 524)] = True
        good = np.flatnonzero(~missing)
        if len(good):
            values[missing] = np.interp(np.flatnonzero(missing), good, values[good])
        else:
            values[:] = 0
    return audio


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--use-bad-crc", action="store_true")
    p.add_argument("--crc-guard", type=int, default=1)
    p.add_argument("--lowpass", type=float, default=15000)
    p.add_argument("--origin", choices=("capture", "first"), default="capture",
                   help="capture pads from RF time zero; first starts at first field")
    a = p.parse_args()
    if not 0 <= a.crc_guard <= 32 or not 0 <= a.lowpass < RATE / 2:
        p.error("invalid filter parameters")
    # Match Rust's second-order Butterworth reconstruction filter.
    from scipy.signal import butter
    sos = butter(2, a.lowpass, fs=RATE, output="sos") if a.lowpass else None
    state = np.zeros((1, 2, 2), dtype=np.float64) if sos is not None else None
    written = 0
    fields = 0
    with a.input.open("rb") as source, sf.SoundFile(
            a.output, "w", samplerate=RATE, channels=2, subtype="PCM_16") as sink:
        capture_rate, records = read_fields(source)
        origin = None
        for field in records:
            if origin is None:
                origin = 0 if a.origin == "capture" else field.sample
            target = round((field.sample - origin) / capture_rate * RATE)
            if target < written - 525:
                raise ValueError("field timestamp moves backwards")
            if target > written:
                silence = np.zeros((min(target - written, 65536), 2), dtype=np.int16)
                while written < target:
                    n = min(len(silence), target - written)
                    sink.write(silence[:n])
                    written += n
                if sos is not None:
                    state[:] = 0
            if not field.present:
                audio = np.zeros((525, 2), dtype=np.float64)
            else:
                audio = decode(field.blocks, a.use_bad_crc, a.crc_guard)
            audio *= 64
            if sos is not None:
                audio, state = sosfilt(sos, audio, axis=0, zi=state)
            pcm = np.clip(audio, -32768, 32767).astype(np.int16)
            overlap = max(0, written - target)
            if overlap < len(pcm):
                sink.write(pcm[overlap:])
                written += len(pcm) - overlap
            fields += 1
    print(f"wrote {fields} fields, {written} WAV frames", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
