#!/usr/bin/env python3
"""Convert timestamped NTSC Video8 fields to synchronized stereo WAV."""
import argparse
from pathlib import Path
import sys
from collections import Counter

import numpy as np
import soundfile as sf
from scipy.signal import sosfilt

from timed import read_fields
from pq import correct_field
from video8pcm import Block, _ntsc_field_audio, crc16_video8

RATE = 31_469


def decode(blocks, use_bad_crc, guard, use_pq=True, correction_counts=None):
    if use_pq and not use_bad_crc:
        blocks, usable, stats = correct_field(blocks)
        if correction_counts is not None:
            correction_counts.update(vars(stats))
    else:
        usable = tuple(
            use_bad_crc or crc16_video8(blocks[address * 13:address * 13 + 11])
            == int.from_bytes(blocks[address * 13 + 11:address * 13 + 13], "little")
            for address in range(132))
    parsed = []
    for address in range(132):
        raw = blocks[address * 13:(address + 1) * 13]
        recorded = int.from_bytes(raw[11:13], "little")
        calculated = crc16_video8(raw[:11])
        if usable[address]:
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
    p.add_argument("--no-pq", action="store_true",
                   help="disable P/Q erasure and CRC-collision correction")
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
    crc_blocks = 0
    crc_errors = 0
    correction_counts = Counter()
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
                crc_blocks += 132
                for address in range(132):
                    raw = field.blocks[address * 13:(address + 1) * 13]
                    if raw[0] != address or crc16_video8(raw[:11]) != int.from_bytes(
                            raw[11:13], "little"):
                        crc_errors += 1
                audio = decode(field.blocks, a.use_bad_crc, a.crc_guard,
                               not a.no_pq, correction_counts)
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
    if crc_blocks:
        print(f"input CRC error rate: {100 * crc_errors / crc_blocks:.4f}% "
              f"({crc_errors}/{crc_blocks} blocks)", file=sys.stderr)
    else:
        print("input CRC error rate: n/a (0 blocks)", file=sys.stderr)
    if not a.no_pq and not a.use_bad_crc:
        print("P/Q: crc_bad={crc_bad} recovered={recovered} "
              "unresolved={unresolved} crc_collision_recovered={crc_collision_recovered} "
              "parity_failures={parity_failures}".format(
                  **{key: correction_counts[key] for key in
                     ("crc_bad", "recovered", "unresolved",
                      "crc_collision_recovered", "parity_failures")}),
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
