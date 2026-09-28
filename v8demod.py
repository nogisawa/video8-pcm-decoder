#!/usr/bin/env python3
"""Demodulate CXADC FLAC to timestamped NTSC Video8 PCM fields."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from timed import FIELD_PERIOD, NTSC_FIELD_BYTES, Field, write_field, write_header
from video8pcm import BLOCK_BITS, NTSC_BIT_RATE, _split_fields, extract
from demod_backend import CPU_BACKEND, DemodBackend
from field_scan import extract_locked, extract_markers
from track_decode import demodulate_tracks
from v8crc import count_file, error_rate


def demodulate(path, output, rate, start, duration, window_ms, phases,
               backend: DemodBackend = CPU_BACKEND, scan_mode: str = "track",
               stats: dict | None = None, workers: int = 1):
    if scan_mode == "track":
        return demodulate_tracks(path, output, rate, start, duration, phases,
                                 stats, workers)
    if scan_mode == "markers":
        valid = extract_markers(path, rate, start, duration, phases, backend)
    elif scan_mode == "locked":
        valid = extract_locked(path, rate, start, duration, window_ms,
                               phases, backend)
    else:
        valid = extract(path, rate, "ntsc", start, duration, window_ms, phases,
                        backend)
    groups = _split_fields(valid, rate)
    step = BLOCK_BITS * rate / NTSC_BIT_RATE
    period = FIELD_PERIOD * rate
    written = 0
    previous_number = None
    previous_start = None
    with sf.SoundFile(path) as source, output.open("wb") as sink:
        write_header(sink, rate)
        for group in groups:
            if len({block.address for block in group}) < 2:
                continue
            zero = float(np.median([b.sample - b.address * step for b in group]))
            start_sample = max(0, round(zero))
            if previous_number is None:
                number = 0
            else:
                gap = max(1, round((zero - previous_start) / period))
                number = previous_number + gap
                for missing in range(previous_number + 1, number):
                    expected = max(0, round(previous_start +
                                            (missing - previous_number) * period))
                    write_field(sink, Field(missing, expected, False,
                                            bytes(NTSC_FIELD_BYTES)))
                    written += 1
            pad = 128
            read_start = max(0, int(zero) - pad)
            read_end = min(source.frames, int(zero + 132 * step) + pad)
            source.seek(read_start)
            samples = source.read(read_end - read_start, dtype="float32")
            filtered = backend.filter_samples(samples, rate)
            known = {b.address: bytes((b.address,)) + b.words +
                     bytes((b.crc_recorded & 255, b.crc_recorded >> 8))
                     for b in group}
            blocks = bytearray()
            for address in range(132):
                blocks.extend(known.get(address) or backend.hard_block_at(
                    filtered, zero + address * step - read_start,
                    rate, NTSC_BIT_RATE))
            write_field(sink, Field(number, start_sample, True, bytes(blocks)))
            written += 1
            previous_number, previous_start = number, zero
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, required=True)
    p.add_argument("--sample-rate", type=float)
    p.add_argument("--system", choices=("ntsc",), default="ntsc")
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--duration", type=float)
    p.add_argument("--window-ms", type=float, default=8)
    p.add_argument("--phases", type=int, default=12)
    available = (len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                 else os.cpu_count() or 1)
    p.add_argument("--workers", type=int, default=min(4, available),
                   help="parallel track decoder processes (default: up to 4 available CPUs)")
    p.add_argument("--scan-mode", choices=("exhaustive", "locked", "markers", "track"),
                   default="track", help="track mode loads and decodes one marked track at a time")
    a = p.parse_args()
    if a.start < 0 or (a.duration is not None and a.duration <= 0) or \
            a.window_ms <= 0 or a.phases < 1 or a.workers < 1:
        p.error("invalid scan parameters")
    with sf.SoundFile(a.input) as source:
        rate = a.sample_rate or float(source.samplerate)
    if rate < 1_000_000:
        rate *= 1000
    stats = {}
    count = demodulate(a.input, a.output, rate, a.start, a.duration,
                       a.window_ms, a.phases, scan_mode=a.scan_mode,
                       stats=stats, workers=a.workers)
    print(f"wrote {count} timestamped fields to {a.output}", file=sys.stderr)
    if stats:
        good, bad, missing = stats["good"], stats["bad"], stats["missing"]
    else:
        good, bad, missing = count_file(a.output)
    print(f"input CRC error rate: {error_rate(bad, good + bad)} "
          f"({bad}/{good + bad} blocks, missing_fields={missing})",
          file=sys.stderr)
    if stats.get("read_error") and not stats.get("expected_eof"):
        return 3
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
