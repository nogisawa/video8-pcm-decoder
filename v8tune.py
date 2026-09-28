#!/usr/bin/env python3
"""Find Video8 demodulation parameters by maximizing valid CRC sequences."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from video8pcm import (NTSC_BIT_RATE, PAL_BIT_RATE, _deduplicate,
                       _keep_sequences)
from demod_backend import CPU_BACKEND, DemodBackend


@dataclass
class Result:
    sample_rate: float
    system: str
    blocks: int
    fields: int
    score: int


def evaluate(samples: np.ndarray, sample_rate: float, system: str,
             phases: int, window_ms: float,
             backend: DemodBackend = CPU_BACKEND) -> Result:
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    max_address = 131 if system == "ntsc" else 156
    window = max(4096, round(window_ms * 1e-3 * sample_rate))
    overlap = round(.6e-3 * sample_rate)
    blocks = []
    position = 0
    pending: list[tuple[np.ndarray, int]] = []
    while position < len(samples):
        chunk = samples[position:min(position + window, len(samples))]
        if len(chunk) < 512:
            break
        pending.append((chunk, position))
        if len(pending) >= 16:
            blocks.extend(backend.scan_windows(pending, sample_rate, bit_rate, phases))
            pending.clear()
        position += max(1, window - overlap)
    if pending:
        blocks.extend(backend.scan_windows(pending, sample_rate, bit_rate, phases))
    blocks = _deduplicate(blocks, sample_rate, bit_rate)
    blocks = _keep_sequences(blocks, sample_rate, bit_rate, max_address)
    starts = sum(block.address == 0 for block in blocks)
    # Complete address runs dominate; field starts break close ties.
    return Result(sample_rate, system, len(blocks), starts,
                  len(blocks) * 100 + starts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="The recommended values can be passed directly to "
               "v8demod.py.")
    parser.add_argument("input", type=Path, help="CXADC FLAC capture")
    parser.add_argument("--system", choices=("auto", "ntsc", "pal"),
                        default="auto", help="try both standards by default")
    parser.add_argument("--sample-rate", type=float,
                        help="centre ADC rate; default: FLAC rate, or x1000 "
                             "when metadata is below 1 MHz")
    parser.add_argument("--range-percent", type=float, default=1.0,
                        help="sample-rate search range on each side (default: 1)")
    parser.add_argument("--steps", type=int, default=9,
                        help="coarse sample-rate candidates (default: 9)")
    parser.add_argument("--refine-passes", type=int, default=2,
                        help="successive searches around the winner (default: 2)")
    parser.add_argument("--start", type=float, default=0.0,
                        help="RF time to begin test, in real seconds")
    parser.add_argument("--duration", type=float, default=.05,
                        help="test duration in real seconds (default: 0.05)")
    parser.add_argument("--phases", type=int, default=8,
                        help="bit phase trials per candidate (default: 8)")
    parser.add_argument("--window-ms", type=float, default=8.0)
    parser.add_argument("--top", type=int, default=8,
                        help="number of ranked candidates to print")
    args = parser.parse_args()
    if args.steps < 3 or args.duration <= 0 or args.range_percent <= 0:
        parser.error("steps must be >=3 and duration/range must be positive")

    with sf.SoundFile(args.input) as source:
        centre = args.sample_rate
        if centre is None:
            centre = float(source.samplerate)
            if centre < 1_000_000:
                centre *= 1000.0
        # Read enough samples for the highest candidate rate.
        high = centre * (1 + args.range_percent / 100)
        first = max(0, round(args.start * centre))
        source.seek(min(first, source.frames))
        samples = source.read(min(round(args.duration * high),
                                  source.frames - min(first, source.frames)),
                              dtype="float32")

    results = []
    systems = ("ntsc", "pal") if args.system == "auto" else (args.system,)
    search_centre = centre
    half_span = centre * args.range_percent / 100
    for pass_number in range(args.refine_passes):
        rates = np.linspace(search_centre - half_span,
                            search_centre + half_span, args.steps)
        print(f"\nPass {pass_number + 1}/{args.refine_passes}: "
              f"{search_centre:.3f} +/- {half_span:.3f} Hz")
        pass_results = []
        for number, (system, rate) in enumerate(
                ((s, r) for s in systems for r in rates), 1):
            result = evaluate(samples, float(rate), system, args.phases,
                              args.window_ms)
            pass_results.append(result)
            print(f"[{number:2d}/{len(rates)*len(systems)}] {system:4s} "
                  f"{rate:12.3f} Hz  CRC-run blocks={result.blocks:4d} "
                  f"field-starts={result.fields}")
        results.extend(pass_results)
        winner = max(pass_results, key=lambda item: item.score)
        search_centre = winner.sample_rate
        half_span = (2 * half_span / (args.steps - 1))
        systems = (winner.system,)

    results.sort(key=lambda item: item.score, reverse=True)
    best = results[0]
    print("\nRank  System  Sample rate (Hz)  CRC-run blocks  Field starts")
    for rank, result in enumerate(results[:args.top], 1):
        print(f"{rank:4d}  {result.system:6s} {result.sample_rate:16.3f}  "
              f"{result.blocks:14d}  {result.fields:12d}")
    print("\nRecommended command:")
    print(f"python3 v8demod.py {args.input} -o pcm.v8t "
          f"--system {best.system} --sample-rate {best.sample_rate:.3f} "
          f"--phases {max(args.phases, 12)} --window-ms {args.window_ms:g}")
    return 0 if best.blocks else 2


if __name__ == "__main__":
    raise SystemExit(main())
