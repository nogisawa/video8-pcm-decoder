#!/usr/bin/env python3
"""Detect Video8 PCM clock run-in bursts and f1-f4 tracking pilots.

This command does not decode PCM. Times refer to the supplied ADC sample rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import soundfile as sf
from scipy.fft import rfft, rfftfreq
from scipy.signal import resample_poly, stft

from pcm_core import NTSC_BIT_RATE, PAL_BIT_RATE

PILOTS = {
    "ntsc": (102_544, 118_951, 165_210, 148_689),
    "pal": (101_024, 117_188, 162_760, 146_484),
}
WINDOW = 8192
HOP = 6144  # 25% overlap; larger hops missed tracks in the reference capture


def tone_hits(samples: np.ndarray, absolute_start: int, sample_rate: float,
              bit_rate: float, threshold: float) -> list[tuple[int, float]]:
    """Find clock-frequency spectral peaks with a local noise-floor ratio."""
    if len(samples) < WINDOW:
        return []
    frequencies, seconds, spectrum = stft(
        samples, sample_rate, nperseg=WINDOW, noverlap=WINDOW-HOP,
        boundary=None, padded=False)
    magnitude = np.abs(spectrum)
    target = np.abs(frequencies - bit_rate) <= 30_000
    background = np.abs(frequencies - bit_rate) <= 300_000
    signal = magnitude[target].max(axis=0)
    noise = np.median(magnitude[background], axis=0)
    ratio = signal / np.maximum(noise, 1e-12)
    return [(absolute_start + round(seconds[i] * sample_rate), float(ratio[i]))
            for i in np.flatnonzero(ratio >= threshold)]


def bursts(hits: list[tuple[int, float]], sample_rate: float
           ) -> list[tuple[int, float]]:
    """Collapse adjacent STFT hits into one peak per tone burst."""
    if not hits:
        return []
    maximum_gap = round(.00045 * sample_rate)
    events = []
    group = [hits[0]]
    for hit in hits[1:]:
        if hit[0] - group[-1][0] <= maximum_gap:
            group.append(hit)
        else:
            events.append(max(group, key=lambda item: item[1]))
            group = [hit]
    events.append(max(group, key=lambda item: item[1]))
    return events


def pairs(events: list[tuple[int, float]], sample_rate: float
          ) -> list[tuple[tuple[int, float], tuple[int, float]]]:
    """Pair a clock run-in with the later all-ones after-record margin."""
    result = []
    at = 0
    while at + 1 < len(events):
        gap = (events[at + 1][0] - events[at][0]) / sample_rate
        if .0022 <= gap <= .0032:
            result.append((events[at], events[at + 1]))
            at += 2
        else:
            at += 1
    return result




def iter_preamble_pairs(source: sf.SoundFile, sample_rate: float,
                        system: str, start: float, duration: float | None,
                        threshold: float, progress=None):
    """Yield paired run-in tones while scanning bounded chunks of the FLAC.

    The burst and pair state is kept across chunk boundaries.  This uses the
    same STFT windows as detect(), but does not retain hits for the whole file.
    """
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    first = min(source.frames, round(start * sample_rate))
    last = (source.frames if duration is None else
            min(source.frames, first + round(duration * sample_rate)))
    chunk = round(.1 * sample_rate)
    context = round(.004 * sample_rate) + WINDOW
    maximum_gap = round(.00045 * sample_rate)
    best = None
    last_hit = None
    pending = None
    for position in range(first, last, chunk):
        left = max(0, position - context)
        right = min(source.frames, position + chunk + context)
        source.seek(left)
        samples = source.read(right - left, dtype="float32")
        for hit in tone_hits(samples, left, sample_rate, bit_rate, threshold):
            if not position <= hit[0] < min(position + chunk, last):
                continue
            if best is None:
                best = hit
            elif hit[0] - last_hit <= maximum_gap:
                if hit[1] > best[1]:
                    best = hit
            else:
                if pending is None:
                    pending = best
                elif .0022 <= (best[0] - pending[0]) / sample_rate <= .0032:
                    yield pending, best
                    pending = None
                else:
                    pending = best
                best = hit
            last_hit = hit[0]
        if progress is not None:
            progress(min(position + chunk, last), last)
    if best is not None:
        if pending is not None and .0022 <= (best[0] - pending[0]) / sample_rate <= .0032:
            yield pending, best


def pilot_at(source: sf.SoundFile, preamble_sample: int, sample_rate: float,
             system: str) -> tuple[str, float, list[float]]:
    # Use the middle of the PCM track. The pilot also exists in that area.
    first = preamble_sample + round(.00025 * sample_rate)
    count = round(.002 * sample_rate)
    if first + count > source.frames:
        return "unknown", 0.0, [0.0] * 4
    source.seek(first)
    raw = source.read(count, dtype="float32")
    reduced = resample_poly(raw, 1, 16)
    spectrum = np.abs(rfft(reduced * np.hanning(len(reduced))))
    frequencies = rfftfreq(len(reduced), 16 / sample_rate)
    values = []
    for centre in PILOTS[system]:
        band = np.abs(frequencies - centre) <= 1_500
        values.append(float(spectrum[band].max()))
    ranking = np.argsort(values)
    confidence = values[ranking[-1]] / max(values[ranking[-2]], 1e-12)
    return f"f{ranking[-1] + 1}", float(confidence), values


def detect(path: Path, sample_rate: float, system: str, start: float,
           duration: float | None, threshold: float):
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    search_started = perf_counter()
    hits: list[tuple[int, float]] = []
    with sf.SoundFile(path) as source:
        first = min(source.frames, round(start * sample_rate))
        last = (source.frames if duration is None else
                min(source.frames, first + round(duration * sample_rate)))
        chunk = round(.1 * sample_rate)
        context = round(.004 * sample_rate) + WINDOW
        for position in range(first, last, chunk):
            left = max(0, position - context)
            right = min(source.frames, position + chunk + context)
            source.seek(left)
            samples = source.read(right - left, dtype="float32")
            hits.extend((sample, ratio) for sample, ratio in
                        tone_hits(samples, left, sample_rate, bit_rate, threshold)
                        if position <= sample < min(position + chunk, last))
    events = bursts(sorted(hits), sample_rate)
    matched = pairs(events, sample_rate)
    scan_seconds = perf_counter() - search_started
    results = []
    with sf.SoundFile(path) as source:
        for preamble, postamble in matched:
            label, confidence, scores = pilot_at(
                source, preamble[0], sample_rate, system)
            results.append({
                "preamble_peak_sample": preamble[0],
                "preamble_peak_seconds": preamble[0] / sample_rate,
                "preamble_ratio": round(preamble[1], 2),
                "postamble_peak_seconds": postamble[0] / sample_rate,
                "pilot": label,
                "pilot_confidence": round(confidence, 2),
                "pilot_scores": [round(value, 6) for value in scores],
            })
    total_seconds = perf_counter() - search_started
    return results, scan_seconds, total_seconds, (last - first) / sample_rate


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("-o", "--output", type=Path, help="JSON Lines; default stdout")
    p.add_argument("--system", choices=PILOTS, default="ntsc")
    p.add_argument("--sample-rate", type=float, help="actual ADC samples per second")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--duration", type=float)
    p.add_argument("--threshold", type=float, default=20.0,
                   help="run-in spectral peak/background ratio (default 20)")
    a = p.parse_args()
    if a.start < 0 or (a.duration is not None and a.duration <= 0) or a.threshold <= 0:
        p.error("invalid range or threshold")
    with sf.SoundFile(a.input) as source:
        rate = a.sample_rate or float(source.samplerate)
    if rate < 1_000_000:
        rate *= 1000
    started = perf_counter()
    count = 0
    pilot_seconds = 0.0
    sink = a.output.open("w", encoding="utf-8") if a.output else sys.stdout
    try:
        with sf.SoundFile(a.input) as scanner, sf.SoundFile(a.input) as pilot_source:
            first = min(scanner.frames, round(a.start * rate))
            last = (scanner.frames if a.duration is None else
                    min(scanner.frames, first + round(a.duration * rate)))
            covered = (last - first) / rate
            for preamble, postamble in iter_preamble_pairs(
                    scanner, rate, a.system, a.start, a.duration, a.threshold):
                pilot_started = perf_counter()
                label, confidence, scores = pilot_at(
                    pilot_source, preamble[0], rate, a.system)
                pilot_seconds += perf_counter() - pilot_started
                row = {
                    "preamble_peak_sample": preamble[0],
                    "preamble_peak_seconds": preamble[0] / rate,
                    "preamble_ratio": round(preamble[1], 2),
                    "postamble_peak_seconds": postamble[0] / rate,
                    "pilot": label,
                    "pilot_confidence": round(confidence, 2),
                    "pilot_scores": [round(value, 6) for value in scores],
                }
                sink.write(json.dumps(row, separators=(",", ":")) + "\n")
                count += 1
                if count % 10 == 0:
                    sink.flush()
    finally:
        if a.output:
            sink.close()
    total = perf_counter() - started
    print(f"input={covered:.3f}s tracks={count} "
          f"preamble_scan={total-pilot_seconds:.3f}s pilot_scan={pilot_seconds:.3f}s "
          f"total={total:.3f}s", file=sys.stderr)
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
