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




UNKNOWN_FRAME_LIMIT = 1 << 62


def read_window(source: sf.SoundFile, left: int, right: int):
    """Read one bounded window, recovering samples before a decoder failure.

    Some FLACs have no total-sample count.  In that case libsndfile reports
    2**63-1 frames and soundfile can raise when a read crosses the real EOF.
    An error poisons the handle, so retries use fresh handles.
    """
    count = max(0, right - left)
    try:
        source.seek(left)
        return source.read(count, dtype="float32"), None
    except sf.LibsndfileError as exc:
        low, high = 0, count + 1
        best = np.empty(0, dtype=np.float32)
        while high - low > 1:
            middle = (low + high) // 2
            try:
                with sf.SoundFile(source.name) as retry:
                    retry.seek(left)
                    data = retry.read(middle, dtype="float32")
            except sf.LibsndfileError:
                high = middle
            else:
                if len(data) < middle:
                    return data, str(exc)
                low, best = middle, data
        return best, str(exc)


def iter_preamble_pairs(source: sf.SoundFile, sample_rate: float,
                        system: str, start: float, duration: float | None,
                        threshold: float, progress=None, state=None):
    """Yield paired run-in tones from bounded chunks, including unknown-length FLACs."""
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    known_length = source.frames < UNKNOWN_FRAME_LIMIT
    first = round(start * sample_rate)
    if known_length:
        first = min(source.frames, first)
    if duration is None:
        last = source.frames if known_length else None
    else:
        last = first + round(duration * sample_rate)
        if known_length:
            last = min(last, source.frames)
    chunk = round(.1 * sample_rate)
    context = round(.004 * sample_rate) + WINDOW
    maximum_gap = round(.00045 * sample_rate)
    best = None
    last_hit = None
    pending = None
    position = first
    if state is not None:
        state.update(known_length=known_length, last_sample=first, read_error=None)
    while last is None or position < last:
        left = max(0, position - context)
        right = position + chunk + context
        if known_length:
            right = min(source.frames, right)
        samples, error = read_window(source, left, right)
        available_end = left + len(samples)
        if state is not None:
            state["last_sample"] = max(state["last_sample"], available_end)
        accepted_end = min(position + chunk, available_end)
        if last is not None:
            accepted_end = min(accepted_end, last)
        for hit in tone_hits(samples, left, sample_rate, bit_rate, threshold):
            if not position <= hit[0] < accepted_end:
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
            progress(accepted_end, last)
        if error is not None:
            if state is not None:
                # Internal seek failure at an unknown-length EOF is a
                # libsndfile limitation; lost sync can also mean corruption.
                state["read_error"] = error
                state["expected_eof"] = not known_length and "psf_fseek" in error
            break
        if len(samples) < right - left:
            break
        position += chunk
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
    """Collect markers for legacy callers; the scanner itself remains bounded."""
    started = perf_counter()
    pilot_seconds = 0.0
    results = []
    state = {}
    with sf.SoundFile(path) as scanner, sf.SoundFile(path) as pilot_source:
        first = round(start * sample_rate)
        for preamble, postamble in iter_preamble_pairs(
                scanner, sample_rate, system, start, duration, threshold,
                state=state):
            pilot_started = perf_counter()
            label, confidence, scores = pilot_at(
                pilot_source, preamble[0], sample_rate, system)
            pilot_seconds += perf_counter() - pilot_started
            results.append({
                "preamble_peak_sample": preamble[0],
                "preamble_peak_seconds": preamble[0] / sample_rate,
                "preamble_ratio": round(preamble[1], 2),
                "postamble_peak_seconds": postamble[0] / sample_rate,
                "pilot": label,
                "pilot_confidence": round(confidence, 2),
                "pilot_scores": [round(value, 6) for value in scores],
            })
    total = perf_counter() - started
    covered = max(0, state["last_sample"] - first) / sample_rate
    if state.get("read_error") and not state.get("expected_eof"):
        print(f"FLAC decode stopped near {covered:.3f}s: "
              f"{state['read_error']}", file=sys.stderr)
    return results, total - pilot_seconds, total, covered


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
    state = {}
    try:
        with sf.SoundFile(a.input) as scanner, sf.SoundFile(a.input) as pilot_source:
            first = round(a.start * rate)
            for preamble, postamble in iter_preamble_pairs(
                    scanner, rate, a.system, a.start, a.duration, a.threshold,
                    state=state):
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
    covered = max(0, state.get("last_sample", first) - first) / rate
    if state.get("read_error"):
        label = ("EOF with unknown FLAC length" if state.get("expected_eof")
                 else "FLAC decode stopped")
        print(f"{label} near {covered:.3f}s: {state['read_error']}",
              file=sys.stderr)
    print(f"input={covered:.3f}s tracks={count} "
          f"preamble_scan={total-pilot_seconds:.3f}s pilot_scan={pilot_seconds:.3f}s "
          f"total={total:.3f}s", file=sys.stderr)
    if state.get("read_error") and not state.get("expected_eof"):
        return 3
    return 0 if count else 2


if __name__ == "__main__":
    raise SystemExit(main())
