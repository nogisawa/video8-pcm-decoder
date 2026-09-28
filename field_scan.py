"""Experimental field-locked scan: search once, then follow the field clock."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from demod_backend import CPU_BACKEND, DemodBackend
from pcm_core import BLOCK_BITS, NTSC_BIT_RATE, Block
from timed import FIELD_PERIOD
from video8pcm import _deduplicate, _keep_sequences, _split_fields, extract


def extract_locked(path: Path, rate: float, start: float, duration: float | None,
                   window_ms: float, phases: int,
                   backend: DemodBackend = CPU_BACKEND) -> list[Block]:
    """Find an initial field, then inspect only predicted track positions.

    A failed prediction is retried in a wider window. The output remains a
    list of CRC-valid blocks for the ordinary field writer.
    """
    with sf.SoundFile(path) as source:
        first = max(0, round(start * rate))
        last = min(source.frames, first + round(duration * rate)) if duration else source.frames
        seed_end = min(last, first + round(.05 * rate))
    seed = extract(path, rate, 'ntsc', start, (seed_end - first) / rate,
                   window_ms, phases, backend)
    groups = [g for g in _split_fields(seed, rate)
              if len({b.address for b in g}) >= 2]
    if not groups:
        # No reliable clock; use the ordinary scan so a late signal is found.
        return extract(path, rate, 'ntsc', start, duration, window_ms, phases,
                       backend)
    step = BLOCK_BITS * rate / NTSC_BIT_RATE
    period = FIELD_PERIOD * rate
    last_group = groups[-1]
    previous_zero = float(np.median([b.sample - b.address * step
                                     for b in last_group]))
    output = seed[:]
    predicted = previous_zero + period
    track_span = 132 * step
    with sf.SoundFile(path) as source:
        while predicted + track_span < last:
            found = []
            for padding in (8000, round(.004 * rate)):
                left = max(first, round(predicted) - padding)
                right = min(last, round(predicted + track_span) + padding)
                source.seek(left)
                samples = source.read(right - left, dtype='float32')
                if len(samples) < 512:
                    break
                candidates = backend.scan_window(samples, left, rate,
                                                 NTSC_BIT_RATE, phases)
                candidates = [b for b in candidates if 0 <= b.address < 132 and
                              abs(b.sample - b.address * step - predicted) < padding / 2]
                found = _keep_sequences(_deduplicate(candidates, rate, NTSC_BIT_RATE),
                                        rate, NTSC_BIT_RATE, 131)
                if len({b.address for b in found}) >= 2:
                    break
            if len({b.address for b in found}) >= 2:
                zero = float(np.median([b.sample - b.address * step for b in found]))
                output.extend(found)
                predicted = zero + period
            else:
                predicted += period
            percent = min(100, round((predicted - first) * 100 / max(1, last - first)))
            print(f'\rtrack scan {percent:3d}%', end='', file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return _deduplicate(output, rate, NTSC_BIT_RATE)


def extract_markers(path: Path, rate: float, start: float,
                    duration: float | None, phases: int,
                    backend: DemodBackend = CPU_BACKEND) -> list[Block]:
    """Use preamble/pilot markers to inspect only the PCM part of each track."""
    from v8markers import detect

    markers, preamble_seconds, total_seconds, covered = detect(
        path, rate, 'ntsc', start, duration, 20.0)
    print(f'markers={len(markers)} preamble_scan={preamble_seconds:.3f}s '
          f'pilot_scan={total_seconds-preamble_seconds:.3f}s', file=sys.stderr)
    step = BLOCK_BITS * rate / NTSC_BIT_RATE
    track_span = 132 * step
    output: list[Block] = []
    with sf.SoundFile(path) as source:
        for index, marker in enumerate(markers, 1):
            peak = marker['preamble_peak_sample']
            expected_zero = peak + .00025 * rate
            found = []
            for margin in (.0002, .0006):
                left = max(0, round(peak - margin * rate))
                right = min(source.frames,
                            round(expected_zero + track_span + margin * rate))
                source.seek(left)
                samples = source.read(right - left, dtype='float32')
                if len(samples) < 512:
                    break
                candidates = backend.scan_window(samples, left, rate,
                                                 NTSC_BIT_RATE, phases)
                candidates = [b for b in candidates if 0 <= b.address < 132 and
                              abs(b.sample - b.address * step - expected_zero)
                              < .0002 * rate]
                found = _keep_sequences(
                    _deduplicate(candidates, rate, NTSC_BIT_RATE),
                    rate, NTSC_BIT_RATE, 131)
                if len({b.address for b in found}) >= 2:
                    break
            output.extend(found)
            if index == len(markers) or index % 10 == 0:
                print(f'\rmarker tracks {index}/{len(markers)}', end='',
                      file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return _deduplicate(output, rate, NTSC_BIT_RATE)
