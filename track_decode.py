"""Experimental one-track-at-a-time Video8 PCM demodulation."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import soundfile as sf

from cpu_backend import (CRC_BASE, CRC_CONTRIBUTIONS, filter_samples,
                         scan_window)
from pcm_core import BLOCK_BITS, NTSC_BIT_RATE, Block, crc16_video8
from timed import FIELD_PERIOD, NTSC_FIELD_BYTES, Field, write_field, write_header
from v8markers import detect
from video8pcm import _deduplicate, _keep_sequences


def decode_track(samples: np.ndarray, absolute_start: int, marker_sample: int,
                 sample_rate: float, phases: int) -> tuple[float, bytes, int] | None:
    """Lock to early CRC blocks, then decode the fixed 107-bit block grid."""
    spb = sample_rate / NTSC_BIT_RATE
    step = BLOCK_BITS * spb
    expected_zero = marker_sample + .00025 * sample_rate
    filtered = filter_samples(samples, sample_rate)
    candidates: list[Block] = []
    for probe_ms in (.8, 1.3):
        probe_end = min(len(samples),
                        round(marker_sample + probe_ms * .001 * sample_rate) - absolute_start)
        if probe_end < 512:
            continue
        found = scan_window(samples[:probe_end], absolute_start,
                            sample_rate, NTSC_BIT_RATE, phases)
        found = [b for b in found if b.address < 55 and
                 abs(b.sample - b.address * step - expected_zero) < .0002 * sample_rate]
        candidates = _keep_sequences(
            _deduplicate(found, sample_rate, NTSC_BIT_RATE),
            sample_rate, NTSC_BIT_RATE, 131)
        if len({b.address for b in candidates}) >= 2:
            break
    if len({b.address for b in candidates}) < 2:
        return None
    zero = float(np.median([b.sample - b.address * step for b in candidates]))
    addresses = np.arange(132)
    nominal = zero - absolute_start + addresses * step
    axis = np.arange(len(filtered))
    bit_cells = np.arange(104) * spb
    selected = [bytes(13) for _ in addresses]
    locked = np.zeros(132, dtype=bool)
    offsets = np.zeros(132, dtype=float)

    def try_phases(targets: np.ndarray, centers: np.ndarray,
                   shifts: np.ndarray) -> None:
        if not len(targets):
            return
        cells = (nominal[targets, None, None] + centers[:, None, None] +
                 shifts[None, :, None] + bit_cells[None, None, :])
        first = np.interp((cells + .25 * spb).ravel(), axis, filtered)
        second = np.interp((cells + .75 * spb).ravel(), axis, filtered)
        bits = (first * second < 0).astype(np.uint8).reshape(
            len(targets), len(shifts), 104)
        packed = np.packbits(bits, axis=2, bitorder='little')
        crc = np.full((len(targets), len(shifts)), CRC_BASE, dtype=np.uint16)
        for column in range(11):
            crc ^= CRC_CONTRIBUTIONS[column, packed[:, :, column]]
        recorded = packed[:, :, 11].astype(np.uint16) | \
            (packed[:, :, 12].astype(np.uint16) << 8)
        valid = (crc == recorded) & (packed[:, :, 0] == targets[:, None])
        order = np.argsort(np.abs(shifts))
        choices = order[np.argmax(valid[:, order], axis=1)]
        for row, address in enumerate(targets):
            if valid[row].any():
                selected[address] = packed[row, choices[row]].tobytes()
                locked[address] = True
                offsets[address] = centers[row] + shifts[choices[row]]
            else:
                selected[address] = packed[row, choices[row]].tobytes()

    # First check a tight grid. Nearby CRC locks predict the phase of gaps.
    try_phases(addresses, np.zeros(132), np.linspace(-.5, .5, 17) * spb)
    for width, count in ((.5, 17), (4.0, 129)):
        remaining = addresses[~locked]
        if not len(remaining):
            break
        known = addresses[locked]
        predicted = (np.interp(remaining, known, offsets[locked])
                     if len(known) else np.zeros(len(remaining)))
        try_phases(remaining, predicted,
                   np.linspace(-width, width, count) * spb)
    for address in addresses[~locked]:
        center = int(round(nominal[address]))
        left = max(0, center - 160)
        right = min(len(samples), center + round(104 * spb) + 160)
        recovered = [block for block in scan_window(
            samples[left:right], absolute_start + left, sample_rate,
            NTSC_BIT_RATE, phases) if block.address == address and
            abs(block.sample - (absolute_start + nominal[address])) < 8 * spb]
        if recovered:
            block = min(recovered, key=lambda b: abs(
                b.sample - (absolute_start + nominal[address])))
            selected[address] = (bytes((address,)) + block.words +
                                 block.crc_recorded.to_bytes(2, 'little'))
    # Exhaustive recovery is reserved for blocks that still fail CRC. Scan
    # every bit alignment and many clock phases over a whole-block margin.
    for address in addresses:
        raw = selected[address]
        if crc16_video8(raw[:11]) == int.from_bytes(raw[11:], 'little'):
            continue
        center = int(round(nominal[address]))
        margin = round(step)
        left = max(0, center - margin)
        right = min(len(samples), center + round(104 * spb) + margin)
        recovered = [block for block in scan_window(
            samples[left:right], absolute_start + left, sample_rate,
            NTSC_BIT_RATE, max(48, phases * 4))
            if block.address == address and
            abs(block.sample - (absolute_start + nominal[address])) < step / 2]
        if recovered:
            block = min(recovered, key=lambda b: abs(
                b.sample - (absolute_start + nominal[address])))
            selected[address] = (bytes((address,)) + block.words +
                                 block.crc_recorded.to_bytes(2, 'little'))
    good = sum(crc16_video8(raw[:11]) == int.from_bytes(raw[11:], 'little')
               for raw in selected)
    return zero, b''.join(selected), good


def demodulate_tracks(path: Path, output: Path, rate: float, start: float,
                      duration: float | None, phases: int) -> int:
    markers, preamble_time, marker_time, _ = detect(
        path, rate, 'ntsc', start, duration, 20.0)
    print(f'markers={len(markers)} preamble_scan={preamble_time:.3f}s '
          f'pilot_scan={marker_time-preamble_time:.3f}s', file=sys.stderr)
    period = FIELD_PERIOD * rate
    written = 0
    previous_peak = None
    previous_number = None
    with sf.SoundFile(path) as source, output.open('wb') as sink:
        write_header(sink, rate)
        for index, marker in enumerate(markers, 1):
            peak = marker['preamble_peak_sample']
            if previous_peak is None:
                number = 0
            else:
                number = previous_number + max(1, round((peak - previous_peak) / period))
                for missing in range(previous_number + 1, number):
                    expected = round(previous_peak + (missing - previous_number) * period +
                                     .00025 * rate)
                    write_field(sink, Field(missing, expected, False,
                                            bytes(NTSC_FIELD_BYTES)))
                    written += 1
            left = max(0, round(peak - .00025 * rate))
            right = min(source.frames, round(peak + .0031 * rate))
            source.seek(left)
            samples = source.read(right - left, dtype='float32')
            result = decode_track(samples, left, peak, rate, phases)
            if result is None:
                write_field(sink, Field(number, round(peak + .00025 * rate),
                                        False, bytes(NTSC_FIELD_BYTES)))
            else:
                zero, blocks, good = result
                write_field(sink, Field(number, max(0, round(zero)), True, blocks))
            written += 1
            previous_peak, previous_number = peak, number
            if index % 10 == 0 or index == len(markers):
                print(f'\rtrack decode {index}/{len(markers)}', end='',
                      file=sys.stderr, flush=True)
    print(file=sys.stderr)
    return written
