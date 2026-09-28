"""NumPy/SciPy implementation of the RF demodulation kernels."""
from __future__ import annotations
import numpy as np
from functools import lru_cache
from scipy.signal import butter, sosfiltfilt
from pcm_core import Block, CRC_TABLE, PAYLOAD_BITS, crc16_video8


def _crc_contributions() -> tuple[np.uint16, np.ndarray]:
    # CRC is linear in the input bits. Precompute each byte's contribution
    # after its remaining bytes, keeping the all-zero preset as a constant.
    table = np.empty((11, 256), dtype=np.uint16)
    for column in range(11):
        for value in range(256):
            crc = 0
            for index in range(11):
                byte = value if index == column else 0
                crc = (crc >> 8) ^ CRC_TABLE[(crc ^ byte) & 255]
            table[column, value] = crc
    return np.uint16(crc16_video8(bytes(11))), table


CRC_BASE, CRC_CONTRIBUTIONS = _crc_contributions()

def _demodulate(signal: np.ndarray, sample_rate: float, bit_rate: float,
                phase: float) -> np.ndarray:
    """Hard-decode biphase mark by comparing the two half-bit polarities."""
    samples_per_bit = sample_rate / bit_rate
    count = int((len(signal) - phase) / samples_per_bit) - 1
    if count <= 0:
        return np.empty(0, dtype=np.uint8)
    cells = phase + np.arange(count) * samples_per_bit
    first_at = cells + .25 * samples_per_bit
    second_at = cells + .75 * samples_per_bit
    first_idx = first_at.astype(np.intp)
    second_idx = second_at.astype(np.intp)
    first_frac = first_at - first_idx
    second_frac = second_at - second_idx
    first = signal[first_idx] + first_frac * (signal[first_idx + 1] - signal[first_idx])
    second = signal[second_idx] + second_frac * (signal[second_idx + 1] - signal[second_idx])
    # A mid-cell transition denotes one.  Boundary transitions cancel out.
    return (first * second < 0).astype(np.uint8)


@lru_cache(maxsize=64)
def _filter_coefficients(sample_rate: float) -> np.ndarray:
    return butter(3, (1.5e6, min(9.0e6, sample_rate * .47)),
                  btype="bandpass", fs=sample_rate, output="sos")


def filter_samples(samples: np.ndarray, sample_rate: float) -> np.ndarray:
    """Band-pass RF before phase search and hard block recovery."""
    return sosfiltfilt(_filter_coefficients(sample_rate), samples).astype(np.float32)


def scan_window(samples: np.ndarray, absolute_start: int, sample_rate: float,
                bit_rate: float, phases: int = 12) -> list[Block]:
    """Find CRC-valid blocks without assuming a sync pattern or bit phase."""
    # Video RF ends where PCM begins, but this band-pass also removes DC and
    # the low-frequency FM audio carrier.
    filtered = filter_samples(samples, sample_rate)
    samples_per_bit = sample_rate / bit_rate
    found: dict[tuple[int, int], Block] = {}

    for phase in np.linspace(0, samples_per_bit, phases, endpoint=False):
        bits = _demodulate(filtered, sample_rate, bit_rate, phase)
        count = len(bits) - PAYLOAD_BITS + 1
        if count <= 0:
            continue
        # Pack each of the eight possible bit alignments once. A 13-byte
        # sliding view then replaces packing every 104-bit candidate window.
        matches: list[tuple[int, bytes]] = []
        for offset in range(min(8, count)):
            rows = (count - 1 - offset) // 8 + 1
            packed = np.packbits(bits[offset:], bitorder="little")
            raw = np.lib.stride_tricks.sliding_window_view(packed, 13)[:rows]
            crc = np.full(rows, CRC_BASE, dtype=np.uint16)
            for column in range(11):
                crc ^= CRC_CONTRIBUTIONS[column, raw[:, column]]
            recorded = raw[:, 11].astype(np.uint16) | \
                (raw[:, 12].astype(np.uint16) << 8)
            for row in np.flatnonzero(crc == recorded):
                matches.append((offset + 8 * int(row), raw[row].tobytes()))
        # Keep the original bit-order collision policy across phase trials.
        for bit_at, payload in sorted(matches):
            sample = absolute_start + phase + bit_at * samples_per_bit
            recorded = payload[11] | (payload[12] << 8)
            block = Block(sample, payload[0], payload[1:11], recorded,
                          crc16_video8(payload[:11]))
            key = (round(block.sample / samples_per_bit), block.address)
            found[key] = block

    return sorted(found.values(), key=lambda item: item.sample)


def hard_block_at(filtered: np.ndarray, address_sample: float,
                  sample_rate: float, bit_rate: float) -> bytes:
    """Decode a missing block and retry nearby phases after a CRC failure."""
    samples_per_bit = sample_rate / bit_rate
    axis = np.arange(len(filtered))
    best_raw = bytes(13)
    best_score = -1.0
    for adjustment in np.linspace(-.55, .55, 19) * samples_per_bit:
        cells = address_sample + adjustment + np.arange(PAYLOAD_BITS) * samples_per_bit
        first = np.interp(cells + .25 * samples_per_bit, axis, filtered)
        second = np.interp(cells + .75 * samples_per_bit, axis, filtered)
        bits = (first * second < 0).astype(np.uint8)
        raw = np.packbits(bits.reshape(13, 8), axis=1,
                          bitorder="little").ravel().tobytes()
        recorded = raw[11] | (raw[12] << 8)
        if crc16_video8(raw[:11]) == recorded:
            return raw
        score = float(np.mean(np.minimum(np.abs(first), np.abs(second))))
        if score > best_score:
            best_score, best_raw = score, raw
    return best_raw
