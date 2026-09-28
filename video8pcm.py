#!/usr/bin/env python3
"""Experimental Video8 PCM block extractor for CXADC FLAC captures.

This first stage deliberately stops at CRC checked, still-interleaved blocks.
It does not yet apply P/Q error correction or turn the data into audio.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, firwin2


from pcm_core import (Block, BLOCK_BITS, NTSC_BIT_RATE, PAL_BIT_RATE,
                      PAYLOAD_BITS, CRC_TABLE, CRC_TABLE_ARRAY, crc16_video8)
from cpu_backend import scan_window, hard_block_at as _hard_block_at
from demod_backend import CPU_BACKEND, DemodBackend


def extract_raw(path: Path, true_sample_rate: float, system: str,
                start: float, duration: float | None, window_ms: float,
                phases: int, backend: DemodBackend = CPU_BACKEND) -> list[bytes]:
    """Return every aligned block, including blocks whose CRC is bad."""
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    block_count = 132 if system == "ntsc" else 157
    valid = extract(path, true_sample_rate, system, start, duration,
                    window_ms, phases, backend)
    fields = _split_fields(valid, true_sample_rate)
    step = BLOCK_BITS * true_sample_rate / bit_rate
    raw_blocks: list[bytes] = []
    with sf.SoundFile(path) as source:
        for field in fields:
            estimates = np.array([b.sample - b.address * step for b in field])
            address_zero = float(np.median(estimates))
            pad = 128
            read_start = max(0, int(address_zero) - pad)
            read_end = min(source.frames,
                           int(address_zero + block_count * step) + pad)
            source.seek(read_start)
            samples = source.read(read_end - read_start, dtype="float32")
            filtered = backend.filter_samples(samples, true_sample_rate)
            known = {b.address: bytes((b.address,)) + b.words +
                     bytes((b.crc_recorded & 0xff, b.crc_recorded >> 8))
                     for b in field}
            for address in range(block_count):
                if address in known:
                    raw_blocks.append(known[address])
                else:
                    relative = address_zero + address * step - read_start
                    raw_blocks.append(backend.hard_block_at(
                        filtered, relative, true_sample_rate, bit_rate))
    return raw_blocks


def _deduplicate(blocks: list[Block], sample_rate: float,
                 bit_rate: float) -> list[Block]:
    tolerance = sample_rate / bit_rate * 2
    output: list[Block] = []
    for block in sorted(blocks, key=lambda item: item.sample):
        if output and block.address == output[-1].address and \
                abs(block.sample - output[-1].sample) < tolerance:
            continue
        output.append(block)
    return output


def _keep_sequences(blocks: list[Block], sample_rate: float, bit_rate: float,
                    max_address: int) -> list[Block]:
    """Discard random CRC collisions which do not join an address run."""
    blocks = [item for item in blocks if item.address <= max_address]
    step = BLOCK_BITS * sample_rate / bit_rate
    keep: set[int] = set()
    for left, a in enumerate(blocks):
        for right in range(left + 1, min(left + 20, len(blocks))):
            b = blocks[right]
            address_gap = b.address - a.address
            if not 1 <= address_gap <= 8:
                continue
            if abs((b.sample - a.sample) - address_gap * step) <= 3:
                keep.add(left)
                keep.add(right)
    return [item for index, item in enumerate(blocks) if index in keep]


def extract(path: Path, true_sample_rate: float, system: str, start: float,
            duration: float | None, window_ms: float, phases: int,
            backend: DemodBackend = CPU_BACKEND) -> list[Block]:
    bit_rate = NTSC_BIT_RATE if system == "ntsc" else PAL_BIT_RATE
    all_blocks: list[Block] = []
    with sf.SoundFile(path) as source:
        first = max(0, round(start * true_sample_rate))
        last = source.frames if duration is None else min(
            source.frames, first + round(duration * true_sample_rate))
        window = max(round(window_ms * 1e-3 * true_sample_rate), 4096)
        overlap = round(0.6e-3 * true_sample_rate)
        position = first
        pending: list[tuple[np.ndarray, int]] = []
        regular_file = Path(path).is_file()
        shown_percent = -1

        def show_progress(done: int) -> None:
            nonlocal shown_percent
            if regular_file and last > first:
                percent = min(100, max(0, int((done - first) * 100 / (last - first))))
                if percent != shown_percent:
                    print(f"\rscan {percent:3d}%  "
                          f"{min(done, last) / true_sample_rate:8.3f} s", end="",
                          file=sys.stderr, flush=True)
                    shown_percent = percent
            else:
                print(f"\rscan {done / true_sample_rate:8.3f} s", end="",
                      file=sys.stderr, flush=True)

        while position < last:
            source.seek(position)
            samples = source.read(min(window, last - position), dtype="float32")
            if len(samples) < 512:
                break
            pending.append((samples, position))
            if len(pending) >= 16:
                all_blocks.extend(backend.scan_windows(
                    pending, true_sample_rate, bit_rate, phases))
                show_progress(min(last, pending[-1][1] + len(pending[-1][0])))
                pending.clear()
            position += max(1, window - overlap)
        if pending:
            all_blocks.extend(backend.scan_windows(
                pending, true_sample_rate, bit_rate, phases))
            show_progress(min(last, pending[-1][1] + len(pending[-1][0])))
        elif last <= first:
            show_progress(last)
    print(file=sys.stderr)
    blocks = _deduplicate(all_blocks, true_sample_rate, bit_rate)
    return _keep_sequences(blocks, true_sample_rate, bit_rate,
                           131 if system == "ntsc" else 156)


def write_jsonl(path: Path, blocks: list[Block], sample_rate: float) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for block in blocks:
            row = {
                "sample": round(block.sample, 3),
                "time": block.sample / sample_rate,
                "address": block.address,
                "data": (block.words[1:5] + block.words[6:10]).hex(),
                "p": block.words[5],
                "q": block.words[0],
                "crc_recorded": f"{block.crc_recorded:04x}",
                "crc_calculated": f"{block.crc_calculated:04x}",
                "crc_ok": block.crc_ok,
            }
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def expand_pcm_byte(value: int, midpoint: bool = True) -> int:
    """Expand Video8's signed nonlinear 8-bit code to signed 10-bit PCM."""
    signed = value if value < 128 else value - 256
    if -16 <= signed <= 15:
        return signed
    if signed > 0:
        if signed < 40:
            base, width = (signed - 8) * 2, 2
        elif signed < 104:
            base, width = (signed - 24) * 4, 4
        else:
            base, width = (signed - 64) * 8, 8
        return base + (width // 2 if midpoint else 0)
    # Negative code intervals are offset by one because two's complement is
    # asymmetric.  Mirroring around -0.5 selects the interval centre.
    return -expand_pcm_byte((-signed) - 1, midpoint)


def _split_fields(blocks: list[Block], sample_rate: float) -> list[list[Block]]:
    """Split at the long non-PCM gap between successive helical tracks."""
    if not blocks:
        return []
    fields: list[list[Block]] = [[blocks[0]]]
    for block in blocks[1:]:
        if block.sample - fields[-1][-1].sample > sample_rate * .005:
            fields.append([])
        fields[-1].append(block)
    return fields


def _ntsc_field_audio(field: list[Block]) -> np.ndarray:
    """Undo the 3 x 44-block shuffle shown in JIS C 5583 figure 28."""
    audio = np.full((525, 2), np.nan, dtype=np.float64)
    bases = (0, 63, 129, 195, 261, 327, 393, 459)
    # Physical tape order in figure 28 is Q,W0,W1,W2,W3,P,W4,W5,W6,W7.
    physical_word = (1, 2, 3, 4, 6, 7, 8, 9)
    for block in field:
        address = block.address
        if not 0 <= address < 132:
            continue
        group, column = divmod(address, 44)
        for word_index, (base, physical_index) in enumerate(
                zip(bases, physical_word)):
            if word_index == 0:
                if column < 2:       # ID0..ID5, not audio
                    continue
                pair = (column - 2) // 2
            else:
                pair = column // 2
            sample_index = base + group + 3 * pair
            channel = column & 1
            audio[sample_index, channel] = expand_pcm_byte(
                block.words[physical_index])
    return audio


def _conceal_missing(audio: np.ndarray) -> np.ndarray:
    for channel in range(audio.shape[1]):
        values = audio[:, channel]
        good = np.flatnonzero(~np.isnan(values))
        if len(good) == 0:
            values[:] = 0
        elif len(good) == 1:
            values[:] = values[good[0]]
        else:
            missing = np.flatnonzero(np.isnan(values))
            values[missing] = np.interp(missing, good, values[good])
    return audio


def noise_reduction_expand(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Approximate the Video8 2:1 playback expander.

    The standard specifies the static 2:1 law, timing and frequency response,
    but an exact implementation is an analogue control-loop model.  This uses
    the specified 3 ms attack, 15 ms hold and 40 ms recovery plus the inverse
    of the compressor's measured high-frequency boost.
    """
    normalized = audio.astype(np.float64) / 512.0

    # JIS C 5583 table 11: compressor response at reference input.  Apply its
    # inverse before the level-controlled expansion to tame boosted HF noise.
    frequencies = np.array((0, 50, 100, 200, 400, 700, 1000, 2000,
                            4000, 7000, 10000, 14000, sample_rate / 2))
    boost_db = np.array((0, 0, 0, 0, 0, .1, .3, 1.2,
                         2.7, 4.1, 4.8, 5.3, 5.3))
    taps = firwin2(257, frequencies / (sample_rate / 2),
                   10 ** (-boost_db / 20))
    equalized = np.column_stack([
        fftconvolve(normalized[:, channel], taps, mode="same")
        for channel in range(normalized.shape[1])
    ])

    attack = np.exp(-1.0 / (sample_rate * .003))
    recovery = np.exp(-1.0 / (sample_rate * .040))
    hold_samples = round(sample_rate * .015)
    output = np.empty_like(equalized)
    for channel in range(equalized.shape[1]):
        envelope = 0.0
        hold = 0
        for index, value in enumerate(equalized[:, channel]):
            level = abs(value)
            if level >= envelope:
                envelope = attack * envelope + (1.0 - attack) * level
                hold = hold_samples
            elif hold:
                hold -= 1
            else:
                envelope = recovery * envelope + (1.0 - recovery) * level
            # For a steady encoded amplitude E, 2:1 expansion is E * E.
            output[index, channel] = value * min(envelope, 1.0)
    return output * 512.0


def write_wav(path: Path, blocks: list[Block], capture_rate: float,
              system: str, noise_reduction: bool = False) -> tuple[int, int]:
    if system != "ntsc":
        raise NotImplementedError("PAL shuffle is not implemented yet")
    fields = _split_fields(blocks, capture_rate)
    decoded = [_ntsc_field_audio(field) for field in fields
               if len({block.address for block in field}) >= 2]
    if not decoded:
        raise ValueError("no complete PCM fields found")
    audio = _conceal_missing(np.concatenate(decoded, axis=0))
    if noise_reduction:
        audio = noise_reduction_expand(audio, 2 * 15_734.264)
    # Scale signed 10-bit to most of signed 16-bit.  The WAV container stores
    # an integer rate; NTSC's exact 2*fH is approximately 31468.528 Hz.
    pcm16 = np.clip(audio * 64, -32768, 32767).astype(np.int16)
    wav_rate = round(2 * 15_734.264)
    sf.write(path, pcm16, wav_rate, subtype="PCM_16")
    return len(fields), int(np.isnan(np.concatenate(decoded)).sum())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("-o", "--output", type=Path, default=Path("blocks.jsonl"))
    parser.add_argument("--wav", type=Path,
                        help="also write deinterleaved stereo PCM audio")
    parser.add_argument("--noise-reduction", action="store_true",
                        help="apply the experimental 2:1 expander")
    parser.add_argument("--sample-rate", type=float, default=28_636_360,
                        help="actual ADC rate; FLAC metadata is commonly /1000")
    parser.add_argument("--system", choices=("ntsc", "pal"), default="ntsc")
    parser.add_argument("--start", type=float, default=0.0,
                        help="actual RF time in seconds")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--window-ms", type=float, default=8.0)
    parser.add_argument("--phases", type=int, default=12)
    args = parser.parse_args()

    blocks = extract(args.input, args.sample_rate, args.system, args.start,
                     args.duration, args.window_ms, args.phases)
    write_jsonl(args.output, blocks, args.sample_rate)
    print(f"wrote {len(blocks)} CRC-valid blocks to {args.output}")
    if args.wav:
        fields, concealed = write_wav(args.wav, blocks, args.sample_rate,
                                      args.system, args.noise_reduction)
        print(f"wrote {fields} fields to {args.wav}; "
              f"concealed {concealed} missing channel samples")
    return 0 if blocks else 2


if __name__ == "__main__":
    raise SystemExit(main())
