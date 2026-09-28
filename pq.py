"""NTSC Video8 P/Q cross-interleave erasure correction.

The 132 physical blocks of a field form one circular parity domain.  A CRC
failure erases all ten words in that block; parity equations recover words
only when every other term is known.  The original CRC bytes are retained so
the corrected data is never mistaken for a clean RF read.
"""
from __future__ import annotations

from dataclasses import dataclass

from pcm_core import crc16_video8

BLOCKS = 132
BLOCK_SIZE = 13
P_OFFSETS = (-59, -44, -29, -15, 15, 29, 44, 59)
P_WORDS = (1, 2, 3, 4, 6, 7, 8, 9)
Q_OFFSETS = (12, 24, 36, 48, 60, 72, 84, 96, 108)
Q_WORDS = (1, 2, 3, 4, 5, 6, 7, 8, 9)


def _checks() -> tuple[tuple[tuple[int, int], ...], ...]:
    result = []
    for address in range(BLOCKS):
        result.append(((address, 5),) + tuple(
            ((address + offset) % BLOCKS, word)
            for offset, word in zip(P_OFFSETS, P_WORDS)))
        result.append(((address, 0),) + tuple(
            ((address + offset) % BLOCKS, word)
            for offset, word in zip(Q_OFFSETS, Q_WORDS)))
    return tuple(result)


CHECKS = _checks()
BLOCK_CHECKS = tuple(frozenset(
    index for index, check in enumerate(CHECKS)
    if any(address == block for address, _ in check))
    for block in range(BLOCKS))


@dataclass(frozen=True)
class CorrectionStats:
    crc_bad: int
    recovered: int
    unresolved: int
    crc_collision_recovered: int
    parity_failures: int


def _solve(values: list[list[int]], known: list[list[bool]]) -> None:
    while True:
        progress = False
        for check in CHECKS:
            missing = [(address, word) for address, word in check
                       if not known[address][word]]
            if len(missing) != 1:
                continue
            value = 0
            for address, word in check:
                if known[address][word]:
                    value ^= values[address][word]
            address, word = missing[0]
            values[address][word] = value
            known[address][word] = True
            progress = True
        if not progress:
            break


def _failed_checks(values: list[list[int]],
                   known: list[list[bool]]) -> list[int]:
    failed = []
    for index, check in enumerate(CHECKS):
        if not all(known[address][word] for address, word in check):
            continue
        residual = 0
        for address, word in check:
            residual ^= values[address][word]
        if residual:
            failed.append(index)
    return failed


def correct_field(blocks: bytes) -> tuple[bytes, tuple[bool, ...], CorrectionStats]:
    """Recover CRC-bad words and unambiguous CRC-valid parity collisions.

    Returns repaired physical blocks, a per-block usability mask, and counts.
    A repaired block keeps its original recorded CRC; callers must use the
    usability mask instead of rechecking that CRC.
    """
    if len(blocks) != BLOCKS * BLOCK_SIZE:
        raise ValueError("NTSC field must contain 132 physical blocks")
    raw = [blocks[i * BLOCK_SIZE:(i + 1) * BLOCK_SIZE]
           for i in range(BLOCKS)]
    crc_good = [block[0] == address
                and crc16_video8(block[:11]) == int.from_bytes(block[11:], "little")
                for address, block in enumerate(raw)]
    values = [list(block[1:11]) for block in raw]
    known = [[good] * 10 for good in crc_good]
    _solve(values, known)

    # A valid CRC can still select the wrong hard decisions.  Attempt a
    # correction only if every nonzero parity check identifies one unique
    # block, and accept it only when the complete field becomes consistent.
    collision_fixed = 0
    failed = _failed_checks(values, known)
    if len(failed) >= 4 and all(all(row) for row in known):
        suspects = [address for address in range(BLOCKS)
                    if all(index in BLOCK_CHECKS[address] for index in failed)]
        if len(suspects) == 1 and crc_good[suspects[0]]:
            address = suspects[0]
            candidate_values = [row.copy() for row in values]
            candidate_known = [row.copy() for row in known]
            candidate_known[address] = [False] * 10
            _solve(candidate_values, candidate_known)
            if all(all(row) for row in candidate_known) and not _failed_checks(
                    candidate_values, candidate_known):
                values, known = candidate_values, candidate_known
                collision_fixed = 1
                failed = []

    # A contradictory parity equation invalidates any recovered word in it.
    inconsistent = {address for index in failed for address, _ in CHECKS[index]}
    usable = tuple(all(known[address]) and
                   (crc_good[address] or address not in inconsistent)
                   for address in range(BLOCKS))
    recovered = sum(not good and usable[address]
                    for address, good in enumerate(crc_good))
    result = b"".join(bytes((address,)) + bytes(values[address]) + raw[address][11:]
                      for address in range(BLOCKS))
    stats = CorrectionStats(
        crc_bad=crc_good.count(False),
        recovered=recovered,
        unresolved=crc_good.count(False) - recovered,
        crc_collision_recovered=collision_fixed,
        parity_failures=len(failed),
    )
    return result, usable, stats
