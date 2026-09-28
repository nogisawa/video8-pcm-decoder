"""Shared Video8 constants, block representation and CRC."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

NTSC_BIT_RATE = 15_734.264 * 368
PAL_BIT_RATE = 15_625.0 * 368
BLOCK_BITS = 107
PAYLOAD_BITS = 104              # address + 8 data + P + Q + CRC16


def _crc_table() -> tuple[int, ...]:
    # Reflected form of x^16 + x^12 + x^5 + 1.  Video8 transmits LSB first.
    result = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0x8408 if crc & 1 else 0)
        result.append(crc)
    return tuple(result)


CRC_TABLE = _crc_table()
CRC_TABLE_ARRAY = np.asarray(CRC_TABLE, dtype=np.uint16)


def crc16_video8(data: bytes) -> int:
    """CRC-16/CCITT, reflected, preset to all ones (no final xor)."""
    crc = 0xFFFF
    for byte in data:
        crc = (crc >> 8) ^ CRC_TABLE[(crc ^ byte) & 0xFF]
    return crc


@dataclass(frozen=True)
class Block:
    sample: float
    address: int
    words: bytes                # physical order: Q,W0,W1,W2,W3,P,W4,W5,W6,W7
    crc_recorded: int
    crc_calculated: int

    @property
    def crc_ok(self) -> bool:
        return self.crc_recorded == self.crc_calculated


