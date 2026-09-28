"""Versioned Video8 field container with RF sample timestamps."""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import BinaryIO, Iterator

MAGIC = b"V8PCMT01"
HEADER = struct.Struct("<8sIdI")  # magic, version, ADC samples/s, system
RECORD = struct.Struct("<QQI")     # field number, RF sample of address 0, flags
NTSC_FIELD_BYTES = 132 * 13
FIELD_PERIOD = 525 / (2 * 15_734.264)
PRESENT = 1


@dataclass(frozen=True)
class Field:
    number: int
    sample: int
    present: bool
    blocks: bytes


def write_header(stream: BinaryIO, sample_rate: float) -> None:
    if sample_rate <= 0:
        raise ValueError("sample rate must be positive")
    stream.write(HEADER.pack(MAGIC, 1, sample_rate, 0))


def write_field(stream: BinaryIO, field: Field) -> None:
    if len(field.blocks) != NTSC_FIELD_BYTES:
        raise ValueError("NTSC field must contain 132 physical blocks")
    stream.write(RECORD.pack(field.number, field.sample,
                             PRESENT if field.present else 0))
    stream.write(field.blocks)


def read_fields(stream: BinaryIO) -> tuple[float, Iterator[Field]]:
    header = stream.read(HEADER.size)
    if len(header) != HEADER.size:
        raise ValueError("truncated container header")
    magic, version, sample_rate, system = HEADER.unpack(header)
    if magic != MAGIC or version != 1 or system != 0 or sample_rate <= 0:
        raise ValueError("unsupported Video8 timed container")

    def records() -> Iterator[Field]:
        previous = -1
        while record := stream.read(RECORD.size):
            if len(record) != RECORD.size:
                raise ValueError("truncated field record")
            number, sample, flags = RECORD.unpack(record)
            if number <= previous or flags & ~PRESENT:
                raise ValueError("invalid field order or flags")
            blocks = stream.read(NTSC_FIELD_BYTES)
            if len(blocks) != NTSC_FIELD_BYTES:
                raise ValueError("truncated field blocks")
            previous = number
            yield Field(number, sample, bool(flags & PRESENT), blocks)

    return sample_rate, records()
