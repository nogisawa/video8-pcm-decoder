#!/usr/bin/env python3
"""Check CRCs in timestamped Video8 PCM fields."""
import argparse
from pathlib import Path
import sys

from timed import read_fields
from video8pcm import crc16_video8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", type=Path)
    p.add_argument("--show-bad", action="store_true")
    a = p.parse_args()
    good = bad = missing = 0
    with a.input.open("rb") as source:
        rate, fields = read_fields(source)
        for field in fields:
            if not field.present:
                missing += 1
                continue
            for address in range(132):
                raw = field.blocks[address * 13:(address + 1) * 13]
                if crc16_video8(raw[:11]) == int.from_bytes(raw[11:13], "little"):
                    good += 1
                else:
                    bad += 1
                    if a.show_bad:
                        print(f"field={field.number} address={address} "
                              f"rf_time={(field.sample / rate):.9f}s", file=sys.stderr)
    print(f"crc_ok={good} crc_bad={bad} missing_fields={missing}")
    return 1 if bad or missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
