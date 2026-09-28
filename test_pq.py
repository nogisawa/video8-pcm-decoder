"""P/Q correction tests with independently generated parity-valid fields."""
from __future__ import annotations

import random
import unittest

from pcm_core import crc16_video8
from pq import P_OFFSETS, P_WORDS, Q_OFFSETS, Q_WORDS, correct_field


def encoded_field(seed=27):
    randomizer = random.Random(seed)
    words = [[randomizer.randrange(256) for _ in range(10)]
             for _ in range(132)]
    for address in range(132):
        value = 0
        for offset, word in zip(P_OFFSETS, P_WORDS):
            value ^= words[(address + offset) % 132][word]
        words[address][5] = value
    for address in range(132):
        value = 0
        for offset, word in zip(Q_OFFSETS, Q_WORDS):
            value ^= words[(address + offset) % 132][word]
        words[address][0] = value
    blocks = []
    for address, row in enumerate(words):
        payload = bytes((address, *row))
        blocks.append(payload + crc16_video8(payload).to_bytes(2, "little"))
    return b"".join(blocks)


class PQTests(unittest.TestCase):
    def test_clean_field_is_unchanged(self):
        source = encoded_field()
        result, usable, stats = correct_field(source)
        self.assertEqual(result, source)
        self.assertTrue(all(usable))
        self.assertEqual(stats.crc_bad, 0)
        self.assertEqual(stats.parity_failures, 0)

    def test_one_erasure_at_field_boundary(self):
        source = encoded_field()
        damaged = bytearray(source)
        for address in (0, 131):
            damaged[address * 13 + 2] ^= 0x55
            damaged[address * 13 + 8] ^= 0x93
        result, usable, stats = correct_field(bytes(damaged))
        self.assertEqual(result, source)
        self.assertTrue(all(usable))
        self.assertEqual(stats.recovered, 2)

    def test_crc_valid_wrong_payload_is_recovered(self):
        source = encoded_field()
        damaged = bytearray(source)
        at = 24 * 13
        for offset, mask in ((8, 0x22), (9, 0x10), (10, 0x34)):
            damaged[at + offset] ^= mask
        damaged[at + 11:at + 13] = crc16_video8(damaged[at:at + 11]).to_bytes(2, "little")
        result, usable, stats = correct_field(bytes(damaged))
        self.assertEqual(result[at:at + 11], source[at:at + 11])
        self.assertEqual(result[at + 11:at + 13], damaged[at + 11:at + 13])
        self.assertTrue(all(usable))
        self.assertEqual(stats.crc_collision_recovered, 1)

    def test_unrecoverable_field_stays_unusable(self):
        source = bytearray(encoded_field())
        for address in range(132):
            source[address * 13 + 2] ^= 1
        _, usable, stats = correct_field(bytes(source))
        self.assertFalse(any(usable))
        self.assertEqual(stats.unresolved, 132)


if __name__ == "__main__":
    unittest.main()
