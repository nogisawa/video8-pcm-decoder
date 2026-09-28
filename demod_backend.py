"""Boundary between RF orchestration and CPU/GPU signal kernels.

The public methods use NumPy arrays at the boundary today. A future GPU
backend may upload a batch of windows internally and return the same Block
and physical-byte results; container and audio code need no changes.
"""
from __future__ import annotations

from typing import Protocol
import numpy as np

from pcm_core import Block
import cpu_backend


class DemodBackend(Protocol):
    def scan_windows(self, windows: list[tuple[np.ndarray, int]],
                     sample_rate: float, bit_rate: float,
                     phases: int) -> list[Block]: ...

    def scan_window(self, samples: np.ndarray, absolute_start: int,
                    sample_rate: float, bit_rate: float,
                    phases: int = 12) -> list[Block]: ...

    def filter_samples(self, samples: np.ndarray,
                       sample_rate: float) -> np.ndarray: ...

    def hard_block_at(self, filtered: np.ndarray, address_sample: float,
                      sample_rate: float, bit_rate: float) -> bytes: ...


class CpuBackend:
    def scan_windows(self, windows: list[tuple[np.ndarray, int]],
                     sample_rate: float, bit_rate: float,
                     phases: int) -> list[Block]:
        return [block for samples, start in windows
                for block in cpu_backend.scan_window(
                    samples, start, sample_rate, bit_rate, phases)]

    scan_window = staticmethod(cpu_backend.scan_window)
    filter_samples = staticmethod(cpu_backend.filter_samples)
    hard_block_at = staticmethod(cpu_backend.hard_block_at)


CPU_BACKEND: DemodBackend = CpuBackend()
