"""Spatial chunking utilities for parallel processing."""

from collections.abc import Iterator
from dataclasses import dataclass


def _split_interval(start: int, end: int, max_size: int) -> Iterator[tuple[int, int]]:
    """Split interval using binary tree, respecting zarr chunk boundaries."""
    n = end - start
    if n > max_size:
        # Find split point that aligns with zarr chunk boundaries
        split_at = ((start + n // 2) // max_size) * max_size
        if split_at <= start:
            split_at = start + max_size
        if split_at >= end:
            yield start, end
        else:
            yield from _split_interval(start, split_at, max_size)
            yield from _split_interval(split_at, end, max_size)
    else:
        yield start, end


@dataclass(frozen=True, order=True)
class SpatialChunk:
    """2D spatial chunk for parallel processing."""

    y_start: int
    y_end: int
    x_start: int
    x_end: int

    def __str__(self) -> str:
        return f"{self.y_start}:{self.y_end}, {self.x_start}:{self.x_end}"

    def immediate_sub_chunks(self, y_size: int, x_size: int) -> list["SpatialChunk"]:  # noqa: PLR0912
        """Subdivide chunk into at most four sub-chunks for parallel processing."""
        y_needs_split = (self.y_end - self.y_start) > y_size
        x_needs_split = (self.x_end - self.x_start) > x_size

        if not y_needs_split and not x_needs_split:
            return [self]

        sub_chunks = []

        if y_needs_split:
            y_mid = ((self.y_start + (self.y_end - self.y_start) // 2) // y_size) * y_size
            if y_mid <= self.y_start:
                y_mid = self.y_start + y_size
            if y_mid >= self.y_end:
                y_ranges = [(self.y_start, self.y_end)]
            else:
                y_ranges = [(self.y_start, y_mid), (y_mid, self.y_end)]
        else:
            y_ranges = [(self.y_start, self.y_end)]

        if x_needs_split:
            x_mid = ((self.x_start + (self.x_end - self.x_start) // 2) // x_size) * x_size
            if x_mid <= self.x_start:
                x_mid = self.x_start + x_size
            if x_mid >= self.x_end:
                x_ranges = [(self.x_start, self.x_end)]
            else:
                x_ranges = [(self.x_start, x_mid), (x_mid, self.x_end)]
        else:
            x_ranges = [(self.x_start, self.x_end)]

        for y_start, y_end in y_ranges:
            for x_start, x_end in x_ranges:
                sub_chunks.append(SpatialChunk(y_start, y_end, x_start, x_end))

        return sub_chunks
