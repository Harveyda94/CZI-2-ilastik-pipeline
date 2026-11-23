"""Tile geometry and the ordered prefetch stream.

`Tiler.tasks` defines the one canonical ordering used everywhere: pages in TZC order,
tiles row-major within a page. That is the order TIFF stores tiles in, and it is what
lets HDF5 resume get away with a single counter instead of a per-tile bitmap.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


def overview_level(w, h, max_px):
    """Smallest power-of-two decimation level that fits the longer side in `max_px`."""
    longest = max(w, h)
    return max(0, int(np.ceil(np.log2(longest / max_px)))) if longest > max_px else 0


@dataclass
class Tiler:
    tile: Sequence[int]                 # (height, width)
    sel_t: Sequence[int]
    sel_z: Sequence[int]
    sel_c: Sequence[int]
    samples: int = 1

    def level_shape(self, rect, level=0):
        step = 1 << level
        h, w = max(1, rect[3] // step), max(1, rect[2] // step)
        base = (len(self.sel_t), len(self.sel_z), len(self.sel_c), h, w)
        return base + ((self.samples,) if self.samples == 3 else ())

    def n_tiles(self, rect, level=0):
        th, tw = self.tile
        _, _, _, h, w = self.level_shape(rect, level)[:5]
        return (len(self.sel_t) * len(self.sel_z) * len(self.sel_c)
                * math.ceil(h / th) * math.ceil(w / tw))

    def tasks(self, rect, level=0):
        """(t, z, c, x, y, w, h) in page order, then tile row-major.

        x/y/w/h are full-resolution coordinates; a level-l consumer decimates by 2**l.
        """
        th, tw = self.tile
        step = 1 << level
        x_off, y_off, W, H = rect
        wl, hl = max(1, W // step), max(1, H // step)
        for t in self.sel_t:
            for z in self.sel_z:
                for c in self.sel_c:
                    for ty in range(0, hl, th):
                        for tx in range(0, wl, tw):
                            yield (t, z, c,
                                   x_off + tx * step, y_off + ty * step,
                                   min(tw, wl - tx) * step, min(th, hl - ty) * step)

    def fit(self, arr):
        """Crop or zero-pad to exactly (tile_h, tile_w[, samples])."""
        th, tw = self.tile
        h, w = arr.shape[:2]
        if h > th or w > tw:
            arr = arr[:th, :tw]
            h, w = arr.shape[:2]
        if h < th or w < tw:
            pad = [(0, th - h), (0, tw - w)] + [(0, 0)] * (arr.ndim - 2)
            arr = np.pad(arr, pad)
        return arr

    def page_index(self, ti, zi, ci):
        return (ti * len(self.sel_z) + zi) * len(self.sel_c) + ci

    def index_maps(self):
        return ({v: i for i, v in enumerate(self.sel_t)},
                {v: i for i, v in enumerate(self.sel_z)},
                {v: i for i, v in enumerate(self.sel_c)})


def stream(tasks: Iterable, work, executor, prefetch: int, progress=None):
    """Run `work` over `tasks` `prefetch` deep, yielding results in submission order.

    Order preservation is the whole point: the writer must never see an out-of-order
    tile, but reads should still overlap compression.
    """
    it, pending = iter(tasks), deque()
    try:
        for _ in range(prefetch):
            pending.append(executor.submit(work, next(it)))
    except StopIteration:
        pass
    while pending:
        fut = pending.popleft()
        try:
            pending.append(executor.submit(work, next(it)))
        except StopIteration:
            pass
        yield fut.result()
        if progress is not None:
            progress.update(1)
