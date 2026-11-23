"""Mosaic tile geometry and the memory-bounded compositor.

The compositor is shared by `refine` and `bigstitcher`: both produce a
{mosaic_index: (dy, dx)} correction dict, and everything downstream is identical.
"""

from __future__ import annotations

import threading
from concurrent.futures import Future

import numpy as np

from ..compute import cast_to
from ..geometry import overview_level


def bbox_xywh(b):
    return (int(b.x), int(b.y), int(b.w), int(b.h))


def mosaic_geometry(source, scene_id, ref_plane):
    """{m: (x, y, w, h)} in pylibCZIrw coordinates, for the reference plane.

    aicspylibczi and pylibCZIrw express mosaic coordinates in the same pixel space, but
    the offset between them is measured rather than assumed.
    """
    czi = source.aics()
    if not czi.is_mosaic():
        raise RuntimeError('This CZI is not a mosaic; use stitch_mode="metadata".')

    dshape = czi.get_dims_shape()[0]
    m0, m1 = dshape.get("M", (0, 1))
    kw = dict(ref_plane)
    if scene_id is not None and "S" in dshape:
        kw["S"] = scene_id

    rects = {}
    for m in range(m0, m1):
        try:
            rects[m] = bbox_xywh(czi.get_mosaic_tile_bounding_box(M=m, **kw))
        except Exception:
            continue                       # tile absent from this scene/plane
    if not rects:
        raise RuntimeError(f"No mosaic tiles found for scene {scene_id}")

    ux = min(r[0] for r in rects.values())
    uy = min(r[1] for r in rects.values())
    span = (max(r[0] + r[2] for r in rects.values()) - ux,
            max(r[1] + r[3] for r in rects.values()) - uy)
    ref = source.scene_rects.get(scene_id, source.total_rect)
    dx, dy = ref[0] - ux, ref[1] - uy
    if (dx, dy) != (0, 0) and abs(span[0] - ref[2]) < 8 and abs(span[1] - ref[3]) < 8:
        rects = {m: (x + dx, y + dy, w, h) for m, (x, y, w, h) in rects.items()}
        rects["_shift"] = (dx, dy)         # popped by the caller for reporting
        shift = rects.pop("_shift")
        return rects, shift
    return rects, (0, 0)


def describe_grid(rects):
    xs = sorted({r[0] for r in rects.values()})
    ys = sorted({r[1] for r in rects.values()})

    def gaps(vals, size):
        d = [b - a for a, b in zip(vals, vals[1:])]
        return (1 - np.median(d) / size) * 100 if d else 0.0

    w = int(np.median([r[2] for r in rects.values()]))
    h = int(np.median([r[3] for r in rects.values()]))
    return dict(n=len(rects), cols=len(xs), rows=len(ys), tile_w=w, tile_h=h,
                overlap_x_pct=gaps(xs, w), overlap_y_pct=gaps(ys, h))


class MosaicCompositor:
    """Composites arbitrary output tiles from corrected mosaic positions.

    Only the mosaic tiles intersecting the requested region are decoded, held in a
    byte-bounded LRU cache. Because the writer walks the output row-major, the cache
    naturally holds about two rows of mosaic tiles and evicts the rest, so peak memory
    is `mosaic_cache_mb` plus the tiles in flight, independent of mosaic size.
    """

    def __init__(self, source, cfg, scene_id, rects, corrections, flatfields=None):
        self.source = source
        self.flatfields = flatfields or {}
        self.cfg = cfg
        self.scene_id = scene_id
        self.rects = {m: (x + corrections.get(m, (0, 0))[1],
                          y + corrections.get(m, (0, 0))[0], w, h)
                      for m, (x, y, w, h) in rects.items()}
        xs = [r[0] for r in self.rects.values()]
        ys = [r[1] for r in self.rects.values()]
        self.rect = (min(xs), min(ys),
                     max(r[0] + r[2] for r in self.rects.values()) - min(xs),
                     max(r[1] + r[3] for r in self.rects.values()) - min(ys))
        self._cache = {}
        self._loading = {}                 # key -> Future, for tiles being decoded now
        self._bytes = 0
        self._limit = int(cfg.mosaic_cache_mb * 1e6)
        self._lock = threading.Lock()
        self._weights = {}
        self._has_S = "S" in source.aics().get_dims_shape()[0]

    # -- tile access --------------------------------------------------------
    def _fetch(self, key):
        """One mosaic tile, decoded at most once however many threads ask together.

        Output tiles are smaller than mosaic tiles, so the read threads routinely start
        on output tiles that all need the same mosaic tile. Without the in-flight record
        each of them would miss the cache, decode its own copy and discard all but one.
        The first to ask decodes; the rest wait for its result (or its exception, which
        is not cached, so a later request tries again).
        """
        t, z, c, m = key
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache[key] = self._cache.pop(key)       # refresh LRU order
                return hit
            pending = self._loading.get(key)
            mine = pending is None
            if mine:
                pending = self._loading[key] = Future()
        if not mine:
            return pending.result()

        kw = {"M": m, "T": t, "Z": z, "C": c}
        if self.scene_id is not None and self._has_S:
            kw["S"] = self.scene_id
        try:
            arr = np.squeeze(self.source.aics().read_image(**kw)[0])
        except BaseException as exc:
            with self._lock:
                self._loading.pop(key, None)
            pending.set_exception(exc)
            raise
        with self._lock:
            self._cache[key] = arr
            self._bytes += arr.nbytes
            while self._bytes > self._limit and len(self._cache) > 1:
                old_key, old_arr = next(iter(self._cache.items()))
                self._cache.pop(old_key)
                self._bytes -= old_arr.nbytes
            self._loading.pop(key, None)
        pending.set_result(arr)
        return arr

    def _weight(self, shape):
        w = self._weights.get(shape)
        if w is None:
            h, wd = shape
            ramp = self.cfg.blend_width or max(8, min(h, wd) // 8)
            iy = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1.0
            ix = np.minimum(np.arange(wd), np.arange(wd)[::-1]) + 1.0
            w = np.minimum(np.minimum(iy[:, None], ix[None, :]) / ramp, 1.0).astype(np.float32)
            np.maximum(w, 1e-3, out=w)
            self._weights[shape] = w
        return w

    # -- compositing --------------------------------------------------------
    def tile(self, t, z, c, x0, y0, w, h, level=0):
        """Composite the full-resolution region (x0, y0, w, h), decimated by 2**level."""
        step = 1 << level
        nx, ny = w // step, h // step
        acc = np.zeros((ny, nx), np.float32)
        wsum = np.zeros((ny, nx), np.float32)
        blend = self.cfg.blend == "linear"

        for m, (mx, my, mw, mh) in self.rects.items():
            if mx >= x0 + w or mx + mw <= x0 or my >= y0 + h or my + mh <= y0:
                continue
            kx0 = max(0, -(-(mx - x0) // step))
            kx1 = min(nx - 1, (mx + mw - 1 - x0) // step)
            ky0 = max(0, -(-(my - y0) // step))
            ky1 = min(ny - 1, (my + mh - 1 - y0) // step)
            if kx1 < kx0 or ky1 < ky0:
                continue
            src = self._fetch((t, z, c, m))
            sy, sx = y0 + ky0 * step - my, x0 + kx0 * step - mx
            ys = slice(sy, sy + (ky1 - ky0) * step + 1, step)
            xs = slice(sx, sx + (kx1 - kx0) * step + 1, step)
            patch = src[ys, xs]
            # Shading is a property of the field, so it must be divided out per mosaic
            # tile before any blending — afterwards the fields are no longer separable.
            ff = self.flatfields.get(c)
            if ff is not None:
                patch = ff.apply(patch, ys, xs)
            dy = slice(ky0, ky0 + patch.shape[0])
            dx = slice(kx0, kx0 + patch.shape[1])
            if blend:
                wt = self._weight(src.shape)[ys, xs]
                acc[dy, dx] += patch * wt
                wsum[dy, dx] += wt
            else:
                acc[dy, dx] = patch
                wsum[dy, dx] = 1.0

        np.maximum(wsum, 1e-6, out=wsum)
        acc /= wsum
        # Round, don't truncate: see compute.cast_to. Ramps are applied on every side of
        # a tile, so single-coverage pixels along the mosaic's outer edge pass through
        # v*w/w too, and truncation would knock ~6% of them down by one.
        return cast_to(acc, self.source.dtype)

    def preview(self, t, z, c, max_px=1200):
        x, y, w, h = self.rect
        level = overview_level(w, h, max_px)
        step = 1 << level
        return self.tile(t, z, c, x, y, (w // step) * step, (h // step) * step, level), level

    @property
    def cache_mb(self):
        return self._bytes / 1e6

    @property
    def tile_shape(self):
        """Shape of one mosaic tile, i.e. of the field a flat-field profile must match."""
        shapes = {(r[3], r[2]) for r in self.rects.values()}
        if len(shapes) != 1:
            raise RuntimeError(f"mosaic tiles are not all the same size: {sorted(shapes)}")
        return shapes.pop()

    def sample_fields(self, channel, n, z_indices, t):
        """Raw mosaic tiles, for estimating an illumination profile."""
        ms = sorted(self.rects)
        picks = []
        for z in z_indices:
            for m in ms:
                picks.append((t, z, channel, m))
        if len(picks) > n:
            idx = np.linspace(0, len(picks) - 1, n).astype(int)
            picks = [picks[i] for i in idx]
        return [self._fetch(k) for k in picks]
