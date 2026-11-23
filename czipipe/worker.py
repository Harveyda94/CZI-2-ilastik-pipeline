"""Construction of the per-tile callable used by every writer and by verification.

One factory, two paths: libCZI ROI reads for `stitch_mode="metadata"`, the mosaic
compositor otherwise. Both return a tile already cast to the output dtype and padded to
exactly the tile shape, so writers never need to know which was used.
"""

from __future__ import annotations

import numpy as np

from .compute import cast_to


def make_worker(source, cfg, backend, tiler, plan, level=0):
    th, tw = tiler.tile
    ranges = plan.ranges
    rescaling = np.dtype(source.out_dtype) != np.dtype(source.dtype)

    def convert(arr, c):
        # Only touch `ranges` when a rescale actually needs them. A 16-bit passthrough
        # has nothing to look up, and indexing here would make an unrelated planning
        # step a hard prerequisite of every tile.
        if not rescaling:
            return arr
        try:
            lo, hi = ranges[c]
        except KeyError:
            raise RuntimeError(
                f"no intensity range for channel {c}. Ranges are built by "
                "Pipeline.plan(); call it before converting, or set manual_ranges."
            ) from None
        return backend.to_out_dtype(arr, lo, hi, source.dtype, source.out_dtype,
                                    cfg.eight_bit_method)

    if plan.compositor is not None:
        comp = plan.compositor

        def work(task):
            t, z, c, x, y, w, h = task
            return tiler.fit(convert(comp.tile(t, z, c, x, y, w, h, level), c))
        return work

    zoom = 1.0 / (1 << level)
    kw = source.read_kwargs(plan.scene_id)
    samples = source.samples
    step = 1 << level
    ox, oy = plan.rect[0], plan.rect[1]
    ffs = plan.flatfields or {}

    def work(task):
        t, z, c, x, y, w, h = task
        # background_pixel is deliberately not passed: some pylibCZIrw builds call
        # Color(*background_pixel) unconditionally, so a scalar raises TypeError even
        # for Gray pixel types. The default is already black.
        arr = np.asarray(source.reader().read(
            roi=(x, y, w, h), plane={"T": t, "Z": z, "C": c}, zoom=zoom, **kw))
        if samples == 1:
            arr = arr[..., 0] if arr.ndim == 3 else arr
        else:
            arr = arr[..., ::-1]                     # BGR -> RGB
        ff = ffs.get(c)
        if ff is not None:
            # Non-mosaic: the whole image is one field, so the profile is indexed by
            # absolute position within it.
            ys = slice(y - oy, y - oy + h, step)
            xs = slice(x - ox, x - ox + w, step)
            # apply() works in float32; bring the result back to the source dtype (rounded,
            # clipped at 0 after dark subtraction) so writers see the dtype they declared
            # and the 8-bit "shift" method still has integers to shift.
            arr = cast_to(ff.apply(arr, ys, xs), source.dtype)
        return tiler.fit(convert(arr, c))

    return work
