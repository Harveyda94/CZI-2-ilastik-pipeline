"""Stage timing, so tuning is driven by measurement rather than guesswork."""

from __future__ import annotations

import time

import numpy as np


def benchmark(pipe, n=8):
    """Time one tile's worth of read, convert and compress in isolation."""
    cfg, source, tiler = pipe.cfg, pipe.source, pipe.tiler
    if not pipe.plans:
        pipe.plan()
    plan = pipe.plans[0]
    th, tw = tiler.tile
    tasks = [(source.sel_t[0], source.sel_z[len(source.sel_z) // 2], source.sel_c[0],
              plan.rect[0] + (i % 4) * tw, plan.rect[1] + (i // 4) * th, tw, th)
             for i in range(n)]
    lo, hi = plan.ranges[source.sel_c[0]]

    if plan.compositor is not None:
        comp = plan.compositor
        comp.tile(*tasks[0], 0)                      # warm the tile cache
        t0 = time.perf_counter()
        raws = [comp.tile(*t, 0) for t in tasks]
        label = "composite"
    else:
        kw = source.read_kwargs(plan.scene_id)
        doc = source.reader()
        t0 = time.perf_counter()
        raws = [np.asarray(doc.read(roi=(x, y, w, h), plane={"T": t, "Z": z, "C": c},
                                    **kw))
                for (t, z, c, x, y, w, h) in tasks]
        label = "read"
    t_read = time.perf_counter() - t0
    raws = [r[..., 0] if (source.samples == 1 and r.ndim == 3) else r for r in raws]

    t0 = time.perf_counter()
    outs = [pipe.backend.to_out_dtype(r, lo, hi, source.dtype, source.out_dtype,
                                      cfg.eight_bit_method) for r in raws]
    t_conv = time.perf_counter() - t0

    t0, ratio = time.perf_counter(), 1.0
    if cfg.compression:
        import imagecodecs
        codec = {"zlib": "zlib", "zstd": "zstd", "lzw": "lzw",
                 "jpeg2000": "jpeg2k"}[cfg.compression]
        enc = getattr(imagecodecs, f"{codec}_encode")
        blobs = [enc(np.ascontiguousarray(o)) for o in outs]
        ratio = sum(o.nbytes for o in outs) / max(sum(len(b) for b in blobs), 1)
    t_comp = time.perf_counter() - t0

    mb = sum(r.nbytes for r in raws) / 1e6
    stages = [(label, t_read), ("convert", t_conv), ("compress", t_comp)]
    pipe.log(f"{n} tiles of {th}x{tw}  ({mb:.0f} MB source)")
    for name, t in stages:
        pipe.log(f"  {name:10s} {t*1000/n:7.1f} ms/tile   "
                 f"{mb/max(t,1e-9):7.1f} MB/s single-threaded")
    pipe.log(f"  compression ratio {ratio:.2f}x   (GPU {'on' if pipe.backend.gpu else 'off'})")
    slowest = max(stages, key=lambda kv: kv[1])[0]
    pipe.log(f"\n  bottleneck: {slowest}")
    pipe.log({
        "read": "  -> raise n_read_threads until it stops helping; then you are disk-bound "
                "and only faster storage helps. Copying the CZI off a network share to "
                "local NVMe is often the single largest win.",
        "composite": "  -> refine/bigstitcher decode whole mosaic tiles. Raise "
                     "mosaic_cache_mb if tiles are being evicted and re-read.",
        "convert": "  -> the only case where use_gpu matters. It rarely is.",
        "compress": "  -> raise n_write_workers, lower compression_level, or switch to zstd.",
    }[slowest])
    return {"stages": dict(stages), "mb": mb, "ratio": ratio, "bottleneck": slowest}
