"""Top-level orchestration: the two calls the notebook actually makes."""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .compute import Backend
from .geometry import Tiler, overview_level
from .plan import ScenePlan, plan_scenes
from .source import Source
from .stitch import mosaic_geometry
from .validate import verify as verify_output, report_h5_check
from .worker import make_libczi_reader, make_worker
from .writers import WRITERS


# ---------------------------------------------------------------- progress
def _null_progress(iterable=None, total=None, initial=0, **kw):
    if iterable is not None:
        return iterable

    class _Bar:
        n = initial

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def update(self, k=1):
            self.n += k
    return _Bar()


def default_progress():
    """tqdm if available, otherwise a no-op that satisfies the same protocol."""
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _null_progress

    def _p(iterable=None, **kw):
        return tqdm(iterable, **kw) if iterable is not None else tqdm(**kw)
    return _p


# ---------------------------------------------------------------- results
@dataclass
class Result:
    plan: ScenePlan
    path: Path
    verified: Optional[bool] = None


class Pipeline:
    """Holds the open source, the backend and the plans for one Config.

    Typical use is two calls — `inspect()` then `run()` — but the pieces stay reachable
    so a notebook can preview the stitching or benchmark before committing.
    """

    def __init__(self, cfg, log=print, progress=None):
        self.cfg = cfg.validate()
        self.log = log
        self.progress = progress or default_progress()
        self.source: Optional[Source] = None
        self.backend: Optional[Backend] = None
        self.tiler: Optional[Tiler] = None
        self.plans: List[ScenePlan] = []
        self.results: List[Result] = []

    # -- setup --------------------------------------------------------------
    def inspect(self):
        cfg = self.cfg
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        self.backend = Backend(cfg.use_gpu)
        self.source = Source(cfg, log=self.log)
        self.tiler = Tiler(tuple(cfg.tile), self.source.sel_t, self.source.sel_z,
                           self.source.sel_c, self.source.samples)
        self.log(f"input       {cfg.czi_path}  ({cfg.czi_path.stat().st_size / 1e9:.1f} GB)")
        self.log(f"output      {cfg.out_dir}  [{cfg.output_format}]")
        mode = cfg.stitch_mode
        if mode == "refine" and cfg.refine.mode == "3d":
            mode += " (3d: XY and Z)"
        self.log(f"stitching   {mode}    compute  {self.backend.name}")
        self.log("")
        self.log(self.source.describe())
        peak = (cfg.prefetch * cfg.tile[0] * cfg.tile[1]
                * np.dtype(self.source.dtype).itemsize * self.source.samples)
        self.log(f"\nestimated pipeline RAM (tiles in flight): {peak / 1e6:.0f} MB"
                 + (f" + {cfg.mosaic_cache_mb} MB mosaic cache"
                    if cfg.stitch_mode != "metadata" else ""))
        return self

    def plan(self):
        if self.source is None:
            self.inspect()
        self.plans = plan_scenes(self.source, self.cfg, self.backend,
                                 log=self.log, progress=self.progress)
        return self.plans

    # -- work ---------------------------------------------------------------
    def _preview_plane(self):
        src = self.source
        return {"T": src.sel_t[0], "Z": src.sel_z[len(src.sel_z) // 2], "C": src.sel_c[0]}

    def preview(self, max_px=1200):
        """Downsampled overview per scene, for eyeballing seams before a long run.

        Returns [(plan, image, level)], the image decimated by 2**level. A plan with a
        compositor is drawn by it, so registration and per-field flat-field show. A
        plain `stitch_mode="metadata"` plan has none and is read through libCZI at the
        same decimation: the stage-position composite that convert() will write.
        """
        if not self.plans:
            self.plan()
        ref = self._preview_plane()
        out = []
        for p in self.plans:
            if p.compositor is not None:
                img, lvl = p.compositor.preview(ref["T"], ref["Z"], ref["C"], max_px)
            else:
                x, y, w, h = p.rect
                lvl = overview_level(w, h, max_px)
                step = 1 << lvl
                read = make_libczi_reader(self.source, p, lvl)
                img = read(ref["T"], ref["Z"], ref["C"], x, y,
                           (w // step) * step, (h // step) * step)
            out.append((p, img, lvl))
        return out

    def tile_rects(self, plan):
        """{m: (x, y, w, h)} mosaic tile positions, for outlining a preview.

        Empty when there is nothing to outline (not a mosaic) or the positions cannot be
        had. With a compositor they are the corrected positions it composites from. A
        plain `metadata` plan never otherwise touches aicspylibczi, so there the outlines
        are strictly optional: a missing package or an unreadable tile index costs the
        outlines, says so, and leaves the overview itself alone.
        """
        if plan.compositor is not None:
            return plan.compositor.rects
        try:
            if not self.source.aics().is_mosaic():
                return {}
            rects, _ = mosaic_geometry(self.source, plan.scene_id, self._preview_plane())
        except ImportError:
            self.log(f"[preview] {plan.label}: tile outlines need aicspylibczi "
                     "(pip install aicspylibczi); showing the overview without them.")
            return {}
        except Exception as exc:
            self.log(f"[preview] {plan.label}: tile positions unavailable for outlines "
                     f"({type(exc).__name__}: {exc}); showing the overview without them.")
            return {}
        return rects

    def convert(self):
        if not self.plans:
            self.plan()
        writer = WRITERS[self.cfg.output_format]
        self.results = []
        for plan in self.plans:
            path = writer(self.source, self.cfg, self.backend, self.tiler, plan,
                          log=self.log, progress=self.progress)
            self.results.append(Result(plan, path))
        self.log("\nwritten:")
        for r in self.results:
            self.log(f"  {r.path}")
        return self.results

    def verify(self, n=None):
        if not self.results:
            # Typically a restarted kernel. all([]) is True, so carrying on would announce
            # VERIFICATION PASSED having checked nothing.
            raise RuntimeError(
                "No converted output is attached to this Pipeline, so there is nothing to "
                "verify. Run convert() first: outputs that are already complete are "
                "recognised and skipped, so after a restart it only re-attaches them.")
        for r in self.results:
            r.verified = verify_output(self.source, self.cfg, self.backend, self.tiler,
                                       r.plan, r.path, n=n, log=self.log,
                                       progress=self.progress)
        ok = all(r.verified is not False for r in self.results)
        self.log("\nVERIFICATION PASSED" if ok else
                 "\nVERIFICATION FAILED - inspect before using output")
        if self.cfg.output_format == "hdf5":
            self.log("")
            for r in self.results:
                report_h5_check(r.path, self.cfg.hdf5_dataset, self.source.h5_axes,
                                log=self.log)
        return ok

    def classify(self):
        if not self.cfg.ilastik.run:
            self.log("ilastik.run is False; nothing to classify.")
            return []
        if not self.results:
            # Typically a restarted kernel. Returning [] here would look like success.
            raise RuntimeError(
                "No converted output is attached to this Pipeline, so there is nothing to "
                "classify. Run convert() first: outputs that are already complete are "
                "recognised and skipped, so after a restart it only re-attaches them.")
        from . import ilastik
        return ilastik.classify(self.source, self.cfg, self.tiler,
                                [(r.plan, r.path) for r in self.results],
                                log=self.log, progress=self.progress)

    def preview_segmentation(self, seg, n_planes=3, max_px=1200):
        """Raw channels next to the segmentation at a few depths, for a quick sanity
        check of a classify() result. `seg` is classify()'s own return value."""
        from . import ilastik
        return ilastik.preview_segmentation(
            self.source, self.cfg, self.tiler,
            [(r.plan, r.path) for r in self.results], seg,
            n_planes=n_planes, max_px=max_px)

    def benchmark(self, n=8):
        """Time read/convert/compress separately, so tuning is evidence-led."""
        from .bench import benchmark
        return benchmark(self, n)

    def run(self, verify=True, classify=True):
        self.inspect()
        self.plan()
        self.convert()
        if verify:
            self.verify()
        if classify and self.cfg.ilastik.run:
            self.classify()
        return self.results

    # -- lifecycle ----------------------------------------------------------
    def close(self):
        if self.source is not None:
            self.source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
