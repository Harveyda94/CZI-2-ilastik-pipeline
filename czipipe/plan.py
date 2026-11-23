"""Per-scene planning: intensity ranges, stitch corrections, compositor.

Everything a writer needs to produce one output file is gathered into a ScenePlan
before any full-resolution reading starts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import flatfield as ff
from .stitch import (MosaicCompositor, mosaic_geometry, describe_grid, refine, refine3d,
                     bigstitcher)


@dataclass
class ScenePlan:
    scene_id: Optional[int]
    rect: tuple
    ranges: dict                          # {source_channel: (lo, hi)}
    corrections: Optional[dict] = None
    compositor: Optional[MosaicCompositor] = None
    stitch_log: Optional[dict] = None
    grid: Optional[dict] = None
    flatfields: dict = field(default_factory=dict)   # {source_channel: FlatField}

    @property
    def label(self):
        return f"scene {self.scene_id if self.scene_id is not None else '-'}"

    @property
    def suffix(self):
        return f"_scene{self.scene_id:02d}" if self.scene_id is not None else ""


# ---------------------------------------------------------------- statistics
def auto_zoom(rect, max_px):
    longest = max(rect[2], rect[3])
    return 1.0 if longest <= max_px else max_px / longest


def compute_ranges(source, cfg, scene_id, rect, progress=None):
    """Global (lo, hi) per source channel, from downsampled sample planes.

    A naive `>> 8` discards the bottom 8 bits, which for typical two-photon data — where
    the useful signal sits in the low hundreds of counts — throws away most of the
    dynamic range. Percentiles are computed once over the whole selection and applied
    uniformly, so intensities stay comparable across the mosaic.
    """
    if cfg.output_bit_depth != 16 and np.dtype(source.dtype) == np.uint8:
        return {c: (0.0, 255.0) for c in source.sel_c}
    if cfg.output_bit_depth == 16:
        return {c: (0.0, 255.0) for c in source.sel_c}
    if cfg.eight_bit_method == "shift":
        return {c: (0.0, 65535.0) for c in source.sel_c}
    if cfg.eight_bit_method == "manual":
        missing = [c for c in source.sel_c if c not in cfg.manual_ranges]
        if missing:
            raise ValueError(f"manual_ranges missing entries for channels {missing}")
        return {c: (float(cfg.manual_ranges[c][0]), float(cfg.manual_ranges[c][1]))
                for c in source.sel_c}

    zoom = auto_zoom(rect, cfg.stats_max_px)
    zs = (source.sel_z if len(source.sel_z) <= cfg.stats_n_planes else
          [source.sel_z[i] for i in
           np.linspace(0, len(source.sel_z) - 1, cfg.stats_n_planes).astype(int)])
    t0 = source.sel_t[len(source.sel_t) // 2]
    kw = source.read_kwargs(scene_id)
    doc = source.reader()
    out = {}
    it = progress(source.sel_c) if progress else source.sel_c
    for c in it:
        pool = [np.asarray(doc.read(roi=rect, plane={"T": t0, "Z": z, "C": c},
                                    zoom=zoom, **kw)).reshape(-1)
                for z in zs]
        flat = np.concatenate(pool)
        lo, hi = np.percentile(flat, cfg.percentiles)
        out[c] = (float(lo), float(hi) if hi > lo else float(lo) + 1.0)
        del pool, flat
    return out


# ---------------------------------------------------------------- flat field
def needs_field_access(cfg):
    """Flat-field correction requires individual fields of view to still be separable."""
    return cfg.flatfield.mode != "none"


def _estimation_stride(shape, max_px):
    longest = max(shape)
    return max(1, int(np.ceil(longest / max_px)))


def build_flatfields(source, cfg, plan, log=print, progress=None):
    """One FlatField per selected channel, or {} when correction is off."""
    ffc = cfg.flatfield
    if ffc.mode == "none":
        return {}

    if plan.compositor is not None:
        field_shape = plan.compositor.tile_shape
        where = f"per mosaic tile ({field_shape[1]} x {field_shape[0]} px)"
    else:
        field_shape = tuple(plan.rect[3:1:-1])       # (h, w)
        where = f"whole field ({field_shape[1]} x {field_shape[0]} px)"
    log(f"{plan.label}: flat-field '{ffc.mode}', applied {where}")

    out = {}
    for ci, c in enumerate(source.sel_c):
        if ffc.mode == "file":
            out[c] = ff.load_from_file(ffc.flat_file, ci, len(source.sel_c),
                                       field_shape, dark=ffc.dark, min_gain=ffc.min_gain)
        else:
            samples = _collect_samples(source, cfg, plan, c, field_shape, log)
            stride = _estimation_stride(field_shape, ffc.estimate_max_px)
            small = [s[::stride, ::stride] for s in samples]
            if ffc.mode == "basic":
                out[c] = ff.estimate_basic(small, ffc, field_shape)
            else:
                out[c] = ff.estimate_from_samples(small, ffc, field_shape, dark=ffc.dark)
        log(ff.format_report(c, out[c], ffc.max_disagreement_pct))

    if ffc.save_profile:
        import tifffile
        path = cfg.out_dir / f"{cfg.stem}{plan.suffix}.flatfield.tif"
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(path, np.stack([out[c].profile for c in source.sel_c]),
                         metadata={"axes": "CYX"})
        log(f"  profile written to {path.name} (open it and check it looks like "
            "illumination, not like your sample)")
    return out


def _collect_samples(source, cfg, plan, channel, field_shape, log):
    """Fields to estimate a profile from: mosaic tiles, or Z planes for a single FOV."""
    ffc = cfg.flatfield
    zs = (source.sel_z if len(source.sel_z) <= ffc.z_samples else
          [source.sel_z[i] for i in
           np.linspace(0, len(source.sel_z) - 1, ffc.z_samples).astype(int)])
    t0 = source.sel_t[len(source.sel_t) // 2]

    if plan.compositor is not None:
        samples = plan.compositor.sample_fields(channel, ffc.n_samples, zs, t0)
    else:
        # No mosaic: the only independent fields are separate planes. That is a much
        # weaker basis for an estimate, since sample structure repeats across Z.
        zs_all = (source.sel_z if len(source.sel_z) <= ffc.n_samples else
                  [source.sel_z[i] for i in
                   np.linspace(0, len(source.sel_z) - 1, ffc.n_samples).astype(int)])
        kw = source.read_kwargs(plan.scene_id)
        doc = source.reader()
        samples = [np.asarray(doc.read(roi=plan.rect,
                                       plane={"T": t0, "Z": z, "C": channel},
                                       **kw))[..., 0] for z in zs_all]
        log("  [warn] not a mosaic: the profile is estimated across Z planes, where sample "
            "structure repeats. Prefer flatfield.mode='file' with a measured flat-field.")
    if len(samples) < 4:
        raise ValueError(
            f"only {len(samples)} field(s) available to estimate a flat-field for channel "
            f"{channel}. Supply one with flatfield.mode='file', or widen the selection.")
    return samples


# ---------------------------------------------------------------- planning
def registration_planes(source, cfg):
    """Z planes `refine` registers on: refine.z if set, else refine.planes slab centres."""
    zs = cfg.refine.z_planes()
    return zs if zs is not None else refine.sample_planes(source.sel_z, cfg.refine.planes)


def reference_plane(source, cfg):
    """The single plane used for probing and tile geometry.

    The middle of the Z selection unless refine.z names planes, in which case the middle
    one of those. It does not depend on refine.planes, so it is the same plane as before
    registration used more than one.
    """
    rc = cfg.refine
    zs = rc.z_planes()
    return {"C": source.sel_c[0] if rc.channel is None else int(rc.channel),
            "Z": source.sel_z[len(source.sel_z) // 2] if zs is None else zs[len(zs) // 2],
            "T": source.sel_t[0] if rc.t is None else int(rc.t)}


def plan_scenes(source, cfg, backend, log=print, progress=None):
    """Build a ScenePlan per output file: geometry, registration, compositor, ranges."""
    jobs = source.jobs
    ref = reference_plane(source, cfg)
    source.probe_scene_arg(jobs, {"T": ref["T"], "Z": ref["Z"], "C": ref["C"]})
    if not source.scene_arg_ok:
        log("[probe] scene= argument rejected; falling back to ROI-only reads. "
            "Overlapping scenes may bleed at edges.")

    # Flat-field correction is per field of view. In "metadata" mode libCZI blends the
    # mosaic internally and the fields are gone by the time we see pixels, so a mosaic
    # that needs correcting is routed through our own compositor with zero corrections:
    # same stage positions, but the tiles stay separable long enough to divide out the
    # shading. This changes the blending from libCZI's to ours, so it is announced.
    use_compositor = cfg.stitch_mode != "metadata"
    if not use_compositor and needs_field_access(cfg):
        # Deliberately not a blanket except. Treating "could not tell" as "not a mosaic"
        # silently corrects a whole stitched plane as if it were one field of view (and
        # reads every plane whole to estimate it), so any failure here must surface.
        try:
            is_mosaic = source.aics().is_mosaic()
        except ImportError as exc:
            raise ImportError(
                "flat-field correction needs aicspylibczi to tell whether this file is a "
                "mosaic, because each field of view has to be corrected before blending: "
                "pip install aicspylibczi") from exc
        if is_mosaic:
            use_compositor = True
            log('[flatfield] mosaic + stitch_mode="metadata": compositing through czipipe '
                "so each field can be corrected before blending. Tile positions are "
                "unchanged (stage metadata); blending is czipipe's feather, not libCZI's.")

    plans = []
    for scene_id, rect in jobs:
        plan = ScenePlan(scene_id=scene_id, rect=rect, ranges={})

        if use_compositor:
            if source.samples == 3:
                raise NotImplementedError(
                    f'stitch_mode="{cfg.stitch_mode}" supports single-sample (grayscale) '
                    f"pixel types only; this file is {source.pixel_type}")
            rects, shift = mosaic_geometry(source, scene_id, ref)
            if shift != (0, 0):
                log(f"[geom] {plan.label}: shifting aics coords by {shift} to match pylibCZIrw")
            plan.grid = describe_grid(rects)
            g = plan.grid
            log(f"{plan.label}: {g['n']} tiles, {g['cols']} x {g['rows']} grid, "
                f"tile {g['tile_w']} x {g['tile_h']} px, overlap "
                f"~{g['overlap_x_pct']:.1f}% x / {g['overlap_y_pct']:.1f}% y")

            if cfg.stitch_mode == "refine" and cfg.refine.mode == "3d":
                corr, slog = refine3d.register(source, cfg, backend, scene_id, rects, ref,
                                               progress=progress)
                log(refine3d.format_report(slog))
            elif cfg.stitch_mode == "refine":
                corr, slog = refine.register(source, cfg, backend, scene_id, rects, ref,
                                             progress=progress,
                                             z_planes=registration_planes(source, cfg))
                log(refine.format_report(slog))
            elif cfg.stitch_mode == "bigstitcher":
                corr, slog = bigstitcher.corrections(cfg, scene_id, rects)
                log(bigstitcher.format_report(slog))
            else:
                corr, slog = {m: (0, 0) for m in rects}, None

            plan.corrections, plan.stitch_log = corr, slog
            plan.compositor = MosaicCompositor(source, cfg, scene_id, rects, corr)
            if plan.compositor.rect != rect:
                log(f"{plan.label}: canvas {rect} -> {plan.compositor.rect} "
                    "(expanded to hold corrected tile positions)")
                plan.rect = plan.compositor.rect

        plan.flatfields = build_flatfields(source, cfg, plan, log=log, progress=progress)
        if plan.compositor is not None:
            # The estimate needs the compositor (it samples raw mosaic tiles), so the
            # compositor exists first and is handed the profiles here. Without this the
            # correction is estimated, reported and recorded, and never applied.
            plan.compositor.flatfields = plan.flatfields
        plan.ranges = compute_ranges(source, cfg, scene_id, plan.rect, progress=progress)
        if plan.flatfields and cfg.output_bit_depth == 8:
            log("  [note] 8-bit ranges are measured from libCZI's uncorrected composite. "
                "The profile is normalised to median 1, so the shift is small, but set "
                "manual_ranges if you need the display range to be exact.")
        if cfg.output_bit_depth == 8:
            pretty = ", ".join(f"C{c}:[{lo:.0f},{hi:.0f}]" for c, (lo, hi) in plan.ranges.items())
            log(f"{plan.label}  {cfg.eight_bit_method}  {pretty}")
        plans.append(plan)
    return plans
