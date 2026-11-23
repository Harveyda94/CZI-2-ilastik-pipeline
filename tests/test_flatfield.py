"""Flat-field estimation and application, against a known synthetic profile."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from czipipe import flatfield as ff
from czipipe.compute import Backend
from czipipe.plan import ScenePlan, build_flatfields
from czipipe.stitch import MosaicCompositor

from conftest import FakeSource, smooth_image

TS = 256
POSITIONS = {m: (x, y) for m, (x, y) in
             enumerate([(x, y) for y in (0, 192, 384) for x in (0, 192, 384)])}


def true_profile(shape=(TS, TS), depth=0.45):
    """Radial vignette: 1.0 at centre falling to (1 - depth) at the corners."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
    cy, cx = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2) / np.sqrt(2)
    p = 1.0 - depth * r ** 2
    return (p / np.median(p)).astype(np.float32)


def shaded_mosaic(n_fields=27, depth=0.45, seed=3, dark=0.0):
    """Fields of independent content, each multiplied by the same vignette."""
    prof = true_profile(depth=depth)
    rng = np.random.default_rng(seed)
    fields = []
    for i in range(n_fields):
        content = smooth_image((TS, TS), seed=seed + i, lo=4000, hi=20000).astype(np.float32)
        fields.append(np.clip(content * prof + dark, 0, 65535).astype(np.uint16))
    return prof, fields


class _FFC:
    """Stand-in for FlatFieldConfig with just the fields the estimator reads."""
    percentile = 50.0
    smooth_mode = "poly"
    poly_degree = 4
    smooth_sigma = 30.0
    min_gain = 0.2
    basic_darkfield = False


class _FFCGauss(_FFC):
    smooth_mode = "gaussian"


# ---------------------------------------------------------------- helpers
def test_gaussian_blur_preserves_mean_and_smooths():
    a = np.random.default_rng(0).normal(10, 3, (128, 128)).astype(np.float32)
    b = ff.gaussian_blur(a, 8)
    assert abs(b.mean() - a.mean()) < 0.2
    assert b.std() < a.std() / 3


@pytest.mark.parametrize("stride,tol", [(2, 0.01), (4, 0.03), (8, 0.06)])
def test_resize_bilinear_error_grows_with_stride(stride, tol):
    """Used only by the 'gaussian' path; 'poly' evaluates at full resolution instead."""
    p = true_profile((256, 256))
    back = ff.resize_bilinear(p[::stride, ::stride], (256, 256))
    assert np.abs(back - p).max() < tol


def test_build_normalises_to_median_one():
    f = ff.build(true_profile() * 7.0)
    assert np.median(f.profile) == pytest.approx(1.0, abs=1e-5)


def test_min_gain_floor_is_applied_and_counted():
    """An unclipped dark corner would turn correction into arbitrary noise amplification."""
    p = true_profile(depth=0.95)
    f = ff.build(p, min_gain=0.5)
    assert f.profile.min() >= 0.5
    assert f.stats["pixels_clipped_to_min_gain"] > 0
    assert f.gain.max() <= 2.0 + 1e-6


# ---------------------------------------------------------------- estimation
def test_measured_estimate_recovers_the_profile():
    prof, fields = shaded_mosaic()
    f = ff.estimate_from_samples([x[::2, ::2] for x in fields], _FFC(), (TS, TS))
    err = np.abs(f.profile - prof)
    assert err.max() < 0.05, f"worst pixel off by {err.max():.3f}"
    assert err.mean() < 0.01


def test_poly_fit_beats_gaussian_at_the_corners():
    """Local averaging is biased at a boundary; a global surface fit is not.

    On a 45% vignette the Gaussian path overestimates the corner gain by ~0.14 because
    edge replication drags it toward the brighter interior. This is why "poly" is the
    default.
    """
    prof, fields = shaded_mosaic()
    small = [x[::2, ::2] for x in fields]
    poly = ff.estimate_from_samples(small, _FFC(), (TS, TS))
    gauss = ff.estimate_from_samples(small, _FFCGauss(), (TS, TS))
    corner_err = lambda f: abs(float(f.profile[0, 0]) - float(prof[0, 0]))
    assert corner_err(poly) < corner_err(gauss) / 3
    assert np.abs(poly.profile - prof).max() < np.abs(gauss.profile - prof).max() / 3


def test_poly_degree_six_starts_fitting_the_sample():
    """Higher order is not better: it starts absorbing structure that is not illumination."""
    prof, fields = shaded_mosaic()
    small = [x[::2, ::2] for x in fields]

    class Deg6(_FFC):
        poly_degree = 6

    d4 = np.abs(ff.estimate_from_samples(small, _FFC(), (TS, TS)).profile - prof).max()
    d6 = np.abs(ff.estimate_from_samples(small, Deg6(), (TS, TS)).profile - prof).max()
    assert d6 > d4


def test_correction_flattens_a_shaded_field():
    prof, fields = shaded_mosaic()
    f = ff.estimate_from_samples([x[::2, ::2] for x in fields], _FFC(), (TS, TS))
    raw = fields[0].astype(np.float32)
    fixed = f.apply(raw)
    truth = raw / prof
    assert np.abs(fixed - truth).mean() / truth.mean() < 0.02
    # and the residual shading is far smaller than before
    def edge_ratio(img):
        return float(img[8:24, 8:24].mean() / img[TS//2-8:TS//2+8, TS//2-8:TS//2+8].mean())
    assert abs(edge_ratio(fixed) - 1) < abs(edge_ratio(raw) - 1) / 3


def test_darkfield_offset_is_subtracted():
    prof, fields = shaded_mosaic(dark=500.0)
    ffc = _FFC()
    f = ff.estimate_from_samples([x[::2, ::2] for x in fields], ffc, (TS, TS), dark=500.0)
    assert f.dark == 500.0
    corrected = f.apply(fields[0].astype(np.float32))
    assert corrected.min() >= -1.0


@pytest.mark.parametrize("n", [9, 27, 48])
def test_split_half_agreement_on_healthy_data(n):
    """Measured range for uncorrelated content is 1.5-2.7% RMS across n=9..48."""
    prof, fields = shaded_mosaic(n_fields=n)
    f = ff.estimate_from_samples([x[::2, ::2] for x in fields], _FFC(), (TS, TS))
    assert f.stats["split_half_disagreement_pct"] < 5.0
    assert "[warn]" not in ff.format_report(0, f)


def test_split_half_detects_sample_structure():
    """If content correlates with position, the estimate is not illumination — say so."""
    prof = true_profile()
    rng = np.random.default_rng(1)
    fields = []
    for i in range(24):
        content = np.full((TS, TS), 8000.0, np.float32)
        # a bright band whose position depends on which half of the sample set we are in
        band = slice(20, 80) if i % 2 == 0 else slice(180, 240)
        content[band] *= 3.0
        content += rng.normal(0, 50, content.shape)
        fields.append(np.clip(content * prof, 0, 65535).astype(np.uint16))
    f = ff.estimate_from_samples([x[::2, ::2] for x in fields], _FFC(), (TS, TS))
    # ~98% RMS here against under 3% on healthy data: a wide, unambiguous margin.
    assert f.stats["split_half_disagreement_pct"] > 50.0
    assert "[warn]" in ff.format_report(0, f)


def test_too_few_samples_refuses():
    prof, fields = shaded_mosaic(n_fields=3)
    with pytest.raises(ValueError, match="at least 4"):
        ff.estimate_from_samples(fields, _FFC(), (TS, TS))


# ---------------------------------------------------------------- file mode
def test_file_mode_applies_the_supplied_profile(tmp_path):
    prof = true_profile()
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, prof)
    f = ff.load_from_file(path, 0, 1, (TS, TS))
    assert np.abs(f.profile - prof).max() < 1e-5
    raw = np.full((TS, TS), 1000.0, np.float32)
    assert np.abs(f.apply(raw) - 1000.0 / prof).max() < 0.01


def test_file_mode_rejects_a_shape_mismatch(tmp_path):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, true_profile((128, 128)))
    with pytest.raises(ValueError, match="must match exactly"):
        ff.load_from_file(path, 0, 1, (TS, TS))


def test_file_mode_selects_the_right_channel_plane(tmp_path):
    a, b = true_profile(depth=0.2), true_profile(depth=0.6)
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.stack([a, b]))
    f0 = ff.load_from_file(path, 0, 2, (TS, TS))
    f1 = ff.load_from_file(path, 1, 2, (TS, TS))
    assert np.abs(f0.profile - a).max() < 1e-5
    assert np.abs(f1.profile - b).max() < 1e-5


def test_file_mode_rejects_wrong_channel_count(tmp_path):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(path, np.stack([true_profile()] * 2))
    with pytest.raises(ValueError, match="3 channels are selected"):
        ff.load_from_file(path, 0, 3, (TS, TS))


# ---------------------------------------------------------------- integration
def make_shaded_compositor(cfg, depth=0.45):
    prof = true_profile(depth=depth)
    gt = smooth_image((640 + 128, 640 + 128), seed=11)
    tiles, rects = {}, {}
    for m, (x, y) in POSITIONS.items():
        clean = gt[y + 64:y + 64 + TS, x + 64:x + 64 + TS].astype(np.float32)
        tiles[m] = np.clip(clean * prof, 0, 65535).astype(np.uint16)
        rects[m] = (x, y, TS, TS)
    dummy = np.zeros((1, 1, 1, 4, 4), np.uint16)
    src = FakeSource(dummy, (0, 0, 640, 640), [0], [0], [0], cfg, tiles=tiles)
    return prof, gt, tiles, rects, src


def test_correction_happens_before_blending(cfg):
    """Applied per tile, the seams flatten; applied after, they could not."""
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    corr = {m: (0, 0) for m in rects}

    plain = MosaicCompositor(src, cfg, None, rects, corr)
    f = ff.build(prof)
    fixed = MosaicCompositor(src, cfg, None, rects, corr, flatfields={0: f})

    r = plain.rect
    a = plain.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0).astype(np.float32)
    b = fixed.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0).astype(np.float32)
    truth = gt[64:64 + r[3], 64:64 + r[2]].astype(np.float32)

    err_a = np.abs(a - truth)[8:-8, 8:-8].mean() / truth.mean()
    err_b = np.abs(b - truth)[8:-8, 8:-8].mean() / truth.mean()
    assert err_b < err_a / 3, f"uncorrected {err_a:.4f}, corrected {err_b:.4f}"
    assert err_b < 0.02


def test_compositor_tile_shape_is_the_field_shape(cfg):
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    comp = MosaicCompositor(src, cfg, None, rects, {m: (0, 0) for m in rects})
    assert comp.tile_shape == (TS, TS)


def test_sample_fields_returns_raw_tiles(cfg):
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    comp = MosaicCompositor(src, cfg, None, rects, {m: (0, 0) for m in rects})
    got = comp.sample_fields(0, 6, [0], 0)
    assert len(got) == 6
    assert all(g.shape == (TS, TS) for g in got)


def test_build_flatfields_end_to_end(cfg):
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    cfg.flatfield.mode = "measured"
    cfg.flatfield.n_samples = 9
    cfg.flatfield.save_profile = False
    comp = MosaicCompositor(src, cfg, None, rects, {m: (0, 0) for m in rects})
    plan = ScenePlan(None, comp.rect, {}, compositor=comp)
    fields = build_flatfields(src, cfg, plan, log=lambda *a: None)
    assert set(fields) == {0}
    assert np.abs(fields[0].profile - prof).mean() < 0.02


def test_flatfield_off_produces_nothing(cfg):
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    comp = MosaicCompositor(src, cfg, None, rects, {m: (0, 0) for m in rects})
    plan = ScenePlan(None, comp.rect, {}, compositor=comp)
    assert build_flatfields(src, cfg, plan, log=lambda *a: None) == {}


def test_profile_is_saved_for_inspection(cfg):
    prof, gt, tiles, rects, src = make_shaded_compositor(cfg)
    cfg.flatfield.mode = "measured"
    cfg.flatfield.n_samples = 9
    comp = MosaicCompositor(src, cfg, None, rects, {m: (0, 0) for m in rects})
    plan = ScenePlan(None, comp.rect, {}, compositor=comp)
    build_flatfields(src, cfg, plan, log=lambda *a: None)
    saved = cfg.out_dir / f"{cfg.stem}.flatfield.tif"
    assert saved.is_file()
    assert tifffile.imread(saved).shape == (1, TS, TS)


# ---------------------------------------------------------------- config
def test_file_mode_requires_a_file(cfg):
    cfg.flatfield.mode = "file"
    with pytest.raises(ValueError, match="flat_file"):
        cfg.validate()


def test_unknown_mode_rejected(cfg):
    cfg.flatfield.mode = "divide-by-blur"
    with pytest.raises(ValueError, match="flatfield.mode"):
        cfg.validate()


def test_min_gain_bounds_enforced(cfg):
    cfg.flatfield.min_gain = 0.0
    with pytest.raises(ValueError, match="min_gain"):
        cfg.validate()


def test_missing_aicspylibczi_is_reported_not_treated_as_single_field(cfg):
    """A blanket except once turned this ImportError into "not a mosaic", so a stitched
    plane was corrected as if it were one field of view, and read whole to estimate it."""
    from czipipe.plan import plan_scenes

    cfg.stitch_mode, cfg.flatfield.mode = "metadata", "measured"
    data = np.zeros((1, 1, 1, 64, 64), np.uint16)
    src = FakeSource(data, (0, 0, 64, 64), [0], [0], [0], cfg)

    def no_aics():
        raise ImportError("No module named 'aicspylibczi'")

    src.aics = no_aics
    with pytest.raises(ImportError, match="pip install aicspylibczi"):
        plan_scenes(src, cfg, Backend(False), log=lambda *a: None)
