"""Registration, compositing and BigStitcher import, against known ground truth."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from czipipe.compute import Backend
from czipipe.stitch import MosaicCompositor
from czipipe.stitch import bigstitcher as bs
from czipipe.stitch import refine

from conftest import FakeSource, smooth_image

TS, STEP = 256, 192
POSITIONS = {m: (tx, ty) for m, (tx, ty) in
             enumerate([(x, y) for y in (0, 192, 384) for x in (0, 192, 384)])}
PERT = {0: (0, 0), 1: (3, -4), 2: (-2, 5), 3: (4, 2),
        4: (-3, -3), 5: (1, 6), 6: (2, -2), 7: (-5, 4), 8: (0, 3)}   # (dy, dx) stage error


PAD = 64          # so tiles displaced toward the edges are still fully inside the truth


def make_mosaic(seed=7, noise=0.0, dead_tile=None):
    """Padded ground truth, per-tile crops at TRUE positions, and erroneous metadata rects.

    Tile m sits at POSITIONS[m] + PERT[m] in truth, but its metadata rect claims
    POSITIONS[m]. Registration should therefore recover exactly PERT (gauge-fixed on
    tile 0). Padding matters: without it the tiles displaced toward the right and bottom
    edges get silently clipped, and every correlation against them is meaningless.
    """
    gt = smooth_image((640 + 2 * PAD, 640 + 2 * PAD), seed=seed)
    rng = np.random.default_rng(seed + 1)
    tiles, meta = {}, {}
    for m, (x, y) in POSITIONS.items():
        tx, ty = x + PERT[m][1] + PAD, y + PERT[m][0] + PAD
        t = gt[ty:ty + TS, tx:tx + TS].astype(np.float32)
        assert t.shape == (TS, TS), f"tile {m} clipped by the ground-truth bounds"
        if noise:
            t = t + rng.normal(0, noise, t.shape)
        tiles[m] = np.clip(t, 0, 65535).astype(np.uint16)
        meta[m] = (x, y, TS, TS)                       # metadata omits the perturbation
    if dead_tile is not None:
        tiles[dead_tile] = rng.integers(1000, 31000, (TS, TS), dtype=np.uint16)
    return gt, tiles, meta


def truth_window(gt, rect):
    """The ground-truth region corresponding to a compositor canvas."""
    return gt[rect[1] + PAD:rect[1] + PAD + rect[3], rect[0] + PAD:rect[0] + PAD + rect[2]]


def covered_mask(comp, shape):
    """Pixels reached by at least one tile; canvas margins are legitimately background."""
    covered = np.zeros(shape, bool)
    for mx, my, mw, mh in comp.rects.values():
        covered[my - comp.rect[1]:my - comp.rect[1] + mh,
                mx - comp.rect[0]:mx - comp.rect[0] + mw] = True
    return covered


def fake_source(cfg, tiles):
    dummy = np.zeros((1, 1, 1, 4, 4), np.uint16)
    return FakeSource(dummy, (0, 0, 640, 640), [0], [0], [0], cfg, tiles=tiles)


REF = {"T": 0, "Z": 0, "C": 0}


# ---------------------------------------------------------------- refine
def test_registration_recovers_exact_offsets(cfg):
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    corr, log = refine.register(src, cfg, Backend(False), None, meta, REF)
    for m in POSITIONS:
        want = (PERT[m][0] - PERT[0][0], PERT[m][1] - PERT[0][1])
        assert corr[m] == want, f"tile {m}: got {corr[m]}, want {want}"
    assert log["rms_residual_px"] < 0.01
    assert log["n_accepted"] == log["n_pairs"]


@pytest.mark.parametrize("noise", [100, 300, 800])
def test_registration_survives_noise(cfg, noise):
    """Shifts are integer-only, so a pixel or two of error under heavy noise is expected.

    Measured envelope on this fixture: mean error stays sub-pixel and the worst tile is
    within 2 px from sd=100 through sd=800 (SNR 300 down to ~37), with essentially every
    pair still accepted. The mean is the meaningful number; the max is one unlucky tile.
    """
    gt, tiles, meta = make_mosaic(noise=noise)
    src = fake_source(cfg, tiles)
    corr, log = refine.register(src, cfg, Backend(False), None, meta, REF)
    errs = [max(abs(corr[m][0] - (PERT[m][0] - PERT[0][0])),
                abs(corr[m][1] - (PERT[m][1] - PERT[0][1]))) for m in POSITIONS]
    assert max(errs) <= 2
    assert np.mean(errs) < 1.0
    assert log["n_accepted"] >= log["n_pairs"] - 1
    assert log["rms_residual_px"] < 1.5


def test_uncorrelated_tile_is_rejected_not_guessed(cfg):
    """A featureless tile must fall back to its metadata position, loudly."""
    gt, tiles, meta = make_mosaic(dead_tile=4)
    src = fake_source(cfg, tiles)
    corr, log = refine.register(src, cfg, Backend(False), None, meta, REF)
    assert corr[4] == (0, 0)
    assert 4 in log["unconstrained_tiles"]
    assert all(corr[m] == PERT[m] for m in (0, 1, 2, 3, 5))
    assert any(not p["accepted"] and 4 in (p["i"], p["j"]) for p in log["pairs"])


def test_inverted_sign_convention_is_caught(cfg, monkeypatch):
    """The NCC guard exists so a sign error cannot silently mirror the mosaic."""
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    real = refine.phase_corr

    def flipped(backend, a, b, ms):
        s, sharp = real(backend, a, b, ms)
        return ((-s[0], -s[1]) if s else None), sharp

    monkeypatch.setattr(refine, "phase_corr", flipped)
    _, log = refine.register(src, cfg, Backend(False), None, meta, REF)
    assert log["n_accepted"] == 0


def test_global_solve_does_not_compound_error():
    """Inconsistent pairwise edges must be averaged, not accumulated round the loop."""
    ms = [0, 1, 2]
    edges = [(0, 1, (10, 0), 1.0), (1, 2, (10, 0), 1.0), (0, 2, (18, 0), 1.0)]
    corr, resid, orphans = refine.solve_global(ms, edges)
    assert orphans == []
    assert 18 <= corr[2][0] <= 20          # compromise between 20 (chained) and 18 (direct)
    assert resid > 0


# ---------------------------------------------------------------- refine, several planes
ZS = [0, 1, 2, 3, 4]
MID = {"T": 0, "Z": 2, "C": 0}          # the plane a single-plane registration would use


def make_stack(perts=None, dead=(), stripes=(), noise=0.0, seed=7):
    """Per-tile (Z, Y, X) stacks with an unrelated image at each plane, and metadata rects.

    `perts[z]` is the stage error at plane z (PERT where not given), planes in `dead`
    hold noise in every tile, and planes in `stripes` vary down the image only, so they
    pin a tile's dy and say nothing whatever about its dx. `noise` is added per tile.
    """
    side = 640 + 2 * PAD
    stacks = {m: [] for m in POSITIONS}
    for z in ZS:
        pert = (perts or {}).get(z, PERT)
        rng = np.random.default_rng(seed + 10 * z + 1)
        if z in stripes:
            gt = np.repeat(rng.integers(1000, 31000, (side, 1), dtype=np.uint16), side, axis=1)
        else:
            gt = smooth_image((side, side), seed=seed + 10 * z)
        for m, (x, y) in POSITIONS.items():
            tx, ty = x + pert[m][1] + PAD, y + pert[m][0] + PAD
            t = gt[ty:ty + TS, tx:tx + TS]
            assert t.shape == (TS, TS), f"tile {m} clipped by the ground-truth bounds"
            if z in dead:
                t = rng.integers(1000, 31000, (TS, TS), dtype=np.uint16)
            elif noise:
                t = np.clip(t + rng.normal(0, noise, t.shape), 0, 65535).astype(np.uint16)
            stacks[m].append(t)
    meta = {m: (x, y, TS, TS) for m, (x, y) in POSITIONS.items()}
    return {m: np.stack(v) for m, v in stacks.items()}, meta


def stack_source(cfg, stacks):
    dummy = np.zeros((1, len(ZS), 1, 4, 4), np.uint16)
    return FakeSource(dummy, (0, 0, 640, 640), [0], ZS, [0], cfg, tiles=stacks)


def test_sample_planes_are_slab_centres():
    assert refine.sample_planes(list(range(100)), 5) == [10, 30, 50, 70, 90]
    assert refine.sample_planes(list(range(7)), 3) == [1, 3, 5]
    assert refine.sample_planes([4, 5, 6], 5) == [4, 5, 6]            # no more than exist
    assert refine.sample_planes(list(range(20, 60, 2)), 1) == [40]    # values, not positions


def test_an_odd_number_of_planes_includes_the_middle_one():
    """So adding planes can only add evidence to what one plane used to see."""
    for count in (1, 2, 9, 10, 101):
        for n in (1, 3, 5, 7):
            sel = list(range(3, 3 + count))
            planes = refine.sample_planes(sel, n)
            assert sel[count // 2] in planes, (count, n)
            assert planes == sorted(set(planes)) and len(planes) == min(n, count), (count, n)


def test_one_plane_gives_no_depth_report(cfg):
    stacks, meta = make_stack()
    _, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta, MID)
    assert log["z_planes"] == [2] and log["depth"] is None
    assert "depth check" not in refine.format_report(log)


def test_a_featureless_middle_plane_no_longer_loses_the_registration(cfg):
    stacks, meta = make_stack(dead={2})
    src = stack_source(cfg, stacks)
    _, alone = refine.register(src, cfg, Backend(False), None, meta, MID)
    assert alone["n_accepted"] == 0                       # all that one plane could give

    corr, log = refine.register(src, cfg, Backend(False), None, meta, MID, z_planes=ZS)
    assert corr == PERT
    assert log["n_accepted"] == log["n_pairs"]
    usable = [p["usable"] for p in log["depth"]["planes"]]
    assert usable[2] == 0 and min(usable[:2] + usable[3:]) == log["n_pairs"]
    assert log["depth"]["pairs_differing"] == 0


def test_planes_that_agree_are_reported_as_agreeing(cfg):
    stacks, meta = make_stack()
    corr, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta,
                                MID, z_planes=ZS)
    assert corr == PERT
    d = log["depth"]
    assert d["pairs_differing"] == 0 and d["max_difference_px"] == 0
    assert d["pairs_compared"] == log["n_accepted"] == log["n_pairs"]
    assert all(p["usable"] == p["agree"] == log["n_pairs"] for p in d["planes"])
    report = refine.format_report(log)
    assert "depth check, Z 0, 1, 2, 3, 4" in report and "[warn]" not in report


@pytest.mark.parametrize("noise", [100, 300, 800, 2000])
def test_noise_alone_is_not_reported_as_a_depth_difference(cfg, noise):
    """Every plane has the same true shift under independent noise. Individual planes
    then propose shifts a pixel or more out, and those must read as undecided, not as
    the sample moving with depth.

    Measured over 12 seeds of this fixture (this test runs one): no pair flagged at
    sd <= 800, one pair in one seed at sd=2000, and about one pair in every third seed
    at sd=4000. Worst-tile error stays at 1 px through sd=2000, where a single plane
    is already out by up to 3 px and has lost a quarter of its pairs."""
    stacks, meta = make_stack(noise=noise)
    src = stack_source(cfg, stacks)
    corr, log = refine.register(src, cfg, Backend(False), None, meta, MID, z_planes=ZS)
    errs = [max(abs(corr[m][0] - PERT[m][0]), abs(corr[m][1] - PERT[m][1])) for m in POSITIONS]
    assert max(errs) <= 1
    assert log["n_accepted"] == log["n_pairs"]
    assert log["depth"]["pairs_differing"] == 0

    alone, _ = refine.register(src, cfg, Backend(False), None, meta, MID)
    worst = max(max(abs(alone[m][0] - PERT[m][0]), abs(alone[m][1] - PERT[m][1]))
                for m in POSITIONS)
    assert max(errs) <= worst                             # never worse than one plane


def test_a_shift_that_changes_with_depth_is_reported_not_averaged(cfg):
    """The centre tile sits 6 px further right in the two deepest planes. One XY shift
    per tile cannot fit both, so the majority must win outright (a blend of the two would
    be wrong at every plane) and the report must say which planes it does not fit."""
    deep = dict(PERT)
    deep[4] = (PERT[4][0], PERT[4][1] + 6)
    stacks, meta = make_stack(perts={3: deep, 4: deep})
    corr, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta,
                                MID, z_planes=ZS)
    assert corr == PERT
    d = log["depth"]
    assert d["max_difference_px"] == 6
    differ = [p["differ"] for p in d["planes"]]
    assert differ[:3] == [0, 0, 0] and differ[3] == differ[4] >= 4
    assert [p["max_difference_px"] for p in d["planes"]] == [0, 0, 0, 6, 6]
    for p in log["pairs"]:
        assert (p["depth_difference_px"] > 0) == (4 in (p["i"], p["j"]))
    report = refine.format_report(log)
    assert "[warn]" in report and "up to 6 px" in report
    import json
    assert json.loads(json.dumps(log))["depth"] == d      # it is written to .stitch.json


def test_two_planes_that_disagree_are_reported_with_the_tile_left_between_them(cfg):
    """With one usable plane each way there is no majority, and each pair settles it on
    its own: some take one plane's shift and some the other's, so the solve leaves the
    tile somewhere between the two, and its inconsistent edges can pull a neighbour by
    a pixel. Nothing better is knowable from two planes; what matters is that the tile
    stays inside their range, the damage stays local, and every pair is reported."""
    deep = dict(PERT)
    deep[4] = (PERT[4][0], PERT[4][1] + 6)
    stacks, meta = make_stack(perts={4: deep}, dead={0, 1, 2})
    corr, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta,
                                MID, z_planes=ZS)
    assert corr[4][0] == PERT[4][0] and PERT[4][1] <= corr[4][1] <= deep[4][1]
    assert all(max(abs(corr[m][0] - PERT[m][0]), abs(corr[m][1] - PERT[m][1])) <= 1
               for m in POSITIONS if m != 4)
    assert log["rms_residual_px"] > 0.5                   # and the fit says so too
    d = log["depth"]
    assert d["pairs_differing"] == 8 and d["max_difference_px"] == 6
    assert all(p["planes_usable"] == 2 for p in log["pairs"])


def test_a_plane_that_cannot_tell_two_shifts_apart_is_not_a_depth_difference(cfg):
    """Structure running across a seam fixes one axis of the shift and leaves the other
    free. Phase correlation still names a shift on the free axis (zero, here), often
    several pixels from the true one, but the chosen shift fits that plane exactly as
    well, so it is no evidence that the shift changes with depth."""
    stacks, meta = make_stack(stripes={4})
    corr, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta,
                                MID, z_planes=ZS)
    assert corr == PERT
    striped = log["depth"]["planes"][4]
    assert striped["usable"] == log["n_pairs"]
    assert striped["ambiguous"] >= 10 and striped["differ"] == 0
    assert log["depth"]["pairs_differing"] == 0
    report = refine.format_report(log)
    assert "undecided between shifts" in report and "[warn]" not in report


def test_a_rejected_pair_names_the_plane_that_came_closest(cfg):
    stacks, meta = make_stack()
    for m in stacks:
        stacks[m][:, :, :] = np.random.default_rng(m).integers(
            1000, 31000, stacks[m].shape, dtype=np.uint16)
    _, log = refine.register(stack_source(cfg, stacks), cfg, Backend(False), None, meta,
                             MID, z_planes=ZS)
    assert log["n_accepted"] == 0 and log["depth"]["pairs_compared"] == 0
    assert all("best of 5 planes, Z " in p["reason"] for p in log["pairs"])
    report = refine.format_report(log)
    assert "featureless at these planes" in report and "depth check" not in report


def test_plan_registers_on_the_configured_planes(cfg, monkeypatch):
    from czipipe import plan as planmod

    stacks, meta = make_stack()
    src = stack_source(cfg, stacks)
    monkeypatch.setattr(planmod, "mosaic_geometry",
                        lambda source, scene_id, ref: (dict(meta), (0, 0)))
    cfg.stitch_mode = "refine"

    def planes(**refine_settings):
        for k, v in refine_settings.items():
            setattr(cfg.refine, k, v)
        plan = planmod.plan_scenes(src, cfg, Backend(False), log=lambda *a: None)[0]
        assert plan.corrections == PERT
        return plan.stitch_log["z_planes"]

    assert planes() == ZS                                 # five by default, all there are
    assert planes(planes=3) == [0, 2, 4]
    assert planes(planes=1) == [2]                        # the old behaviour
    assert planes(z=[1, 3, 3]) == [1, 3]                  # explicit planes win
    assert planes(z=4) == [4]


def test_refine_planes_must_be_a_positive_int(cfg):
    for bad in (0, -1, 2.5, True, "3"):
        cfg.refine.planes = bad
        with pytest.raises(ValueError, match="refine.planes"):
            cfg.validate()


def test_refine_z_must_be_plane_indices(cfg):
    # Looped, not parametrized: tools/minipytest.py unpacks a list-valued parameter.
    for bad in ([], -1, [0, -2], "3", [1.5], True, 2.0):
        cfg.refine.z = bad
        with pytest.raises(ValueError, match="refine.z"):
            cfg.validate()


def test_refine_z_accepts_numpy_indices_and_stays_serialisable(cfg):
    import json

    cfg.refine.z = np.arange(2, 8, 2)
    assert cfg.validate().refine.z_planes() == [2, 4, 6]
    assert json.loads(json.dumps(cfg.to_dict()))["refine"]["z"] == [2, 4, 6]


# ---------------------------------------------------------------- compositor
def test_composite_is_bit_exact(cfg):
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    full = comp.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0)
    covered = covered_mask(comp, full.shape)
    assert np.array_equal(full[covered], truth_window(gt, r)[covered])


def test_tiled_access_matches_whole_composite(cfg):
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    full = comp.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0)
    for y0 in range(0, r[3] - 64, 64):
        for x0 in range(0, r[2] - 64, 64):
            t = comp.tile(0, 0, 0, r[0] + x0, r[1] + y0, 64, 64, 0)
            assert np.array_equal(t, full[y0:y0 + 64, x0:x0 + 64])


def test_cache_stays_within_limit(cfg):
    cfg.mosaic_cache_mb = 1                       # far smaller than the mosaic
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    for y0 in range(0, r[3] - 64, 64):
        comp.tile(0, 0, 0, r[0], r[1] + y0, r[2] // 64 * 64, 64, 0)
    assert comp.cache_mb <= 1 + tiles[0].nbytes / 1e6


def test_pyramid_levels_have_expected_shape(cfg):
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    for lvl in (1, 2):
        step = 1 << lvl
        t = comp.tile(0, 0, 0, r[0], r[1], (r[2] // step) * step, (r[3] // step) * step, lvl)
        assert t.shape == (r[3] // step, r[2] // step)


# ---------------------------------------------------------------- bigstitcher
def write_bdv_xml(path, positions, shifts, vox=(0.23, 0.23, 1.0),
                  extra_lin=None, angles=1, chans=2):
    sd = ET.Element("SpimData", version="0.2")
    seq = ET.SubElement(sd, "SequenceDescription")
    vss = ET.SubElement(seq, "ViewSetups")
    sid, smap = 0, {}
    for ch in range(chans):
        for ang in range(angles):
            for m in sorted(positions):
                vs = ET.SubElement(vss, "ViewSetup")
                ET.SubElement(vs, "id").text = str(sid)
                ET.SubElement(vs, "size").text = "256 256 100"
                v = ET.SubElement(vs, "voxelSize")
                ET.SubElement(v, "unit").text = "micron"
                ET.SubElement(v, "size").text = " ".join(map(str, vox))
                a = ET.SubElement(vs, "attributes")
                ET.SubElement(a, "illumination").text = "0"
                ET.SubElement(a, "channel").text = str(ch)
                ET.SubElement(a, "tile").text = str(m)
                ET.SubElement(a, "angle").text = str(ang)
                smap[sid] = m
                sid += 1
    vrs = ET.SubElement(sd, "ViewRegistrations")
    for s, m in smap.items():
        vr = ET.SubElement(vrs, "ViewRegistration", timepoint="0", setup=str(s))
        gx = (positions[m][0] + shifts[m][1]) * vox[0]
        gy = (positions[m][1] + shifts[m][0]) * vox[1]
        vt = ET.SubElement(vr, "ViewTransform", type="affine")
        ET.SubElement(vt, "Name").text = "Stitching Transform"
        ET.SubElement(vt, "affine").text = f"1 0 0 {gx} 0 1 0 {gy} 0 0 1 0"
        vt2 = ET.SubElement(vr, "ViewTransform", type="affine")
        ET.SubElement(vt2, "Name").text = "calibration"
        lin = np.diag(vox).astype(float)
        if extra_lin is not None and s == 0:
            lin = lin @ extra_lin
        aff = np.zeros((3, 4))
        aff[:3, :3] = lin
        ET.SubElement(vt2, "affine").text = " ".join(f"{v:.10g}" for v in aff.reshape(-1))
    ET.ElementTree(sd).write(str(path))
    return path


def rects_from_positions():
    return {m: (x, y, TS, TS) for m, (x, y) in POSITIONS.items()}


def test_bigstitcher_transforms_recovered_exactly(cfg, tmp_path):
    xml = write_bdv_xml(tmp_path / "ds.xml", POSITIONS, PERT)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    corr, log = bs.corrections(cfg, None, rects_from_positions())
    for m in POSITIONS:
        assert corr[m] == (PERT[m][0] - PERT[0][0], PERT[m][1] - PERT[0][1])
    assert log["max_subpixel_discarded_px"] == pytest.approx(0.0, abs=1e-6)


def test_matching_is_independent_of_setup_id_order(cfg, tmp_path):
    """Setups carry no CZI mosaic index, so tiles are matched by position."""
    xml = write_bdv_xml(tmp_path / "ds.xml", POSITIONS, PERT)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    expected, _ = bs.corrections(cfg, None, rects_from_positions())

    tree = ET.parse(str(xml))
    perm = [4, 7, 1, 8, 0, 3, 6, 2, 5]
    for vs in tree.getroot().findall(".//ViewSetup"):
        t = vs.find("attributes/tile")
        t.text = str(perm[int(t.text)])
    shuffled = tmp_path / "shuf.xml"
    tree.write(str(shuffled))
    cfg.bigstitcher.xml = shuffled
    got, _ = bs.corrections(cfg, None, rects_from_positions())
    assert got == expected


def test_rotation_is_refused(cfg, tmp_path):
    th = np.deg2rad(3)
    R = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    xml = write_bdv_xml(tmp_path / "rot.xml", POSITIONS, PERT, extra_lin=R)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    with pytest.raises(NotImplementedError, match="rotation/shear/scale"):
        bs.corrections(cfg, None, rects_from_positions())


def test_multi_angle_is_refused(cfg, tmp_path):
    xml = write_bdv_xml(tmp_path / "ang.xml", POSITIONS, PERT, angles=2)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    with pytest.raises(NotImplementedError, match="angles"):
        bs.corrections(cfg, None, rects_from_positions())


def test_mismatched_xml_aborts(cfg, tmp_path):
    """An XML describing a different scene must not be applied silently."""
    wrong = {m: (x + (500 if m % 2 else 0), y) for m, (x, y) in POSITIONS.items()}
    xml = write_bdv_xml(tmp_path / "bad.xml", wrong, {m: (0, 0) for m in POSITIONS})
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    with pytest.raises(RuntimeError, match="max_shift"):
        bs.corrections(cfg, None, rects_from_positions())


def test_subpixel_discard_is_reported(cfg, tmp_path):
    sub = {m: (PERT[m][0] + 0.1 * m, PERT[m][1] - 0.07 * m) for m in POSITIONS}
    xml = write_bdv_xml(tmp_path / "sub.xml", POSITIONS, sub)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    _, log = bs.corrections(cfg, None, rects_from_positions())
    assert log["max_subpixel_discarded_px"] > 0.3


def test_bigstitcher_composite_matches_ground_truth(cfg, tmp_path):
    gt, tiles, meta = make_mosaic()
    xml = write_bdv_xml(tmp_path / "ds.xml", POSITIONS, PERT)
    cfg.stitch_mode, cfg.bigstitcher.xml = "bigstitcher", xml
    corr, _ = bs.corrections(cfg, None, meta)
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, corr)
    r = comp.rect
    full = comp.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0)
    covered = covered_mask(comp, full.shape)
    assert np.array_equal(full[covered], truth_window(gt, r)[covered])


def test_composite_is_bit_exact_with_a_non_dyadic_ramp(cfg):
    """The default ramp here is 32 px, a power of two, so v*w/w is exact in float32 and
    the test above could never see truncation. Real tiles (1920 px -> 240 px ramp) are
    not so lucky: truncating knocked ~6% of feathered pixels down by one."""
    cfg.blend_width = 13
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    full = comp.tile(0, 0, 0, r[0], r[1], r[2], r[3], 0)
    covered = covered_mask(comp, full.shape)
    assert np.array_equal(full[covered], truth_window(gt, r)[covered])


# ---------------------------------------------------------------- shared tile cache
class _SlowCountingAics:
    """Serves mosaic tiles slowly, counting decodes: slow enough that every thread asking
    for a tile arrives while the first is still decoding it."""

    def __init__(self, inner, fail_first=0):
        import threading
        self.inner, self.counts, self.fail_first = inner, {}, fail_first
        self.lock = threading.Lock()

    def get_dims_shape(self):
        return self.inner.get_dims_shape()

    def read_image(self, **kw):
        import time
        with self.lock:
            self.counts[kw["M"]] = self.counts.get(kw["M"], 0) + 1
            fail = self.fail_first > 0
            self.fail_first -= 1
        time.sleep(0.05)
        if fail:
            raise OSError("simulated read error")
        return self.inner.read_image(**kw)


def _in_threads(n, fn):
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(n) as ex:
        futures = [ex.submit(fn) for _ in range(n)]
    return futures


def test_concurrent_requests_decode_each_mosaic_tile_once(cfg):
    """Output tiles are smaller than mosaic tiles, so the read threads start on tiles
    that all need the same mosaic tile. Each used to decode its own copy and discard all
    but one; now the first decodes and the rest wait for it."""
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    slow = _SlowCountingAics(src.aics())
    src.aics = lambda: slow
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect
    want = MosaicCompositor(fake_source(cfg, tiles), cfg, None, meta, PERT).tile(
        0, 0, 0, r[0], r[1], 64, 64, 0)

    futures = _in_threads(8, lambda: comp.tile(0, 0, 0, r[0], r[1], 64, 64, 0))
    assert all(np.array_equal(f.result(), want) for f in futures)
    assert slow.counts == {0: 1}, f"decodes per mosaic tile: {slow.counts}"


def test_a_failed_decode_reaches_every_waiter_and_is_retried_afterwards(cfg):
    """Waiters must not hang on a decode that failed, and the failure must not be cached."""
    gt, tiles, meta = make_mosaic()
    src = fake_source(cfg, tiles)
    slow = _SlowCountingAics(src.aics(), fail_first=1)
    src.aics = lambda: slow
    comp = MosaicCompositor(src, cfg, None, meta, PERT)
    r = comp.rect

    futures = _in_threads(4, lambda: comp.tile(0, 0, 0, r[0], r[1], 64, 64, 0))
    assert all(isinstance(f.exception(), OSError) for f in futures)
    assert slow.counts == {0: 1}
    assert comp.tile(0, 0, 0, r[0], r[1], 64, 64, 0).shape == (64, 64)
    assert slow.counts == {0: 2}


# ---------------------------------------------------------------- preview outlines
class _Box:
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h


def _metadata_pipeline(cfg, tiles, meta, msgs):
    """A plain `metadata` plan (no compositor) over a mosaic whose tile index is readable."""
    from czipipe.pipeline import Pipeline
    from czipipe.plan import ScenePlan

    src = fake_source(cfg, tiles)
    real = src.aics

    def aics():
        h = real()
        h.get_mosaic_tile_bounding_box = lambda M, **kw: _Box(*meta[M])
        return h

    src.aics = aics
    p = Pipeline(cfg, log=msgs.append)
    plan = ScenePlan(None, src.rect, {})
    p.source, p.plans = src, [plan]
    return p, plan


def test_metadata_preview_outlines_are_the_stage_positions(cfg):
    gt, tiles, meta = make_mosaic()
    msgs = []
    p, plan = _metadata_pipeline(cfg, tiles, meta, msgs)
    assert p.tile_rects(plan) == meta
    assert not msgs


def test_outlines_follow_the_compositor_when_there_is_one(cfg):
    """With registration the outlines must be where the tiles were actually placed."""
    gt, tiles, meta = make_mosaic()
    p, plan = _metadata_pipeline(cfg, tiles, meta, [])
    plan.compositor = MosaicCompositor(p.source, cfg, None, meta, PERT)
    assert p.tile_rects(plan) is plan.compositor.rects
    assert p.tile_rects(plan) != meta


def test_missing_aicspylibczi_costs_the_outlines_not_the_preview(cfg):
    """Plain `metadata` conversion never needs aicspylibczi; neither may its preview."""
    gt, tiles, meta = make_mosaic()
    msgs = []
    p, plan = _metadata_pipeline(cfg, tiles, meta, msgs)

    def no_aics():
        raise ImportError("No module named 'aicspylibczi'")

    p.source.aics = no_aics
    assert p.tile_rects(plan) == {}
    assert len(msgs) == 1 and "pip install aicspylibczi" in msgs[0]
    [(_, img, lvl)] = p.preview(max_px=1200)
    assert img.shape == (640, 640) and lvl == 0


def test_an_unreadable_tile_index_is_reported_not_raised(cfg):
    gt, tiles, meta = make_mosaic()
    msgs = []
    p, plan = _metadata_pipeline(cfg, tiles, {}, msgs)     # no tile resolves
    assert p.tile_rects(plan) == {}
    assert len(msgs) == 1 and "No mosaic tiles found" in msgs[0]
