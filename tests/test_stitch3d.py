"""3D registration (`refine.mode = "3d"`) and Z-shifted compositing, against known truth."""

from __future__ import annotations

import functools
import json

import numpy as np
import pytest

from czipipe.compute import Backend
from czipipe.stitch import MosaicCompositor, refine, refine3d

from conftest import FakeSource

TS, PAD = 256, 64
POSITIONS = {m: p for m, p in
             enumerate([(x, y) for y in (0, 192, 384) for x in (0, 192, 384)])}
NZ, ZPAD = 12, 4      # planes per tile, and spare truth either side for tiles offset in Z

# (dy, dx, dz) stage error per tile. Tile 0 is at the origin and the median dz is 0, which
# is where registration fixes what it cannot measure, so it should return exactly this.
PERT3 = {0: (0, 0, 0), 1: (3, -4, 1), 2: (-2, 5, -1), 3: (4, 2, 0), 4: (-3, -3, 2),
         5: (1, 6, 0), 6: (2, -2, -2), 7: (-5, 4, 1), 8: (0, 3, 0)}
Z_ONLY = {m: (0, 0, dz) for m, (_, _, dz) in PERT3.items()}
XY_ONLY = {m: (dy, dx, 0) for m, (dy, dx, _) in PERT3.items()}
FLAT = {m: (0, 0, 0) for m in PERT3}


@functools.lru_cache(maxsize=4)
def smooth_volume(nz, seed, featureless=()):
    """Band-limited noise in 3D: structure for phase correlation to lock onto, with
    neighbouring planes alike but not identical, as in a real stack. Planes listed in
    `featureless` are replaced by a constant."""
    side = 640 + 2 * PAD
    shape = (nz, side, side)
    rng = np.random.default_rng(seed)
    kz, ky, kx = np.meshgrid(np.fft.fftfreq(nz), np.fft.fftfreq(side),
                             np.fft.rfftfreq(side), indexing="ij")
    k = np.sqrt((0.3 * kz) ** 2 + ky ** 2 + kx ** 2)
    g = np.fft.irfftn(np.fft.rfftn(rng.normal(0, 1, shape)) * np.exp(-k * 30),
                      s=shape, axes=(0, 1, 2))
    g = (g - g.min()) / (g.max() - g.min())
    g = (g * 30000 + 1000).astype(np.uint16)
    g[list(featureless)] = 16000
    g.setflags(write=False)
    return g


def make_mosaic(pert=PERT3, nz=NZ, seed=3, noise=0.0, deep=None, featureless=()):
    """Truth volume, per-tile (Z, Y, X) stacks cut at their TRUE positions, metadata rects.

    Tile m's plane k shows the truth at plane k + dz, displaced by (dy, dx): its stack
    was acquired dz planes off, so its content belongs dz planes further along than the
    file says. `deep` = (first_plane, pert) swaps in a different error from that truth
    plane on, for a tile whose offset changes with depth.
    """
    gt = smooth_volume(nz + 2 * ZPAD, seed, tuple(z + ZPAD for z in featureless))
    rng = np.random.default_rng(seed + 1)
    tiles, meta = {}, {}
    for m, (x, y) in POSITIONS.items():
        planes = []
        for k in range(nz):
            dy, dx, dz = (deep[1] if deep and k >= deep[0] else pert)[m]
            y0, x0 = y + dy + PAD, x + dx + PAD
            planes.append(gt[k + dz + ZPAD, y0:y0 + TS, x0:x0 + TS])
        t = np.stack(planes).astype(np.float32)
        assert t.shape == (nz, TS, TS), f"tile {m} clipped by the truth bounds"
        if noise:
            t = t + rng.normal(0, noise, t.shape)
        tiles[m] = np.clip(t, 0, 65535).astype(np.uint16)
        meta[m] = (x, y, TS, TS)                       # metadata omits the perturbation
    return gt, tiles, meta


def source(cfg, tiles, nz=NZ, sel_z=None):
    """A source whose file holds planes 0..nz-1 of every tile."""
    sel_z = list(range(nz)) if sel_z is None else sel_z
    src = FakeSource(np.zeros((1, len(sel_z), 1, 4, 4), np.uint16), (0, 0, 640, 640),
                     [0], sel_z, [0], cfg, tiles=tiles)
    aics = src.aics()
    aics.dims["Z"] = (0, nz)
    src.aics = lambda: aics
    return src


def register(cfg, tiles, meta, nz=NZ, **kw):
    return refine3d.register(source(cfg, tiles, nz), cfg, Backend(False), None, meta,
                             {"T": 0, "Z": nz // 2, "C": 0}, **kw)


@pytest.fixture
def cfg3(cfg):
    cfg.stitch_mode = "refine"
    cfg.refine.mode, cfg.refine.slab = "3d", NZ
    return cfg


# ---------------------------------------------------------------- registration
def test_registration_recovers_xy_and_z_exactly(cfg3):
    gt, tiles, meta = make_mosaic()
    corr, log = register(cfg3, tiles, meta)
    assert corr == PERT3
    assert log["n_accepted"] == log["n_pairs"] == 20
    assert log["rms_residual_px"] < 0.01 and log["rms_residual_z"] < 0.01
    assert log["mode"] == "3d" and log["z_correction_range"] == [-2, 2]
    assert log["corrections"]["4"] == [-3, -3, 2]


def test_a_z_offset_is_found_where_the_stage_xy_is_already_right(cfg3):
    """What a plane-by-plane registration cannot see at all."""
    gt, tiles, meta = make_mosaic(Z_ONLY)
    corr, log = register(cfg3, tiles, meta)
    assert corr == Z_ONLY
    assert log["max_correction_px"] == 0


def test_no_z_offset_gives_the_xy_shifts_2d_registration_finds(cfg3):
    gt, tiles, meta = make_mosaic(XY_ONLY)
    corr, log = register(cfg3, tiles, meta)
    assert corr == XY_ONLY
    assert log["z_correction_range"] == [0, 0] and log["incomplete_planes"] == []

    flat, _ = refine.register(source(cfg3, tiles), cfg3, Backend(False), None, meta,
                              {"T": 0, "Z": NZ // 2, "C": 0})
    assert {m: v[:2] for m, v in corr.items()} == flat


@pytest.mark.parametrize("noise", [300, 800, 2000])
def test_registration_is_exact_under_noise(cfg3, noise):
    """Phase correlation under noise puts a peak a step off as often as not. The NCC walk
    is what settles it: over 6 seeds every tile is exact up to sd=2000, where taking the
    peaks as they come left a third of the tiles a step out."""
    gt, tiles, meta = make_mosaic(noise=noise)
    corr, log = register(cfg3, tiles, meta)
    assert corr == PERT3
    assert log["n_accepted"] == log["n_pairs"]
    assert log["rms_residual_px"] < 0.01 and log["rms_residual_z"] < 0.01


def test_heavy_noise_costs_a_step_at_most(cfg3):
    """At sd=4000 the noise is as large as the signal. Measured over 6 seeds: 52 of 54
    tiles exact and none out by more than one step, with one pair in 120 rejected."""
    gt, tiles, meta = make_mosaic(noise=4000)
    corr, log = register(cfg3, tiles, meta)
    assert max(abs(a - b) for m in POSITIONS for a, b in zip(corr[m], PERT3[m])) <= 1
    assert log["n_accepted"] >= log["n_pairs"] - 1


# ---------------------------------------------------------------- NCC decides
def relative(pair, pert=PERT3):
    return tuple(b - a for a, b in zip(pert[pair["i"]], pert[pair["j"]]))


def with_proposals(monkeypatch, change):
    """Make phase correlation propose `change(dy, dx, dz)` instead of what it found."""
    real = refine3d.peaks

    def wrong(backend, a, b, n=refine3d.PEAKS, workers=1):
        shifts, sharp = real(backend, a, b, n, workers)
        return [change(*s) for s in shifts], sharp

    monkeypatch.setattr(refine3d, "peaks", wrong)


@pytest.mark.parametrize("change", [
    lambda dy, dx, dz: (dy, dx, -dz),             # Z sign inverted
    lambda dy, dx, dz: (-dy, -dx, dz),            # XY sign inverted
    lambda dy, dx, dz: (-dy, -dx, -dz),           # both
    lambda dy, dx, dz: (dy + 40, dx - 40, dz),    # nowhere near
])
def test_a_wrong_proposal_is_not_what_gets_accepted(cfg3, monkeypatch, change):
    """Phase correlation only proposes. A shift is accepted where the NCC stops
    improving, so a proposal of the wrong sign, or nowhere near, is either walked to the
    true shift or rejected: it is never accepted as proposed. Measured on this mosaic,
    17 to 19 of the 20 pairs end exactly right in each case. (The rest are corner
    overlaps, 64 px square, where nothing within reach matches well.)"""
    gt, tiles, meta = make_mosaic()
    with_proposals(monkeypatch, change)
    corr, log = register(cfg3, tiles, meta)
    right = 0
    for p in log["pairs"]:
        want = relative(p)
        proposed = change(*want)
        if p["accepted"]:
            if max(abs(a - b) for a, b in zip(want, proposed)) > 1:      # a wrong proposal
                assert max(abs(a - b) for a, b in zip(p["shift"], proposed)) > 1, p
            right += tuple(p["shift"]) == want
    assert right >= 17
    assert sum(corr[m] == PERT3[m] for m in POSITIONS) >= 7


def test_the_stage_position_is_a_proposal_too(cfg3, monkeypatch):
    """With every correlation peak nowhere near, a pair whose true offset is within
    reach of the stage position is still found, by walking from there."""
    one = dict(FLAT)
    one[4] = (-3, 5, 2)
    gt, tiles, meta = make_mosaic(one)
    with_proposals(monkeypatch, lambda dy, dx, dz: (dy + 40, dx - 40, dz))
    corr, log = register(cfg3, tiles, meta)
    assert corr == one
    assert log["n_accepted"] == log["n_pairs"] and log["n_confirming_metadata"] == 12


def test_a_walk_that_never_settles_is_rejected(cfg3, monkeypatch):
    """Still improving after CLIMB steps means the proposal was nowhere near a match.
    Keeping wherever the walk had got to would be a guess with a good score."""
    real = refine3d.peaks

    def three_planes_off(backend, a, b, n=refine3d.PEAKS, workers=1):
        (dy, dx, dz), sharp = real(backend, a, b, 1)[0][0], 1.0
        return [(dy, dx, dz + 3)], sharp

    monkeypatch.setattr(refine3d, "peaks", three_planes_off)
    cfg3.refine.max_shift_z = 5
    gt, tiles, meta = make_mosaic()

    monkeypatch.setattr(refine3d, "CLIMB", 2)
    _, log = register(cfg3, tiles, meta)
    unsettled = [p for p in log["pairs"] if "no settled match within 2 steps" in p["reason"]]
    assert len(unsettled) >= 5
    assert all(tuple(p["shift"]) == relative(p) for p in log["pairs"] if p["accepted"])

    monkeypatch.setattr(refine3d, "CLIMB", 16)
    corr, log = register(cfg3, tiles, meta)
    assert corr == PERT3 and log["n_accepted"] >= log["n_pairs"] - 2


def test_a_z_offset_beyond_max_shift_z_is_rejected_not_clipped(cfg3):
    cfg3.refine.max_shift_z = 1
    gt, tiles, meta = make_mosaic(Z_ONLY)
    corr, log = register(cfg3, tiles, meta)
    for p in log["pairs"]:
        dz = abs(relative(p, Z_ONLY)[2])
        assert p["accepted"] == (dz <= 1), p
        if dz > 1:
            assert "exceeds the +/-1 plane(s) searched" in p["reason"]
    assert log["max_shift_z"] == 1


def test_an_offset_one_plane_past_the_range_is_not_accepted_at_its_edge(cfg3):
    """Found on a real CZI. A slab of 8 planes searches +/-3, and for a pair 4 planes
    apart the strongest peak inside that range sits at 3: a real improvement on the
    stage position, so it was accepted, a plane short. The walk now steps on to 4,
    finds it better, and the pair is rejected as out of range."""
    cfg3.refine.slab, cfg3.refine.planes = 8, 1
    gt, tiles, meta = make_mosaic()
    corr, log = register(cfg3, tiles, meta)
    assert log["max_shift_z"] == 3
    for p in log["pairs"]:
        want = relative(p)
        if abs(want[2]) > 3:
            assert not p["accepted"] and "exceeds the +/-3 plane(s) searched" in p["reason"]
        else:
            assert p["accepted"] and tuple(p["shift"]) == want, p
    assert corr == PERT3 and log["rms_residual_z"] == 0


def test_a_sliver_of_overlap_is_not_scored(cfg3):
    """An overlap too small to attempt is too small to score: a few pixels of it can
    match anything, and a walk must not be able to end there."""
    gt, tiles, meta = make_mosaic(XY_ONLY)
    for c in refine3d.candidates(meta, 32, 100):
        (va, vb), = refine3d.load(lambda z, m: tiles[m][z], list(range(NZ)), [c], np.uint16)
        oy, ox = c["overlap"]
        if oy == ox:
            continue                                          # corners: either axis would do
        # tile j is the lower or right-hand one, so moving it further that way narrows
        # the 64 px they share
        thin = (33, 0, 0) if oy < ox else (0, 33, 0)          # leaves 31 px
        okay = (32, 0, 0) if oy < ox else (0, 32, 0)          # leaves 32
        assert refine3d.score(c, va, vb, thin) == 0.0
        assert refine3d.score(c, va, vb, okay) != 0.0


def test_the_search_along_z_is_limited_by_the_slab(cfg3):
    """A slab can only vouch for a Z offset that leaves most of it overlapping."""
    cfg3.refine.slab, cfg3.refine.max_shift_z = 6, 5
    gt, tiles, meta = make_mosaic(FLAT)
    _, log = register(cfg3, tiles, meta)
    assert log["max_shift_z"] == 2
    assert "searched to +/-2 planes" in refine3d.format_report(log)


def test_uncorrelated_tile_is_rejected_not_guessed(cfg3):
    gt, tiles, meta = make_mosaic()
    tiles[4] = np.random.default_rng(0).integers(1000, 31000, tiles[4].shape, dtype=np.uint16)
    corr, log = register(cfg3, tiles, meta)
    assert corr[4] == (0, 0, 0) and log["unconstrained_tiles"] == [4]
    assert all(not p["accepted"] for p in log["pairs"] if 4 in (p["i"], p["j"]))
    assert all(corr[m][:2] == PERT3[m][:2] for m in POSITIONS if m != 4)
    assert "[warn] 1 tile(s) unconstrained [4]" in refine3d.format_report(log)


def test_stage_positions_that_are_already_right_are_confirmed(cfg3):
    """A pair that matches best where the stage put it is evidence too. Throwing it away
    would leave every such tile unconstrained, and free to drift with its neighbours."""
    gt, tiles, meta = make_mosaic(FLAT)
    corr, log = register(cfg3, tiles, meta)
    assert corr == FLAT
    assert log["n_accepted"] == log["n_confirming_metadata"] == log["n_pairs"]
    assert log["unconstrained_tiles"] == []
    report = refine3d.format_report(log)
    assert "match best where the stage put them" in report and "[warn]" not in report


def test_one_displaced_tile_moves_alone(cfg3):
    """Its neighbours agree with each other where they are, so they must stay there."""
    one = dict(FLAT)
    one[4] = (-3, 5, 2)
    gt, tiles, meta = make_mosaic(one)
    corr, log = register(cfg3, tiles, meta)
    assert corr == one
    assert log["n_confirming_metadata"] == 12 and log["n_accepted"] == 20


# ---------------------------------------------------------------- the global solve
def test_solve_does_not_compound_error():
    ms = [0, 1, 2]
    edges = [(0, 1, (10, 0, 1), 1.0), (1, 2, (10, 0, 1), 1.0), (0, 2, (18, 0, 2), 1.0)]
    corr, rms, rms_z, orphans, apart, dropped = refine3d.solve(ms, edges)
    assert orphans == [] and corr[0][:2] == (0, 0)
    assert 18 <= corr[2][0] <= 20 and rms > 0 and rms_z == 0
    assert dropped == [] and max(abs(v) for d in apart for v in d) <= 1


def test_z_keeps_the_median_tile_on_its_own_planes():
    """XY is pinned at the first tile; pinning Z there would shift three stacks to suit
    one, and every shifted stack loses planes at an end."""
    ms = [0, 1, 2, 3]
    edges = [(0, m, (0, 5 * m, 3), 1.0) for m in (1, 2, 3)]       # tile 0 is the odd one
    corr, *_ = refine3d.solve(ms, edges)
    assert corr == {0: (0, 0, -3), 1: (0, 5, 0), 2: (0, 10, 0), 3: (0, 15, 0)}


def test_a_group_not_joined_to_the_first_tile_keeps_its_median_in_place():
    ms = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1, (2, 0, 0), 1.0),
             (3, 4, (0, 6, 1), 1.0), (4, 5, (0, 6, 1), 1.0)]
    corr, _, _, orphans, _, _ = refine3d.solve(ms, edges)
    assert corr[0] == (0, 0, 0) and corr[1] == (2, 0, 0)
    assert orphans == [2] and corr[2] == (0, 0, 0)
    assert [corr[m] for m in (3, 4, 5)] == [(0, -6, -1), (0, 0, 0), (0, 6, 1)]


def test_no_accepted_pair_leaves_everything_at_metadata():
    corr, rms, rms_z, orphans, apart, dropped = refine3d.solve([0, 1], [])
    assert corr == {0: (0, 0, 0), 1: (0, 0, 0)} and rms is None and orphans == [0, 1]
    assert apart == [] and dropped == []


# ---------------------------------------------------------------- pairs that are wrong
GRID = {0: (0, 0, 0), 1: (3, -4, 1), 2: (-2, 5, -1), 3: (4, 2, 0)}       # a 2 x 2 mosaic
LINKS = [(0, 1), (0, 2), (1, 3), (2, 3), (0, 3), (1, 2)]


def links(wrong=None, weight=1.0):
    """Every pair of the 2 x 2 mosaic measured exactly, bar those in `wrong`, which are
    out by the amount given."""
    out = []
    for i, j in LINKS:
        s = tuple(b - a for a, b in zip(GRID[i], GRID[j]))
        if wrong and (i, j) in wrong:
            s = tuple(a + b for a, b in zip(s, wrong[(i, j)]))
        out.append((i, j, s, weight))
    return out


def test_one_wrong_pair_does_not_move_the_tiles_the_others_agree_on():
    """Found on a real stack: two pairs out by 20 px, among twelve that agreed to within a
    few, pulled four tiles 6 px with them. Plain least squares shares a wrong
    measurement out among every tile it touches."""
    corr, rms, rms_z, orphans, apart, dropped = refine3d.solve(
        [0, 1, 2, 3], links({(1, 3): (20, -6, 0)}))
    assert corr == GRID and dropped == [2] and orphans == []
    assert apart[2] == (-20, 6, 0) and rms == rms_z == 0        # of the five in the fit


def test_a_pair_wrong_along_z_is_left_out_too():
    corr, *_, apart, dropped = refine3d.solve([0, 1, 2, 3], links({(0, 3): (0, 0, 3)}))
    assert corr == GRID and dropped == [4] and apart[4] == (0, 0, -3)


def mosaic_links(wrong=None):
    """All 20 pairs of the 3 x 3 mosaic measured exactly, bar those in `wrong`."""
    pairs = [(c["i"], c["j"]) for c in refine3d.candidates(
        {m: (x, y, TS, TS) for m, (x, y) in POSITIONS.items()}, 32, 100)]
    out = []
    for i, j in pairs:
        s = tuple(b - a for a, b in zip(PERT3[i], PERT3[j]))
        if wrong and (i, j) in wrong:
            s = tuple(a + b for a, b in zip(s, wrong[(i, j)]))
        out.append((i, j, s, 1.0))
    return pairs, out


def test_two_wrong_pairs_go_one_at_a_time():
    """As on the real stack: two of twenty out by 20 px or so, the rest agreeing."""
    pairs, edges = mosaic_links({(1, 4): (20, 6, 0), (6, 7): (-24, 5, 2)})
    corr, rms, rms_z, _, apart, dropped = refine3d.solve(sorted(POSITIONS), edges)
    assert corr == PERT3 and rms == rms_z == 0
    assert sorted(pairs[k] for k in dropped) == [(1, 4), (6, 7)]
    assert apart[pairs.index((6, 7))] == (24, -5, -2)


def test_two_wrong_pairs_in_six_cannot_be_told_apart():
    """With a third of the measurements wrong and every tile touched by one, there is no
    agreeing majority, and nothing is left out: the fit is the least-squares compromise."""
    *_, dropped = refine3d.solve([0, 1, 2, 3], links({(1, 3): (20, 0, 0), (0, 2): (-24, 5, 2)}))
    assert dropped == []


def test_a_tiles_only_link_is_never_left_out():
    """Nothing contradicts it, so there is nothing for it to be far from."""
    chain = [(0, 1, (50, 0, 0), 1.0), (1, 2, (-70, 30, 4), 1.0), (2, 3, (9, 9, -5), 1.0)]
    corr, rms, rms_z, _, apart, dropped = refine3d.solve([0, 1, 2, 3], chain)
    assert dropped == [] and rms == rms_z == 0
    assert corr[3][:2] == (-11, 39)


def test_pairs_that_all_disagree_a_little_all_stay():
    """Tissue that moved between tiles leaves every pair a few pixels from any fit.
    There is no agreeing majority to side with, and leaving pairs out until the rest
    agreed would keep whichever few happened to."""
    scatter = {(0, 1): (5, -4, 0), (0, 2): (-5, 3, 1), (1, 3): (4, 5, -1),
               (2, 3): (-4, -5, 0), (0, 3): (5, 5, 1), (1, 2): (-5, 4, -1)}
    corr, rms, _, _, apart, dropped = refine3d.solve([0, 1, 2, 3], links(scatter))
    assert dropped == [] and rms > 3
    assert max(abs(a - b) for m in GRID for a, b in zip(corr[m], GRID[m])) <= 4


def test_one_route_against_one_pair_is_settled_by_weight_not_by_leaving_out():
    """Three tiles in a ring: against any one pair there is only the other two, taken
    together, so none can be called wrong. Each counts for as much as it matched."""
    ring = [(0, 1, (10, 0, 0), 1.0), (1, 2, (10, 0, 0), 1.0), (0, 2, (40, 0, 0), 1.0)]
    corr, *_, dropped = refine3d.solve([0, 1, 2], ring)
    assert dropped == [] and corr[2] == (33, 0, 0)        # 20 px shared out three ways
    ring[2] = (0, 2, (40, 0, 0), 0.05)                    # a far weaker match
    corr, *_, dropped = refine3d.solve([0, 1, 2], ring)
    assert dropped == [] and corr[2] == (20, 0, 0)


def test_a_wrong_measurement_is_left_out_of_the_mosaic(cfg3, monkeypatch):
    """Through the registration itself: one pair made to report a confident wrong shift."""
    real = refine3d.measure

    def one_pair_wrong(backend, rc, max_z, c, va, vb, workers=1):
        o = real(backend, rc, max_z, c, va, vb, workers)
        if (c["i"], c["j"]) == (4, 5):
            o["shift"] = (o["shift"][0] + 25, o["shift"][1] - 9, o["shift"][2] + 2)
            o["got"] = o["at"][o["shift"]] = 0.9
        return o

    monkeypatch.setattr(refine3d, "measure", one_pair_wrong)
    gt, tiles, meta = make_mosaic()
    corr, log = register(cfg3, tiles, meta)
    assert corr == PERT3
    assert log["left_out_of_fit"] == [[4, 5]] and log["left_out_by"] == [25, 2]
    wrong = next(p for p in log["pairs"] if (p["i"], p["j"]) == (4, 5))
    assert wrong["accepted"] and wrong["left_out_of_fit"] and wrong["from_fit"] == [-25, 9, -2]
    assert sum(p.get("left_out_of_fit", False) for p in log["pairs"]) == 1
    assert log["rms_residual_px"] == 0
    report = refine3d.format_report(log)
    assert "20/20 pairs accepted, RMS residual 0.00 px, 0.00 planes in Z over the 19 in the fit" in report
    assert "[warn] 1 accepted pair(s) left out of the fit (4-5): up to 25 px / 2 planes" in report
    json.dumps(log)


# ---------------------------------------------------------------- slabs
def test_slabs_are_spread_and_never_overlap():
    assert refine3d.sample_slabs(list(range(100)), 5, 16) == [
        list(range(a, a + 16)) for a in (2, 22, 42, 62, 82)]
    assert refine3d.sample_slabs(list(range(40)), 5, 16) == [
        list(range(2, 18)), list(range(22, 38))]                  # only two fit
    assert refine3d.sample_slabs(list(range(10, 22)), 5, 16) == [list(range(10, 22))]
    assert refine3d.sample_slabs(list(range(0, 60, 2)), 1, 8) == [list(range(25, 33))]


@pytest.mark.parametrize("count,n,depth", [(count, n, depth)
                                           for count in (3, 16, 17, 50, 101)
                                           for n in (1, 3, 5) for depth in (3, 8, 16)])
def test_an_odd_number_of_slabs_includes_the_middle_plane(count, n, depth):
    sel = list(range(4, 4 + count))
    slabs = refine3d.sample_slabs(sel, n, depth)
    flat = [z for s in slabs for z in s]
    assert len(flat) == len(set(flat)) and set(flat) <= set(sel)
    assert len({len(s) for s in slabs}) == 1 and len(slabs[0]) == min(depth, count)
    if len(slabs) % 2:
        assert sel[count // 2] in flat


def test_named_planes_are_slab_centres_pulled_inside_the_file():
    sel = list(range(20, 60))
    assert refine3d.sample_slabs(sel, 5, 8, centres=[40]) == [list(range(36, 44))]
    assert refine3d.sample_slabs(sel, 5, 8, centres=[2, 99], bounds=(0, 80)) == [
        list(range(0, 8)), list(range(72, 80))]
    assert refine3d.sample_slabs(sel, 5, 8, centres=[21, 22]) == [list(range(20, 28))]


def test_too_few_planes_for_3d_says_so():
    with pytest.raises(ValueError, match='refine.mode="2d"'):
        refine3d.sample_slabs([7, 8], 5, 16)


# ---------------------------------------------------------------- several slabs
DEEP_NZ = 24
# An 8-plane slab can vouch for a Z offset of 3 planes at most, and PERT3 has neighbours
# 4 apart, so the stacks measured in thin slabs use offsets of a plane either way.
PERT_DEEP = {m: (dy, dx, int(np.sign(dz))) for m, (dy, dx, dz) in PERT3.items()}


def test_a_featureless_slab_does_not_lose_the_registration(cfg3):
    cfg3.refine.slab, cfg3.refine.planes = 8, 3
    gt, tiles, meta = make_mosaic(PERT_DEEP, nz=DEEP_NZ, featureless=range(6, 18))
    corr, log = register(cfg3, tiles, meta, nz=DEEP_NZ)
    assert log["z_slabs"] == [[0, 7], [8, 15], [16, 23]]
    assert corr == PERT_DEEP and log["n_accepted"] == log["n_pairs"]
    usable = [s["usable"] for s in log["depth"]["slabs"]]
    assert usable == [20, 0, 20] and log["depth"]["pairs_differing"] == 0
    assert "[warn]" not in refine3d.format_report(log)


def test_slabs_that_agree_are_reported_as_agreeing(cfg3):
    cfg3.refine.slab, cfg3.refine.planes = 8, 3
    gt, tiles, meta = make_mosaic(PERT_DEEP, nz=DEEP_NZ)
    corr, log = register(cfg3, tiles, meta, nz=DEEP_NZ)
    assert corr == PERT_DEEP
    d = log["depth"]
    assert d["pairs_compared"] == 20 and d["pairs_differing"] == 0
    assert all(s["usable"] == s["agree"] == 20 for s in d["slabs"])
    report = refine3d.format_report(log)
    assert "depth check, slab by slab" in report and "no depth fits a different shift" in report


def test_an_offset_that_changes_with_depth_is_reported_not_averaged(cfg3):
    """The centre tile sits 6 px further right in the deepest third of the stack. One
    translation per tile cannot fit both, so the two slabs that agree must win outright
    (a blend would be wrong everywhere) and the report must say where it does not fit."""
    cfg3.refine.slab, cfg3.refine.planes = 8, 3
    moved = dict(PERT_DEEP)
    moved[4] = (PERT_DEEP[4][0], PERT_DEEP[4][1] + 6, PERT_DEEP[4][2])
    gt, tiles, meta = make_mosaic(PERT_DEEP, nz=DEEP_NZ, deep=(16, moved))
    corr, log = register(cfg3, tiles, meta, nz=DEEP_NZ)
    assert corr == PERT_DEEP
    d = log["depth"]
    assert [s["differ"] for s in d["slabs"]] == [0, 0, 8] and d["max_difference"] == 6
    for p in log["pairs"]:
        assert (p["depth_difference"] > 0) == (4 in (p["i"], p["j"]))
    report = refine3d.format_report(log)
    assert "[warn] 8 pair(s) fit a different shift" in report and "up to 6 px/planes" in report


def test_a_single_slab_gives_no_depth_report(cfg3):
    gt, tiles, meta = make_mosaic()
    _, log = register(cfg3, tiles, meta)
    assert log["depth"] is None and "depth check" not in refine3d.format_report(log)


# ---------------------------------------------------------------- memory
class _CountingAics:
    """Counts tile reads, which now arrive from several threads at once."""

    def __init__(self, inner):
        import threading
        self.inner, self.dims, self.reads = inner, inner.dims, 0
        self.lock = threading.Lock()

    def get_dims_shape(self):
        return self.inner.get_dims_shape()

    def read_image(self, **kw):
        with self.lock:
            self.reads += 1
        return self.inner.read_image(**kw)


def test_batches_bound_memory_without_changing_the_answer(cfg3):
    """`mosaic_cache_mb` caps the overlap volumes held at once. A cap too small for the
    whole mosaic costs extra reads of the tiles shared between batches, never accuracy."""
    gt, tiles, meta = make_mosaic()

    def run(mb):
        cfg3.mosaic_cache_mb = mb
        src = source(cfg3, tiles)
        counting = _CountingAics(src.aics())
        src.aics = lambda: counting
        corr, log = refine3d.register(src, cfg3, Backend(False), None, meta,
                                      {"T": 0, "Z": NZ // 2, "C": 0})
        return corr, log, counting.reads

    whole, log_whole, reads_whole = run(768)
    split, log_split, reads_split = run(12)
    assert whole == split == PERT3
    assert log_whole.pop("memory")["batches_per_slab"] == 1
    assert log_split.pop("memory")["batches_per_slab"] > 1
    assert json.dumps(log_whole, sort_keys=True) == json.dumps(log_split, sort_keys=True)
    assert reads_whole == len(POSITIONS) * NZ               # every tile, once per plane
    assert reads_whole < reads_split < len(log_whole["pairs"]) * 2 * NZ


def test_a_single_pair_over_budget_is_still_measured():
    pair = {"area": 1000}
    assert refine3d.batches([pair, pair, pair], depth=10, itemsize=2, budget=1) == [
        [pair], [pair], [pair]]
    assert refine3d.batches([pair, pair, pair], depth=10, itemsize=2, budget=40000) == [
        [pair, pair], [pair]]


def test_threads_do_not_change_the_answer(cfg3):
    """Tiles are read and pairs measured on several threads. Each pair belongs to one
    thread at a time, so the record must be the same however many there are."""
    gt, tiles, meta = make_mosaic(noise=2000)
    logs = []
    for n in (1, 2, 8):
        cfg3.n_read_threads = n
        corr, log = register(cfg3, tiles, meta)
        assert corr == PERT3 and log.pop("memory")["threads"] == n
        logs.append(json.dumps(log, sort_keys=True))
    assert logs[0] == logs[1] == logs[2]


# ---------------------------------------------------------------- the run's settings
def test_the_transform_runs_on_the_runs_threads_not_the_machines(cfg3, monkeypatch):
    """It used to ask scipy for every CPU on the machine, whatever n_read_threads said
    and whatever a container or scheduler allowed the process."""
    try:
        import scipy.fft as sfft
    except ImportError:
        pytest.skip("scipy not installed")
    asked = []

    def recording(real):
        def call(*a, **kw):
            asked.append(kw.get("workers"))
            return real(*a, **kw)
        return call

    monkeypatch.setattr(sfft, "rfftn", recording(sfft.rfftn))
    monkeypatch.setattr(sfft, "irfftn", recording(sfft.irfftn))
    gt, tiles, meta = make_mosaic()
    for n in (1, 3):
        asked.clear()
        cfg3.n_read_threads = n
        corr, _ = register(cfg3, tiles, meta)
        assert corr == PERT3 and asked and set(asked) == {n}


def test_registration_memory_fits_the_cache_setting(cfg3):
    """Overlap volumes and working memory together, not the volumes alone: the transform
    and the scoring used to come on top of the cap, set by nothing and reported nowhere."""
    gt, tiles, meta = make_mosaic()

    def run(mb):
        cfg3.mosaic_cache_mb = mb
        corr, log = register(cfg3, tiles, meta)
        assert corr == PERT3
        return log["memory"], refine3d.format_report(log)

    m, report = run(768)
    assert m["budget_mb"] == 768 and m["batches_per_slab"] == 1 and m["threads"] == 4
    assert m["peak_mb"] <= 768 and abs(m["peak_mb"] - m["volumes_mb"] - m["working_mb"]) <= 1
    assert f"up to {m['peak_mb']} MB" in report and "within mosaic_cache_mb = 768" in report

    m, report = run(12)                           # room for a few pairs at a time
    assert m["peak_mb"] <= 12 and m["batches_per_slab"] > 1
    assert "batches a slab" in report and "[warn]" not in report

    m, report = run(4)                            # less than one pair and one transform
    assert m["peak_mb"] > 4 and m["batches_per_slab"] == 20
    assert "[warn] that is over mosaic_cache_mb = 4" in report


def test_the_memory_it_reports_is_what_it_uses(cfg3):
    """The figure in the report is a ceiling the registration has to stay under, so it is
    checked against what numpy actually allocated."""
    import tracemalloc

    gt, tiles, meta = make_mosaic()
    src = source(cfg3, tiles)
    register(cfg3, tiles, meta)                               # imports, FFT plans
    tracemalloc.start()
    try:
        base = tracemalloc.get_traced_memory()[0]
        tracemalloc.reset_peak()
        _, log = refine3d.register(src, cfg3, Backend(False), None, meta,
                                   {"T": 0, "Z": NZ // 2, "C": 0})
        used = (tracemalloc.get_traced_memory()[1] - base) / 1e6
    finally:
        tracemalloc.stop()
    m = log["memory"]
    assert m["volumes_mb"] < used <= m["peak_mb"], (used, m)


def test_threads_give_way_to_memory_for_very_large_tiles():
    """Each thread holds a whole tile while it is read. Thirty-two 4096 px tiles at once
    would be more than the cap on their own, so fewer threads are used."""
    plan = refine3d.working_plan
    assert plan([], 2 * 1920 * 1920, 16, 2, 32, 768e6)[0] == 32
    assert plan([], 2 * 4096 * 4096, 16, 2, 32, 768e6)[0] == 11
    assert plan([], 2 * 4096 * 4096, 16, 2, 32, 16e6)[0] == 1
    rects = {m: (x, y, 1920, 1920) for m, (x, y) in
             enumerate((x, y) for y in (0, 1728) for x in (0, 1728, 3456))}
    cands = refine3d.candidates(rects, 32, 100)
    for c in cands:
        c["step"] = refine3d.score_step(c["overlap"][0] * c["overlap"][1] * 16)
    threads, room, m = plan(cands, 2 * 1920 * 1920, 16, 2, 8, 768e6)
    assert (threads, m["batches_per_slab"]) == (8, 1)
    assert m == {"budget_mb": 768, "threads": 8, "working_mb": 112, "volumes_mb": 273,
                 "peak_mb": 385, "batches_per_slab": 1}


def test_volume_ncc_is_the_ncc():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 60000, (7, 90, 110)).astype(np.uint16)
    b = (0.6 * a + rng.normal(0, 9000, a.shape)).clip(0, 65535).astype(np.uint16)
    assert refine3d.volume_ncc(a, b) == pytest.approx(refine.ncc(a, b), abs=1e-6)
    assert refine3d.volume_ncc(a[:, ::2, 1::3], b[:, ::2, 1::3]) == pytest.approx(
        refine.ncc(a[:, ::2, 1::3], b[:, ::2, 1::3]), abs=1e-6)
    assert refine3d.volume_ncc(a, np.full_like(a, 7)) == 0.0          # nothing to match
    assert refine3d.volume_ncc(a[:0], b[:0]) == 0.0


def test_a_large_overlap_is_scored_on_a_lattice():
    assert [refine3d.score_step(v) for v in (0, 3_999_999, 4_000_000, 15_999_999,
                                             16_000_000, 10**9)] == [1, 1, 2, 2, 4, 8]


def test_lattice_scores_agree_with_full_ones(cfg3):
    """What the lattice is for: a quarter (or a sixteenth) of the voxels, the same NCC to
    well inside min_gain, and the same ordering of the shifts being compared."""
    gt, tiles, meta = make_mosaic(noise=800)
    c = refine3d.candidates(meta, 32, 100)[0]
    (va, vb), = refine3d.load(lambda z, m: tiles[m][z], list(range(NZ)), [c], np.uint16)
    true = tuple(b - a for a, b in zip(PERT3[c["i"]], PERT3[c["j"]]))
    shifts = [true, (0, 0, 0), (true[0] + 1, true[1], true[2]), (true[0], true[1], true[2] + 1)]
    full = [refine3d.score(c, va, vb, s) for s in shifts]
    for step in (2, 4):
        coarse = [refine3d.score(dict(c, step=step), va, vb, s) for s in shifts]
        assert max(abs(a - b) for a, b in zip(full, coarse)) < 0.01
        assert np.argsort(coarse).tolist() == np.argsort(full).tolist()


def test_scoring_on_the_kept_pieces_matches_scoring_on_whole_tiles(cfg3):
    """Only the overlap grown by max_shift is held, on the claim that this loses nothing
    for any shift within max_shift. Checked against the 2D score on the whole tiles."""
    gt, tiles, meta = make_mosaic(XY_ONLY)
    for c in refine3d.candidates(meta, 32, 20):
        (va, vb), = refine3d.load(lambda z, m: tiles[m][z], [5], [c], np.uint16)
        for d in [(0, 0), (20, -20), (-20, 20), (7, -13), (-20, -20)]:
            want = refine.score(tiles[c["i"]][5], meta[c["i"]], tiles[c["j"]][5], meta[c["j"]], d)
            assert refine3d.score(c, va, vb, d + (0,)) == pytest.approx(want, abs=1e-6)


# ---------------------------------------------------------------- compositor
def covered(comp, shape, z):
    """Pixels reached by at least one tile that has a plane to give at output plane z."""
    mask = np.zeros(shape, bool)
    for m, (mx, my, mw, mh) in comp.rects.items():
        if comp.source_plane(m, z) is not None:
            mask[my - comp.rect[1]:my - comp.rect[1] + mh,
                 mx - comp.rect[0]:mx - comp.rect[0] + mw] = True
    return mask


def truth_plane(gt, rect, z):
    return gt[z + ZPAD, rect[1] + PAD:rect[1] + PAD + rect[3],
              rect[0] + PAD:rect[0] + PAD + rect[2]]


@pytest.mark.parametrize("blend_width", [None, 13])
def test_composite_is_bit_exact_at_every_plane(cfg3, blend_width):
    """Each tile must be taken from the plane its Z offset says belongs here: with the
    offsets applied, every output plane is one plane of the truth, seams included."""
    cfg3.blend_width = blend_width
    gt, tiles, meta = make_mosaic()
    comp = MosaicCompositor(source(cfg3, tiles), cfg3, None, meta, PERT3)
    r = comp.rect
    for z in range(NZ):
        full = comp.tile(0, z, 0, r[0], r[1], r[2], r[3], 0)
        mask = covered(comp, full.shape, z)
        assert np.array_equal(full[mask], truth_plane(gt, r, z)[mask]), f"plane {z}"
        assert not full[~mask].any()


def test_ignoring_the_z_offsets_does_not_match_the_truth(cfg3):
    """The control for the test above: the same XY corrections without the Z ones."""
    gt, tiles, meta = make_mosaic()
    comp = MosaicCompositor(source(cfg3, tiles), cfg3, None, meta, XY_ONLY)
    r = comp.rect
    full = comp.tile(0, 6, 0, r[0], r[1], r[2], r[3], 0)
    assert not np.array_equal(full, truth_plane(gt, r, 6))


def test_a_tile_with_no_plane_to_give_is_left_out_not_filled(cfg3):
    gt, tiles, meta = make_mosaic()
    comp = MosaicCompositor(source(cfg3, tiles), cfg3, None, meta, PERT3)
    assert comp.z_shift == {m: v[2] for m, v in PERT3.items() if v[2]}
    assert comp.source_plane(4, 0) is None and comp.source_plane(4, 1) is None   # dz +2
    assert comp.source_plane(4, 2) == 0 and comp.source_plane(4, 11) == 9
    assert comp.source_plane(6, 9) == 11 and comp.source_plane(6, 10) is None    # dz -2
    assert comp.source_plane(0, 0) == 0
    x, y, w, h = comp.rects[4]
    middle = comp.tile(0, 0, 0, x + 80, y + 80, 64, 64, 0)       # only tile 4 reaches here
    assert not middle.any()


def test_planes_outside_the_selection_fill_the_ends(cfg3):
    """A tile shifted along Z is taken from wherever its planes are in the file, so a
    selection that stops short of the ends of the stack loses nothing."""
    gt, tiles, meta = make_mosaic()
    sel = list(range(2, NZ - 2))
    src = source(cfg3, tiles, sel_z=sel)
    comp = MosaicCompositor(src, cfg3, None, meta, PERT3)
    r = comp.rect
    for z in sel:
        full = comp.tile(0, z, 0, r[0], r[1], r[2], r[3], 0)
        mask = covered(comp, full.shape, z)
        assert mask.sum() == covered(comp, full.shape, 6).sum()        # nothing missing
        assert np.array_equal(full[mask], truth_plane(gt, r, z)[mask])
    assert refine3d.incomplete_planes(sel, PERT3, (0, NZ)) == []
    assert refine3d.incomplete_planes(list(range(NZ)), PERT3, (0, NZ)) == [0, 1, 10, 11]


def test_two_component_corrections_composite_as_before(cfg3):
    gt, tiles, meta = make_mosaic(XY_ONLY)
    flat = {m: v[:2] for m, v in XY_ONLY.items()}
    a = MosaicCompositor(source(cfg3, tiles), cfg3, None, meta, flat)
    b = MosaicCompositor(source(cfg3, tiles), cfg3, None, meta, XY_ONLY)
    assert a.z_shift == b.z_shift == {} and a.rects == b.rects
    r = a.rect
    assert np.array_equal(a.tile(0, 5, 0, r[0], r[1], r[2], r[3], 0),
                          b.tile(0, 5, 0, r[0], r[1], r[2], r[3], 0))


# ---------------------------------------------------------------- through the pipeline
def plan_for(cfg, tiles, meta, monkeypatch, nz=NZ, msgs=None):
    from czipipe import plan as planmod

    monkeypatch.setattr(planmod, "mosaic_geometry",
                        lambda source, scene_id, ref: (dict(meta), (0, 0)))
    src = source(cfg, tiles, nz)
    log = (lambda *a: None) if msgs is None else msgs.append
    return src, planmod.plan_scenes(src, cfg, Backend(False), log=log)[0]


def test_plan_runs_3d_registration_when_asked(cfg3, monkeypatch):
    gt, tiles, meta = make_mosaic()
    msgs = []
    src, plan = plan_for(cfg3, tiles, meta, monkeypatch, msgs=msgs)
    assert plan.corrections == PERT3 and plan.stitch_log["mode"] == "3d"
    assert plan.compositor.z_shift[4] == 2
    report = "\n".join(str(m) for m in msgs)
    assert "3D registration on 1 slab(s) of 12 planes (Z 0-11)" in report
    assert "Z corrections -2 to +2 planes" in report
    assert "[note] 4 of 12 output plane(s) lack some tiles (Z 0-1, 10-11)" in report
    json.dumps(plan.stitch_log)                                 # goes into .stitch.json


def test_plan_stays_2d_by_default(cfg, monkeypatch):
    gt, tiles, meta = make_mosaic(XY_ONLY)
    cfg.stitch_mode = "refine"
    src, plan = plan_for(cfg, tiles, meta, monkeypatch)
    assert plan.corrections == {m: v[:2] for m, v in XY_ONLY.items()}
    assert "mode" not in plan.stitch_log and plan.compositor.z_shift == {}


def test_plan_takes_slabs_from_refine_planes_and_refine_z(cfg3, monkeypatch):
    gt, tiles, meta = make_mosaic(PERT_DEEP, nz=DEEP_NZ)

    def slabs(**settings):
        for k, v in settings.items():
            setattr(cfg3.refine, k, v)
        _, plan = plan_for(cfg3, tiles, meta, monkeypatch, nz=DEEP_NZ)
        assert plan.corrections == PERT_DEEP
        return plan.stitch_log["z_slabs"]

    assert slabs(slab=8) == [[0, 7], [8, 15], [16, 23]]         # as many of 5 as fit
    assert slabs(planes=1) == [[8, 15]]
    assert slabs(slab=6, planes=2) == [[3, 8], [15, 20]]
    assert slabs(z=[4, 19]) == [[1, 6], [16, 21]]               # named planes win
    assert slabs(z=23) == [[18, 23]]


def test_written_output_holds_the_z_corrected_planes(cfg3, monkeypatch):
    """End to end: plan, write a TIFF, read it back, verify."""
    import tifffile

    from czipipe.geometry import Tiler
    from czipipe.pipeline import _null_progress
    from czipipe.validate import verify
    from czipipe.writers import write_tiff

    gt, tiles, meta = make_mosaic()
    src, plan = plan_for(cfg3, tiles, meta, monkeypatch)
    tiler = Tiler(cfg3.tile, [0], list(range(NZ)), [0])
    path = write_tiff(src, cfg3, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(path) as tf:
        stack = np.stack([p.asarray() for p in tf.pages])
    r = plan.rect
    assert stack.shape == (NZ, r[3], r[2])
    for z in range(NZ):
        mask = covered(plan.compositor, stack[z].shape, z)
        assert np.array_equal(stack[z][mask], truth_plane(gt, r, z)[mask]), f"plane {z}"
    assert verify(src, cfg3, Backend(False), tiler, plan, path, log=lambda *a: None)


def test_fingerprint_changes_only_when_a_tile_moves_along_z(cfg3):
    """An output is reused when the settings that determine its pixels match. A Z offset
    changes the pixels; 3D registration that finds none does not, and existing outputs
    and partial exports must keep being recognised."""
    from czipipe import writers
    from czipipe.geometry import Tiler
    from czipipe.plan import ScenePlan

    gt, tiles, meta = make_mosaic()
    src = source(cfg3, tiles)
    tiler = Tiler(cfg3.tile, [0], list(range(NZ)), [0])

    def payload(corr):
        comp = MosaicCompositor(src, cfg3, None, meta, corr)
        plan = ScenePlan(None, comp.rect, {}, corrections=corr, compositor=comp)
        return writers._settings_payload(src, cfg3, plan, tiler.level_shape(plan.rect, 0))

    flat = payload({m: v[:2] for m, v in XY_ONLY.items()})
    assert "corrections_z" not in flat
    assert payload(XY_ONLY) == flat                              # dz of 0 everywhere
    moved = payload(PERT3)
    assert moved["corrections_z"] == {str(m): v[2] for m, v in PERT3.items() if v[2]}
    assert {k: v for k, v in moved.items() if k != "corrections_z"} == flat
    assert writers._digest(moved) != writers._digest(flat)


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("bad", ["3D", "xyz", 3, None, True])
def test_refine_mode_must_be_2d_or_3d(cfg, bad):
    cfg.refine.mode = bad
    with pytest.raises(ValueError, match="refine.mode"):
        cfg.validate()


@pytest.mark.parametrize("name,bad", [("slab", 2), ("slab", 16.0), ("slab", True),
                                      ("max_shift_z", 0), ("max_shift_z", "2")])
def test_3d_settings_are_checked(cfg, name, bad):
    setattr(cfg.refine, name, bad)
    with pytest.raises(ValueError, match=f"refine.{name}"):
        cfg.validate()


def test_3d_settings_are_serialisable_and_valid_by_default(cfg):
    cfg.refine.mode = "3d"
    d = json.loads(json.dumps(cfg.validate().to_dict()))["refine"]
    assert (d["mode"], d["slab"], d["max_shift_z"]) == ("3d", 16, 5)


# ---------------------------------------------------------------- the FFT underneath
def test_phase_correlation_finds_a_known_shift_in_a_volume():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 48, 40)).astype(np.float32)
    b = np.roll(a, (-2, 5, -3), axis=(0, 1, 2))             # b[k, i, j] = a[k+2, i-5, j+3]
    shifts, sharp = refine3d.peaks(Backend(False), a, b)
    assert shifts[0] == (-5, 3, 2) and sharp > 100


def test_phase_correlation_of_planes_is_unchanged():
    """The 2D path must not move: existing registrations have to come out the same."""
    rng = np.random.default_rng(1)
    a = rng.integers(0, 60000, (96, 64)).astype(np.uint16)
    b = rng.integers(0, 60000, (96, 64)).astype(np.uint16)
    fa, fb = a.astype(np.float32), b.astype(np.float32)
    fa, fb = fa - fa.mean(), fb - fb.mean()
    win = np.hanning(96)[:, None] * np.hanning(64)[None, :]
    R = np.fft.rfft2(fa * win) * np.conj(np.fft.rfft2(fb * win))
    R /= np.maximum(np.abs(R), 1e-12)
    assert np.array_equal(Backend(False).fft_phase_correlation(a, b),
                          np.fft.irfft2(R, s=a.shape))


# ---------------------------------------------------------------- the GPU path
def gpu_stand_in():
    """A Backend whose GPU branch runs, with numpy standing in for CuPy.

    There is no CUDA device to run the real thing on here, and until this the branch had
    never been executed for a volume at all. It exercises the branch's own logic (the
    window, the axes, the dtypes, the lock under threads), not the device or CuPy itself.
    """
    import types

    backend = Backend(False)
    backend.cp = types.SimpleNamespace(asarray=np.asarray, asnumpy=np.asarray, fft=np.fft,
                                       conj=np.conj, maximum=np.maximum, abs=np.abs)
    backend.gpu = True
    return backend


def test_gpu_branch_correlates_a_volume_as_the_cpu_one_does():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(12, 48, 40)).astype(np.float32)
    b = np.roll(a, (-2, 5, -3), axis=(0, 1, 2))
    on_gpu = gpu_stand_in().fft_phase_correlation(a, b)
    on_cpu = Backend(False).fft_phase_correlation(a, b)
    assert on_gpu.shape == on_cpu.shape == a.shape
    assert np.unravel_index(on_gpu.argmax(), a.shape) == (2, 43, 3)      # (2, -5, 3)
    assert np.allclose(on_gpu, on_cpu, atol=1e-4)


def test_gpu_branch_correlates_planes_as_the_cpu_one_does():
    rng = np.random.default_rng(1)
    a = rng.integers(0, 60000, (96, 64)).astype(np.uint16)
    b = rng.integers(0, 60000, (96, 64)).astype(np.uint16)
    assert np.allclose(gpu_stand_in().fft_phase_correlation(a, b),
                       Backend(False).fft_phase_correlation(a, b), rtol=0, atol=1e-12)


def test_registration_through_the_gpu_branch(cfg3):
    gt, tiles, meta = make_mosaic(noise=800)
    corr, log = refine3d.register(source(cfg3, tiles), cfg3, gpu_stand_in(), None, meta,
                                  {"T": 0, "Z": NZ // 2, "C": 0})
    assert corr == PERT3 and log["n_accepted"] == log["n_pairs"]
