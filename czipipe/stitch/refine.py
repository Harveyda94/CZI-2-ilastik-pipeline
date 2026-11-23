"""Translation registration by phase correlation plus a global least-squares solve.

Each candidate shift is scored against the metadata: the overlap NCC is recomputed at
the proposed shift and at zero shift, and the pair is kept only if the shift genuinely
improves the match. That guard is deliberate. Sign and axis-order errors are the classic
failure mode in stitching code and are invisible in the output until you notice the
mosaic is mirrored; if the convention were wrong here, essentially every pair would be
rejected and the report would say so.

Registration is 2D. Each pair is measured at several Z planes and the planes are combined
into one XY shift per pair, which makes the result independent of any single plane having
structure in the overlaps, and lets the report say whether the planes agree. It does not
make the stitch 3D: one XY correction per tile is applied to every plane, and a Z offset
between tiles is not measured at all.
"""

from __future__ import annotations

import numpy as np


def phase_corr(backend, a, b, max_shift):
    """Peak shift s such that b[i, j] ~ a[i + sy, j + sx], plus a peak-sharpness score."""
    cc = backend.fft_phase_correlation(a, b)
    py, px = np.unravel_index(np.argmax(cc), cc.shape)
    sy = py - cc.shape[0] if py > cc.shape[0] // 2 else py
    sx = px - cc.shape[1] if px > cc.shape[1] // 2 else px
    sharp = float(cc.max() / (np.abs(cc).mean() + 1e-12))
    if abs(sy) > max_shift or abs(sx) > max_shift:
        return None, sharp
    return (int(sy), int(sx)), sharp


def overlap_slices(ra, rb, d=(0, 0)):
    """Pixel slices of the region where rect a and shifted rect b overlap."""
    ax, ay, aw, ah = ra
    bx, by, bw, bh = rb
    bx += d[1]
    by += d[0]
    x0, x1 = max(ax, bx), min(ax + aw, bx + bw)
    y0, y1 = max(ay, by), min(ay + ah, by + bh)
    if x1 - x0 < 1 or y1 - y0 < 1:
        return None
    return ((slice(y0 - ay, y1 - ay), slice(x0 - ax, x1 - ax)),
            (slice(y0 - by, y1 - by), slice(x0 - bx, x1 - bx)))


def ncc(a, b):
    a = np.asarray(a, np.float32).ravel()
    b = np.asarray(b, np.float32).ravel()
    a = a - a.mean()
    b = b - b.mean()
    n = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / n) if n > 1e-9 else 0.0


def score(A, ra, B, rb, d):
    sl = overlap_slices(ra, rb, d)
    return 0.0 if sl is None else ncc(A[sl[0]], B[sl[1]])


def solve_global(ms, edges):
    """Weighted least squares over the tile graph, gauge-fixed at the first tile.

    Solving all constraints at once rather than accumulating tile by tile is what stops
    error compounding across the mosaic. Tiles with no accepted edge fall out at zero,
    i.e. they keep their metadata position.
    """
    idx = {m: k for k, m in enumerate(ms)}
    corr = {m: (0, 0) for m in ms}
    if not edges:
        return corr, None, list(ms)

    A = np.zeros((len(edges) + 1, len(ms)), np.float64)
    rhs = np.zeros((len(edges) + 1, 2), np.float64)
    wts = np.ones(len(edges) + 1)
    for k, (mi, mj, s, w) in enumerate(edges):
        A[k, idx[mi]], A[k, idx[mj]] = -1.0, 1.0
        rhs[k] = s
        wts[k] = w
    A[-1, 0], wts[-1] = 1.0, 10.0                    # gauge: pin the first tile

    sol, *_ = np.linalg.lstsq(A * wts[:, None], rhs * wts[:, None], rcond=None)
    sol = np.rint(sol - sol[0]).astype(int)
    corr = {m: (int(sol[idx[m], 0]), int(sol[idx[m], 1])) for m in ms}
    r = A[:-1] @ sol - rhs[:-1]
    residual = float(np.sqrt((r ** 2).sum(axis=1).mean()))
    linked = {m for e in edges for m in e[:2]}
    return corr, residual, [m for m in ms if m not in linked]


AGREE_PX = 1          # shifts are integers, so a true half-pixel shift reads as either side


def sample_planes(sel_z, n):
    """`n` Z planes spread through the selection: the centres of `n` equal slabs.

    Slab centres rather than end-to-end spacing, because the first and last planes of a
    stack are the ones most often above or below the sample. One plane gives the middle
    of the selection, and any odd `n` includes it.
    """
    count = len(sel_z)
    n = max(1, min(int(n), count))
    return [sel_z[i] for i in sorted({int((k + 0.5) * count / n) for k in range(n)})]


def _apart(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def measure(backend, rc, z, A, ra, B, rb, sl):
    """One pair at one plane: the shift it proposes and how that scores against metadata.

    `reason` is empty when the plane is usable, i.e. the shift is within bounds, matches
    well enough, and matches better than leaving the tiles where the stage put them.
    `at` holds this plane's NCC at each shift scored on it, starting with its own.
    """
    s, sharp = phase_corr(backend, A[sl[0]], B[sl[1]], rc.max_shift)
    base = score(A, ra, B, rb, (0, 0))
    got = None if s is None else score(A, ra, B, rb, s)
    if s is None:
        reason = f"shift exceeds max_shift ({rc.max_shift} px)"
    elif got < rc.min_ncc:
        reason = f"ncc {got:.3f} < min_ncc"
    elif got < base + rc.min_gain:
        reason = f"no gain over metadata ({got:.3f} vs {base:.3f})"
    else:
        reason = ""
    return {"z": z, "shift": s, "sharpness": sharp, "base": base, "got": got,
            "at": {} if s is None else {s: got}, "reason": reason, "status": "unusable"}


def choose(obs):
    """The proposed shift that matches best over all of a pair's usable planes together.

    Each usable plane proposes one shift, and by now has scored every proposal (`at`).
    Summing those scores, rather than taking a vote or an average, means a plane counts
    for as much as it has structure to match, a shift fitted to one plane's noise loses
    to one that fits them all, and the answer is always a shift some plane measured,
    never a blend that no plane supports. Ties go to the shift more planes proposed.
    """
    good = [o for o in obs if not o["reason"]]
    if not good:
        return None
    proposed = [o["shift"] for o in good]
    return max(dict.fromkeys(proposed),
               key=lambda s: (sum(o["at"][s] for o in good), proposed.count(s)))


def register(source, cfg, backend, scene_id, rects, ref_plane, progress=None,
             z_planes=None):
    """Per-tile integer corrections {m: (dy, dx)} plus a diagnostics record.

    Each overlapping pair is measured independently at every plane in `z_planes` (the
    reference plane alone if not given), and the planes are then combined per pair, so a
    pair needs only one plane with usable structure. Tiles are read one plane at a time:
    peak memory is one plane of the mosaic however many planes are used.

    The correction is still a single XY shift per tile. Planes that fit a different
    shift from the one chosen are therefore reported rather than acted on; see
    `depth_summary`.
    """
    rc = cfg.refine
    czi = source.aics()
    zs = [ref_plane["Z"]] if z_planes is None else z_planes
    zs = list(dict.fromkeys(int(z) for z in zs))
    if not zs:
        raise ValueError("refine needs at least one Z plane to register on")
    kw = {k: v for k, v in ref_plane.items() if k != "Z"}
    if scene_id is not None and "S" in czi.get_dims_shape()[0]:
        kw["S"] = scene_id

    def read_planes(wanted):
        """(z, {m: tile}) for each (z, [m, ...]) wanted: one plane in memory at a time."""
        todo = [(z, m) for z, need in wanted for m in need]
        count = {z: len(need) for z, need in wanted}
        tiles = {}
        for z, m in (progress(todo) if progress and todo else todo):
            tiles[m] = np.squeeze(czi.read_image(M=m, Z=z, **kw)[0])
            if len(tiles) == count[z]:
                yield z, tiles
                tiles = {}

    ms = sorted(rects)
    cands = []                                   # (mi, mj, slices, overlap, observations)
    for i, mi in enumerate(ms):
        for mj in ms[i + 1:]:
            sl = overlap_slices(rects[mi], rects[mj])
            if sl is None:
                continue
            oy = sl[0][0].stop - sl[0][0].start
            ox = sl[0][1].stop - sl[0][1].start
            if min(oy, ox) < rc.min_overlap:
                continue
            cands.append((mi, mj, sl, [int(oy), int(ox)], []))

    for z, tiles in read_planes([(z, ms) for z in zs]):
        for mi, mj, sl, _, obs in cands:
            obs.append(measure(backend, rc, z, tiles[mi], rects[mi], tiles[mj], rects[mj], sl))

    # Where a pair's usable planes proposed more than one shift, every proposal is scored
    # at each of those planes, so that the choice between them rests on how well each
    # fits the whole stack. That needs the pixels again, for those pairs only.
    contested = {}                               # z -> [(mi, mj, proposals, observation)]
    for mi, mj, _, _, obs in cands:
        good = [o for o in obs if not o["reason"]]
        proposals = {o["shift"] for o in good}
        if len(proposals) > 1:
            for o in good:
                contested.setdefault(o["z"], []).append((mi, mj, proposals, o))
    wanted = [(z, sorted({m for mi, mj, _, _ in contested[z] for m in (mi, mj)}))
              for z in zs if z in contested]
    for z, tiles in read_planes(wanted):
        for mi, mj, proposals, o in contested[z]:
            for s in proposals - set(o["at"]):
                o["at"][s] = score(tiles[mi], rects[mi], tiles[mj], rects[mj], s)

    edges, picks = [], []
    for mi, mj, _, _, obs in cands:
        shift = choose(obs)
        picks.append(shift)
        if shift is None:
            continue
        good = [o for o in obs if not o["reason"]]
        for o in good:
            # A plane that proposed another shift is only evidence of an offset that
            # changes with depth if the chosen shift is measurably worse *at that plane*.
            # Where it scores about as well, the plane cannot tell the two apart
            # (structure running along the seam, or noise moving a weak peak), and
            # calling that a difference would be a false alarm.
            if _apart(o["shift"], shift) <= AGREE_PX:
                o["status"] = "agrees"
            elif o["got"] - o["at"][shift] >= rc.min_gain:
                o["status"] = "differs"
            else:
                o["status"] = "ambiguous"
        edges.append((mi, mj, shift,
                      max(sum(o["at"][shift] - o["base"] for o in good), 1e-3)))

    pairs = []
    for (mi, mj, _, overlap, obs), shift in zip(cands, picks):
        if shift is None:
            # A rejected pair reports the plane that came closest, so the reason given
            # is the one that would have to change for the pair to be accepted.
            shown = max(obs, key=lambda o: -1.0 if o["got"] is None else o["got"])
        else:
            shown = max((o for o in obs if not o["reason"] and o["shift"] == shift),
                        key=lambda o: o["got"])
        reason = shown["reason"]
        if reason and len(zs) > 1:
            reason += f" (best of {len(zs)} planes, Z {shown['z']})"
        differing = [_apart(o["shift"], shift) for o in obs if o["status"] == "differs"]
        pairs.append({
            "i": mi, "j": mj, "overlap_px": overlap,
            "shift": list(shown["shift"]) if shown["shift"] else None,
            "sharpness": round(shown["sharpness"], 1),
            "ncc_metadata": round(shown["base"], 4),
            "ncc_shifted": None if shown["got"] is None else round(shown["got"], 4),
            "accepted": shift is not None, "reason": reason, "z": shown["z"],
            "planes_usable": sum(o["status"] != "unusable" for o in obs),
            "depth_difference_px": max(differing, default=0),
            "planes": [_plane_record(o, shift) for o in obs]})

    corr, residual, orphans = solve_global(ms, edges)
    n_ok = sum(p["accepted"] for p in pairs)
    mags = [abs(v[0]) + abs(v[1]) for v in corr.values()]
    log = {"scene": scene_id, "source": "refine", "reference_plane": ref_plane,
           "z_planes": zs,
           "n_pairs": len(pairs), "n_accepted": n_ok, "rms_residual_px": residual,
           "unconstrained_tiles": orphans, "max_correction_px": max(mags) if mags else 0,
           "depth": depth_summary(zs, pairs) if len(zs) > 1 else None,
           "pairs": pairs, "corrections": {str(m): list(v) for m, v in corr.items()}}
    return corr, log


def _plane_record(o, chosen):
    rec = {"z": o["z"], "shift": list(o["shift"]) if o["shift"] else None,
           "sharpness": round(o["sharpness"], 1), "ncc_metadata": round(o["base"], 4),
           "ncc_shifted": None if o["got"] is None else round(o["got"], 4),
           "status": o["status"]}
    if o["reason"]:
        rec["reason"] = o["reason"]
    elif o["shift"] != chosen:
        rec["ncc_at_chosen_shift"] = round(o["at"][chosen], 4)
    return rec


def depth_summary(zs, pairs):
    """How far the accepted pairs' shifts agree between the planes they were measured at.

    Per plane, of the accepted pairs: how many were usable there, and how many of those
    fit a shift more than AGREE_PX from the one chosen *and* fit it measurably better.
    This is the only view the pipeline has of an XY offset that changes with depth. It
    says nothing about tiles being offset in Z, which no step here measures.
    """
    acc = [p for p in pairs if p["accepted"]]
    planes = []
    for k, z in enumerate(zs):
        obs = [(p, p["planes"][k]) for p in acc]
        differ = [_apart(o["shift"], p["shift"]) for p, o in obs if o["status"] == "differs"]
        planes.append({"z": z,
                       "usable": sum(o["status"] != "unusable" for _, o in obs),
                       "agree": sum(o["status"] == "agrees" for _, o in obs),
                       "ambiguous": sum(o["status"] == "ambiguous" for _, o in obs),
                       "differ": len(differ),
                       "max_difference_px": max(differ, default=0)})
    return {"planes": planes,
            "pairs_compared": sum(p["planes_usable"] > 1 for p in acc),
            "pairs_single_plane": sum(p["planes_usable"] == 1 for p in acc),
            "pairs_differing": sum(p["depth_difference_px"] > 0 for p in acc),
            "max_difference_px": max((p["depth_difference_px"] for p in acc), default=0)}


def format_report(log):
    res = (f"RMS residual {log['rms_residual_px']:.2f} px"
           if log["rms_residual_px"] is not None else "no global fit")
    sid = log["scene"]
    out = [f"scene {sid if sid is not None else '-'}: "
           f"{log['n_accepted']}/{log['n_pairs']} pairs accepted, {res}"]
    orphans = log["unconstrained_tiles"]
    if orphans:
        out.append(f"  [warn] {len(orphans)} tile(s) unconstrained {orphans[:8]}"
                   f"{' ...' if len(orphans) > 8 else ''} - left at metadata positions")
    out.append(f"  max |correction| {log['max_correction_px']} px")
    zs = log.get("z_planes") or [log["reference_plane"]["Z"]]
    if log["n_accepted"] == 0 and log["n_pairs"]:
        where = "this Z" if len(zs) == 1 else "these planes"
        out.append("  [warn] nothing accepted. Check refine.min_ncc / refine.max_shift, or "
                   f"the overlaps may be featureless at {where}. Using metadata positions.")
    elif log["rms_residual_px"] is not None and log["rms_residual_px"] > 2.0:
        out.append(f"  [warn] residual {log['rms_residual_px']:.2f} px is high for pure "
                   "translation; tiles may need rotation/affine (see BigStitcher).")
    out.extend(_format_depth(log, zs))
    return "\n".join(out)


def _format_depth(log, zs):
    d = log.get("depth")
    n = log["n_accepted"]
    if not d or not n:
        return []
    width = max(len(str(z)) for z in zs)
    out = [f"  depth check, Z {', '.join(str(z) for z in zs)}:"]
    for p in d["planes"]:
        line = f"    Z {p['z']:>{width}}: {p['usable']}/{n} accepted pairs usable"
        if p["differ"]:
            line += (f", {p['differ']} fit a different shift "
                     f"(up to {p['max_difference_px']} px away)")
        if p["ambiguous"]:
            line += f", {p['ambiguous']} undecided between shifts"
        out.append(line)
    if d["pairs_differing"]:
        out.append(f"  [warn] {d['pairs_differing']} pair(s) fit a different shift at some "
                   f"depths, by up to {d['max_difference_px']} px. One XY correction per "
                   "tile cannot fit every plane: expect seams offset by about that much "
                   "at the planes marked above.")
    elif d["pairs_compared"]:
        out.append(f"    no depth fits a different shift, in any of the "
                   f"{d['pairs_compared']} pair(s) usable at more than one plane")
    if d["pairs_single_plane"]:
        out.append(f"    {d['pairs_single_plane']} pair(s) were usable at one plane only, "
                   "so were not compared between depths")
    return out
