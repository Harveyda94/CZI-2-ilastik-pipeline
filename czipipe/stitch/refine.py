"""Translation registration by phase correlation plus a global least-squares solve.

Each candidate shift is scored against the metadata: the overlap NCC is recomputed at
the proposed shift and at zero shift, and the pair is kept only if the shift genuinely
improves the match. That guard is deliberate. Sign and axis-order errors are the classic
failure mode in stitching code and are invisible in the output until you notice the
mosaic is mirrored; if the convention were wrong here, essentially every pair would be
rejected and the report would say so.
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


def register(source, cfg, backend, scene_id, rects, ref_plane, progress=None):
    """Per-tile integer corrections {m: (dy, dx)} plus a diagnostics record."""
    rc = cfg.refine
    czi = source.aics()
    kw = dict(ref_plane)
    if scene_id is not None and "S" in czi.get_dims_shape()[0]:
        kw["S"] = scene_id

    ms = sorted(rects)
    planes = {}
    for m in (progress(ms) if progress else ms):
        planes[m] = np.squeeze(czi.read_image(M=m, **kw)[0])

    pairs, edges = [], []
    for i, mi in enumerate(ms):
        for mj in ms[i + 1:]:
            sl = overlap_slices(rects[mi], rects[mj])
            if sl is None:
                continue
            oy = sl[0][0].stop - sl[0][0].start
            ox = sl[0][1].stop - sl[0][1].start
            if min(oy, ox) < rc.min_overlap:
                continue
            s, sharp = phase_corr(backend, planes[mi][sl[0]], planes[mj][sl[1]], rc.max_shift)
            base = score(planes[mi], rects[mi], planes[mj], rects[mj], (0, 0))
            rec = {"i": mi, "j": mj, "overlap_px": [int(oy), int(ox)],
                   "shift": list(s) if s else None, "sharpness": round(sharp, 1),
                   "ncc_metadata": round(base, 4), "ncc_shifted": None,
                   "accepted": False, "reason": ""}
            if s is None:
                rec["reason"] = f"shift exceeds max_shift ({rc.max_shift} px)"
            else:
                got = score(planes[mi], rects[mi], planes[mj], rects[mj], s)
                rec["ncc_shifted"] = round(got, 4)
                if got < rc.min_ncc:
                    rec["reason"] = f"ncc {got:.3f} < min_ncc"
                elif got < base + rc.min_gain:
                    rec["reason"] = f"no gain over metadata ({got:.3f} vs {base:.3f})"
                else:
                    rec["accepted"] = True
                    edges.append((mi, mj, s, max(got - base, 1e-3)))
            pairs.append(rec)
    del planes

    corr, residual, orphans = solve_global(ms, edges)
    n_ok = sum(p["accepted"] for p in pairs)
    mags = [abs(v[0]) + abs(v[1]) for v in corr.values()]
    log = {"scene": scene_id, "source": "refine", "reference_plane": ref_plane,
           "n_pairs": len(pairs), "n_accepted": n_ok, "rms_residual_px": residual,
           "unconstrained_tiles": orphans, "max_correction_px": max(mags) if mags else 0,
           "pairs": pairs, "corrections": {str(m): list(v) for m, v in corr.items()}}
    return corr, log


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
    if log["n_accepted"] == 0 and log["n_pairs"]:
        out.append("  [warn] nothing accepted. Check refine.min_ncc / refine.max_shift, or "
                   "the overlaps may be featureless at this Z. Using metadata positions.")
    elif log["rms_residual_px"] is not None and log["rms_residual_px"] > 2.0:
        out.append(f"  [warn] residual {log['rms_residual_px']:.2f} px is high for pure "
                   "translation; tiles may need rotation/affine (see BigStitcher).")
    return "\n".join(out)
