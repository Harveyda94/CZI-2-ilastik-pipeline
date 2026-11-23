"""Apply per-tile translations from a BigStitcher (BDV SpimData) XML.

This never touches BigStitcher's resaved image data. BigStitcher computes the
registration on its multi-resolution pyramid, where the work is cheap; this package
does the full-resolution compositing straight from the CZI. RAM and disk are unchanged.

Two guards, because the failure here is silent. BigStitcher setups carry no CZI mosaic
index, so tiles are matched by position rather than by assuming the IDs line up; a wrong
match shows up as a correction of about one tile spacing and aborts the run. And since
the compositor can only translate, any view whose transform carries rotation, shear or
scale beyond the calibration is refused rather than having that component dropped.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def parse_bdv_xml(xml_path):
    """View setups and their concatenated affine transforms."""
    root = ET.parse(str(xml_path)).getroot()

    setups = {}
    for vs in root.findall(".//ViewSetups/ViewSetup"):
        sid = int(vs.findtext("id"))
        attrs = {}
        node = vs.find("attributes")
        if node is not None:
            for ch in node:
                try:
                    attrs[ch.tag] = int(ch.text)
                except (TypeError, ValueError):
                    attrs[ch.tag] = ch.text
        size = [int(float(v)) for v in (vs.findtext("size") or "0 0 0").split()]
        setups[sid] = {"attrs": attrs, "size": size}

    regs = {}
    for vr in root.findall(".//ViewRegistrations/ViewRegistration"):
        tp, sid = int(vr.get("timepoint")), int(vr.get("setup"))
        M = np.eye(4)
        for vt in vr.findall("ViewTransform"):
            txt = vt.findtext("affine")
            if not txt:
                continue
            A = np.eye(4)
            A[:3, :] = np.array([float(v) for v in txt.split()], np.float64).reshape(3, 4)
            M = M @ A          # BDV concatenates in listed order; first listed applies last
        regs[(tp, sid)] = M
    return setups, regs


def pixel_positions(setups, regs, timepoint):
    """{setup_id: (x_px, y_px)}, the per-axis scale, and views that are not translations."""
    out, scales, bad = {}, [], []
    for sid in sorted(setups):
        M = regs.get((timepoint, sid))
        if M is None:
            continue
        lin, trans = M[:3, :3], M[:3, 3]
        s = np.linalg.norm(lin, axis=0)                # per-axis scale (the calibration)
        if np.any(s < 1e-12):
            bad.append((sid, "degenerate transform"))
            continue
        resid = lin / s[None, :] - np.eye(3)           # what remains after removing scale
        if np.abs(resid).max() > 1e-3:
            bad.append((sid, f"non-translation component {np.abs(resid).max():.3g}"))
            continue
        scales.append(s)
        out[sid] = (trans[0] / s[0], trans[1] / s[1])
    return out, (scales[0] if scales else np.ones(3)), bad, np.array(scales) if scales else None


def match_by_position(bs_xy, meta_xy):
    """Assign BigStitcher setups to CZI mosaic tiles by position, global offset removed."""
    sids, ms = list(bs_xy), list(meta_xy)
    A = np.array([bs_xy[s] for s in sids], np.float64)
    B = np.array([meta_xy[m] for m in ms], np.float64)
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    cost = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=2)
    try:
        from scipy.optimize import linear_sum_assignment
        ri, ci = linear_sum_assignment(cost)
    except ImportError:                                # greedy fallback
        ri, ci, used_r, used_c = [], [], set(), set()
        for r, c in sorted(np.ndindex(cost.shape), key=lambda rc: cost[rc]):
            if r not in used_r and c not in used_c:
                used_r.add(r)
                used_c.add(c)
                ri.append(r)
                ci.append(c)
        ri, ci = np.array(ri), np.array(ci)
    return {ms[c]: sids[r] for r, c in zip(ri, ci)}, cost[ri, ci]


def corrections(cfg, scene_id, rects):
    """Per-tile integer corrections {m: (dy, dx)} plus a diagnostics record."""
    bc = cfg.bigstitcher
    xml = Path(bc.xml)
    if not xml.is_file():
        raise FileNotFoundError(f"bigstitcher.xml not found: {xml}")
    setups, regs = parse_bdv_xml(xml)

    for key in ("angle", "illumination"):
        vals = {s["attrs"].get(key) for s in setups.values() if key in s["attrs"]}
        if len(vals) > 1:
            raise NotImplementedError(
                f"XML has {len(vals)} {key}s. Fusing multiple {key}s needs BigStitcher's own "
                "fusion; fuse there, then run this package on the fused output with "
                'stitch_mode="metadata".')

    chans = sorted({s["attrs"]["channel"] for s in setups.values() if "channel" in s["attrs"]})
    sub = {sid: s for sid, s in setups.items()
           if not chans or s["attrs"].get("channel") == chans[0]}

    bs_xy, scale, bad, all_scales = pixel_positions(sub, regs, bc.timepoint)
    warnings = []
    if all_scales is not None and np.ptp(all_scales, axis=0).max() > 1e-6 * max(
            1.0, np.abs(all_scales).max()):
        warnings.append(f"view scales are not uniform: min {all_scales.min(axis=0)} "
                        f"max {all_scales.max(axis=0)}")
    if bad and not bc.force:
        raise NotImplementedError(
            f"{len(bad)} view(s) carry rotation/shear/scale beyond calibration "
            f"(e.g. setup {bad[0][0]}: {bad[0][1]}). This compositor only translates. "
            "Fuse in BigStitcher instead, or set bigstitcher.force = True to ignore the "
            "non-translation part (this will misalign those tiles).")
    if not bs_xy:
        raise RuntimeError(f"No usable ViewRegistrations at timepoint {bc.timepoint}")
    if len(bs_xy) != len(rects):
        warnings.append(f"XML has {len(bs_xy)} tiles at this timepoint, the CZI scene has "
                        f"{len(rects)}; matching the overlap only")

    meta_xy = {m: (r[0], r[1]) for m, r in rects.items()}
    pairing, match_cost = match_by_position(bs_xy, meta_xy)

    ref = sorted(pairing)[0]
    dref = np.array(bs_xy[pairing[ref]]) - np.array(meta_xy[ref])
    corr, subpix = {}, []
    for m in sorted(rects):
        if m not in pairing:
            corr[m] = (0, 0)
            continue
        d = np.array(bs_xy[pairing[m]]) - np.array(meta_xy[m]) - dref
        corr[m] = (int(round(d[1])), int(round(d[0])))       # (dy, dx)
        subpix.append(float(np.abs(d - np.round(d)).max()))

    mags = [max(abs(v[0]), abs(v[1])) for v in corr.values()]
    worst = max(mags) if mags else 0
    if worst > bc.max_shift and not bc.force:
        raise RuntimeError(
            f"A correction of {worst} px exceeds bigstitcher.max_shift ({bc.max_shift}). "
            "Corrections that large almost always mean tiles were matched to the wrong "
            "BigStitcher views, not that the stage was that far out. Check that the XML "
            "describes this CZI and this scene. Override with bigstitcher.force = True "
            "only if you are sure.")

    log = {"scene": scene_id, "source": "bigstitcher", "xml": str(xml),
           "timepoint": bc.timepoint, "n_matched": len(pairing),
           "scale_xyz": [float(v) for v in scale],
           "tile_match_residual_px": float(match_cost.max()),
           "max_correction_px": int(worst),
           "max_subpixel_discarded_px": float(max(subpix) if subpix else 0),
           "rejected_views": [[int(a), b] for a, b in bad],
           "warnings": warnings,
           "corrections": {str(m): list(v) for m, v in corr.items()}}
    return corr, log


def format_report(log):
    sid = log["scene"]
    out = [f"scene {sid if sid is not None else '-'}: "
           f"{log['n_matched']} tiles matched from {Path(log['xml']).name}",
           f"  scale (x,y,z) {np.round(log['scale_xyz'], 4)} | "
           f"tile match residual max {log['tile_match_residual_px']:.1f} px",
           f"  max |correction| {log['max_correction_px']} px | sub-pixel discarded by "
           f"rounding {log['max_subpixel_discarded_px']:.2f} px"]
    out += [f"  [warn] {w}" for w in log.get("warnings", [])]
    return "\n".join(out)
