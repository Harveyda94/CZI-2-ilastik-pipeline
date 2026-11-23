"""Verification of written output, and the ilastik HDF5 compatibility check.

Verification re-reads random tiles from the output and compares them against the same
ROI read fresh from the CZI with the identical transform applied. This catches
tile-ordering bugs, off-by-one errors in the ROI arithmetic and silent codec problems —
the failure modes that otherwise surface months later as a mysteriously shifted mosaic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import tifffile

from .worker import make_worker
from .writers import read_output_tile

ILASTIK_DTYPES = {"uint8", "uint16", "uint32", "int8", "int16", "int32", "float32", "float64"}


def verify(source, cfg, backend, tiler, plan, path, n=None, seed=0, log=print, progress=None):
    n = cfg.verify_samples if n is None else n
    if not n:
        log("verification skipped (verify_samples = 0)")
        return True

    work = make_worker(source, cfg, backend, tiler, plan, 0)
    rng = np.random.default_rng(seed)
    th, tw = tiler.tile
    _, _, _, H, W = tiler.level_shape(plan.rect, 0)[:5]
    n_z, n_c = len(source.sel_z), len(source.sel_c)
    is_h5 = Path(path).suffix.lower() in (".h5", ".hdf5")

    bad = 0
    if is_h5:
        import h5py
        ctx = h5py.File(path, "r")
    else:
        ctx = tifffile.TiffFile(path)
    with ctx as src:
        dset = src[cfg.hdf5_dataset] if is_h5 else None
        it = progress(range(n), desc="verify", leave=False) if progress else range(n)
        for _ in it:
            ti = int(rng.integers(len(source.sel_t)))
            zi = int(rng.integers(n_z))
            ci = int(rng.integers(n_c))
            y0 = int(rng.integers(0, max(1, H)) // th * th)
            x0 = int(rng.integers(0, max(1, W)) // tw * tw)
            h, w = min(th, H - y0), min(tw, W - x0)

            expect = work((source.sel_t[ti], source.sel_z[zi], source.sel_c[ci],
                           plan.rect[0] + x0, plan.rect[1] + y0, w, h))[:h, :w]
            if is_h5:
                got = np.asarray(dset[ti, zi, ci, y0:y0 + h, x0:x0 + w])
            else:
                got = read_output_tile(src, tiler.page_index(ti, zi, ci), y0, x0,
                                       source.samples)[:h, :w]

            if not np.array_equal(expect, got):
                bad += 1
                d = np.abs(expect.astype(np.int32) - got.astype(np.int32))
                log(f"  MISMATCH t={ti} z={zi} c={ci} y={y0} x={x0}  "
                    f"max|d|={d.max()}  n_diff={int((d > 0).sum())}")

    log(f"{Path(path).name}: {n - bad}/{n} sampled tiles identical")
    return bad == 0


def check_no_spaces(*paths):
    """ilastik's CLI mis-parses values containing spaces (cpython bug 22433); only
    --export_source is exempt. Fail here rather than three hours in."""
    bad = [str(p) for p in paths if " " in str(p)]
    if bad:
        raise ValueError("ilastik cannot handle paths containing spaces: " + "; ".join(bad)
                         + "\nMove the file(s) or symlink to a space-free path.")


def check_h5_for_ilastik(path, dataset="/data", expect_axes=None):
    """Validate an HDF5 against what ilastik's loader actually requires.

    Checks follow ilastik's headless documentation and the axistags convention used in
    lazyflow's own reader tests: a vigra-format `axistags` attribute whose length matches
    ndim, an accepted dtype, chunking (unchunked forces whole-array reads), completeness,
    and a path free of spaces.
    """
    import h5py

    ok, notes = True, []
    path = Path(path)
    with h5py.File(path, "r") as f:
        try:
            d = f[dataset]
        except KeyError:
            return False, [f"FAIL dataset {dataset} not present in {path.name}; "
                           f"top-level keys: {list(f.keys())}"]
        notes.append(f"shape {d.shape}  dtype {d.dtype}  chunks {d.chunks}")

        raw = d.attrs.get("axistags")
        if raw is None:
            ok = False
            notes.append("FAIL no 'axistags' attribute; ilastik would have to guess "
                         "(--input_axes is a fallback, but tags are safer)")
        else:
            raw = raw.decode() if isinstance(raw, bytes) else raw
            keys = None
            try:
                keys = "".join(a["key"] for a in json.loads(raw)["axes"])
            except Exception as exc:
                ok = False
                notes.append(f"FAIL axistags is not vigra-format JSON ({exc})")
            if keys is not None:
                notes.append(f"axistags '{keys}'")
                if len(keys) != d.ndim:
                    ok = False
                    notes.append(f"FAIL axistags '{keys}' has {len(keys)} axes but the "
                                 f"dataset has {d.ndim} dimensions (ilastik raises "
                                 "UnsuitedAxistagsException on this)")
                if set(keys) - set("xyzct"):
                    ok = False
                    notes.append(f"FAIL axis keys outside xyzct: {set(keys) - set('xyzct')}")
                for ax in "xy":
                    if ax not in keys:
                        ok = False
                        notes.append(f"FAIL no '{ax}' axis; ilastik requires x and y")
                if expect_axes and keys != expect_axes:
                    notes.append(f"note  axistags '{keys}' != --input_axes '{expect_axes}'; "
                                 "these must agree")

        if d.dtype.name not in ILASTIK_DTYPES:
            ok = False
            notes.append(f"FAIL dtype {d.dtype} is not one ilastik handles")
        if d.chunks is None:
            ok = False
            notes.append("FAIL dataset is contiguous, not chunked; ilastik could not read "
                         "it block-wise and would pull the whole array into RAM")
        if "progress" in f:
            p = f["progress"]
            n = int(p.attrs.get("tiles_written", 0))
            tot = int(p.attrs.get("total_tiles", 0))
            if not bool(p.attrs.get("complete", False)):
                ok = False
                notes.append(f"FAIL conversion is incomplete ({n}/{tot} tiles written); "
                             "re-run the convert step to resume before classifying")
    try:
        check_no_spaces(path)
    except ValueError as exc:
        ok = False
        notes.append(f"FAIL {exc}")

    notes.append(f"ilastik input path: {path}/{dataset.lstrip('/')}")
    return ok, notes


def report_h5_check(path, dataset="/data", expect_axes=None, log=print):
    ok, notes = check_h5_for_ilastik(path, dataset, expect_axes)
    log(f"ilastik compatibility of {Path(path).name}: {'OK' if ok else 'PROBLEMS'}")
    for n in notes:
        log("   " + n)
    return ok
