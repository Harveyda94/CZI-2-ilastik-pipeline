"""Output writers: tiled BigTIFF and chunked, resumable HDF5.

Both stream from the same tile iterator, so peak memory is set by how far ahead we read
rather than by the size of the image.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import itertools
import json
import math
import os
import sys
import time
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile

from .geometry import stream
from .worker import make_worker

MICRON = "\u00b5m"
FINGERPRINT_TAG = 65001          # private TIFF tag: the settings a file was written with


# ---------------------------------------------------------------- TIFF codecs
TIFF_CODECS = (None, "none", "zlib", "lzw", "zstd", "jpeg2000")     # all used losslessly


def tiff_compressionargs(compression, level):
    """Codec arguments, chosen so that every supported codec is lossless.

    `level` means different things to different codecs. For zlib and zstd it only trades
    speed for size. For jpeg2000, imagecodecs reads it as a PSNR target in dB, and any
    value from 1 to 1000 switches to the irreversible 9/7 wavelet \u2014 `level=5` would ask
    for 5 dB, which destroys the image. So it is never passed for jpeg2000; leaving it out
    selects the reversible 5/3 wavelet, which is bit-exact.
    """
    if compression in ("zlib", "zstd"):
        return {"level": level}
    return None


def check_tiff_codec(compression, level, what="compression"):
    """Fail at configuration time unless `compression` round-trips bit-exactly here.

    Probes with the exact arguments the writer will use, so a codec that needs a missing
    package (lzw and jpeg2000 always need imagecodecs; zstd does before Python 3.14), or
    settings that would make it lossy, are caught before a long run rather than after.
    """
    if compression not in TIFF_CODECS:
        raise ValueError(f"{what}={compression!r} is not supported; use one of "
                         "None, 'zlib', 'lzw', 'zstd' or 'jpeg2000' (all lossless)")
    probe = np.random.default_rng(0).integers(0, 65535, (32, 32), dtype=np.uint16)
    buf = io.BytesIO()
    try:
        tifffile.imwrite(buf, probe, tile=(16, 16), compression=compression,
                         compressionargs=tiff_compressionargs(compression, level))
        buf.seek(0)
        back = tifffile.imread(buf)
    except (KeyError, ImportError, ValueError) as exc:
        if "imagecodecs" in str(exc):
            raise ImportError(f'{what}="{compression}" needs the imagecodecs package: '
                              "pip install imagecodecs") from exc
        raise
    if not np.array_equal(back, probe):
        raise ValueError(f"{what}={compression!r} is lossy with these settings")


# ---------------------------------------------------------------- atomic output
@contextlib.contextmanager
def atomic_output(path):
    """Yield a temporary sibling of `path`, moved onto `path` only if the block succeeds.

    The final name therefore only ever holds a finished file, which is what lets a re-run
    treat "exists" as "done". A failed or interrupted block removes the temporary file;
    one left by a run that was killed outright is replaced on the next attempt.
    """
    path = Path(path)
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        partial.unlink()
    try:
        yield partial
    except BaseException:
        with contextlib.suppress(OSError):
            partial.unlink(missing_ok=True)
        raise
    try:
        os.replace(partial, path)
    except OSError as exc:
        raise RuntimeError(
            f"{partial.name} was written completely but could not be renamed to "
            f"{path.name} ({exc}). Close any program that has {path.name} open, then "
            "rename the file by hand; its contents are complete.") from exc


# ---------------------------------------------------------------- metadata
def ome_metadata(source, cfg, plan):
    meta = {"axes": "TZCYXS" if source.samples == 3 else "TZCYX",
            "Channel": {"Name": source.channel_names}}
    if plan.scene_id is not None:
        meta["Name"] = f"{cfg.stem}_scene{plan.scene_id:02d}"
    for axis, key in (("X", "PhysicalSizeX"), ("Y", "PhysicalSizeY"), ("Z", "PhysicalSizeZ")):
        if axis in source.scaling:
            meta[key] = source.scaling[axis]
            meta[key + "Unit"] = MICRON
    return meta


def extra_tags(source, cfg):
    """Original CZI XML verbatim in a private tag.

    OME-XML carries what downstream readers understand; the CZI XML records objective
    and detector settings, laser powers, timestamps and stage positions that have no OME
    equivalent, so it is preserved byte-for-byte rather than translated and truncated.

    The XML must be handed over as `bytes`. tifffile enforces `value.encode("ascii")` on
    `str` values written with TIFF type 's', and Zeiss writes a micro sign (U+00B5) in
    the scaling units, so passing a `str` raises "TIFF strings must be 7-bit ASCII" on
    essentially every real file. `bytes` skips that check and is written through
    verbatim; reading the tag back still yields a `str` with the micro sign intact.
    """
    if not cfg.embed_czi_xml or not source.raw_xml:
        return ()
    xml = source.raw_xml
    return [(65000, "s", 0, xml.encode("utf-8") if isinstance(xml, str) else xml, True)]


def provenance(source, cfg, plan, out_path, elapsed, tiles):
    return {
        "source": {"path": str(cfg.czi_path), "bytes": cfg.czi_path.stat().st_size,
                   "pixel_type": source.pixel_type,
                   "bounding_box": {k: list(v) for k, v in source.total_bbox.items()},
                   "scaling_um": source.scaling},
        "output": {"path": str(out_path), "bytes": out_path.stat().st_size,
                   "format": cfg.output_format, "dtype": np.dtype(source.out_dtype).name,
                   "tile": list(cfg.tile), "compression": cfg.compression,
                   "compression_level": cfg.compression_level,
                   "pyramid_levels": cfg.pyramid_levels if cfg.write_pyramid else 0},
        "selection": {"scene": plan.scene_id, "rect_xywh": list(plan.rect),
                      "T": source.sel_t, "Z": source.sel_z, "C": source.sel_c},
        "bit_depth": {"output_bits": cfg.output_bit_depth,
                      "method": cfg.eight_bit_method if cfg.output_bit_depth == 8 else None,
                      "ranges": {str(k): list(v) for k, v in plan.ranges.items()}},
        "flatfield": {"mode": cfg.flatfield.mode,
                      "dark": cfg.flatfield.dark,
                      "min_gain": cfg.flatfield.min_gain,
                      "per_channel": {str(c): f.stats for c, f in plan.flatfields.items()}},
        "stitching": {"mode": cfg.stitch_mode,
                      "blend": cfg.blend if cfg.stitch_mode != "metadata" else None,
                      "log": plan.stitch_log},
        "runtime": {"seconds": round(elapsed, 1), "tiles": tiles,
                    "read_threads": cfg.n_read_threads, "prefetch": cfg.prefetch},
        "versions": {"tifffile": tifffile.__version__, "numpy": np.__version__,
                     "python": sys.version.split()[0]},
    }


def write_sidecars(source, cfg, plan, out_path, elapsed, tiles):
    if cfg.write_sidecars:
        # Appended, never Path.with_suffix(): that treats everything after the last dot in
        # the stem as a suffix, so "brain_0.5x_scene00" would become "brain_0.<sidecar>"
        # and every scene (and every file sharing the prefix) would overwrite the same one.
        base = f"{cfg.stem}{plan.suffix}"
        if source.raw_xml:
            (cfg.out_dir / f"{base}.czi_metadata.xml").write_text(source.raw_xml,
                                                                  encoding="utf-8")
        (cfg.out_dir / f"{base}.conversion.json").write_text(
            json.dumps(provenance(source, cfg, plan, out_path, elapsed, tiles), indent=2))
    if cfg.stitch_report and plan.stitch_log:
        (cfg.out_dir / f"{cfg.stem}{plan.suffix}.stitch.json").write_text(
            json.dumps(plan.stitch_log, indent=2))


# ---------------------------------------------------------------- TIFF
def tiff_fingerprint(source, cfg, plan, shape):
    """Everything that must match for an existing TIFF to stand in for a new one."""
    payload = _settings_payload(source, cfg, plan, shape)
    payload.update({
        "format": "tiff", "compression": cfg.compression,
        "level": cfg.compression_level if cfg.compression in ("zlib", "zstd") else None,
        "pyramid_levels": cfg.pyramid_levels if cfg.write_pyramid else 0,
    })
    return _digest(payload), payload


def tiff_reuse_state(out_path, want_hash, want_payload):
    """('complete'|'mismatch'|'unknown'|'unreadable', detail) for an existing TIFF.

    TIFFs are written under a temporary name and renamed when finished, so a file under
    the final name is complete; what remains is whether it was made with these settings.
    """
    try:
        with tifffile.TiffFile(out_path) as tf:
            tag = tf.pages[0].tags.get(FINGERPRINT_TAG)
            raw = tag.value if tag is not None else None
    except Exception as exc:
        return "unreadable", f"{type(exc).__name__}: {exc}"
    if raw is None:
        return "unknown", None
    try:
        stored = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except ValueError:
        return "unknown", None
    if stored.get("fingerprint") != want_hash:
        if _same_but_uncorrected(stored.get("config"), want_payload):
            return "complete", UNAPPLIED_NOTE
        return "mismatch", _describe_diffs(stored.get("config"), want_payload)
    return "complete", None


def write_tiff(source, cfg, backend, tiler, plan, log=print, progress=None):
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.stem}{plan.suffix}.ome.tif"

    levels = cfg.pyramid_levels if cfg.write_pyramid else 0
    shape0 = tiler.level_shape(plan.rect, 0)
    total = sum(tiler.n_tiles(plan.rect, l) for l in range(levels + 1))
    fp_hash, fp_payload = tiff_fingerprint(source, cfg, plan, shape0)

    if out_path.exists() and not cfg.overwrite:
        state, detail = tiff_reuse_state(out_path, fp_hash, fp_payload)
        if state == "complete":
            log(f"{plan.label}: {out_path.name} is already complete and was written with "
                "these settings; skipping. Set overwrite=True to rebuild.")
            if detail:
                log(f"  [note] {detail}; reused as the uncorrected output it is.")
            return out_path
        if state == "mismatch":
            raise FileExistsError(
                f"{out_path.name} exists but was written with different settings:\n  - "
                + "\n  - ".join(detail) + "\nSet overwrite=True to replace it, or restore "
                "the settings it was made with.")
        why = ("it does not record the settings it was made with (written by an older "
               "version or another program)" if state == "unknown"
               else f"it could not be read ({detail})")
        raise FileExistsError(f"{out_path} exists and cannot be reused: {why}. "
                              "Set overwrite=True to replace it.")

    common = dict(tile=tuple(cfg.tile), dtype=source.out_dtype,
                  photometric="rgb" if source.samples == 3 else "minisblack",
                  compression=cfg.compression,
                  compressionargs=tiff_compressionargs(cfg.compression,
                                                       cfg.compression_level),
                  maxworkers=cfg.n_write_workers)
    stamp = json.dumps({"fingerprint": fp_hash, "config": fp_payload}, sort_keys=True)
    tags = [*extra_tags(source, cfg), (FINGERPRINT_TAG, "s", 0, stamp.encode("ascii"), True)]

    log(f"{plan.label}: {shape0}  {np.dtype(source.out_dtype).name}  "
        f"-> {out_path.name}  ({total} tiles)")

    t0, ok = time.perf_counter(), False
    try:
        # Written as <name>.partial and renamed on success, so the final name only ever
        # holds a finished file and a re-run can safely reuse it.
        with atomic_output(out_path) as tmp, \
                ThreadPoolExecutor(cfg.n_read_threads) as ex, \
                progress(total=total, unit="tile", desc=plan.label) as bar, \
                tifffile.TiffWriter(tmp, bigtiff=True, ome=True) as tif:
            tif.write(
                data=stream(tiler.tasks(plan.rect, 0),
                            make_worker(source, cfg, backend, tiler, plan, 0),
                            ex, cfg.prefetch, bar),
                shape=shape0, subifds=levels or None,
                metadata=ome_metadata(source, cfg, plan),
                extratags=tags, **common)
            for lvl in range(1, levels + 1):
                tif.write(
                    data=stream(tiler.tasks(plan.rect, lvl),
                                make_worker(source, cfg, backend, tiler, plan, lvl),
                                ex, cfg.prefetch, bar),
                    shape=tiler.level_shape(plan.rect, lvl),
                    subfiletype=1, metadata=None, **common)
        ok = True
    finally:
        if not ok:
            log(f"[abort] discarded incomplete {out_path.name}.partial  "
                "(TIFF is written sequentially, so the next run starts it again)")

    elapsed = time.perf_counter() - t0
    log(f"  done in {elapsed/60:.1f} min -> {out_path.stat().st_size/1e9:.2f} GB "
        f"({total/max(elapsed, 1e-9):.1f} tiles/s)")
    write_sidecars(source, cfg, plan, out_path, elapsed, total)
    return out_path


def read_output_tile(tf, page_index, y0, x0, samples=1):
    """Decode exactly one tile from a written TIFF page, without loading the page."""
    p = tf.pages[page_index]
    across = math.ceil(p.imagewidth / p.tilewidth)
    idx = (y0 // p.tilelength) * across + (x0 // p.tilewidth)
    count = p.databytecounts[idx]
    if count:
        tf.filehandle.seek(p.dataoffsets[idx])
        data = tf.filehandle.read(count)
    else:
        data = None                                   # sparse / unwritten tile
    seg, _, _ = p.decode(data, idx)
    arr = np.asarray(seg)[0]
    return arr if samples == 3 else arr[..., 0]


# ---------------------------------------------------------------- HDF5
def vigra_axistags(order):
    """ilastik reads axis order from this attribute rather than guessing.

    Format matches vigra's AxisTags.toJSON(), which is what lazyflow's own reader tests
    write. typeFlags: Channels=1, Space=2, Time=8.
    """
    flags = {"x": 2, "y": 2, "z": 2, "c": 1, "t": 8, "s": 1}
    return json.dumps({"axes": [{"key": k, "typeFlags": flags[k],
                                 "resolution": 0, "description": ""} for k in order]})


def h5_chunks(cfg, shape, samples):
    """One plane deep: the pipeline fills a whole (t, z, c) plane before moving on, so a
    chunk spanning z would be decompressed and rewritten once per plane."""
    return ((1, 1, 1, min(cfg.tile[0], shape[3]), min(cfg.tile[1], shape[4]))
            + ((samples,) if samples == 3 else ()))


def h5_filters(cfg):
    if cfg.hdf5_compression == "gzip":
        return dict(compression="gzip", compression_opts=cfg.hdf5_level,
                    shuffle=cfg.hdf5_shuffle)
    if cfg.hdf5_compression == "lzf":
        return dict(compression="lzf", shuffle=cfg.hdf5_shuffle)
    return {}


def h5_direct_packer(cfg, dset):
    """tile -> the exact bytes HDF5 would store for its chunk, or None to leave it to HDF5.

    HDF5 runs its filters inside the library, one chunk at a time, so gzip there uses a
    single thread however many are reading. One tile is exactly one chunk (h5_chunks),
    which lets the read threads do the shuffle and deflate themselves - zlib releases the
    GIL - and hand the writer finished chunks to place with write_direct_chunk. The file
    is no different: the same filters are declared and any HDF5 reader decodes it.

    Only gzip is done this way; lzf has no encoder outside h5py. A chunk is always stored
    whole, so a partial tile at the right or bottom edge goes in zero-padded, exactly as
    HDF5 would pad it.
    """
    if not cfg.hdf5_parallel_compress or cfg.hdf5_compression != "gzip":
        return None
    if not hasattr(getattr(dset, "id", None), "write_direct_chunk"):
        return None
    dtype = np.dtype(dset.dtype)
    if sys.byteorder != "little" or not dtype.isnative:
        return None                       # chunk bytes must be in the file's byte order
    ch, cw = dset.chunks[3], dset.chunks[4]
    level, size = int(cfg.hdf5_level), dtype.itemsize
    shuffle = bool(cfg.hdf5_shuffle) and size > 1      # HDF5 skips it for 1-byte types

    def pack(tile):
        a = np.ascontiguousarray(tile[:ch, :cw], dtype=dtype)
        if shuffle:                       # byte 0 of every element, then byte 1, ...
            a = np.ascontiguousarray(a.view(np.uint8).reshape(-1, size).T)
        return zlib.compress(a, level)

    return pack


def _settings_payload(source, cfg, plan, shape):
    """The settings that determine an output's pixel values, common to both formats."""
    blended = plan.compositor is not None
    payload = {
        "source": str(cfg.czi_path), "source_bytes": cfg.czi_path.stat().st_size,
        "axes": source.h5_axes,
        "shape": [int(v) for v in shape], "dtype": np.dtype(source.out_dtype).name,
        "tile": list(cfg.tile), "rect": [int(v) for v in plan.rect], "scene": plan.scene_id,
        "T": list(source.sel_t), "Z": list(source.sel_z), "C": list(source.sel_c),
        "bit_depth": cfg.output_bit_depth,
        "eight_bit_method": cfg.eight_bit_method if cfg.output_bit_depth == 8 else None,
        "ranges": {str(k): [round(float(v[0]), 6), round(float(v[1]), 6)]
                   for k, v in sorted(plan.ranges.items())},
        "stitch_mode": cfg.stitch_mode,
        "blend": cfg.blend if blended else None,
        "blend_width": cfg.blend_width if blended else None,
        "flatfield_mode": cfg.flatfield.mode,
        "flatfield_dark": float(cfg.flatfield.dark) if plan.flatfields else None,
        "flatfield_digest": {str(c): round(float(f.profile.sum()), 3)
                             for c, f in sorted(plan.flatfields.items())},
        "corrections": {str(k): [int(v[0]), int(v[1])]
                        for k, v in sorted((plan.corrections or {}).items())},
    }
    if plan.flatfields and blended:
        # Present only in this case, so every other fingerprint is unchanged and existing
        # outputs keep being recognised. See _unapplied_flatfield for what it separates.
        payload["flatfield_applied"] = "per mosaic tile"
    return payload


def _digest(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


UNAPPLIED_NOTE = ("it was written by an earlier version that estimated a flat-field for "
                  "this mosaic but did not apply it, so its pixels are uncorrected")


def _unapplied_flatfield(old):
    """Was `old` recorded by a version that estimated a mosaic flat-field without applying it?

    Until the profiles were handed to the compositor, a mosaic converted with
    flatfield.mode set came out uncorrected while its record said otherwise. Such a
    record names a flat-field on a blended (composited) output and lacks the
    `flatfield_applied` marker that corrected output carries.
    """
    return (isinstance(old, dict) and "flatfield_applied" not in old
            and old.get("flatfield_mode") not in (None, "none")
            and old.get("blend") is not None)


def _same_but_uncorrected(old, new):
    """True if `old` is an unapplied-flat-field record of exactly the job `new` describes
    with the flat-field off: the same pixels, so the file can stand in for it."""
    if not _unapplied_flatfield(old) or new.get("flatfield_mode") != "none":
        return False
    as_written = dict(old, flatfield_mode="none", flatfield_dark=None, flatfield_digest={})
    return as_written == json.loads(json.dumps(new))


def _describe_diffs(old, new):
    """Human-readable list of the settings that differ between two payloads."""
    if not isinstance(old, dict):
        return ["the settings differ (the stored config could not be read back, so which "
                "setting changed cannot be shown)"]
    keys = set(old) | set(new)
    diffs = [f"{k}: was {old.get(k)!r}, now {new.get(k)!r}"
             for k in sorted(keys) if old.get(k) != new.get(k)]
    if _unapplied_flatfield(old):
        diffs.append(f"note: {UNAPPLIED_NOTE}. Set flatfield.mode = \"none\" to keep "
                     "using it as it is, or overwrite=True to rebuild it corrected")
    return diffs or ["(the stored fingerprint predates this version)"]


def h5_fingerprint(source, cfg, plan, shape):
    """Everything that must match for a partial file to be safe to continue."""
    payload = _settings_payload(source, cfg, plan, shape)
    payload.update({"dataset": cfg.hdf5_dataset, "compression": cfg.hdf5_compression,
                    "level": cfg.hdf5_level, "shuffle": bool(cfg.hdf5_shuffle)})
    return _digest(payload), payload


def h5_resume_state(h5py, out_path, want_hash, want_payload):
    """('new'|'resume'|'complete'|'mismatch'|'unreadable', tiles_done, detail)."""
    if not out_path.exists():
        return "new", 0, None
    try:
        with h5py.File(out_path, "r") as f:
            prog = f["progress"]
            got = prog.attrs["fingerprint"]
            got = got.decode() if isinstance(got, bytes) else got
            done = int(prog.attrs["tiles_written"])
            complete = bool(prog.attrs.get("complete", False))
            old = None
            try:
                raw = prog["config"][()]
                old = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            except Exception:
                pass
    except Exception as exc:
        return "unreadable", 0, f"{type(exc).__name__}: {exc}"

    if str(got) != want_hash:
        if not _same_but_uncorrected(old, want_payload):
            return "mismatch", done, _describe_diffs(old, want_payload)
        return ("complete" if complete else "resume"), done, UNAPPLIED_NOTE
    return ("complete" if complete else "resume"), done, None


def write_hdf5(source, cfg, backend, tiler, plan, log=print, progress=None):
    import h5py

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = cfg.out_dir / f"{cfg.stem}{plan.suffix}.h5"
    shape = tiler.level_shape(plan.rect, 0)
    total = tiler.n_tiles(plan.rect, 0)
    fp_hash, fp_payload = h5_fingerprint(source, cfg, plan, shape)
    state, done, detail = h5_resume_state(h5py, out_path, fp_hash, fp_payload)

    if state in ("complete", "resume") and detail and not cfg.overwrite:
        log(f"{plan.label}: [note] {out_path.name}: {detail}; "
            + ("reused" if state == "complete" else "continued")
            + " as the uncorrected output it is.")
    if state == "complete" and not cfg.overwrite:
        log(f"{plan.label}: {out_path.name} is already complete ({total} tiles); skipping. "
            "Set overwrite=True to rebuild.")
        return out_path
    if state == "mismatch" and not cfg.overwrite:
        raise RuntimeError(
            f"{out_path.name} exists but was written with different settings, so it cannot "
            "be continued:\n  - " + "\n  - ".join(detail) +
            "\nSet overwrite=True to rebuild it, or restore the settings it was made with.")
    if state == "unreadable" and not cfg.overwrite:
        raise RuntimeError(
            f"{out_path.name} exists but its progress record could not be read ({detail}). "
            "It was probably truncated before the first checkpoint. Set overwrite=True "
            "to rebuild.")

    resuming = cfg.hdf5_resume and state == "resume"
    if not resuming:
        done = 0
    if resuming and done:
        log(f"{plan.label}: resuming {out_path.name} at tile {done}/{total} "
            f"({100*done/max(total,1):.1f}% already written)")
    else:
        log(f"{plan.label}: {shape}  {np.dtype(source.out_dtype).name}  "
            f"-> {out_path.name} ({total} tiles)")
    if cfg.write_pyramid:
        log("  [note] write_pyramid is ignored for HDF5 output")

    ti_of, zi_of, ci_of = tiler.index_maps()
    t0, ok, n_done = time.perf_counter(), False, done
    try:
        with h5py.File(out_path, "a" if resuming else "w") as f, \
                ThreadPoolExecutor(cfg.n_read_threads) as ex, \
                progress(total=total, initial=done, unit="tile", desc=plan.label) as bar:

            if resuming:
                dset, prog = f[cfg.hdf5_dataset], f["progress"]
            else:
                dset = f.create_dataset(cfg.hdf5_dataset, shape=shape,
                                        dtype=source.out_dtype,
                                        chunks=h5_chunks(cfg, shape, source.samples),
                                        **h5_filters(cfg))
                dset.attrs["axistags"] = vigra_axistags(source.h5_axes)
                if source.scaling:
                    dset.attrs["element_size_um"] = np.array(
                        [source.scaling.get("Z", 1.0), source.scaling.get("Y", 1.0),
                         source.scaling.get("X", 1.0)], dtype=np.float64)  # (z, y, x)
                dset.attrs["channel_names"] = source.channel_names

                st = h5py.string_dtype(encoding="utf-8")
                g = f.create_group("metadata")        # up front, so a partial file is usable
                g.create_dataset("czi_xml", shape=(), dtype=st, data=source.raw_xml)

                prog = f.create_group("progress")
                prog.attrs["fingerprint"] = fp_hash
                prog.attrs["total_tiles"] = total
                prog.attrs["tiles_written"] = 0
                prog.attrs["complete"] = False
                prog.create_dataset("config", shape=(), dtype=st,
                                    data=json.dumps(fp_payload, sort_keys=True, indent=1))
                f.flush()

            work = make_worker(source, cfg, backend, tiler, plan, 0)
            pack = h5_direct_packer(cfg, dset)
            if pack is not None:
                read, tail = work, (0,) * (len(shape) - 5)
                work = lambda task: pack(read(task))      # compressed in the read thread
                log(f"  gzip-{cfg.hdf5_level} compressed in the {cfg.n_read_threads} read "
                    "threads (hdf5_parallel_compress)")
            dests = itertools.islice(tiler.tasks(plan.rect, 0), done, None)
            srcs = itertools.islice(tiler.tasks(plan.rect, 0), done, None)

            for task, tile in zip(dests, stream(srcs, work, ex, cfg.prefetch, bar)):
                _t, _z, _c, x, y, w, h = task
                yi, xi = y - plan.rect[1], x - plan.rect[0]
                if pack is not None:
                    dset.id.write_direct_chunk(
                        (ti_of[_t], zi_of[_z], ci_of[_c], yi, xi) + tail, tile)
                else:
                    dset[(ti_of[_t], zi_of[_z], ci_of[_c],
                          slice(yi, yi + h), slice(xi, xi + w))] = tile[:h, :w]
                n_done += 1
                if n_done % cfg.hdf5_checkpoint_every == 0:
                    # Data first, then the claim that it is there. A crash between the two
                    # costs a few redone tiles; the reverse would leave a silent hole.
                    f.flush()
                    prog.attrs["tiles_written"] = n_done
                    f.flush()

            prog.attrs["tiles_written"] = n_done
            prog.attrs["complete"] = True
            f.flush()
        ok = True
    finally:
        if not ok and out_path.exists():
            if cfg.hdf5_resume:
                log(f"[abort] {out_path.name} left in place at ~{n_done}/{total} tiles; "
                    "re-run to continue from the last checkpoint.")
            else:
                out_path.unlink()
                log(f"[abort] removed incomplete {out_path.name}")

    elapsed = time.perf_counter() - t0
    log(f"  done in {elapsed/60:.1f} min -> {out_path.stat().st_size/1e9:.2f} GB "
        f"({n_done - done} tiles this run)")
    write_sidecars(source, cfg, plan, out_path, elapsed, total)
    return out_path


WRITERS = {"tiff": write_tiff, "hdf5": write_hdf5}
