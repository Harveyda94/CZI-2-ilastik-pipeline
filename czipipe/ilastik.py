"""Headless ilastik pixel classification.

RAM is capped by LAZYFLOW_TOTAL_RAM_MB, not by the image: lazyflow sizes its own blocks
to stay under it and applies the halo its feature scales need at every block boundary.
That halo handling is the main reason not to hand-roll block-wise classification — get
it wrong and you get seams at every block edge that look plausible and are not.

Results go to HDF5 (chunked, written incrementally) and are then re-streamed into a
tiled BigTIFF through the same writer used elsewhere, which guarantees BigTIFF past 4 GB
with bounded memory.

For long runs, `cfg.ilastik.chunk_z` splits the classifier export itself into
checkpointed Z-chunks (see run_classify_chunked): each chunk builds its own small,
ordinary input h5 covering its Z-range plus a halo on each side (sized from the
project's own max feature sigma), and runs a completely normal ilastik invocation
against it — the exact same command shape as the non-chunked path. Only the middle of
the result (dropping the halo) gets kept, which is what avoids the seam risk above
without hand-waving: the classifier sees the same Z-context it would in one monolithic
run. (--cutout_subregion looked like a cleaner way to do this with no halo bookkeeping
at all, but it's confirmed broken against a real trained project on this ilastik
build — see build_command.) The cost is rereading a halo-plane overlap per chunk
boundary, plus ilastik reloading — and, if the project has no saved classifier,
retraining — on every chunk. Worth it once a run is long enough (hours+) that losing
it all to one crash costs more than that overhead.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import tifffile

from . import config
from .validate import check_no_spaces
from .writers import (read_output_tile, h5_chunks, h5_filters, vigra_axistags, MICRON,
                      atomic_output, tiff_compressionargs, extra_tags)

EXPORT_DONE_ATTR = "czipipe_export_complete"
# Where ilastik is asked to put its export inside the HDF5 file. No leading slash: ilastik
# joins this to the file name with a slash of its own, and splits "out.h5//exported_data"
# by the rules for file paths. On Windows a doubled slash opens a network share, so
# ilastik 1.4.2 there took the whole of "//exported_data" for the group and was left with
# no dataset name: it made a *group* called exported_data and wrote no image at all.
EXPORT_NAME = "exported_data"


# ---------------------------------------------------------------- finding ilastik
def _version_key(path):
    """Natural sort, so ilastik-1.4.10 ranks above ilastik-1.4.9."""
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", str(path))]


def _install_locations(windows):
    """Default install locations as (root, pattern), most preferred first."""
    if windows:
        roots = []
        for var in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)"):
            v = os.environ.get(var)
            if v and Path(v) not in roots:
                roots.append(Path(v))
        if os.environ.get("LOCALAPPDATA"):
            roots.append(Path(os.environ["LOCALAPPDATA"]) / "Programs")
        return [(r, "ilastik*/ilastik.exe") for r in roots]
    app = "ilastik*.app/Contents/ilastik-release/run_ilastik.sh"
    return [(Path("/Applications"), app), (Path.home() / "Applications", app),
            (Path("/opt"), "ilastik*/run_ilastik.sh"),
            (Path.home(), "ilastik*/run_ilastik.sh")]


def find_exe(cfg, windows=None):
    if cfg.ilastik.exe:
        return str(cfg.ilastik.exe)
    windows = (os.name == "nt") if windows is None else windows
    for name in ("run_ilastik.sh", "ilastik", "ilastik.exe"):
        p = shutil.which(name)
        if p:
            return p
    for root, pattern in _install_locations(windows):
        hits = sorted(root.glob(pattern), key=_version_key) if root.is_dir() else []
        if hits:
            return str(hits[-1])
    raise FileNotFoundError(
        "ilastik not found; set cfg.ilastik.exe to its launcher, for example\n"
        "  macOS:   /Applications/ilastik-1.4.1-OSX.app/Contents/ilastik-release/"
        "run_ilastik.sh\n"
        "  Windows: C:\\Program Files\\ilastik-1.4.1\\ilastik.exe\n"
        "  Linux:   ~/ilastik-1.4.1-Linux/run_ilastik.sh")


def jobs(cfg):
    """Normalise cfg.ilastik.projects into (tag, path) pairs."""
    out = []
    for entry in cfg.ilastik.projects:
        if isinstance(entry, (tuple, list)):
            tag, p = str(entry[0]), Path(entry[1])
        else:
            p = Path(entry)
            tag = p.stem
        if not p.is_file():
            raise FileNotFoundError(f"ilastik project not found: {p}")
        out.append((tag, p))
    return out


def inspect_project(ilp):
    """Pull the sanity-check facts out of a project file (an .ilp is just HDF5)."""
    import h5py

    info = {"path": str(ilp), "workflow": None, "classes": [], "channels": None,
            "axes": None, "max_sigma": None, "n_features": None}
    with h5py.File(ilp, "r") as f:
        if "workflowName" in f:
            info["workflow"] = f["workflowName"][()].decode(errors="replace")
        try:
            info["classes"] = [x.decode(errors="replace") if isinstance(x, bytes) else str(x)
                               for x in f["PixelClassification/LabelNames"][()]]
        except Exception:
            pass
        try:
            lane = f["Input Data/infos/lane0000/Raw Data"]
            shape = tuple(int(v) for v in lane["shape"][()])
            keys = [a["key"] for a in json.loads(lane["axistags"][()].decode())["axes"]]
            info["axes"] = "".join(keys)
            if "c" in keys:
                info["channels"] = shape[keys.index("c")]
        except Exception:
            pass
        try:
            sel = f["FeatureSelections/SelectionMatrix"][()]
            scales = f["FeatureSelections/Scales"][()]
            used = [float(scales[j]) for j in range(sel.shape[1]) if sel[:, j].any()]
            info["max_sigma"] = max(used) if used else None
            info["n_features"] = int(np.count_nonzero(sel))
        except Exception:
            pass
    return info


def feature_halo(max_sigma):
    """ilastik's own rule of thumb for how far a feature at this scale reaches."""
    return int(np.ceil(max_sigma * 3.5)) if max_sigma else 0


def report_projects(source, cfg, tiler, plans, log=print):
    out = []
    for tag, ilp in jobs(cfg):
        d = inspect_project(ilp)
        out.append((tag, ilp, d))
        log(f"[{tag}] {ilp.name}")
        log(f"   workflow   {d['workflow']}")
        log(f"   classes    {len(d['classes']) or '?'}  {d['classes']}")
        log(f"   trained on {d['channels']} channel(s), axes {d['axes']}, "
            f"{d['n_features']} features, max sigma {d['max_sigma']}")
        if d["channels"] is not None and d["channels"] != len(source.sel_c):
            log(f"   [WARN] project expects {d['channels']} channel(s) but this conversion "
                f"has {len(source.sel_c)} (channels={source.sel_c}). ilastik will refuse or "
                "produce nonsense; align the selection with the training data.")
        if d["max_sigma"]:
            log(f"   -> ilastik will use a halo of roughly "
                f"{feature_halo(d['max_sigma'])} px per block edge")
        vox = sum(int(np.prod(np.array(tiler.level_shape(p.rect, 0))[[0, 1, 3, 4]]))
                  for p in plans)
        if cfg.ilastik.export_source == "Probabilities":
            per = (len(d["classes"]) or 2) * (4 if cfg.ilastik.export_dtype == "float32" else 1)
        else:
            per = 1
        gb = vox * per / 1e9
        log(f"   projected uncompressed export: {gb:.1f} GB"
            + ("   <-- consider 'Simple Segmentation'" if gb > 200 else ""))
    return out


def build_command(cfg, exe, ilp, raw_h5, internal, out_h5, axes):
    """Pixel classification takes its input as the LAST POSITIONAL argument.

    (--raw_data is for the multi-input workflows: object classification, multicut,
    tracking.) HDF5 inputs use the h5ls convention: file.h5/internal/path.

    NOTE: ilastik also exposes --cutout_subregion, which looks like exactly the right
    tool for exporting a Z-range without touching the input (see dataExportApplet.py).
    It doesn't actually work on this build for a real trained project: it throws
    lazyflow.slot.Slot.SlotNotReadyError deep in OpSubRegion before ever running,
    confirmed against real data — not just theoretically risky, empirically broken.
    Do not reach for it; see run_classify_chunked for the halo-overlap alternative
    that only uses this same plain, proven command shape.
    """
    check_no_spaces(ilp, raw_h5, out_h5)
    ic = cfg.ilastik
    return [
        exe, "--headless", "--readonly",
        f"--project={ilp}",
        f"--input_axes={axes}",
        f"--export_source={ic.export_source}",       # exempt from the space rule
        f"--export_dtype={ic.export_dtype}",
        "--output_format=hdf5",
        f"--output_filename_format={out_h5}",
        f"--output_internal_path={EXPORT_NAME}",     # no leading slash: see EXPORT_NAME
        f"--output_axis_order={axes}",
        f"{raw_h5}/{internal.lstrip('/')}",          # positional input, must be last
    ]


def _proc_tree_cpu_seconds(pid):
    """Total CPU seconds used by pid and all its descendants, or None if unavailable.

    ilastik.exe is a thin launcher; the real work happens in a child python.exe, so
    watching the launcher's own CPU time alone would look idle even when it's busy.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        root = psutil.Process(pid)
        total = 0.0
        for p in [root] + root.children(recursive=True):
            try:
                t = p.cpu_times()
                total += t.user + t.system
            except psutil.Error:
                pass
        return total
    except psutil.Error:
        return None


def _heartbeat(pid, t0, last_cpu, log):
    """Say ilastik is still alive, and warn if its process tree has stopped using CPU.

    A long export can print nothing for many minutes; this is what tells "busy" from
    "hung" without leaving the notebook. Returns the CPU total for the next comparison.
    """
    elapsed = (time.perf_counter() - t0) / 60
    cpu = _proc_tree_cpu_seconds(pid)
    if cpu is None:
        log(f"  ... still running ({elapsed:.0f} min elapsed)")
        return None
    stalled = last_cpu is not None and cpu - last_cpu < 1.0
    log(f"  ... still running ({elapsed:.0f} min elapsed, {cpu / 60:.1f} CPU-min used "
        "across the process tree)"
        + ("  [WARN: no CPU activity in the process tree since the last check - it may "
           "be hung]" if stalled else ""))
    return cpu


def _kill_tree(proc, grace=5.0):
    """Stop ilastik and everything it started.

    The launcher (run_ilastik.sh, ilastik.exe) may start the Python that does the work as
    a child instead of replacing itself with it, so killing only `proc` can leave the real
    worker running: holding the export open, using the RAM it was given, and colliding
    with the next attempt. The child is started in its own process group/session so the
    whole tree can be signalled at once.
    """
    if os.name == "nt":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGTERM)
        with contextlib.suppress(subprocess.TimeoutExpired):
            proc.wait(timeout=grace)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(proc.pid, signal.SIGKILL)
    with contextlib.suppress(OSError):
        proc.kill()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=30)


# The last line of a Python traceback: "ValueError: ...", "pkg.mod.SlotNotReadyError: ...".
_EXC_LINE = re.compile(r"[\w.]*(Error|Exception)\w*(:|\s*$)")
# ilastik's excepthook writes this when an exception escapes its main thread. Headless,
# that is the end of the run, whatever the exit code says -- unless the export had
# already finished, which the workflow announces with _DONE_LINE.
_FATAL_LINE = "Unhandled exception in thread: 'MainThread'"
_DONE_LINE = "Completed Batch Processing"


def _log_reason(log_path, limit=400):
    """The last exception line in an ilastik log, or '' when there is none to show."""
    try:
        with open(log_path, encoding="utf-8", errors="replace") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return ""
    for line in reversed(lines):
        if _EXC_LINE.match(line):
            return line.strip()[:limit]
    return ""


def run_command(cfg, cmd, log_path, log=print, heartbeat_s=300):
    env = os.environ.copy()
    # ilastik is its own Python installation, and this variable is addressed to the
    # notebook's: a Jupyter kernel sets it to its inline backend, which ilastik's
    # matplotlib does not have. ilastik 1.4.2 imports matplotlib while loading its
    # plugins, rejects the value and stops before it has opened the project.
    env.pop("MPLBACKEND", None)
    env["LAZYFLOW_TOTAL_RAM_MB"] = str(cfg.ilastik_ram_mb)
    env["LAZYFLOW_THREADS"] = str(cfg.ilastik_threads)
    env.setdefault("PYTHONIOENCODING", "utf-8")     # so ilastik writes what we decode
    log(f"  LAZYFLOW_TOTAL_RAM_MB={env['LAZYFLOW_TOTAL_RAM_MB']}  "
        f"LAZYFLOW_THREADS={env['LAZYFLOW_THREADS']}"
        "   (ilastik documents this cap as approximate - leave headroom)")
    if not cfg.ilastik.ram_mb:
        total = config.physical_ram_mb()
        log("  RAM cap chosen automatically: "
            + (f"{config.ILASTIK_RAM_FRACTION:.0%} of the {total / 1024:.0f} GB installed"
               if total else
               "the installed RAM could not be read, so the 4 GB default")
            + ". Set ilastik.ram_mb to choose it yourself.")
    log("  " + " \\\n    ".join(cmd))
    if cfg.ilastik.dry_run:
        log("  [dry run] not executed")
        return False

    # Decode explicitly. Left to the default, Windows decodes with the ANSI code page and
    # the first byte outside it raises mid-run; errors="replace" means a stray byte can
    # only ever garble a log line, never stop the job.
    kw = dict(env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
              encoding="utf-8", errors="replace", bufsize=1)
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kw["start_new_session"] = True

    t0 = time.perf_counter()
    deadline = time.monotonic() + cfg.ilastik.timeout_h * 3600
    proc = subprocess.Popen(cmd, **kw)

    # Output is read on a thread so the deadline can be enforced while ilastik is still
    # writing; reading it inline would block until ilastik closed stdout, i.e. finished.
    lines = queue.Queue()

    def pump():
        try:
            for line in proc.stdout:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    threading.Thread(target=pump, name="ilastik-output", daemon=True).start()
    timed_out = False
    log_ok = True
    exported, fatal, reason = False, False, ""
    last_beat, last_cpu = time.monotonic(), _proc_tree_cpu_seconds(proc.pid)
    try:
        with open(log_path, "w", encoding="utf-8") as logf:
            while True:
                now = time.monotonic()
                if now > deadline:
                    timed_out = True
                    break
                if heartbeat_s and now - last_beat >= heartbeat_s:
                    last_cpu = _heartbeat(proc.pid, t0, last_cpu, log)
                    last_beat = now
                try:
                    line = lines.get(timeout=0.5)
                except queue.Empty:
                    continue
                if line is None:
                    break
                if log_ok:
                    try:
                        logf.write(line)
                        logf.flush()
                    except OSError as e:
                        # A full disk or a dropped network share costs the log file, not
                        # an hours-long run; the key lines still reach the notebook below.
                        log_ok = False
                        log(f"  [warn] couldn't write ilastik log ({e}); continuing without it")
                if _DONE_LINE in line:
                    exported = True
                elif _FATAL_LINE in line and not exported:
                    fatal = True
                elif _EXC_LINE.match(line):
                    reason = line.strip()[:400]
                if any(k in line for k in ("ERROR", "Error", "Traceback", "%", "Exporting")):
                    log("   | " + line.rstrip()[:400])
            if not timed_out:
                try:
                    proc.wait(timeout=max(0.0, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
    except BaseException:                       # KeyboardInterrupt included
        _kill_tree(proc)
        raise
    if timed_out:
        _kill_tree(proc)
        raise RuntimeError(f"ilastik exceeded timeout_h ({cfg.ilastik.timeout_h} h); "
                           "it and any processes it started were stopped")
    # The exit code alone is not enough: the Windows launcher has returned 0 after ilastik
    # stopped on an unhandled exception, having exported nothing.
    if proc.returncode != 0 or fatal:
        what = (f"ilastik exited {proc.returncode}" if proc.returncode else
                "ilastik stopped on an unhandled error, although it exited with code 0")
        raise RuntimeError(what + (f":\n    {reason}\n" if reason else "; ")
                           + f"see {log_path}")
    log(f"  classified in {(time.perf_counter()-t0)/60:.1f} min")
    return True


class ExportMissing(RuntimeError):
    """ilastik reported success, but its output does not hold the export asked for."""


def _export_dataset(f):
    """The export in an open ilastik output file, or None if it holds no such image.

    A group of that name is not one. It is what a run that wrote no image has left
    behind (see EXPORT_NAME), and it has neither a shape nor pixels to read.
    """
    d = f[EXPORT_NAME] if EXPORT_NAME in f else None
    return d if hasattr(d, "shape") else None


def _check_export(out_h5, log_path, want=None, what="ilastik"):
    """Raise ExportMissing unless `out_h5` holds an export (of shape `want`, if given).

    Called after every run that exited cleanly and before anything is built on its
    output, so a run that wrote nothing is reported as that, with ilastik's own last
    error, instead of as a missing HDF5 object several steps later.
    """
    import h5py

    out_h5 = Path(out_h5)
    if not out_h5.is_file():
        problem = f"it wrote no output file ({out_h5.name})"
    else:
        try:
            with h5py.File(out_h5, "r") as f:
                d = _export_dataset(f)
                if d is None and EXPORT_NAME in f:
                    problem = (f"'{EXPORT_NAME}' in {out_h5.name} is a group with no "
                               "image in it")
                elif d is None:
                    problem = f"{out_h5.name} holds no '{EXPORT_NAME}'"
                elif want is not None and tuple(d.shape) != tuple(want):
                    problem = (f"the export in {out_h5.name} has shape {tuple(d.shape)}, "
                               f"not the expected {tuple(want)}")
                else:
                    return
        except Exception as exc:
            problem = f"{out_h5.name} could not be read ({type(exc).__name__}: {exc})"
    reason = _log_reason(log_path)
    raise ExportMissing(
        f"{what} exited without an error code, but {problem}.\n"
        + (f"ilastik's log ends with:\n    {reason}\n" if reason else "")
        + f"Full log: {log_path}")


# ---------------------------------------------------------------- restart bookkeeping
def _identity(path):
    """Enough to tell whether a file is the one a record was made from."""
    st = Path(path).stat()
    return {"path": str(path), "bytes": st.st_size, "mtime_ns": st.st_mtime_ns}


def _as_str(v):
    return v.decode() if isinstance(v, bytes) else v


def _job_key(cfg, ilp, raw_h5, internal):
    """What an ilastik export depends on: the project, the input, the export settings."""
    return json.dumps({"project": _identity(ilp), "input": _identity(raw_h5),
                       "internal": internal, "export_source": cfg.ilastik.export_source,
                       "export_dtype": cfg.ilastik.export_dtype}, sort_keys=True)


def _export_is_complete(out_h5, key):
    """True if `out_h5` is a finished export of exactly this job.

    ilastik leaves no completeness marker of its own, so one is added once it exits
    successfully. Without it, an interruption while converting the export to TIFF would
    throw away the (much longer) classification itself.
    """
    if not Path(out_h5).exists():
        return False
    import h5py
    try:
        with h5py.File(out_h5, "r") as f:
            # The marker alone is not proof. It was once written to a file ilastik had
            # never touched, and once to one holding a group where the image should be;
            # such a file must be classified again, not reused.
            return (_as_str(f.attrs.get(EXPORT_DONE_ATTR)) == key
                    and _export_dataset(f) is not None)
    except Exception:
        return False


def _mark_export_complete(out_h5, key):
    import h5py
    # "r+", never "a": marking must not be able to create the file it vouches for.
    with h5py.File(out_h5, "r+") as f:
        f.attrs[EXPORT_DONE_ATTR] = key
        f.flush()


def stage_h5(source, cfg, tiler, plan, path, log=print, progress=None):
    """Return (h5_path, internal_path, is_temporary) suitable as ilastik input."""
    import h5py

    path = Path(path)
    if path.suffix.lower() in (".h5", ".hdf5"):
        return path, cfg.hdf5_dataset, False

    tmp = path.with_suffix(".ilastik_input.h5")
    made_from = json.dumps(_identity(path), sort_keys=True)
    if tmp.exists() and not cfg.overwrite:
        try:
            with h5py.File(tmp, "r") as f:
                p = f["progress"].attrs
                reusable = (bool(p.get("complete", False))
                            and _as_str(p.get("source_identity")) == made_from)
        except Exception:
            reusable = False
        if reusable:
            log(f"  reusing existing {tmp.name}")
            return tmp, cfg.hdf5_dataset, True
        log(f"  {tmp.name} exists but is incomplete or was staged from a different "
            f"{path.name}; rewriting")

    shape = tiler.level_shape(plan.rect, 0)
    H, W = shape[3], shape[4]
    th, tw = tiler.tile
    total = tiler.n_tiles(plan.rect, 0)
    log(f"  staging {path.name} -> {tmp.name} (chunked HDF5 for ilastik)")
    with h5py.File(tmp, "w") as f, tifffile.TiffFile(path) as tf, \
            progress(total=total, unit="tile", desc="stage") as bar:
        dset = f.create_dataset(cfg.hdf5_dataset, shape=shape, dtype=source.out_dtype,
                                chunks=h5_chunks(cfg, shape, source.samples),
                                **h5_filters(cfg))
        dset.attrs["axistags"] = vigra_axistags(source.h5_axes)
        prog = f.create_group("progress")
        prog.attrs["total_tiles"] = total
        prog.attrs["tiles_written"] = 0
        prog.attrs["complete"] = False
        prog.attrs["source_identity"] = made_from
        for ti in range(len(source.sel_t)):
            for zi in range(len(source.sel_z)):
                for ci in range(len(source.sel_c)):
                    page = tiler.page_index(ti, zi, ci)
                    for y0 in range(0, H, th):
                        for x0 in range(0, W, tw):
                            h, w = min(th, H - y0), min(tw, W - x0)
                            dset[ti, zi, ci, y0:y0 + h, x0:x0 + w] = \
                                read_output_tile(tf, page, y0, x0, source.samples)[:h, :w]
                            bar.update(1)
        prog.attrs["tiles_written"] = total
        prog.attrs["complete"] = True
        f.flush()
    return tmp, cfg.hdf5_dataset, True


def seg_to_tiff(source, cfg, tiler, h5_path, out_tiff, class_names, progress=None):
    """Stream ilastik's HDF5 export into a tiled BigTIFF, one chunk at a time."""
    import h5py

    with h5py.File(h5_path, "r") as f:
        d = f["/exported_data"]
        shape = tuple(int(v) for v in d.shape)
        while len(shape) < 5:
            shape = (1,) + shape
        nt, nz, nc, H, W = shape[:5]
        th, tw = tiler.tile
        total = nt * nz * nc * math.ceil(H / th) * math.ceil(W / tw)

        def gen(bar):
            for t in range(nt):
                for z in range(nz):
                    for c in range(nc):
                        for y0 in range(0, H, th):
                            for x0 in range(0, W, tw):
                                h, w = min(th, H - y0), min(tw, W - x0)
                                blk = np.asarray(d[t, z, c, y0:y0 + h, x0:x0 + w])
                                bar.update(1)
                                yield tiler.fit(blk.reshape(h, w))

        names = ((class_names or [f"class{i}" for i in range(nc)])[:nc]
                 if cfg.ilastik.export_source == "Probabilities" else ["Segmentation"][:nc])
        meta = {"axes": "TZCYX", "Channel": {"Name": names}, "Name": Path(out_tiff.stem).stem}
        for axis, key in (("X", "PhysicalSizeX"), ("Y", "PhysicalSizeY"),
                          ("Z", "PhysicalSizeZ")):
            if axis in source.scaling:
                meta[key] = source.scaling[axis]
                meta[key + "Unit"] = MICRON
        if cfg.ilastik.export_source != "Probabilities" and class_names:
            # A single-channel label map has nowhere else to record what each pixel
            # value means -- without this, "1" and "2" are meaningless months later.
            mapping = ", ".join(f"{i + 1}={name}" for i, name in enumerate(class_names))
            meta["Description"] = (f"Pixel classification ({cfg.ilastik.export_source}) "
                                   f"from {cfg.czi_path.name}. Label values: {mapping}")

        cargs = tiff_compressionargs(cfg.ilastik.seg_compression, cfg.compression_level)
        # Written as <name>.partial and renamed when complete: classify() treats an
        # existing segmentation TIFF as done, so a half-written one must never carry the
        # final name.
        with atomic_output(out_tiff) as tmp, \
                tifffile.TiffWriter(tmp, bigtiff=True, ome=True) as tif, \
                progress(total=total, unit="tile", desc="-> tiff") as bar:
            tif.write(data=gen(bar), shape=(nt, nz, nc, H, W), dtype=d.dtype,
                      tile=tuple(cfg.tile), photometric="minisblack",
                      compression=cfg.ilastik.seg_compression, compressionargs=cargs,
                      maxworkers=cfg.write_workers, metadata=meta,
                      extratags=extra_tags(source, cfg))
    return out_tiff


# ---------------------------------------------------------------- chunked export
def _classifier_export_shape(cfg, tiler, plan):
    """Expected (T,Z,C,Y,X) of ilastik's own export h5, given export_source.

    Simple Segmentation always collapses to a single label channel regardless of how
    many channels the raw input had; Probabilities keeps one channel per class.
    """
    shape = list(tiler.level_shape(plan.rect, 0))
    if cfg.ilastik.export_source != "Probabilities":
        shape[2] = 1
    return tuple(shape)


def _chunk_fingerprint(cfg, ilp, raw_h5, axes, export_shape, halo_z):
    """Everything that must match for a partially-chunked export to be safe to
    continue (mirrors writers.h5_fingerprint's role for the main converter)."""
    ilp_stat = Path(ilp).stat()
    raw_stat = Path(raw_h5).stat()
    payload = {
        "raw_h5": str(raw_h5), "raw_h5_bytes": raw_stat.st_size,
        "raw_h5_mtime": raw_stat.st_mtime,
        "ilp": str(ilp), "ilp_bytes": ilp_stat.st_size, "ilp_mtime": ilp_stat.st_mtime,
        "axes": axes, "shape": [int(v) for v in export_shape], "halo_z": int(halo_z),
        "export_source": cfg.ilastik.export_source, "export_dtype": cfg.ilastik.export_dtype,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _chunk_resume_state(h5py, out_h5, want_hash, want_payload):
    """('new'|'resume'|'complete'|'mismatch'|'unreadable', z_done, detail)."""
    if not out_h5.is_file():
        return "new", 0, None
    try:
        with h5py.File(out_h5, "r") as f:
            if "progress" not in f and _export_dataset(f) is None:
                # Nothing to continue and nothing to lose, e.g. the file an earlier
                # version marked complete after a run that had exported nothing, or
                # only a group where the image should be.
                return "new", 0, None
            prog = f["progress"]
            got = prog.attrs["fingerprint"]
            got = got.decode() if isinstance(got, bytes) else got
            z_done = int(prog.attrs["z_done"])
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
        if not isinstance(old, dict):
            return "mismatch", z_done, ["the settings differ (the stored config could not "
                                        "be read back, so which setting changed can't be shown)"]
        keys = set(old) | set(want_payload)
        diffs = [f"{k}: was {old.get(k)!r}, now {want_payload.get(k)!r}"
                 for k in sorted(keys) if old.get(k) != want_payload.get(k)]
        return "mismatch", z_done, diffs or ["(the stored fingerprint predates this version)"]
    return ("complete" if complete else "resume"), z_done, None


def planes_computed(z_from, Z, n, halo):
    """Z-planes ilastik must read and classify to export [z_from, Z) in chunks of n:
    each chunk's own planes plus up to `halo` either side of it."""
    return sum(min(Z, z0 + n + halo) - max(0, z0 - halo) for z0 in range(z_from, Z, n))


def halo_cost_report(z_from, Z, n, halo):
    """One line on what the Z halo costs at this chunk_z, and what larger chunks would.

    The halo is fixed by the project's largest feature sigma, so its share of the work
    depends only on the chunk size: a 35-plane halo makes each 50-plane chunk classify
    120 planes. That is easy to miss, because the run still finishes and is still right.
    """
    todo = Z - z_from
    if not halo or todo <= 0:
        return None
    cost = planes_computed(z_from, Z, n, halo)
    line = (f"  halo cost: {cost} planes classified to export {todo} "
            f"({cost / todo:.1f}x) at chunk_z={n}")
    if cost / todo >= 1.25 and n < todo:
        bigger = sorted({min(m, todo) for m in (2 * n, 4 * n)})
        line += (". Larger chunks cost less but lose more to a crash: "
                 + ", ".join(f"chunk_z={m} -> {planes_computed(z_from, Z, m, halo) / todo:.1f}x"
                             for m in bigger)
                 + ". chunk_z may be changed between runs of a partial export.")
    return line


def run_classify_chunked(cfg, exe, ilp, raw_h5, internal, out_h5, axes, export_shape,
                         halo_z=0, log=print):
    """Classify in Z-chunks, checkpointed so a crash loses at most one chunk.

    Each chunk builds its own small, ordinary input h5 covering [z0-halo_z, z1+halo_z]
    (clamped to the volume), runs a completely normal ilastik invocation against it —
    the exact same command shape as the non-chunked path, nothing exotic — and then
    only the middle [z0:z1] of the result (dropping the halo) gets copied into the
    checkpointed master file. The halo gives the feature/classifier operators the same
    Z-context they'd see in one monolithic run, so there's no seam at chunk boundaries.

    (--cutout_subregion looked like the right tool for this and needed no halo
    juggling at all, but it does not actually work on this build for a real trained
    project — confirmed against real data, see build_command — hence this approach.)

    The real cost is rereading a halo_z-plane overlap per chunk boundary, plus ilastik
    reloading the project — and, if it has no saved classifier, retraining — on every
    chunk. Only worth it once a full run is long enough (hours+) that losing it all to
    one crash is worse than that overhead.
    """
    import h5py

    T, Z, C, Y, X = export_shape
    n = int(cfg.ilastik.chunk_z)
    n_chunks = math.ceil(Z / n)
    want_hash, want_payload = _chunk_fingerprint(cfg, ilp, raw_h5, axes, export_shape, halo_z)

    state, z_done, detail = _chunk_resume_state(h5py, out_h5, want_hash, want_payload)
    if state == "mismatch":
        raise RuntimeError(
            f"{out_h5.name} exists but was started with different settings, so it cannot "
            "be continued:\n  - " + "\n  - ".join(detail) +
            "\nSet overwrite=True to rebuild it, or restore the settings it was started with.")
    if state == "unreadable":
        raise RuntimeError(f"{out_h5.name} exists but its progress record could not be read "
                           f"({detail}). Set overwrite=True to rebuild.")
    if state == "complete":
        log(f"[{out_h5.name}] already complete ({Z} Z-planes, {n_chunks} chunks); "
            "skipping ilastik entirely")
        return True

    if state == "new":
        log(f"[{out_h5.name}] {Z} Z-planes in {n_chunks} chunks of {n} (halo {halo_z} px)")
        with h5py.File(out_h5, "w") as f:
            f.create_dataset("/exported_data", shape=export_shape, dtype=cfg.ilastik.export_dtype,
                             chunks=(1, min(n, Z), C, min(1024, Y), min(1024, X)))
            prog = f.create_group("progress")
            prog.attrs["fingerprint"] = want_hash
            prog.attrs["z_done"] = 0
            prog.attrs["complete"] = False
            prog.create_dataset("config", data=json.dumps(want_payload))

    # Continue from exactly where the record stops, not from a multiple of chunk_z, so
    # the chunk size can be changed between runs without redoing exported planes. With
    # an unchanged chunk_z the two are the same, and so is the numbering below.
    todo = [(z0, min(z0 + n, Z)) for z0 in range(z_done, Z, n)]
    start_chunk = math.ceil(z_done / n)
    n_chunks = start_chunk + len(todo)
    if state != "new":
        log(f"[{out_h5.name}] resuming at Z={z_done}/{Z} ({start_chunk}/{n_chunks} "
           "chunks already done)")
    cost = halo_cost_report(z_done, Z, n, halo_z)
    if cost:
        log(cost)

    chunk_in_h5 = out_h5.with_name(out_h5.stem + ".chunk_in.h5")
    chunk_out_h5 = out_h5.with_name(out_h5.stem + ".chunk_out.h5")
    t0 = time.perf_counter()
    for i, (z0, z1) in enumerate(todo, start_chunk):
        pad0, pad1 = max(0, z0 - halo_z), min(Z, z1 + halo_z)
        log(f"\n[{out_h5.name}] chunk {i + 1}/{n_chunks}  Z={z0}:{z1}  "
           f"(computing over {pad0}:{pad1} for halo context)")

        # A previous crashed/interrupted attempt at this exact chunk can leave a stale,
        # partially-written chunk_out_h5 sitting at this path. ilastik trying to write
        # into that existing file (rather than a fresh one) is exactly the kind of thing
        # that produced the earlier "Couldn't delete link" / "bad object header version"
        # corruption errors -- always start this chunk's output from a clean slate.
        chunk_in_h5.unlink(missing_ok=True)
        chunk_out_h5.unlink(missing_ok=True)

        stage_t0 = time.perf_counter()
        with h5py.File(raw_h5, "r") as fsrc:
            src = fsrc[internal]
            in_shape = (T, pad1 - pad0) + tuple(src.shape[2:])
            with h5py.File(chunk_in_h5, "w") as f:
                dset = f.create_dataset("/data", shape=in_shape, dtype=src.dtype,
                                        chunks=(1, 1, 1, min(1024, Y), min(1024, X)))
                tags = src.attrs.get("axistags")
                dset.attrs["axistags"] = tags if tags is not None else vigra_axistags(axes)
                for zi, z in enumerate(range(pad0, pad1)):   # one Z-plane at a time; never
                    for t in range(T):                       # materialise more than that in
                        dset[t, zi] = src[t, z]              # RAM. Every timepoint, not just 0.
                    if zi and zi % 50 == 0:
                        log(f"  staging input: plane {zi}/{pad1 - pad0} "
                           f"({time.perf_counter() - stage_t0:.0f}s elapsed)")

        cmd = build_command(cfg, exe, ilp, chunk_in_h5, "/data", chunk_out_h5, axes)
        chunk_log = out_h5.with_name(f"{out_h5.stem}.chunk_z{z0:06d}.ilastik.log")
        if not run_command(cfg, cmd, chunk_log, log=log):
            return False   # dry_run: printed the next pending chunk's command and stopped

        trim0 = z0 - pad0   # offset of the wanted [z0:z1] range within the padded chunk
        this = f"chunk {i + 1}/{n_chunks} (Z={z0}:{z1})"
        retry = ("\nNothing already exported is lost: running classify() again retries "
                 "this chunk (its stale output is cleared automatically).")
        try:
            _check_export(chunk_out_h5, chunk_log, want=(T, pad1 - pad0, C, Y, X),
                          what=f"{this}: ilastik")
            with h5py.File(chunk_out_h5, "r") as cf, h5py.File(out_h5, "r+") as f:
                csrc, dst = cf["/exported_data"], f["/exported_data"]
                for zi in range(z1 - z0):
                    dst[:, z0 + zi] = csrc[:, trim0 + zi]
                f["progress"].attrs["z_done"] = z1
        except ExportMissing as e:
            raise ExportMissing(f"{e}{retry}") from None
        except Exception as e:
            raise RuntimeError(
                f"{this}: copying ilastik's output into {out_h5.name} failed "
                f"({type(e).__name__}: {e}). ilastik has been seen to log a fatal error yet "
                f"still exit 0 -- check {chunk_log} for what actually happened.{retry}"
            ) from e
        chunk_in_h5.unlink(missing_ok=True)
        chunk_out_h5.unlink(missing_ok=True)

        elapsed = time.perf_counter() - t0
        done = i + 1 - start_chunk
        remaining = (n_chunks - i - 1) * (elapsed / done)
        log(f"[{out_h5.name}] {i + 1}/{n_chunks} chunks done "
           f"({elapsed/60:.0f} min elapsed, ~{remaining/60:.0f} min remaining)")

    with h5py.File(out_h5, "r+") as f:
        f["progress"].attrs["complete"] = True
    return True


def classify(source, cfg, tiler, results, log=print, progress=None):
    """Run every configured project over every converted output. Returns [(scene, tag, path)]."""
    from .validate import report_h5_check

    exe = find_exe(cfg)
    log(f"ilastik executable: {exe}\n")
    projects = report_projects(source, cfg, tiler, [r[0] for r in results], log=log)

    # Restart granularity is one (scene, project) job. A finished segmentation TIFF is
    # skipped; a finished ilastik export whose TIFF conversion was interrupted is reused;
    # an interrupted ilastik run starts again, since ilastik itself cannot resume - unless
    # cfg.ilastik.chunk_z is set, when it continues from the last finished Z-chunk.
    seg = []
    for plan, path in results:
        raw_h5, internal, is_tmp = stage_h5(source, cfg, tiler, plan, path,
                                            log=log, progress=progress)
        if not report_h5_check(raw_h5, internal, source.h5_axes, log=log) \
                and not cfg.ilastik.dry_run:
            raise RuntimeError(f"{Path(raw_h5).name} is not in a form ilastik will accept "
                               "(see the FAIL lines above)")
        finished = False
        try:
            for tag, ilp, info in projects:
                out_h5 = cfg.out_dir / f"{cfg.stem}{plan.suffix}_{tag}.h5"
                out_tiff = cfg.out_dir / f"{cfg.stem}{plan.suffix}_{tag}.ome.tif"
                if out_tiff.exists() and not cfg.overwrite:
                    # Safe to trust: the TIFF only gets its final name once complete.
                    log(f"[{tag}] {out_tiff.name} already complete; skipping "
                        "(overwrite=False)")
                    seg.append((plan.scene_id, tag, out_tiff))
                    continue
                log(f"\n[{tag}] {plan.label}")
                key = _job_key(cfg, ilp, raw_h5, internal)
                if cfg.ilastik.chunk_z:
                    # Resumes from its own per-chunk record inside out_h5, so a partial
                    # export is continued here rather than removed.
                    if cfg.overwrite and not cfg.ilastik.dry_run:
                        out_h5.unlink(missing_ok=True)
                    if not run_classify_chunked(
                            cfg, exe, ilp, raw_h5, internal, out_h5, source.h5_axes,
                            _classifier_export_shape(cfg, tiler, plan),
                            halo_z=feature_halo(info["max_sigma"]), log=log):
                        continue
                elif not cfg.overwrite and _export_is_complete(out_h5, key):
                    log(f"  reusing {out_h5.name}: ilastik finished it on an earlier run")
                else:
                    cmd = build_command(cfg, exe, ilp, raw_h5, internal, out_h5,
                                        source.h5_axes)
                    if out_h5.exists() and not cfg.ilastik.dry_run:
                        log(f"  removing {out_h5.name} left by an interrupted run")
                        out_h5.unlink()
                    run_log = cfg.out_dir / f"{cfg.stem}{plan.suffix}_{tag}.ilastik.log"
                    if not run_command(cfg, cmd, run_log, log=log):
                        continue
                    _check_export(out_h5, run_log, what=f"[{tag}] ilastik")
                    _mark_export_complete(out_h5, key)
                seg_to_tiff(source, cfg, tiler, out_h5, out_tiff, info["classes"],
                            progress=progress)
                log(f"  -> {out_tiff.name}  ({out_tiff.stat().st_size/1e9:.2f} GB)")
                seg.append((plan.scene_id, tag, out_tiff))
                if not cfg.ilastik.keep_h5:
                    out_h5.unlink(missing_ok=True)
            finished = True
        finally:
            if is_tmp and not cfg.ilastik.keep_h5 and not cfg.ilastik.dry_run:
                if finished:
                    Path(raw_h5).unlink(missing_ok=True)
                elif Path(raw_h5).exists():
                    log(f"  kept {Path(raw_h5).name} so the next run need not stage it again")
    return seg


def preview_segmentation(source, cfg, tiler, results, seg, n_planes=3, max_px=1200):
    """Raw channels next to the segmentation at a few Z-planes spread across depth, for
    a quick visual sanity check without leaving the notebook (Fiji/napari still the
    thorough check, but this catches an obviously-wrong or degenerate classifier fast).

    Returns [(tag, seg_path, [(z, [raw_channel_images...], seg_image), ...]), ...] for
    the notebook to plot -- matching Pipeline.preview()'s split of "return the data
    here, draw it in the notebook" so this module doesn't need matplotlib.
    """
    import h5py
    import tifffile

    by_scene = {plan.scene_id: path for plan, path in results}
    nz = len(source.sel_z)
    n_planes = max(1, min(n_planes, nz))
    zis = sorted({round(i * (nz - 1) / max(n_planes - 1, 1)) for i in range(n_planes)})

    def downsample(a):
        step = max(1, int(np.ceil(max(a.shape[:2]) / max_px)))
        return a[::step, ::step]

    out = []
    for scene_id, tag, seg_path in seg:
        raw_path = by_scene.get(scene_id)
        if raw_path is None:
            continue
        is_h5 = Path(raw_path).suffix.lower() in (".h5", ".hdf5")
        planes = []
        with tifffile.TiffFile(seg_path) as segtf:
            seg_channels = max(1, len(segtf.pages) // nz)   # 1 for Simple Segmentation
            raw_ctx = h5py.File(raw_path, "r") if is_h5 else tifffile.TiffFile(raw_path)
            with raw_ctx as rawsrc:
                dset = rawsrc[cfg.hdf5_dataset] if is_h5 else None
                for zi in zis:
                    if is_h5:
                        chans = [downsample(np.asarray(dset[0, zi, ci]))
                                for ci in range(len(source.sel_c))]
                    else:
                        chans = [downsample(rawsrc.pages[tiler.page_index(0, zi, ci)].asarray())
                                for ci in range(len(source.sel_c))]
                    seg_img = downsample(segtf.pages[zi * seg_channels].asarray())
                    planes.append((source.sel_z[zi], chans, seg_img))
        out.append((tag, seg_path, planes))
    return out
