"""Chunked ilastik classification (cfg.ilastik.chunk_z), the run heartbeat, and the
segmentation preview.

ilastik is faked by a classifier whose answer at each plane depends on the planes either
side of it, like a real feature filter with a Z extent. That makes a missing or
mis-trimmed halo visible as a seam at every chunk boundary, so "chunked equals
monolithic" is a real test of the halo logic rather than a tautology. These tests use
the real h5py, since the resume record lives in HDF5 attributes.
"""

from __future__ import annotations

import sys
import textwrap

import numpy as np
import pytest
import tifffile

from czipipe import ilastik, validate
from czipipe.compute import Backend
from czipipe.geometry import Tiler
from czipipe.pipeline import _null_progress
from czipipe.plan import ScenePlan
from czipipe.writers import vigra_axistags, write_tiff

from conftest import FakeSource

try:
    import h5py
except ImportError:          # tools/minipytest.py has no importorskip, so skip per test
    h5py = None


@pytest.fixture
def real_h5py():
    if h5py is None:
        pytest.skip("h5py not installed")
    return h5py

SHAPE = (1, 23, 2, 24, 20)          # T, Z, C, Y, X
CHUNK = 5                           # -> 5 chunks, the last one short


def classifier(vol):
    """Label 2 where the mean over this plane and its Z-neighbours is bright, else 1."""
    nz = vol.shape[1]
    out = np.empty((vol.shape[0], nz, 1) + vol.shape[3:], np.uint8)
    for z in range(nz):
        window = vol[:, max(0, z - 1):min(nz, z + 2)].astype(np.float64)
        out[:, z, 0] = np.where(window.mean(axis=(1, 2)) > 32768, 2, 1)
    return out


class FakeIlastik:
    """Stands in for ilastik.run_command: classifies the input named on the command line."""

    def __init__(self):
        self.calls = 0
        self.fail_on = None          # (call number, exception) to raise instead of running
        self.bad_shape_on = None     # call number that "succeeds" with a malformed export
        self.dry = False             # behave like cfg.ilastik.dry_run: print, don't run

    def __call__(self, cfg, cmd, log_path, log=print):
        self.calls += 1
        if self.dry:
            return False
        if self.fail_on and self.calls == self.fail_on[0]:
            raise self.fail_on[1]
        out = [a.split("=", 1)[1] for a in cmd if a.startswith("--output_filename_format=")][0]
        h5_path, internal = cmd[-1].rsplit(".h5/", 1)
        with h5py.File(h5_path + ".h5", "r") as f:
            result = classifier(f[internal][()])
        if self.calls == self.bad_shape_on:
            result = result[:, :-1]          # ilastik logged an error but still exited 0
        with h5py.File(out, "w") as f:
            f.create_dataset("exported_data", data=result)
        return True


@pytest.fixture
def run(real_h5py, cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(ilastik, "check_no_spaces", lambda *p: None)
    fake = FakeIlastik()
    monkeypatch.setattr(ilastik, "run_command", fake)

    vol = np.random.default_rng(11).integers(0, 65535, SHAPE, dtype=np.uint16)
    raw = tmp_path / "raw.h5"
    with h5py.File(raw, "w") as f:
        f.create_dataset("data", data=vol).attrs["axistags"] = vigra_axistags("tzcyx")
    ilp = tmp_path / "proj.ilp"
    ilp.write_bytes(b"project")
    cfg.ilastik.chunk_z = CHUNK
    out = cfg.out_dir / "seg.h5"
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    export_shape = SHAPE[:2] + (1,) + SHAPE[3:]
    logs = []

    def go(halo_z=1):
        logs.clear()
        return ilastik.run_classify_chunked(cfg, "ilastik", ilp, raw, "/data", out, "tzcyx",
                                            export_shape, halo_z=halo_z, log=logs.append)

    go.fake, go.vol, go.out, go.ilp, go.logs = fake, vol, out, ilp, logs
    return go


def exported(path):
    with h5py.File(path, "r") as f:
        return f["exported_data"][()], dict(f["progress"].attrs)


# ---------------------------------------------------------------- correctness
def test_chunked_export_matches_one_monolithic_run(run):
    assert run() is True
    data, prog = exported(run.out)
    assert np.array_equal(data, classifier(run.vol))
    assert run.fake.calls == 5
    assert prog["complete"] and prog["z_done"] == SHAPE[1]
    assert not list(run.out.parent.glob("*.chunk_in.h5"))
    assert not list(run.out.parent.glob("*.chunk_out.h5"))


def test_without_a_halo_the_chunk_seams_show(run):
    """Guards the test above: it would pass vacuously if the halo had no effect."""
    run(halo_z=0)
    data, _ = exported(run.out)
    differs = np.flatnonzero((data != classifier(run.vol)).any(axis=(0, 2, 3, 4)))
    assert set(differs) <= {4, 5, 9, 10, 14, 15, 19, 20}    # only either side of a seam
    assert len(differs) >= 4


def test_every_timepoint_is_classified(real_h5py, cfg, tmp_path, monkeypatch):
    """Each chunk's input was once staged from timepoint 0 alone, so every later
    timepoint was classified from zeros: a plausible label map, and no error."""
    monkeypatch.setattr(ilastik, "check_no_spaces", lambda *p: None)
    monkeypatch.setattr(ilastik, "run_command", FakeIlastik())
    shape = (2,) + SHAPE[1:]
    vol = np.random.default_rng(12).integers(0, 65535, shape, dtype=np.uint16)
    raw = tmp_path / "raw_t.h5"
    with h5py.File(raw, "w") as f:
        f.create_dataset("data", data=vol).attrs["axistags"] = vigra_axistags("tzcyx")
    ilp = tmp_path / "proj.ilp"
    ilp.write_bytes(b"project")
    cfg.ilastik.chunk_z = CHUNK
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / "seg_t.h5"
    ilastik.run_classify_chunked(cfg, "ilastik", ilp, raw, "/data", out, "tzcyx",
                                 shape[:2] + (1,) + shape[3:], halo_z=1,
                                 log=lambda *a: None)
    data, _ = exported(out)
    assert np.array_equal(data, classifier(vol))


# ---------------------------------------------------------------- restart semantics
def test_interrupted_run_resumes_at_the_next_unfinished_chunk(run):
    run.fake.fail_on = (3, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run()
    _, prog = exported(run.out)
    assert prog["z_done"] == 10 and not prog["complete"]

    run.fake.fail_on, run.fake.calls = None, 0
    assert run() is True
    assert run.fake.calls == 3
    assert any("resuming at Z=10/23" in m for m in run.logs)
    data, _ = exported(run.out)
    assert np.array_equal(data, classifier(run.vol))


def test_chunk_size_can_change_between_runs_without_redoing_planes(run, cfg):
    """The halo makes small chunks expensive, and the log says so; acting on that mid-way
    must not cost the planes already exported. Resuming used to snap back to a multiple
    of the new chunk_z, which here would have gone back to plane 7 and taken three runs."""
    run.fake.fail_on = (3, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run()
    cfg.ilastik.chunk_z = 7
    run.fake.fail_on, run.fake.calls = None, 0
    assert run() is True
    assert run.fake.calls == 2                           # 10:17 and 17:23, nothing redone
    data, prog = exported(run.out)
    assert np.array_equal(data, classifier(run.vol))
    assert prog["complete"] and prog["z_done"] == SHAPE[1]


def test_halo_cost_is_counted_and_reported(run):
    # The field run this came from: 440 planes, max sigma 10 -> a 35-plane halo.
    assert ilastik.planes_computed(0, 440, 50, 35) == 1000
    assert ilastik.planes_computed(0, 440, 200, 35) == 580
    assert ilastik.planes_computed(100, 440, 50, 35) == 795      # resuming at Z=100
    assert ilastik.planes_computed(0, 440, 50, 0) == 440

    run()
    line = [m for m in run.logs if "halo cost" in m]
    assert len(line) == 1
    assert "31 planes classified to export 23 (1.3x) at chunk_z=5" in line[0]
    assert "chunk_z=10 -> 1.2x" in line[0] and "chunk_z=20 -> 1.1x" in line[0]


def test_no_halo_cost_line_without_a_halo(run):
    run(halo_z=0)
    assert not any("halo cost" in m for m in run.logs)


def test_finished_export_is_not_rerun(run):
    run()
    run.fake.calls = 0
    assert run() is True
    assert run.fake.calls == 0
    assert any("already complete" in m for m in run.logs)


def test_partial_export_from_a_different_project_is_refused(run):
    run.fake.fail_on = (2, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        run()
    run.ilp.write_bytes(b"project, retrained")
    run.fake.fail_on = None
    with pytest.raises(RuntimeError, match="different settings") as e:
        run()
    assert "ilp_bytes" in str(e.value)


def test_export_with_the_wrong_shape_fails_that_chunk_and_is_retried(run):
    run.fake.bad_shape_on = 2
    with pytest.raises(RuntimeError, match=r"chunk 2/5 \(Z=5:10\)"):
        run()
    run.fake.bad_shape_on, run.fake.calls = None, 0
    assert run() is True
    assert run.fake.calls == 4
    data, _ = exported(run.out)
    assert np.array_equal(data, classifier(run.vol))


def test_dry_run_stops_after_printing_the_first_chunk(run):
    run.fake.dry = True
    assert run() is False
    assert run.fake.calls == 1
    _, prog = exported(run.out)
    assert prog["z_done"] == 0 and not prog["complete"]


# ---------------------------------------------------------------- through classify()
RECT = (0, 0, 40, 30)
SEL_T, SEL_Z, SEL_C = [0], list(range(7)), [0]


@pytest.fixture
def job(real_h5py, cfg, monkeypatch):
    monkeypatch.setattr(ilastik, "check_no_spaces", lambda *p: None)
    monkeypatch.setattr(validate, "check_no_spaces", lambda *p: None)
    fake = FakeIlastik()
    monkeypatch.setattr(ilastik, "run_command", fake)
    monkeypatch.setattr(ilastik, "inspect_project",
                        lambda p: {"path": str(p), "workflow": "Pixel Classification",
                                   "classes": ["background", "vessel"], "channels": None,
                                   "axes": None, "max_sigma": 0.25, "n_features": None})
    cfg.overwrite, cfg.output_format = False, "tiff"
    cfg.ilastik.run, cfg.ilastik.exe, cfg.ilastik.chunk_z = True, "ilastik", 3
    ilp = cfg.out_dir.parent / "proj.ilp"
    ilp.write_bytes(b"project")
    cfg.ilastik.projects = [("seg", str(ilp))]

    data = np.random.default_rng(4).integers(
        0, 65535, (len(SEL_T), len(SEL_Z), len(SEL_C), RECT[3], RECT[2]), dtype=np.uint16)
    src = FakeSource(data, RECT, SEL_T, SEL_Z, SEL_C, cfg)
    tiler = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, 1)
    plan = ScenePlan(None, RECT, {0: (0.0, 65535.0)})
    tif = write_tiff(src, cfg, Backend(False), tiler, plan,
                     log=lambda *a: None, progress=_null_progress)
    return src, tiler, plan, tif, data, fake


def test_classify_with_chunk_z_produces_a_labelled_segmentation(cfg, job):
    src, tiler, plan, tif, data, fake = job
    seg = ilastik.classify(src, cfg, tiler, [(plan, tif)], log=lambda *a: None,
                           progress=_null_progress)
    assert fake.calls == 3                               # 7 planes in chunks of 3
    (scene, tag, path), = seg
    with tifffile.TiffFile(path) as tf:
        back = np.stack([p.asarray() for p in tf.pages])
        ome = tf.ome_metadata
    assert np.array_equal(back, classifier(data).reshape(back.shape))
    assert "Label values: 1=background, 2=vessel" in ome
    assert not (cfg.out_dir / "fake_seg.h5").exists()    # keep_h5=False tidies up


def test_preview_shows_raw_and_labels_at_spread_depths(cfg, job):
    src, tiler, plan, tif, data, _ = job
    seg = ilastik.classify(src, cfg, tiler, [(plan, tif)], log=lambda *a: None,
                           progress=_null_progress)
    (tag, path, planes), = ilastik.preview_segmentation(src, cfg, tiler, [(plan, tif)], seg,
                                                         n_planes=3, max_px=1200)
    assert tag == "seg" and [z for z, _, _ in planes] == [0, 3, 6]
    expected = classifier(data)
    for z, chans, seg_img in planes:
        assert len(chans) == 1 and np.array_equal(chans[0], data[0, z, 0])
        assert np.array_equal(seg_img, expected[0, z, 0])


# ---------------------------------------------------------------- heartbeat
def test_heartbeat_reports_progress_and_flags_an_idle_process(cfg, tmp_path):
    try:
        import psutil  # noqa: F401  (the CPU figures need it)
    except ImportError:
        pytest.skip("psutil not installed")
    script = tmp_path / "quiet_ilastik.py"
    script.write_text(textwrap.dedent("""
        import time
        time.sleep(1.8)
    """))
    logs = []
    assert ilastik.run_command(cfg, [sys.executable, str(script)], tmp_path / "run.log",
                               log=logs.append, heartbeat_s=0.4)
    beats = [m for m in logs if "still running" in m]
    assert len(beats) >= 2
    assert "CPU-min used across the process tree" in beats[0]
    assert any("may be hung" in m for m in beats[1:])
