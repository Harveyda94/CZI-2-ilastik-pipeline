"""HDF5 output, crash/resume semantics, and the ilastik compatibility check."""

from __future__ import annotations

import json
import sys
import threading

import numpy as np
import pytest

from czipipe.compute import Backend
from czipipe.geometry import Tiler
from czipipe.plan import ScenePlan
from czipipe.pipeline import _null_progress
from czipipe import writers, validate

from conftest import FakeH5, FakeSource, snapshot_of

RECT = (-13, 7, 300, 220)
SEL_T, SEL_Z, SEL_C = [0, 1], [0, 1, 2], [0, 1]
TOTAL = 2 * 3 * 2 * 4 * 5          # (220/64->4) x (300/64->5) tiles per plane


@pytest.fixture
def h5(monkeypatch):
    """Install the flush-aware h5py shim for the duration of a test."""
    monkeypatch.setitem(sys.modules, "h5py", FakeH5)
    return FakeH5


def build(cfg, bit_depth=16):
    cfg.output_format = "hdf5"
    cfg.output_bit_depth = bit_depth
    cfg.hdf5_checkpoint_every = 25
    cfg.overwrite = False
    data = np.random.default_rng(4).integers(
        0, 65535, (len(SEL_T), len(SEL_Z), len(SEL_C), RECT[3], RECT[2]), dtype=np.uint16)
    src = FakeSource(data, RECT, SEL_T, SEL_Z, SEL_C, cfg)
    tiler = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, 1)
    plan = ScenePlan(None, RECT, {c: (0.0, 65535.0) for c in SEL_C})
    return src, tiler, plan, data


def write(cfg, src, tiler, plan, crash_after=None):
    backend = Backend(False)
    if crash_after is None:
        return writers.write_hdf5(src, cfg, backend, tiler, plan,
                                  log=lambda *a: None, progress=_null_progress)
    real = writers.make_worker
    n, lock = [0], threading.Lock()

    def crashing(*a, **k):
        w = real(*a, **k)

        def f(task):
            with lock:
                n[0] += 1
                if n[0] > crash_after:
                    raise RuntimeError("simulated crash")
            return w(task)
        return f

    writers.make_worker = crashing
    try:
        return writers.write_hdf5(src, cfg, backend, tiler, plan,
                                  log=lambda *a: None, progress=_null_progress)
    finally:
        writers.make_worker = real


def _item(path, name):
    items = snapshot_of(path)["items"]
    return items.get(name) or items[f"/{name}"]


def dataset_of(path):
    return _item(path, "data")[1]


def attrs_of(path):
    return _item(path, "data")[5]


def progress_of(path):
    return _item(path, "progress")[1]["attrs"]


# ---------------------------------------------------------------- basics
def test_hdf5_content_is_bit_exact(cfg, h5):
    src, tiler, plan, data = build(cfg)
    path = write(cfg, src, tiler, plan)
    assert np.array_equal(dataset_of(path), data)
    assert progress_of(path)["complete"] is True


def test_axistags_and_scaling_written(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    path = write(cfg, src, tiler, plan)
    attrs = attrs_of(path)
    keys = [a["key"] for a in json.loads(attrs["axistags"])["axes"]]
    assert keys == list("tzcyx")
    assert list(attrs["element_size_um"]) == [1.0, 0.23, 0.23]     # (z, y, x)
    assert list(attrs["channel_names"]) == ["C0", "C1"]


def test_chunks_are_one_plane_deep(cfg, h5):
    """A chunk spanning z would be rewritten once per plane, given the write order."""
    src, tiler, plan, _ = build(cfg)
    path = write(cfg, src, tiler, plan)
    assert _item(path, "data")[4] == (1, 1, 1, 64, 64)


def test_verify_passes_on_hdf5(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    path = write(cfg, src, tiler, plan)
    assert validate.verify(src, cfg, Backend(False), tiler, plan, path, n=40,
                             log=lambda *a: None) is True


# ---------------------------------------------------------------- resume
def test_checkpoint_never_overstates_progress(cfg, h5):
    """Data is flushed before the counter, so a crash can only under-report."""
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=118)
    done = progress_of(cfg.out_dir / "fake.h5")["tiles_written"]
    assert done <= 118
    assert done % cfg.hdf5_checkpoint_every == 0


def test_resume_produces_bit_exact_file(cfg, h5):
    src, tiler, plan, data = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=118)
    path = write(cfg, src, tiler, plan)
    assert np.array_equal(dataset_of(path), data)
    assert progress_of(path)["tiles_written"] == TOTAL
    assert progress_of(path)["complete"] is True


def test_partial_file_is_left_in_place(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)
    assert (cfg.out_dir / "fake.h5").exists()


def test_partial_file_deleted_when_resume_disabled(cfg, h5):
    cfg.hdf5_resume = False
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)
    assert not (cfg.out_dir / "fake.h5").exists()


def test_completed_file_is_skipped(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    write(cfg, src, tiler, plan)
    msgs = []
    writers.write_hdf5(src, cfg, Backend(False), tiler, plan,
                       log=msgs.append, progress=_null_progress)
    assert any("already complete" in m for m in msgs)


def test_metadata_survives_a_crash(cfg, h5):
    """Written up front, so a partial file is still self-describing."""
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)
    assert _item(cfg.out_dir / "fake.h5", "metadata") is not None


@pytest.mark.parametrize("field,value", [
    ("output_bit_depth", 8),
    ("tile", (128, 128)),
    ("hdf5_compression", "lzf"),
    ("stitch_mode", "refine"),
])
def test_changed_settings_refuse_to_resume(cfg, h5, field, value):
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)

    setattr(cfg, field, value)
    src2 = FakeSource(src.data, RECT, SEL_T, SEL_Z, SEL_C, cfg)
    tiler2 = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, 1)
    with pytest.raises(RuntimeError, match="different settings"):
        writers.write_hdf5(src2, cfg, Backend(False), tiler2, plan,
                           log=lambda *a: None, progress=_null_progress)


def test_mismatch_names_the_changed_setting(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)
    cfg.tile = (128, 128)
    tiler2 = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, 1)
    with pytest.raises(RuntimeError) as e:
        writers.write_hdf5(src, cfg, Backend(False), tiler2, plan,
                           log=lambda *a: None, progress=_null_progress)
    assert "tile: was [64, 64], now [128, 128]" in str(e.value)


# ---------------------------------------------------------------- ilastik check
def test_valid_file_passes_ilastik_check(cfg, h5):
    src, tiler, plan, _ = build(cfg)
    path = write(cfg, src, tiler, plan)
    ok, notes = validate.check_h5_for_ilastik(path, cfg.hdf5_dataset, src.h5_axes)
    assert ok, notes


def test_incomplete_file_fails_ilastik_check(cfg, h5):
    """A half-converted image must not quietly reach a classifier."""
    src, tiler, plan, _ = build(cfg)
    with pytest.raises(RuntimeError):
        write(cfg, src, tiler, plan, crash_after=50)
    ok, notes = validate.check_h5_for_ilastik(cfg.out_dir / "fake.h5", cfg.hdf5_dataset)
    assert not ok
    assert any("incomplete" in n for n in notes)


@pytest.mark.parametrize("mutate,expect", [
    (lambda d: d.attrs.__setitem__("axistags", writers.vigra_axistags("yxc")), "5 dimensions"),
    (lambda d: d.attrs.pop("axistags"), "no 'axistags'"),
    (lambda d: setattr(d, "chunks", None), "not chunked"),
    (lambda d: setattr(d, "dtype", np.dtype("complex64")), "not one ilastik handles"),
])
def test_broken_files_fail_ilastik_check(cfg, h5, mutate, expect):
    src, tiler, plan, _ = build(cfg)
    path = write(cfg, src, tiler, plan)
    with FakeH5.File(path, "a") as f:
        mutate(f["data"])
        f.flush()
    ok, notes = validate.check_h5_for_ilastik(path, cfg.hdf5_dataset)
    assert not ok
    assert any(expect in n for n in notes), notes


def test_paths_with_spaces_are_rejected():
    with pytest.raises(ValueError, match="spaces"):
        validate.check_no_spaces("/tmp/my data/scan.h5")
