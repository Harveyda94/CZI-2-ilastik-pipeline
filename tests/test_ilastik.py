"""Pixel classification: restart semantics, process control and locating ilastik.

ilastik itself is faked in-process (it reads the staged input and writes an export) so
the orchestration around it can be interrupted at each stage. The process-control tests
run real subprocesses, since that is where the platform differences live.
"""

from __future__ import annotations

import os
import sys
import textwrap
import time

import numpy as np
import pytest
import tifffile

from czipipe import ilastik, validate
from czipipe.compute import Backend
from czipipe.geometry import Tiler
from czipipe.pipeline import Pipeline, _null_progress
from czipipe.plan import ScenePlan
from czipipe.writers import write_tiff

from conftest import FakeH5, FakeSource, ilastik_export_target

RECT = (0, 0, 150, 100)
SEL_T, SEL_Z, SEL_C = [0], [0, 1], [0]


# ---------------------------------------------------------------- harness
@pytest.fixture
def job(cfg, tmp_path, monkeypatch):
    """A converted TIFF, a project file, and a fake ilastik that counts its runs."""
    monkeypatch.setitem(sys.modules, "h5py", FakeH5)
    # The harness's temp dir may contain spaces (Windows user names); the guard against
    # them is tested separately in test_hdf5.
    monkeypatch.setattr(ilastik, "check_no_spaces", lambda *p: None)
    monkeypatch.setattr(validate, "check_no_spaces", lambda *p: None)

    cfg.overwrite = False
    cfg.output_format = "tiff"
    cfg.ilastik.run = True
    cfg.ilastik.exe = "ilastik-not-needed"
    ilp = tmp_path / "proj.ilp"
    ilp.write_bytes(b"project")
    cfg.ilastik.projects = [("seg", str(ilp))]
    monkeypatch.setattr(ilastik, "inspect_project",
                        lambda p: {"path": str(p), "workflow": "Pixel Classification",
                                   "classes": ["a", "b"], "channels": None, "axes": None,
                                   "max_sigma": None, "n_features": None})

    data = np.random.default_rng(3).integers(
        0, 65535, (len(SEL_T), len(SEL_Z), len(SEL_C), RECT[3], RECT[2]), dtype=np.uint16)
    src = FakeSource(data, RECT, SEL_T, SEL_Z, SEL_C, cfg)
    tiler = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, 1)
    plan = ScenePlan(None, RECT, {0: (0.0, 65535.0)})
    tif = write_tiff(src, cfg, Backend(False), tiler, plan,
                     log=lambda *a: None, progress=_null_progress)

    runs = []

    def fake_run(cfg_, cmd, log_path, log=print):
        out = [a.split("=", 1)[1] for a in cmd if a.startswith("--output_filename_format=")][0]
        h5_path, internal = cmd[-1].rsplit(".h5/", 1)
        with FakeH5.File(h5_path + ".h5", "r") as fi:
            d = np.asarray(fi[internal][:])
        group, name = ilastik_export_target(cmd)
        with FakeH5.File(out, "w") as fo:
            if name:
                fo.create_dataset(name, shape=d.shape, dtype=np.uint8,
                                  chunks=None)[...] = (d >> 8).astype(np.uint8)
            else:           # ilastik makes the group, then has no name for the dataset
                fo.create_group(group.strip("/"))
            fo.flush()
        runs.append(out)
        return True

    monkeypatch.setattr(ilastik, "run_command", fake_run)

    class Job:
        pass

    j = Job()
    j.cfg, j.src, j.tiler, j.plan, j.tif, j.data, j.runs, j.ilp = \
        cfg, src, tiler, plan, tif, data, runs, ilp
    j.seg = cfg.out_dir / "fake_seg.ome.tif"
    j.staged = tif.with_suffix(".ilastik_input.h5")
    j.export = cfg.out_dir / "fake_seg.h5"
    return j


def classify(j, progress=_null_progress, log=None):
    return ilastik.classify(j.src, j.cfg, j.tiler, [(j.plan, j.tif)],
                            log=log or (lambda *a: None), progress=progress)


def dies_during(stage, at=0.5):
    """A progress factory that raises KeyboardInterrupt part-way through `stage`."""
    def factory(iterable=None, total=None, desc="", **kw):
        if desc != stage:
            return _null_progress(iterable, total=total, **kw)

        class Bar:
            n = 0

            def __enter__(self):
                return self

            def __exit__(self, *e):
                return False

            def update(self, k=1):
                self.n += k
                if self.n >= max(1, int(total * at)):
                    raise KeyboardInterrupt
        return Bar()
    return factory


def seg_is_correct(j):
    with tifffile.TiffFile(j.seg) as tf:
        back = np.stack([p.asarray() for p in tf.pages])
    return np.array_equal(back, (j.data >> 8).astype(np.uint8).reshape(back.shape))


# ---------------------------------------------------------------- restart semantics
def test_clean_run_produces_segmentation_and_tidies_up(job):
    out = classify(job)
    assert out == [(None, "seg", job.seg)]
    assert seg_is_correct(job)
    assert not job.staged.exists() and not job.export.exists()


def test_interrupted_tiff_conversion_leaves_no_segmentation(job):
    """The bug: a half-written TIFF under the final name was skipped as done next time."""
    with pytest.raises(KeyboardInterrupt):
        classify(job, progress=dies_during("-> tiff"))
    assert not job.seg.exists()
    assert not job.seg.with_name(job.seg.name + ".partial").exists()


def test_rerun_after_tiff_interruption_reuses_the_export(job):
    with pytest.raises(KeyboardInterrupt):
        classify(job, progress=dies_during("-> tiff"))
    msgs = []
    classify(job, log=msgs.append)
    assert seg_is_correct(job)
    assert len(job.runs) == 1, "ilastik was run again although its export was complete"
    assert any("reusing" in m and job.export.name in m for m in msgs)


def test_export_is_not_reused_when_the_project_changed(job):
    with pytest.raises(KeyboardInterrupt):
        classify(job, progress=dies_during("-> tiff"))
    time.sleep(0.01)
    job.ilp.write_bytes(b"retrained project")
    classify(job)
    assert len(job.runs) == 2


def test_failed_ilastik_run_keeps_the_staged_input(job, monkeypatch):
    def broken(*a, **k):
        raise RuntimeError("ilastik exited 1")

    monkeypatch.setattr(ilastik, "run_command", broken)
    with pytest.raises(RuntimeError):
        classify(job)
    assert job.staged.exists()


def test_rerun_reuses_the_staged_input(job, monkeypatch):
    real = ilastik.run_command
    monkeypatch.setattr(ilastik, "run_command",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        classify(job)
    monkeypatch.setattr(ilastik, "run_command", real)
    msgs = []
    classify(job, log=msgs.append)
    assert any("reusing existing" in m for m in msgs)
    assert seg_is_correct(job)
    assert not job.staged.exists()                      # removed once everything finished


def test_staged_input_is_rebuilt_when_the_tiff_changed(job, monkeypatch):
    monkeypatch.setattr(ilastik, "run_command",
                        lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        classify(job)
    time.sleep(0.01)
    job.cfg.overwrite = True                            # reconvert the source TIFF
    write_tiff(job.src, job.cfg, Backend(False), job.tiler, job.plan,
               log=lambda *a: None, progress=_null_progress)
    job.cfg.overwrite = False
    msgs = []
    with pytest.raises(KeyboardInterrupt):
        classify(job, log=msgs.append)
    assert any("rewriting" in m for m in msgs)


def test_finished_segmentation_is_skipped_and_still_returned(job):
    classify(job)
    msgs = []
    out = classify(job, log=msgs.append)
    assert out == [(None, "seg", job.seg)]
    assert len(job.runs) == 1
    assert any("skipping" in m for m in msgs)


def test_classify_without_convert_says_what_to_do(job):
    """After a kernel restart pipe.results is empty; this used to return [] silently."""
    p = Pipeline.__new__(Pipeline)
    p.cfg, p.log, p.progress, p.results = job.cfg, (lambda *a: None), _null_progress, []
    with pytest.raises(RuntimeError, match="convert"):
        p.classify()


# ---------------------------------------------------------------- runs that export nothing
DIED_AT_STARTUP = """\
ERROR 2026-11-24 14:43:14,982 excepthooks 17180 27656 Unhandled exception in thread: 'MainThread'
ERROR 2026-11-24 14:43:14,998 excepthooks 17180 27656 Traceback (most recent call last):
  File "matplotlib/__init__.py", line 774, in __setitem__
    raise ValueError(f"Key {key}: {ve}") from None
ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is not a valid value

ilastik error:
The operation completed successfully.
"""


def test_run_that_writes_nothing_fails_and_marks_nothing_complete(job, monkeypatch):
    """ilastik on Windows exited 0 after stopping at start-up. Marking the export complete
    then created it, empty; the conversion failed on a missing HDF5 object, and every
    later run reused the empty file and failed the same way."""
    real = ilastik.run_command

    def wrote_nothing(cfg_, cmd, log_path, log=print):
        log_path.write_text(DIED_AT_STARTUP, encoding="utf-8")
        return True

    monkeypatch.setattr(ilastik, "run_command", wrote_nothing)
    with pytest.raises(RuntimeError, match="wrote no output file") as e:
        classify(job)
    assert "ValueError: Key backend" in str(e.value)
    assert "fake_seg.ilastik.log" in str(e.value)
    assert not job.export.exists() and not job.seg.exists()

    monkeypatch.setattr(ilastik, "run_command", real)
    classify(job)
    assert seg_is_correct(job) and len(job.runs) == 1


def test_output_without_an_export_fails_and_is_classified_again(job, monkeypatch):
    real = ilastik.run_command

    def wrote_an_empty_file(cfg_, cmd, log_path, log=print):
        out = [a.split("=", 1)[1] for a in cmd if a.startswith("--output_filename_format=")][0]
        with FakeH5.File(out, "w") as fo:
            fo.flush()
        return True

    monkeypatch.setattr(ilastik, "run_command", wrote_an_empty_file)
    with pytest.raises(RuntimeError, match="holds no 'exported_data'"):
        classify(job)
    assert not job.seg.exists()

    monkeypatch.setattr(ilastik, "run_command", real)
    msgs = []
    classify(job, log=msgs.append)
    assert seg_is_correct(job) and len(job.runs) == 1
    assert not any("reusing" in m and job.export.name in m for m in msgs)


def test_export_path_names_a_dataset_to_ilastik_on_windows(cfg):
    """'--output_internal_path=/exported_data' reached ilastik's writer as
    '//exported_data', which Windows path rules read as a network share with no file
    name. ilastik 1.4.2 made a group called exported_data and wrote no image."""
    cmd = ilastik.build_command(cfg, "ilastik", "p.ilp", "raw.h5", "/data", "out.h5", "tzcyx")
    assert ilastik_export_target(cmd) == ("/", "exported_data")
    # The form that failed, so the line above cannot pass by the stand-in being lenient.
    assert ilastik_export_target(["--output_internal_path=/exported_data"]) \
        == ("//exported_data", "")


def test_export_that_is_a_group_fails_and_is_classified_again(job, monkeypatch):
    """What that ilastik left. The check passed it and it was marked complete; the
    conversion, and every later run, stopped on "'Group' object has no attribute
    'shape'"."""
    real = ilastik.run_command

    def wrote_a_group(cfg_, cmd, log_path, log=print):
        out = [a.split("=", 1)[1] for a in cmd if a.startswith("--output_filename_format=")][0]
        with FakeH5.File(out, "w") as fo:
            fo.create_group("exported_data")
            fo.flush()
        return True

    monkeypatch.setattr(ilastik, "run_command", wrote_a_group)
    with pytest.raises(RuntimeError, match="is a group with no image in it"):
        classify(job)
    assert not job.seg.exists()

    monkeypatch.setattr(ilastik, "run_command", real)
    msgs = []
    classify(job, log=msgs.append)
    assert seg_is_correct(job) and len(job.runs) == 1
    assert not any("reusing" in m and job.export.name in m for m in msgs)


def test_completion_marker_on_a_file_with_no_export_is_not_trusted(job):
    """What the earlier versions left behind. It has to be classified again, not reused."""
    with FakeH5.File(job.export, "w") as f:
        f.attrs[ilastik.EXPORT_DONE_ATTR] = "this job"
        f.flush()
    assert not ilastik._export_is_complete(job.export, "this job")

    with FakeH5.File(job.export, "w") as f:
        f.create_group("exported_data")
        f.attrs[ilastik.EXPORT_DONE_ATTR] = "this job"
        f.flush()
    assert not ilastik._export_is_complete(job.export, "this job")

    with FakeH5.File(job.export, "w") as f:
        f.create_dataset("exported_data", shape=(1, 2, 1, 4, 4), dtype=np.uint8)
        f.attrs[ilastik.EXPORT_DONE_ATTR] = "this job"
        f.flush()
    assert ilastik._export_is_complete(job.export, "this job")
    assert not ilastik._export_is_complete(job.export, "another job")


# ---------------------------------------------------------------- process control
def _script(tmp_path, body):
    path = tmp_path / "fake_ilastik.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(path)]


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_dead(pid, seconds=10.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if not _alive(pid):
            return True
        time.sleep(0.05)
    return False


SPAWNS_A_WORKER = """
    import subprocess, sys, time
    worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    open(sys.argv[1], "w").write(str(worker.pid))
    print("Exporting 10%", flush=True)
    time.sleep(120)
"""


def test_non_ascii_and_undecodable_output_does_not_stop_the_run(cfg, tmp_path):
    """Windows decoded ilastik's output with the ANSI code page; one stray byte raised
    UnicodeDecodeError mid-run and left ilastik running unsupervised."""
    cmd = _script(tmp_path, r"""
        import sys
        sys.stdout.buffer.write("scaling 0.23 µm\n".encode("utf-8"))
        sys.stdout.buffer.write(b"raw byte \xff\x9d here\n")
        sys.stdout.flush()
    """)
    log_path = tmp_path / "ilastik.log"
    assert ilastik.run_command(cfg, cmd, log_path, log=lambda *a: None) is True
    text = log_path.read_text(encoding="utf-8")
    assert "µm" in text and "raw byte" in text


def test_timeout_fires_while_ilastik_is_still_running(cfg, tmp_path):
    """wait(timeout=...) used to start only once stdout closed, i.e. after ilastik had
    already finished, so timeout_h could never fire."""
    pidfile = tmp_path / "worker.pid"
    cmd = _script(tmp_path, SPAWNS_A_WORKER) + [str(pidfile)]
    cfg.ilastik.timeout_h = 1.5 / 3600
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="timeout_h"):
        ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=lambda *a: None)
    assert time.monotonic() - t0 < 30
    if os.name != "nt":
        assert _wait_dead(int(pidfile.read_text())), "ilastik's worker process survived"


def test_interrupt_stops_the_whole_process_tree(cfg, tmp_path):
    """Killing only the launcher left a child worker running, holding the export open."""
    if os.name == "nt":
        pytest.skip("process-group check is POSIX-only; Windows uses taskkill /T")
    pidfile = tmp_path / "worker.pid"
    cmd = _script(tmp_path, SPAWNS_A_WORKER) + [str(pidfile)]

    def interrupt_on_progress(msg):
        if "Exporting" in msg:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=interrupt_on_progress)
    assert _wait_dead(int(pidfile.read_text())), "ilastik's worker process survived"


def test_nonzero_exit_is_reported(cfg, tmp_path):
    cmd = _script(tmp_path, "import sys; print('ERROR bad project'); sys.exit(3)")
    with pytest.raises(RuntimeError, match="exited 3"):
        ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=lambda *a: None)


def test_unhandled_error_is_a_failure_even_with_exit_code_zero(cfg, tmp_path):
    """The Windows launcher printed "The operation completed successfully" and returned 0
    after ilastik had stopped on an exception while loading its plugins."""
    cmd = _script(tmp_path, """
        print("ERROR excepthooks 1 2 Unhandled exception in thread: 'MainThread'")
        print("Traceback (most recent call last):")
        print("ValueError: Key backend: 'module://matplotlib_inline.backend_inline' is bad")
    """)
    with pytest.raises(RuntimeError, match="exited with code 0") as e:
        ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=lambda *a: None)
    assert "ValueError: Key backend" in str(e.value)
    assert "ilastik.log" in str(e.value)


def test_error_after_the_export_finished_does_not_fail_the_run(cfg, tmp_path):
    cmd = _script(tmp_path, """
        print("INFO ilastik.workflows.pixelClassification: Completed Batch Processing")
        print("ERROR excepthooks 1 2 Unhandled exception in thread: 'MainThread'")
    """)
    assert ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log",
                               log=lambda *a: None) is True


def test_notebook_matplotlib_backend_is_not_passed_to_ilastik(cfg, tmp_path, monkeypatch):
    """A Jupyter kernel sets MPLBACKEND to its inline backend. ilastik 1.4.2 imports
    matplotlib at start-up, does not have that backend, and stopped there."""
    monkeypatch.setitem(os.environ, "MPLBACKEND", "module://matplotlib_inline.backend_inline")
    cmd = _script(tmp_path, """
        import os
        print("Exporting with MPLBACKEND =", os.environ.get("MPLBACKEND"), flush=True)
    """)
    msgs = []
    ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=msgs.append)
    assert any("MPLBACKEND = None" in m for m in msgs)
    assert os.environ["MPLBACKEND"].startswith("module://")    # the notebook keeps its own


def test_ram_cap_defaults_to_three_quarters_of_the_installed_ram(cfg, monkeypatch):
    """A fixed 4 GB default starved ilastik on a workstation; it is sized from the machine
    unless a number is given."""
    from czipipe import config

    monkeypatch.setattr(config, "physical_ram_mb", lambda: 65536)
    assert cfg.ilastik.ram_mb is None
    assert cfg.ilastik_ram_mb == 49152
    monkeypatch.setattr(config, "physical_ram_mb", lambda: 16384)
    assert cfg.ilastik_ram_mb == 12288
    monkeypatch.setattr(config, "physical_ram_mb", lambda: 2400)
    assert cfg.ilastik_ram_mb == 2048                    # floor on a small machine
    monkeypatch.setattr(config, "physical_ram_mb", lambda: None)
    assert cfg.ilastik_ram_mb == 4096                    # cannot tell: the old default
    cfg.ilastik.ram_mb = 8192
    assert cfg.ilastik_ram_mb == 8192                    # an explicit setting always wins


def test_installed_ram_is_readable_here():
    from czipipe import config

    assert config.physical_ram_mb() > 512


def test_ram_cap_reaches_ilastik_and_the_log_says_where_it_came_from(cfg, tmp_path,
                                                                     monkeypatch):
    from czipipe import config

    monkeypatch.setattr(config, "physical_ram_mb", lambda: 65536)
    cmd = _script(tmp_path, """
        import os
        print("Exporting with", os.environ["LAZYFLOW_TOTAL_RAM_MB"], flush=True)
    """)
    msgs = []
    assert ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=msgs.append) is True
    assert any("Exporting with 49152" in m for m in msgs)
    assert any("75% of the 64 GB installed" in m for m in msgs)

    cfg.ilastik.ram_mb = 8192
    msgs.clear()
    ilastik.run_command(cfg, cmd, tmp_path / "ilastik.log", log=msgs.append)
    assert any("Exporting with 8192" in m for m in msgs)
    assert not any("installed" in m for m in msgs)


def test_ram_cap_must_be_positive(cfg):
    cfg.ilastik.ram_mb = 0
    with pytest.raises(ValueError, match="ram_mb"):
        cfg.validate()


# ---------------------------------------------------------------- locating ilastik
def test_windows_install_is_found_and_newest_wins(cfg, tmp_path, monkeypatch):
    for v in ("1.4.1", "1.4.9", "1.4.10"):
        d = tmp_path / f"ilastik-{v}"
        d.mkdir()
        (d / "ilastik.exe").write_bytes(b"")
    for var in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        monkeypatch.setitem(os.environ, var, str(tmp_path))
    monkeypatch.setattr(ilastik.shutil, "which", lambda name: None)
    cfg.ilastik.exe = None
    assert ilastik.find_exe(cfg, windows=True) == str(tmp_path / "ilastik-1.4.10" /
                                                      "ilastik.exe")


def test_missing_ilastik_names_the_setting(cfg, tmp_path, monkeypatch):
    for var in ("ProgramW6432", "ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA"):
        monkeypatch.setitem(os.environ, var, str(tmp_path))
    monkeypatch.setattr(ilastik.shutil, "which", lambda name: None)
    cfg.ilastik.exe = None
    with pytest.raises(FileNotFoundError, match="cfg.ilastik.exe"):
        ilastik.find_exe(cfg, windows=True)
