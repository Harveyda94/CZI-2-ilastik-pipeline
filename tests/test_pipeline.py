"""Geometry, ordered streaming, TIFF output and verification."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tifffile

from czipipe.compute import Backend
from czipipe.geometry import Tiler, stream
from czipipe.plan import ScenePlan
from czipipe.pipeline import _null_progress
from czipipe.validate import verify
from czipipe.worker import make_worker
from czipipe.writers import write_tiff

from conftest import FakeSource

RECT = (-13, 7, 300, 220)          # deliberately non-zero, non-tile-aligned origin
SEL_T, SEL_Z, SEL_C = [0, 1], [0, 1, 2], [0, 1]


def build(cfg, bit_depth=16, samples=1):
    cfg.output_bit_depth = bit_depth
    data = np.random.default_rng(4).integers(
        0, 65535, (len(SEL_T), len(SEL_Z), len(SEL_C), RECT[3], RECT[2]), dtype=np.uint16)
    src = FakeSource(data, RECT, SEL_T, SEL_Z, SEL_C, cfg)
    tiler = Tiler(cfg.tile, SEL_T, SEL_Z, SEL_C, samples)
    plan = ScenePlan(None, RECT, {c: (0.0, 65535.0) for c in SEL_C})
    return src, tiler, plan, data


def test_task_count_matches_n_tiles(cfg):
    src, tiler, plan, _ = build(cfg)
    assert len(list(tiler.tasks(RECT, 0))) == tiler.n_tiles(RECT, 0)


def test_level_shape_halves_each_level(cfg):
    src, tiler, plan, _ = build(cfg)
    assert tiler.level_shape(RECT, 0)[3:5] == (220, 300)
    assert tiler.level_shape(RECT, 1)[3:5] == (110, 150)
    assert tiler.level_shape(RECT, 2)[3:5] == (55, 75)


def test_fit_crops_and_pads(cfg):
    tiler = Tiler((64, 64), SEL_T, SEL_Z, SEL_C)
    assert tiler.fit(np.ones((30, 40), np.uint16)).shape == (64, 64)
    assert tiler.fit(np.ones((90, 90), np.uint16)).shape == (64, 64)
    padded = tiler.fit(np.ones((30, 40), np.uint16))
    assert padded[:30, :40].all() and not padded[30:, :].any()


def test_stream_preserves_order_under_concurrency():
    """The writer must never see an out-of-order tile, however unevenly work completes."""
    import random
    import time as _t

    def work(i):
        _t.sleep(random.random() * 0.002)
        return i

    with ThreadPoolExecutor(8) as ex:
        got = list(stream(range(200), work, ex, prefetch=16))
    assert got == list(range(200))


def test_tiff_roundtrip_is_bit_exact(cfg):
    src, tiler, plan, data = build(cfg)
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(path) as tf:
        back = np.stack([p.asarray() for p in tf.pages]).reshape(data.shape)
    assert np.array_equal(back, data)
    assert tf.is_bigtiff


def test_verify_passes_on_good_tiff(cfg):
    src, tiler, plan, _ = build(cfg)
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    assert verify(src, cfg, Backend(False), tiler, plan, path, n=40,
                  log=lambda *a: None) is True


def test_verify_detects_corruption(cfg):
    """A verification that cannot fail is worthless; prove it fails on damaged output."""
    src, tiler, plan, _ = build(cfg)
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    raw = bytearray(path.read_bytes())
    raw[len(raw) // 2:len(raw) // 2 + 400] = b"\x00" * 400     # damage tile data
    path.write_bytes(bytes(raw))
    msgs = []
    try:
        ok = verify(src, cfg, Backend(False), tiler, plan, path, n=40, log=msgs.append)
    except Exception:
        ok = False                                             # a decode error also counts
    assert ok is False


def test_eight_bit_conversion_uses_percentile_range(cfg):
    src, tiler, plan, data = build(cfg, bit_depth=8)
    plan.ranges = {c: (0.0, 65535.0) for c in SEL_C}
    work = make_worker(src, cfg, Backend(False), tiler, plan, 0)
    tile = work((0, 0, 0, RECT[0], RECT[1], 64, 64))
    assert tile.dtype == np.uint8
    expect = (data[0, 0, 0, :64, :64].astype(np.float32) * (255.0 / 65535.0)) \
        .clip(0, 255).astype(np.uint8)
    assert np.array_equal(tile, expect)


def test_pyramid_levels_written(cfg):
    cfg.write_pyramid, cfg.pyramid_levels = True, 2
    src, tiler, plan, _ = build(cfg)
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(path) as tf:
        assert [p.shape for p in tf.pages[0].pages] == [(110, 150), (55, 75)]


def test_czi_xml_embedded_in_private_tag(cfg):
    src, tiler, plan, _ = build(cfg)
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(path) as tf:
        assert tf.pages[0].tags[65000].value.startswith("<ImageDocument>")
        assert 'PhysicalSizeX="0.23"' in tf.ome_metadata


def _write(cfg, src, tiler, plan, log=None):
    return write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=log or (lambda *a: None), progress=_null_progress)


def test_complete_tiff_is_reused_when_settings_match(cfg):
    """After a kernel restart, convert() must re-attach a finished TIFF, not refuse it;
    refusing left no route to classify() short of rewriting the whole conversion."""
    cfg.overwrite = False
    src, tiler, plan, _ = build(cfg)
    path = _write(cfg, src, tiler, plan)
    before = path.stat().st_mtime_ns
    msgs = []
    assert _write(cfg, src, tiler, plan, log=msgs.append) == path
    assert path.stat().st_mtime_ns == before
    assert any("skipping" in m for m in msgs)


def test_changed_settings_refuse_to_reuse_tiff(cfg):
    cfg.overwrite = False
    src, tiler, plan, _ = build(cfg)
    _write(cfg, src, tiler, plan)
    cfg.compression_level = 7
    with pytest.raises(FileExistsError, match="level: was 3, now 7"):
        _write(cfg, src, tiler, plan)


def test_tiff_without_fingerprint_is_not_reused(cfg):
    """A file this version did not write (or an older one did) proves nothing."""
    cfg.overwrite = False
    src, tiler, plan, _ = build(cfg)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(cfg.out_dir / "fake.ome.tif", np.zeros((8, 8), np.uint16))
    with pytest.raises(FileExistsError, match="cannot be reused"):
        _write(cfg, src, tiler, plan)


def test_interrupted_tiff_leaves_nothing_under_any_name(cfg, monkeypatch):
    from czipipe import writers
    src, tiler, plan, _ = build(cfg)
    real, n = writers.make_worker, [0]

    def dying(*a, **k):
        w = real(*a, **k)

        def f(task):
            n[0] += 1
            if n[0] > 10:
                raise KeyboardInterrupt
            return w(task)
        return f

    monkeypatch.setattr(writers, "make_worker", dying)
    with pytest.raises(KeyboardInterrupt):
        _write(cfg, src, tiler, plan)
    assert list(cfg.out_dir.iterdir()) == []


def test_interrupted_rewrite_keeps_the_previous_complete_file(cfg, monkeypatch):
    """overwrite=True must not destroy a good file before its replacement exists."""
    from czipipe import writers
    src, tiler, plan, data = build(cfg)
    path = _write(cfg, src, tiler, plan)
    monkeypatch.setattr(writers, "make_worker",
                        lambda *a, **k: (lambda task: (_ for _ in ()).throw(RuntimeError())))
    with pytest.raises(RuntimeError):
        _write(cfg, src, tiler, plan)
    with tifffile.TiffFile(path) as tf:
        back = np.stack([p.asarray() for p in tf.pages]).reshape(data.shape)
    assert np.array_equal(back, data)
    assert not path.with_name(path.name + ".partial").exists()


# ---------------------------------------------------------------- lossless codecs
def test_jpeg2000_is_written_losslessly(cfg):
    """imagecodecs reads `level` as a PSNR target for JPEG 2000; level=5 meant 5 dB."""
    try:
        import imagecodecs  # noqa: F401
    except ImportError:
        pytest.skip("imagecodecs not installed")
    cfg.compression, cfg.compression_level = "jpeg2000", 5
    src, tiler, plan, data = build(cfg)
    path = _write(cfg, src, tiler, plan)
    with tifffile.TiffFile(path) as tf:
        back = np.stack([p.asarray() for p in tf.pages]).reshape(data.shape)
    assert np.array_equal(back, data)


@pytest.mark.parametrize("codec", ["jpeg", "webp", "lerc"])
def test_lossy_or_unknown_codecs_are_rejected(cfg, codec):
    from czipipe.writers import check_tiff_codec
    with pytest.raises(ValueError, match="not supported"):
        check_tiff_codec(codec, 5)


@pytest.mark.parametrize("codec", [None, "zlib"])
def test_default_codecs_pass_the_probe_without_imagecodecs(cfg, codec):
    from czipipe.writers import check_tiff_codec
    check_tiff_codec(codec, 5)


def test_codec_needing_imagecodecs_fails_at_configuration(cfg):
    try:
        import imagecodecs  # noqa: F401
        pytest.skip("imagecodecs is installed")
    except ImportError:
        pass
    from czipipe.writers import check_tiff_codec
    with pytest.raises(ImportError, match="pip install imagecodecs"):
        check_tiff_codec("lzw", 5)


def test_validate_runs_the_codec_probe(cfg):
    cfg.compression = "jpeg"
    with pytest.raises(ValueError, match="not supported"):
        cfg.validate()


# ---------------------------------------------------------------- flat-field, single field
def test_single_field_flatfield_keeps_the_output_dtype(cfg):
    """FlatField.apply works in float32. The single-field path used to hand that float
    straight to the writer: TIFF refused it, HDF5 silently truncated it."""
    from czipipe import flatfield as ff
    src, tiler, plan, data = build(cfg, bit_depth=16)
    yy, xx = np.mgrid[0:RECT[3], 0:RECT[2]]
    profile = 0.7 + 0.6 * (xx / RECT[2]) * (yy / RECT[3])
    plan.flatfields = {c: ff.build(profile, dark=100.0) for c in SEL_C}
    work = make_worker(src, cfg, Backend(False), tiler, plan, 0)
    tile = work((0, 0, 0, RECT[0], RECT[1], 64, 64))
    assert tile.dtype == np.uint16
    f = plan.flatfields[0]
    expect = (data[0, 0, 0, :64, :64].astype(np.float32) - np.float32(100.0)) \
        * f.gain[:64, :64]
    assert np.array_equal(tile, np.clip(np.rint(expect), 0, 65535).astype(np.uint16))


# ---------------------------------------------------------------- field regressions
# Each of these reproduces a failure hit on a real 74 GB Zeiss Z1 file.

def test_czi_xml_with_micro_sign_is_writable(cfg):
    """tifffile rejects non-ASCII `str` for TIFF type 's'; Zeiss writes U+00B5 in Scaling.

    The XML has to be handed over as bytes. Reading the tag back still gives a str.
    """
    src, tiler, plan, _ = build(cfg)
    src.raw_xml = ('<ImageDocument><Metadata><Scaling><Items><Distance Id="X">'
                   '<Value>2.3e-7</Value><DefaultUnitFormat>\u00b5m</DefaultUnitFormat>'
                   "</Distance></Items></Scaling></Metadata></ImageDocument>")
    path = write_tiff(src, cfg, Backend(False), tiler, plan,
                      log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(path) as tf:
        got = tf.pages[0].tags[65000].value
    got = got if isinstance(got, str) else got.decode("utf-8")
    assert got == src.raw_xml
    assert "\u00b5m" in got


def test_extra_tags_hands_tifffile_bytes(cfg):
    from czipipe.writers import extra_tags
    src, tiler, plan, _ = build(cfg)
    src.raw_xml = "<X>\u00b5m</X>"
    (code, dtype, count, value, once), = extra_tags(src, cfg)
    assert code == 65000 and dtype == "s"
    assert isinstance(value, bytes)


def test_extra_tags_empty_when_disabled(cfg):
    from czipipe.writers import extra_tags
    cfg.embed_czi_xml = False
    src, tiler, plan, _ = build(cfg)
    assert extra_tags(src, cfg) == ()


def test_no_read_passes_background_pixel(cfg):
    """FakeReader raises TypeError on background_pixel, mirroring the real library."""
    src, tiler, plan, _ = build(cfg)
    work = make_worker(src, cfg, Backend(False), tiler, plan, 0)
    work((0, 0, 0, RECT[0], RECT[1], 64, 64))          # would raise if reintroduced


def test_sixteen_bit_passthrough_needs_no_ranges(cfg):
    """A 16-bit run must not depend on the intensity-statistics step having run."""
    src, tiler, plan, data = build(cfg, bit_depth=16)
    plan.ranges = {}                                    # as if planning were skipped
    work = make_worker(src, cfg, Backend(False), tiler, plan, 0)
    tile = work((0, 0, 0, RECT[0], RECT[1], 64, 64))
    assert np.array_equal(tile, data[0, 0, 0, :64, :64])


def test_eight_bit_without_ranges_says_what_to_do(cfg):
    src, tiler, plan, _ = build(cfg, bit_depth=8)
    plan.ranges = {}
    work = make_worker(src, cfg, Backend(False), tiler, plan, 0)
    with pytest.raises(RuntimeError, match="Pipeline.plan"):
        work((0, 0, 0, RECT[0], RECT[1], 64, 64))


def test_probe_reraises_errors_that_are_not_about_scene(cfg):
    """A blanket except here once blamed `scene=` for a background_pixel TypeError,
    silently disabling scene-restricted reads for a whole run."""
    src, tiler, plan, _ = build(cfg)

    class Boom:
        def read(self, **kw):
            raise TypeError("Color() argument after * must be an iterable, not int")

    src.reader = lambda: Boom()
    src.probe_scene_arg = FakeSource.probe_scene_arg.__get__(src)
    with pytest.raises(TypeError, match="Color"):
        src.probe_scene_arg([(0, RECT)], {"T": 0, "Z": 0, "C": 0})


def test_probe_disables_scene_arg_when_it_is_the_cause(cfg):
    src, tiler, plan, _ = build(cfg)

    class NoScene:
        def read(self, **kw):
            raise TypeError("read() got an unexpected keyword argument 'scene'")

    src.reader = lambda: NoScene()
    src.probe_scene_arg = FakeSource.probe_scene_arg.__get__(src)
    assert src.probe_scene_arg([(0, RECT)], {"T": 0, "Z": 0, "C": 0}) is False
    assert src.scene_arg_ok is False
