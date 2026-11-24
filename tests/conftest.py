"""Fakes standing in for pylibCZIrw, aicspylibczi and h5py.

These let the whole pipeline be exercised against data whose correct output is known
exactly, on a machine with none of the microscopy stack installed.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

from czipipe.config import Config


# ---------------------------------------------------------------- image data
def smooth_image(shape, seed=0, lo=1000, hi=31000):
    """Band-limited noise: has structure for phase correlation to lock onto."""
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, shape)
    k = np.hypot(*np.meshgrid(np.fft.fftfreq(shape[0]), np.fft.rfftfreq(shape[1]),
                              indexing="ij"))
    g = np.fft.irfft2(np.fft.rfft2(b) * np.exp(-k * 30), s=shape)
    g = (g - g.min()) / (g.max() - g.min())
    return (g * (hi - lo) + lo).astype(np.uint16)


# ---------------------------------------------------------------- fake CZI
class FakeReader:
    """Stands in for a pylibCZIrw reader over an in-memory (T,Z,C,Y,X) array.

    `background_pixel` is rejected on purpose. Real pylibCZIrw builds do
    `Color(*background_pixel)` unconditionally, so passing a scalar raises TypeError even
    for Gray pixel types. Modelling that here means any reintroduction of the argument
    fails the suite instead of only failing on a real microscope file.
    """

    def __init__(self, data, rect, sel_t, sel_z, sel_c):
        self.data, self.rect = data, rect
        self.sel_t, self.sel_z, self.sel_c = sel_t, sel_z, sel_c

    def read(self, roi=None, plane=None, zoom=1.0, **kw):
        if "background_pixel" in kw:
            raise TypeError(
                "pylibCZIrw.czi.Color() argument after * must be an iterable, not int")
        x, y, w, h = roi
        f = self.data[self.sel_t.index(plane["T"]),
                      self.sel_z.index(plane["Z"]),
                      self.sel_c.index(plane["C"])]
        out = np.zeros((h, w), self.data.dtype)
        sy, sx = y - self.rect[1], x - self.rect[0]
        ys, xs = max(0, sy), max(0, sx)
        ye, xe = min(f.shape[0], sy + h), min(f.shape[1], sx + w)
        if ye > ys and xe > xs:
            out[ys - sy:ye - sy, xs - sx:xe - sx] = f[ys:ye, xs:xe]
        if zoom != 1.0:
            step = max(1, int(round(1 / zoom)))
            out = out[::step, ::step]
        return out[..., None]


class FakeAics:
    """Stands in for aicspylibczi, serving whole mosaic tiles by index.

    A tile is a (Y, X) array, the same at every plane, or a (Z, Y, X) stack indexed by
    the Z asked for.
    """

    def __init__(self, tiles, n_c=1, n_z=1, n_t=1, has_scene=False):
        self.tiles = tiles
        self.dims = {"M": (0, len(tiles)), "C": (0, n_c), "Z": (0, n_z), "T": (0, n_t)}
        if has_scene:
            self.dims["S"] = (0, 1)

    def is_mosaic(self):
        return True

    def get_dims_shape(self):
        return [self.dims]

    def read_image(self, **kw):
        tile = self.tiles[kw["M"]]
        if tile.ndim == 3:
            tile = tile[kw["Z"]]
        return tile[None, None], None


class FakeSource:
    """A Source-compatible object backed by arrays, with no CZI on disk."""

    def __init__(self, data, rect, sel_t, sel_z, sel_c, cfg,
                 dtype=np.uint16, samples=1, tiles=None, scene_rects=None):
        self.cfg = cfg
        self.data, self.rect = data, rect
        self.sel_t, self.sel_z, self.sel_c = list(sel_t), list(sel_z), list(sel_c)
        self.c0 = 0
        self.dtype, self.samples = dtype, samples
        self.out_dtype = np.uint8 if cfg.output_bit_depth == 8 else dtype
        self.h5_axes = "tzcyxs" if samples == 3 else "tzcyx"
        self.scene_arg_ok = False
        self.total_rect = rect
        self.scene_rects = scene_rects or {}
        self.total_bbox = {"T": (0, len(self.sel_t)), "Z": (0, len(self.sel_z)),
                           "C": (0, len(self.sel_c))}
        self.pixel_type = "Gray16" if dtype == np.uint16 else "Gray8"
        self.raw_xml = "<ImageDocument><Metadata/></ImageDocument>"
        self.scaling = {"X": 0.23, "Y": 0.23, "Z": 1.0}
        self.channel_info = [{"name": f"C{i}"} for i in range(max(self.sel_c) + 1)]
        self._tiles = tiles
        self._lock = threading.Lock()

    @property
    def jobs(self):
        return [(None, self.rect)]

    @property
    def channel_names(self):
        return [self.channel_info[c]["name"] for c in self.sel_c]

    def reader(self):
        return FakeReader(self.data, self.rect, self.sel_t, self.sel_z, self.sel_c)

    def aics(self):
        return FakeAics(self._tiles, n_c=max(self.sel_c) + 1,
                        n_z=max(self.sel_z) + 1, n_t=max(self.sel_t) + 1)

    def read_kwargs(self, scene_id):
        return {}

    # Deliberately the real implementation, so the narrowed exception handling is
    # exercised rather than stubbed out.
    from czipipe.source import Source as _RealSource
    probe_scene_arg = _RealSource.probe_scene_arg

    def close(self):
        pass

    def describe(self):
        return "fake source"


# ---------------------------------------------------------------- h5py shim
class _Scalar:
    def __init__(self, v):
        self.v = v

    def __getitem__(self, k):
        return self.v


class _DS:
    def __init__(self, shape, dtype, chunks):
        self._a = np.zeros(shape, dtype)
        self.shape, self.dtype = tuple(shape), np.dtype(dtype)
        self.chunks, self.attrs, self.ndim = chunks, {}, len(shape)

    def __setitem__(self, k, v):
        self._a[k] = v

    def __getitem__(self, k):
        return self._a[k]


class _Group(dict):
    def __init__(self):
        super().__init__()
        self.attrs = {}

    def create_dataset(self, name, shape=None, dtype=None, chunks=None, data=None, **kw):
        if data is not None and shape == ():
            self[name] = _Scalar(data)
            return self[name]
        d = _DS(shape, dtype, chunks)
        self[name] = d
        return d

    def create_group(self, name):
        g = _Group()
        self[name] = g
        return g

    def _key(self, key):
        """h5py treats '/data' and 'data' as the same path; so must the shim."""
        for k in (key, key.lstrip("/"), "/" + key.lstrip("/")):
            if dict.__contains__(self, k):
                return k
        raise KeyError(key)

    def __getitem__(self, key):
        return dict.__getitem__(self, self._key(key))

    def __contains__(self, key):
        try:
            self._key(key)
            return True
        except KeyError:
            return False


class FakeH5:
    """Persists across 'process restarts' and models flush() semantics.

    Only data that has been flushed survives a simulated crash, which is what makes the
    resume ordering guarantee testable.
    """

    STORE = {}

    class File(_Group):
        def __init__(self, path, mode="r"):
            super().__init__()
            self.path = str(path)
            store = FakeH5.STORE
            if mode == "w" or self.path not in store:
                store[self.path] = None
            else:
                snap = store[self.path]
                if snap is None:
                    raise OSError("unable to open (file signature not found)")
                _restore(self, snap)
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            # Like real HDF5, opening an existing file read-only leaves it untouched on
            # disk; restart logic compares file size and mtime, so this must hold.
            if mode != "r" or not Path(self.path).exists():
                Path(self.path).write_bytes(b"fake-hdf5")

        def flush(self):
            FakeH5.STORE[self.path] = _snapshot(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    @staticmethod
    def string_dtype(encoding=None):
        return object

    @classmethod
    def reset(cls):
        cls.STORE.clear()


def _snapshot(g):
    out = {"attrs": dict(g.attrs), "items": {}}
    for k, v in g.items():
        if isinstance(v, _DS):
            out["items"][k] = ("ds", v._a.copy(), v.shape, v.dtype, v.chunks, dict(v.attrs))
        elif isinstance(v, _Group):
            out["items"][k] = ("g", _snapshot(v))
        elif isinstance(v, _Scalar):
            out["items"][k] = ("scal", v.v)
    return out


def _restore(g, snap):
    g.attrs.update(snap["attrs"])
    for k, v in snap["items"].items():
        if v[0] == "ds":
            d = _DS(v[2], v[3], v[4])
            d._a = v[1].copy()
            d.attrs.update(v[5])
            g[k] = d
        elif v[0] == "g":
            sub = _Group()
            _restore(sub, v[1])
            g[k] = sub
        else:
            g[k] = _Scalar(v[1])


def snapshot_of(path):
    return FakeH5.STORE[str(path)]


# ---------------------------------------------------------------- ilastik's export path
def ilastik_export_target(cmd):
    """(group, dataset name) that ilastik on Windows writes its HDF5 export to.

    Follows lazyflow 1.4.2. OpExportSlot joins the file name and --output_internal_path
    with a slash, PathComponents keeps what comes after the extension, and
    OpH5N5WriterBigDataset divides that with os.path.split: the group is created if it
    is missing, and the dataset is created in it under the name. On Windows that is
    ntpath.split, for which (since Python 3.11) a path opening with two slashes names a
    network share, and one with nothing after the share is all drive and no file name.
    The rule is written out here so the answer does not depend on the Python running
    the tests.
    """
    arg = next((a.split("=", 1)[1] for a in cmd
                if a.startswith("--output_internal_path=")), "exported_data")
    internal = "/" + arg
    if internal.startswith("//") and "/" not in internal[2:]:
        return internal, ""
    group, _, name = internal.rpartition("/")
    return group or "/", name


# ---------------------------------------------------------------- fixtures
@pytest.fixture
def cfg(tmp_path):
    czi = tmp_path / "fake.czi"
    czi.write_bytes(b"not-really-a-czi")
    return Config(czi_path=czi, out_dir=tmp_path / "out", tile=(64, 64),
                  n_read_threads=4, prefetch=8, n_write_workers=2,
                  use_gpu=False, verify_samples=40, write_sidecars=False,
                  overwrite=True, compression_level=3)


@pytest.fixture(autouse=True)
def _clean_h5():
    FakeH5.reset()
    yield
    FakeH5.reset()
