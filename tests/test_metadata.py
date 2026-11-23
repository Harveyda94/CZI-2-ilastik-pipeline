"""CZI metadata: files whose XML cannot be read, and the manual pixel-size override.

The unreadable case was found on a real 136 GB light-sheet file, where pylibCZIrw raised
"Invalid MetadataSegment-magic" for the XML although every pixel read back fine.
"""

from __future__ import annotations

import json
import sys
import types

import numpy as np
import pytest
import tifffile

from czipipe import writers
from czipipe.compute import Backend
from czipipe.geometry import Tiler
from czipipe.pipeline import _null_progress
from czipipe.plan import ScenePlan
from czipipe.source import Source

from conftest import FakeSource

XML = """<ImageDocument><Metadata>
<Scaling><Items>
  <Distance Id="X"><Value>6.5e-07</Value></Distance>
  <Distance Id="Y"><Value>6.5e-07</Value></Distance>
  <Distance Id="Z"><Value>2e-06</Value></Distance>
</Items></Scaling>
<Information><Image><Dimensions><Channels>
  <Channel Name="dsRed"/><Channel Name="FITC"/>
</Channels></Dimensions></Image></Information>
</Metadata></ImageDocument>"""

BROKEN = RuntimeError("Illegal data detected at offset 4928 -> Invalid MetadataSegment-magic")


@pytest.fixture
def czi(monkeypatch):
    """Install a stand-in pylibCZIrw whose document returns (or raises) `raw_metadata`."""
    def install(raw_metadata):
        class Doc:
            total_bounding_box = {"T": (0, 1), "Z": (0, 4), "C": (0, 2),
                                  "X": (0, 64), "Y": (0, 48)}
            total_bounding_rectangle = (0, 0, 64, 48)
            scenes_bounding_rectangle = {}
            pixel_types = {0: "Gray16", 1: "Gray16"}

            @property
            def raw_metadata(self):
                if isinstance(raw_metadata, Exception):
                    raise raw_metadata
                return raw_metadata

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        czi_mod = types.ModuleType("pylibCZIrw.czi")
        czi_mod.open_czi = lambda path: Doc()
        pkg = types.ModuleType("pylibCZIrw")
        pkg.czi = czi_mod
        monkeypatch.setitem(sys.modules, "pylibCZIrw", pkg)
        monkeypatch.setitem(sys.modules, "pylibCZIrw.czi", czi_mod)
    return install


# ---------------------------------------------------------------- reading the CZI
def test_unreadable_metadata_still_opens_with_a_warning(cfg, czi):
    czi(BROKEN)
    logs = []
    src = Source(cfg, log=logs.append)
    assert src.raw_xml is None
    assert src.scaling == {}
    assert src.channel_names == ["C0", "C1"]
    assert src.sel_z == [0, 1, 2, 3]
    assert any("[warn] could not read raw CZI XML metadata" in m
               and "MetadataSegment-magic" in m for m in logs)


def test_manual_scaling_supplies_what_unreadable_metadata_lost(cfg, czi):
    czi(BROKEN)
    cfg.manual_scaling = {"X": 0.9128, "Y": 0.9128, "Z": 4.9357}
    src = Source(cfg, log=lambda *a: None)
    assert src.scaling == {"X": 0.9128, "Y": 0.9128, "Z": 4.9357}


def test_manual_scaling_overrides_only_the_axes_it_names(cfg, czi):
    czi(XML)
    cfg.manual_scaling = {"Z": 4.0}
    src = Source(cfg, log=lambda *a: None)
    assert src.scaling["X"] == pytest.approx(0.65)
    assert src.scaling["Y"] == pytest.approx(0.65)
    assert src.scaling["Z"] == 4.0
    assert src.channel_names == ["dsRed", "FITC"]


def test_readable_metadata_is_used_without_an_override(cfg, czi):
    czi(XML)
    logs = []
    src = Source(cfg, log=logs.append)
    assert src.scaling["Z"] == pytest.approx(2.0)
    assert not any("[warn]" in m or "[info]" in m for m in logs)


# ---------------------------------------------------------------- config checks
@pytest.mark.parametrize("value, message", [
    ({"W": 1.0}, 'keys must be "X", "Y" and/or "Z"'),
    ({"Z": 0}, "must be positive"),
    ({"X": -0.5}, "must be positive"),
])
def test_bad_manual_scaling_is_refused(cfg, value, message):
    cfg.manual_scaling = value
    with pytest.raises(ValueError, match=message):
        cfg.validate()


@pytest.mark.parametrize("chunk_z", [0, -5])
def test_non_positive_chunk_z_is_refused(cfg, chunk_z):
    cfg.ilastik.chunk_z = chunk_z
    with pytest.raises(ValueError, match="chunk_z must be a positive int"):
        cfg.validate()


# ---------------------------------------------------------------- writing without XML
RECT = (0, 0, 70, 50)


def _no_xml_source(cfg):
    data = np.random.default_rng(5).integers(0, 65535, (1, 2, 1, RECT[3], RECT[2]),
                                             dtype=np.uint16)
    src = FakeSource(data, RECT, [0], [0, 1], [0], cfg)
    src.raw_xml, src.scaling = None, {}
    return src, Tiler(cfg.tile, [0], [0, 1], [0], 1), ScenePlan(None, RECT, {0: (0.0, 65535.0)})


def test_tiff_without_czi_xml_writes_no_xml_tag_or_sidecar(cfg):
    cfg.write_sidecars, cfg.embed_czi_xml = True, True
    src, tiler, plan = _no_xml_source(cfg)
    out = writers.write_tiff(src, cfg, Backend(False), tiler, plan,
                             log=lambda *a: None, progress=_null_progress)
    with tifffile.TiffFile(out) as tf:
        assert 65000 not in tf.pages[0].tags
        back = np.stack([p.asarray() for p in tf.pages])
    assert np.array_equal(back, src.data.reshape(back.shape))
    assert not list(cfg.out_dir.glob("*.czi_metadata.xml"))
    (prov,) = cfg.out_dir.glob("*.conversion.json")
    assert json.loads(prov.read_text())["source"]["scaling_um"] == {}


def test_hdf5_without_czi_xml_is_written_and_reads_back(cfg):
    try:
        import h5py
    except ImportError:
        pytest.skip("h5py not installed")
    cfg.output_format = "hdf5"
    src, tiler, plan = _no_xml_source(cfg)
    out = writers.write_hdf5(src, cfg, Backend(False), tiler, plan,
                             log=lambda *a: None, progress=_null_progress)
    with h5py.File(out, "r") as f:
        assert np.array_equal(f[cfg.hdf5_dataset][()], src.data)
        assert "element_size_um" not in f[cfg.hdf5_dataset].attrs
        assert bool(f["progress"].attrs["complete"])
