"""The real readers on a real file: a mosaic CZI written with pylibCZIrw and read back
through Source and aicspylibczi, then registered, converted and verified.

Every other test here runs on fakes, so that the suite needs none of the microscopy
stack. That leaves the calls into the real libraries untested: mosaic tile indices and
bounding boxes, a tile read at a Z plane, the Z extent of the file, a reader handle per
thread. These tests cover them, and are skipped where the libraries are not installed.
The image content is still synthetic, cut from a known volume at known offsets.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from czipipe import Config, Pipeline

from test_stitch3d import NZ, PERT3, XY_ONLY, covered, make_mosaic, truth_plane

try:
    from pylibCZIrw import czi as pyczi
    import aicspylibczi  # noqa: F401
except ImportError:             # tools/minipytest.py has no importorskip, so skip per test
    pyczi = None


@pytest.fixture
def write_czi(tmp_path):
    if pyczi is None:
        pytest.skip("pylibCZIrw and aicspylibczi not installed")

    def write(tiles, meta):
        path = tmp_path / "mosaic.czi"
        with pyczi.create_czi(str(path), exist_ok=True) as doc:
            for z in range(NZ):
                for m in sorted(tiles):          # the same order at every plane, which
                    x, y, _, _ = meta[m]         # is what gives a tile the same M index
                    doc.write(np.ascontiguousarray(tiles[m][z])[..., None], location=(x, y),
                              plane={"C": 0, "Z": z, "T": 0}, scene=0)
        return path
    return write


def convert(tmp_path, czi, mode):
    cfg = Config(czi_path=czi, out_dir=tmp_path / "out", stitch_mode="refine",
                 tile=(256, 256), use_gpu=False, compression=None, write_sidecars=False)
    cfg.refine.mode, cfg.refine.slab = mode, NZ
    msgs = []
    with Pipeline(cfg, log=lambda *a: msgs.append(" ".join(str(v) for v in a))) as p:
        p.inspect()
        plan = p.plan()[0]
        path = p.convert()[0].path
        assert p.verify()
        with tifffile.TiffFile(path) as tf:
            stack = tf.asarray().reshape(NZ, plan.rect[3], plan.rect[2])
        masks = [covered(plan.compositor, stack[z].shape, z) for z in range(NZ)]
    return plan, stack, masks, "\n".join(msgs)


def test_3d_registration_on_a_real_czi(write_czi, tmp_path):
    gt, tiles, meta = make_mosaic()
    plan, stack, masks, report = convert(tmp_path, write_czi(tiles, meta), "3d")
    assert plan.corrections == PERT3
    assert "20/20 pairs accepted" in report and "Z corrections -2 to +2 planes" in report
    assert "lack some tiles (Z 0-1, 10-11)" in report          # the file's own Z extent
    for z in range(NZ):
        assert np.array_equal(stack[z][masks[z]], truth_plane(gt, plan.rect, z)[masks[z]]), z
        assert not stack[z][~masks[z]].any()
    assert masks[0].mean() < masks[5].mean()                    # tiles left out at the end


def test_2d_registration_on_a_real_czi(write_czi, tmp_path):
    gt, tiles, meta = make_mosaic(XY_ONLY)
    plan, stack, masks, report = convert(tmp_path, write_czi(tiles, meta), "2d")
    assert plan.corrections == {m: v[:2] for m, v in XY_ONLY.items()}
    for z in range(NZ):
        assert np.array_equal(stack[z][masks[z]], truth_plane(gt, plan.rect, z)[masks[z]]), z
