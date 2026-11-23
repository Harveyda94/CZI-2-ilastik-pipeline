"""Generate the thin driver notebook."""

import json
import pathlib

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


md(r"""
# CZI conversion

Driver for the `czipipe` package. The logic lives in the `.py` files under `czipipe/`;
this notebook only configures a run and calls it, so it stays readable and the code stays
testable.

Keep this layout — the notebook imports the package from alongside itself:

```
your-folder/
├── convert_czi.ipynb      <- open this
├── czipipe/               <- the supporting modules
├── tests/
└── tools/
```

| | |
|---|---|
| RAM | Nothing larger than one tile is materialised. Peak &asymp; `prefetch x tile` (plus the mosaic cache when stitching), independent of file size. |
| Stitching | `metadata` (stage positions, free) / `refine` (registered here) / `bigstitcher` (transforms from a BDV XML). |
| Shading | Optional flat-field correction, applied per field of view before blending. |
| Output | Tiled BigTIFF, or chunked HDF5 that resumes after a crash. |
| Segmentation | Optional headless ilastik, up to 3 projects, one BigTIFF each. |

```bash
pip install pylibCZIrw tifffile imagecodecs numpy tqdm h5py matplotlib scipy psutil
pip install aicspylibczi      # stitch_mode "refine" / "bigstitcher", and flat-field on mosaics
pip install basicpy           # only for flatfield.mode = "basic"
pip install cupy-cuda12x      # optional, NVIDIA only; rarely the bottleneck (see Benchmark)
```

Use Python 3.12 (3.9-3.13 work). Neither CZI package ships wheels for 3.14 yet, and pip
will fall back to compiling libCZI from source. pylibCZIrw ships ready-built for
Apple-silicon Macs, 64-bit Windows and Linux; on an Intel Mac it compiles from source.

Run the tests after editing anything in `czipipe/`: `python -m pytest` from the package
root, or `python tools/minipytest.py` if pytest is not installed.
""")

md("""
## 0. Environment

Makes `czipipe` importable regardless of where the kernel was started, and enables
autoreload so edits to the `.py` files take effect without restarting the kernel.
""")

code(r'''
%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path

# The package sits next to this notebook. Jupyter's working directory is not always the
# notebook's directory, so locate it explicitly rather than relying on cwd.
here = Path.cwd()
for cand in (here, *here.parents):
    if (cand / "czipipe" / "__init__.py").is_file():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break
else:
    raise ImportError(
        "czipipe/ not found next to this notebook or in any parent directory.\n"
        f"Looked upward from {here}. Keep convert_czi.ipynb alongside the czipipe folder.")

import czipipe
print(f"czipipe {czipipe.__version__}  from {Path(czipipe.__file__).parent}")
print(f"python  {sys.version.split()[0]}")
if sys.version_info >= (3, 14):
    print("\n[WARN] Python 3.14+ is not supported: pylibCZIrw has no wheels for it.\n"
          "       Create a Python 3.12 environment and select it as this notebook's kernel.")

missing = []
for mod, why in [("pylibCZIrw", "required"), ("tifffile", "required"),
                 ("numpy", "required"), ("h5py", 'output_format="hdf5" / ilastik'),
                 ("imagecodecs", "lzw / jpeg2000 compression (zstd before Python 3.14)"),
                 ("aicspylibczi", 'stitch_mode "refine" / "bigstitcher", flat-field'),
                 ("basicpy", 'flatfield.mode "basic"'),
                 ("tqdm", "progress bars"), ("matplotlib", "stitching preview"),
                 ("scipy", "faster tile matching for bigstitcher"),
                 ("psutil", "ilastik heartbeat hang-detection during classify()")]:
    try:
        __import__(mod)
    except ImportError:
        missing.append(f"  {mod:14s} ({why})")
print("\nnot installed:\n" + "\n".join(missing) if missing else "\nall optional deps present")
''')

md("""
## 1. Setup

The only cell to edit. Everything below reads from this `Config`.
""")

code(r'''
from czipipe import Config, Pipeline

cfg = Config(
    czi_path = Path(r"/path/to/your_image.czi"),
    out_dir  = Path(r"/path/to/output"),

    # ---- what to convert ---------------------------------------------------
    channels        = None,        # None = all, else e.g. [0, 2]
    z_range         = None,        # None = all, else (start, stop)
    t_range         = None,
    scenes          = None,
    split_scenes    = True,

    # ---- output ------------------------------------------------------------
    output_format    = "tiff",     # "tiff" (OME-TIFF) | "hdf5" (chunked, resumable)
    output_bit_depth = 16,         # 16 = lossless; 8 is irreversible, breaks linearity
    eight_bit_method = "percentile",
    percentiles      = (0.1, 99.9),

    # ---- tiling / codec ----------------------------------------------------
    tile              = (1024, 1024),
    compression       = "zlib",    # None | zlib | lzw | zstd | jpeg2000 (all lossless)
    compression_level = 5,

    # ---- performance -------------------------------------------------------
    n_read_threads  = 8,
    n_write_workers = 8,
    prefetch        = 24,          # peak RAM scales with this
    use_gpu         = True,

    # ---- stitching ---------------------------------------------------------
    stitch_mode = "metadata",      # "metadata" | "refine" | "bigstitcher"

    # ---- extras ------------------------------------------------------------
    write_pyramid  = False,
    verify_samples = 12,
    overwrite      = False,
)

# --- pixel size override ----------------------------------------------------
# Normally read from the CZI. Set it when Inspect warns that the CZI metadata could not
# be read, or the recorded value is wrong. Axes left out keep the CZI's own value.
cfg.manual_scaling = {}            # um/pixel, e.g. {"X": 0.65, "Y": 0.65, "Z": 2.0}

# --- stitching detail (only read for the matching stitch_mode) --------------
cfg.refine.channel   = None        # correlation channel; None -> first selected
cfg.refine.z         = None        # correlation plane;   None -> middle of selection
cfg.refine.min_ncc   = 0.20
cfg.refine.max_shift = 100

cfg.bigstitcher.xml  = None        # dataset.xml saved by BigStitcher

# --- flat-field (shading) correction ----------------------------------------
# Blending hides seams; it does not remove vignetting. Turn this on for anything
# you intend to quantify across a mosaic.
cfg.flatfield.mode      = "none"   # "none" | "measured" | "file" | "basic"
cfg.flatfield.flat_file = None     # for "file": 2D TIFF, or one plane per channel
cfg.flatfield.dark      = 0.0      # camera offset subtracted before dividing
cfg.flatfield.n_samples = 48       # fields sampled per channel for "measured"
cfg.flatfield.min_gain  = 0.2      # floor, so dark corners are not amplified into noise

# --- HDF5 -------------------------------------------------------------------
cfg.hdf5_dataset          = "/data"
cfg.hdf5_compression      = "gzip"
cfg.hdf5_resume           = True   # re-run this notebook to continue after a crash
cfg.hdf5_checkpoint_every = 200

# --- ilastik ----------------------------------------------------------------
cfg.ilastik.run           = False
cfg.ilastik.exe           = None   # None = autodetect; else run_ilastik.sh / ilastik.exe,
                                   # e.g. r"C:\Program Files\ilastik-1.4.1\ilastik.exe"
cfg.ilastik.projects      = []     # e.g. ["/path/to/project_a.ilp", ("tag_b", "/path/to/project_b.ilp")]
cfg.ilastik.export_source = "Simple Segmentation"   # "Probabilities" is ~50x larger
cfg.ilastik.ram_mb        = None   # None = half the installed RAM; or a number of MB
cfg.ilastik.dry_run       = True   # print the command first; verify before a long run
cfg.ilastik.chunk_z       = None   # e.g. 150: classify 150 Z-planes per ilastik run, so an
                                   # interruption loses one chunk, not the whole run. Each
                                   # chunk also classifies a halo either side of it, so
                                   # small chunks cost far more; the log prints the cost

cfg.validate()
''')

md("""
## 2. Inspect

Reads only the XML header and the subblock directory, so this is fast even on 150 GB.
""")

code(r'''
pipe = Pipeline(cfg)
pipe.inspect()
''')

md("""
## 3. Plan

Mosaic geometry, registration, flat-field profiles and intensity statistics. Nothing is
written yet.

If flat-field correction is on, check the printed **split-half disagreement**: it
compares profiles estimated from two independent halves of the sample fields. Healthy
estimates measure 1.5&ndash;3% RMS. A large value means the estimate is tracking your sample
rather than the illumination, and the profile should not be used &mdash; take more samples, or
supply a measured flat-field with `mode="file"`. The saved `.flatfield.tif` is worth
opening: it should look like illumination, not like your specimen.

If `stitch_mode` is `refine`, the two numbers that matter appear here: the fraction of
pairs accepted (low means featureless overlaps at this Z — try `cfg.refine.z` or a
brighter `cfg.refine.channel`) and the RMS residual (above ~2 px means the tiles are not
related by pure translation, and no tuning here will fix that).
""")

code(r'''
plans = pipe.plan()
''')

md("""
## 4. Preview the stitching

A downsampled overview of the middle Z plane (first selected channel) per scene, with the
mosaic tiles outlined. Look at the seams before committing to a long run.

Every `stitch_mode` draws one. In `metadata` mode it is libCZI's composite from the stage
positions &mdash; the same pixels Convert will write &mdash; so this is the image to judge before
deciding whether `refine` is worth it. The outlines there need aicspylibczi; without it
the overview is shown bare.
""")

code(r'''
import matplotlib.pyplot as plt
import numpy as np

previews = pipe.preview(max_px=1200)
fig, axes = plt.subplots(1, len(previews), figsize=(6 * len(previews), 6), squeeze=False)
for ax, (plan, img, lvl) in zip(axes[0], previews):
    lo, hi = np.percentile(img, (1, 99.5))
    hi = max(hi, lo + 1)
    if img.ndim == 3:                  # RGB: imshow ignores vmin/vmax, so scale here
        img = np.clip((img - lo) / (hi - lo), 0, 1)
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    s = 1 << lvl
    for mx, my, mw, mh in pipe.tile_rects(plan).values():
        ax.add_patch(plt.Rectangle(((mx - plan.rect[0]) / s, (my - plan.rect[1]) / s),
                                   mw / s, mh / s, fill=False, ec="tab:orange",
                                   lw=0.5, alpha=0.7))
    ax.set_title(f"{plan.label}  {cfg.stitch_mode}  (1/{s})")
    ax.axis("off")
plt.tight_layout(); plt.show()
''')

md("""
## 5. Where is the time going?

Run before tuning. Times read, convert and compress separately and names the bottleneck.
""")

code(r'''
pipe.benchmark(n=8)
''')

md("""
## 6. Convert

Safe to interrupt. With `output_format="hdf5"` and `hdf5_resume=True`, re-running this
cell continues from the last checkpoint; TIFF is written sequentially, so an interrupted
TIFF starts again. Either way, an output that already finished with the same settings is
recognised and skipped, so after a kernel restart this cell is quick: it re-attaches the
finished files for verification and classification.
""")

code(r'''
results = pipe.convert()
''')

md("""
## 7. Verify

Re-reads random tiles from the output and compares them against the same ROI read fresh
from the CZI, with the identical transform applied. Catches tile-ordering bugs,
off-by-one errors in the ROI arithmetic and silent codec problems.

For HDF5 this also reports ilastik compatibility: axistags, dtype, chunking and
completeness.
""")

code(r'''
pipe.verify()
''')

md("""
## 8. Pixel classification

Runs each project headlessly and streams the result into a tiled BigTIFF. Leave
`cfg.ilastik.dry_run = True` for the first pass and check the printed command against
`run_ilastik.sh --help` on your install.

Safe to interrupt; stopping the cell stops ilastik and every process it started. Re-running
resumes per scene and project: finished segmentations are skipped, an ilastik export that
completed before its TIFF conversion was interrupted is reused, and an interrupted ilastik
run starts again (ilastik itself cannot resume). With `cfg.ilastik.chunk_z` set, it
continues from the last finished Z-chunk instead, and prints what the Z halo costs at that
chunk size; `chunk_z` can be changed between runs without losing finished planes. After a
kernel restart, run Inspect, Plan and Convert first (Convert skips finished outputs), then
this cell.
""")

code(r'''
segmentations = pipe.classify()
''')

md("""
## 9. Preview the segmentation

Raw channels next to the segmentation at a few depths spread through the stack — enough
to catch an obviously-wrong or degenerate classifier (e.g. one that labels every pixel
the same, which is what a project with no saved classifier does) without leaving the
notebook. This is a quick sanity check, not a replacement for opening the result in Fiji
or napari before trusting it for real analysis.
""")

code(r'''
import matplotlib.pyplot as plt
import numpy as np

previews = pipe.preview_segmentation(segmentations, n_planes=3, max_px=800)
for tag, seg_path, planes in previews:
    n_ch = len(planes[0][1])
    fig, axes = plt.subplots(len(planes), n_ch + 1,
                             figsize=(5 * (n_ch + 1), 5 * len(planes)), squeeze=False)
    for row, (z, chans, seg_img) in enumerate(planes):
        for c, img in enumerate(chans):
            lo, hi = np.percentile(img, (1, 99.5))
            axes[row, c].imshow(img, cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            axes[row, c].set_title(f"Z={z} raw ch{c}")
            axes[row, c].axis("off")
        axes[row, n_ch].imshow(seg_img, cmap="viridis")
        axes[row, n_ch].set_title(f"Z={z} segmentation")
        axes[row, n_ch].axis("off")
    fig.suptitle(tag)
    plt.tight_layout()
    plt.show()
''')

md("""
## 10. Notes

**Sanity-check the output in Fiji or napari before deleting anything.** Verification
proves the tiles match; it cannot tell you the 8-bit display range you picked was a good
one. For anything you intend to quantify, keep `output_bit_depth = 16`.

**Which stitching mode.** Try `metadata` first and look at it — libCZI already composites
from the stage positions, and on a well-calibrated Z1 that is often enough. `refine` adds
translation registration computed here. `bigstitcher` lets BigStitcher do the
registration on its pyramid while this package still composites at full resolution from
the CZI, so no resaved copy is needed. All three feed the same compositor.

`bigstitcher` is translation-only here, and refuses rather than silently dropping
rotation, shear, or multi-angle/multi-illumination data. Those need BigStitcher's own
fusion; fuse there and run this over the fused output with `metadata`.

**Registration is computed once**, on one reference plane, and applied to every Z, C and
T. Right for a stage-scanned mosaic, wrong if you have per-plane drift. Chromatic offsets
between channels are not corrected.

**Flat-field correction is per field of view**, so it only works where individual fields
are still separable. On a mosaic that means the per-tile path: `refine`, `bigstitcher`,
or `metadata` with `flatfield.mode` set, which routes through the same compositor using
the stage positions unchanged. Once libCZI has blended a mosaic the fields are gone and
no per-field correction is possible &mdash; the notebook says so when it reroutes.

`measured` estimates the profile from the median across many fields, which assumes your
sample content is uncorrelated with position *within* the field. That holds for a large
mosaic of dispersed structures and fails for, say, a vessel that runs down the middle of
every tile. The split-half diagnostic is there to catch exactly that. `file` is the
rigorous option if you have a flat-field acquired from a dye slide. `basic` delegates to
BaSiC, which also fits a darkfield term.

Correction happens **before** blending, and the 8-bit display ranges are measured from
libCZI's uncorrected composite. The profile is normalised to a median of 1 so the shift
is small, but set `manual_ranges` if you need it exact.

**ilastik.** `Probabilities` gives one float32 channel per class — for a 100 GB input with
three classes, roughly 600 GB. `Simple Segmentation` is a single uint8 label map. Cell 3
prints the projection for your data. Note that a classifier trained on 16-bit will not
transfer to an 8-bit conversion, and ilastik's CLI cannot handle paths containing spaces.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                  "name": "python3"},
                   "language_info": {"name": "python", "version": "3.12"}},
      "nbformat": 4, "nbformat_minor": 5}

out = pathlib.Path(__file__).resolve().parent.parent / "convert_czi.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}  ({len(cells)} cells, "
      f"{sum(1 for c in cells if c['cell_type'] == 'code')} code)")
