# czipipe

Streaming conversion of large Zeiss `.czi` files (70–150 GB) to tiled OME-TIFF or
chunked HDF5, with optional mosaic stitching and ilastik pixel classification.

Peak memory is set by `prefetch × tile`, not by the size of the file, so a 150 GB scan
converts on a laptop.

## Install

You need [conda](https://docs.conda.io/en/latest/miniconda.html) and
[git](https://git-scm.com/downloads). Then, in a terminal (Anaconda Prompt on Windows):

```bash
git clone https://github.com/Harveyda94/CZI-2-ilastik-pipeline.git
cd CZI-2-ilastik-pipeline
conda create -n czipipe python=3.12 -y
conda activate czipipe
pip install -e ".[stitch,hdf5,ilastik]" ipykernel
python -m ipykernel install --user --name czipipe --display-name "Python 3.12 (czipipe)"
```

Open `convert_czi.ipynb` from that folder and choose the **Python 3.12 (czipipe)**
kernel. Add `flatfield` to the extras for `flatfield.mode "basic"`, or `gpu` for NVIDIA
CUDA 12 (e.g. `".[stitch,hdf5,ilastik,flatfield]"`). To update later, run `git pull`
in the folder.

To call czipipe from your own scripts without the notebook, one command is enough:

```bash
pip install "czipipe[stitch,hdf5,ilastik] @ git+https://github.com/Harveyda94/CZI-2-ilastik-pipeline.git"
```

## Dependencies

What each package is for, if you would rather install them individually:

```bash
pip install pylibCZIrw tifffile imagecodecs numpy tqdm h5py matplotlib scipy psutil
pip install aicspylibczi      # stitch_mode "refine" / "bigstitcher", and flat-field on mosaics
pip install basicpy           # only for flatfield.mode "basic"
pip install cupy-cuda12x      # optional, NVIDIA only
```

Use **Python 3.12** (3.9–3.13 work). Neither CZI package ships wheels for 3.14 yet, and
pip falls back to compiling libCZI from source (needs CMake plus MSVC on Windows or the
Xcode command-line tools on macOS). pylibCZIrw ships ready-built for Apple-silicon Macs,
64-bit Windows and Linux; on an Intel Mac it always compiles. A clean environment:

```bash
conda create -n czipipe python=3.12 -y
conda run -n czipipe pip install pylibCZIrw aicspylibczi tifffile imagecodecs numpy tqdm h5py matplotlib scipy psutil pytest ipykernel
conda run -n czipipe python -m ipykernel install --user --name czipipe --display-name "Python 3.12 (czipipe)"
```

then choose that kernel in Jupyter. Cell 0 of the notebook warns if the kernel is 3.14+.

## Use

Open **`convert_czi.ipynb`** and run the cells top to bottom. Keep it alongside the
`czipipe/` folder — cell 0 locates the package by searching upward from the working
directory, so it works whether Jupyter was started in this folder or a parent.

```
your-folder/
├── convert_czi.ipynb      <- open this
├── czipipe/               <- supporting modules
├── tests/
└── tools/
```

Cell 0 also enables `%autoreload`, so edits to the `.py` files take effect without
restarting the kernel, and prints which optional dependencies are missing.

The notebook is a thin driver over the same API you can call from a script:

```python
from czipipe import Config, Pipeline

cfg = Config(czi_path="scan.czi", out_dir="out", output_format="hdf5")
with Pipeline(cfg) as p:
    p.run()                       # inspect -> plan -> convert -> verify -> classify
```

Or step through it, which is what `convert_czi.ipynb` does:

```python
p = Pipeline(cfg)
p.inspect()        # dimensions, metadata, RAM estimate
p.plan()           # mosaic geometry, registration, intensity ranges
p.preview()        # downsampled overview per scene, in every stitch_mode
p.benchmark()      # read vs convert vs compress; names the bottleneck
p.convert()
p.verify()
seg = p.classify()
p.preview_segmentation(seg)   # raw channels beside the labels at a few depths
```

### Threads and memory

Left at `None`, these are sized from the machine the run is on:

| Setting | When left at `None` |
|---|---|
| `n_read_threads` | every usable CPU, at most 32 |
| `n_write_workers` (TIFF compression) | every usable CPU, at most 32 |
| `prefetch` (tiles in flight) | 3 per read thread |
| `cfg.ilastik.threads` | every usable CPU |
| `cfg.ilastik.ram_mb` | 75% of the installed RAM |

Give any of them a number to choose it yourself. "Usable" means the logical CPUs the
process is allowed to run on, which is fewer than the machine has inside a container or
under a scheduler. `inspect()` prints the counts in use and which of them were chosen
for you; `benchmark()` names the stage they are limiting. Peak conversion memory is
still `prefetch × tile`, so it now grows with the CPU count: about 200 MB of 16-bit
1024 px tiles at the 32-thread ceiling.

3D registration keeps to the same settings. It reads tiles, measures pairs and runs its
volume transform on `n_read_threads` threads and no others, and everything it holds,
overlap volumes and working memory together, fits `mosaic_cache_mb`: the compositor's
cache, which is empty until registration is over. `plan()` prints the threads and MB it
used, and warns if one pair alone needs more than the cap.

## Registration with `stitch_mode="refine"`

`refine` corrects the stage positions by a single integer XY shift per tile. By default
it is 2D: the same shift is applied to every Z plane, channel and timepoint. Set
`cfg.refine.mode = "3d"` to register along Z as well ([below](#3d-registration)).

Each overlapping pair of tiles is phase-correlated at several Z planes
(`cfg.refine.planes`, default 5, taken at the centres of equal slabs of the Z selection;
or name them with `cfg.refine.z`). A plane is *usable* for a pair if the shift it
proposes matches well and beats the stage position. Every proposed shift is then scored
at all of the pair's usable planes and the best overall is kept, so a pair needs only
one plane with structure in the overlap, and a shift fitted to one plane's noise loses
to one that fits the stack. The cost is one read of every mosaic tile per plane, plus a
second read where planes disagreed; memory stays at one plane of tiles.

`plan()` prints what happened, and the same record goes into `<name>.stitch.json`:

```
scene 0: 20/20 pairs accepted, RMS residual 0.00 px
  max |correction| 9 px
  depth check, Z 10, 30, 50, 70, 90:
    Z 10: 20/20 accepted pairs usable
    Z 30: 20/20 accepted pairs usable
    Z 50: 20/20 accepted pairs usable
    Z 70: 20/20 accepted pairs usable, 8 fit a different shift (up to 6 px away)
    Z 90: 20/20 accepted pairs usable, 8 fit a different shift (up to 6 px away)
  [warn] 8 pair(s) fit a different shift at some depths, by up to 6 px. One XY
  correction per tile cannot fit every plane: expect seams offset by about that much at
  the planes marked above.
```

- **usable** counts show which planes carried the registration. A plane at 0 had nothing
  to correlate in the overlaps.
- **fit a different shift** means that at that plane another shift, more than 1 px from
  the one chosen, matches measurably better (by `refine.min_gain`). That is an offset
  that changes with depth, which this mode reports and does not correct.
- **undecided between shifts** means a plane proposed another shift but the chosen one
  matches it about as well, as happens with structure running along a seam, or noise.
  It is not counted as a difference.

The depth check compares XY shifts between planes. It cannot see tiles that are offset
from each other in Z; that takes `refine.mode = "3d"`. `refine.planes = 1` is the old
single-plane behaviour, with no depth check.

### 3D registration

```python
cfg.stitch_mode = "refine"
cfg.refine.mode = "3d"        # default "2d"
cfg.refine.slab = 16          # consecutive Z planes correlated together
cfg.refine.max_shift_z = 5    # planes; larger implied Z corrections are rejected
```

A tile's stack can sit a plane or more above or below its neighbour's. Registering one
plane at a time cannot see that, and blends each tile against the wrong planes of the
next however good the XY fit. In `"3d"` mode each overlapping pair is phase-correlated
as a volume, over a *slab* of `refine.slab` consecutive planes of the overlap, which
gives a shift (dy, dx, dz) in one measurement. Every tile then gets an integer
correction in XY **and** Z, and during conversion each tile is taken from the plane its
Z offset says belongs at the plane being written.

- **Where it measures.** `refine.planes` slabs (default 5) spread through the Z
  selection without overlapping, or one slab around each plane named in `refine.z`. A
  stack too shallow for that many gets as many as fit. At most `(slab - 1) // 2` planes
  of Z offset are searched, so a thin slab limits `max_shift_z`; the report states the
  range actually searched.
- **How a shift is found.** Phase correlation only proposes: its five strongest distinct
  peaks, and the stage position itself. Each is scored by the NCC of the two overlap
  volumes, and the best is then walked a step at a time, in X, Y or Z, to where the
  match stops improving. The shift kept is therefore one that no neighbouring step
  beats, which is what gets the last pixel or plane right under noise. A proposal that
  was wrong is walked to the true shift or rejected, never accepted as proposed, and an
  offset beyond the search range walks out of it and is rejected rather than accepted a
  step short at its edge.
- **What is accepted.** As in 2D, a shift must match well (`min_ncc`) and measurably
  better than the stage position (`min_gain`). The XY part and the Z part are each held
  to that on their own: a Z offset is kept only if it beats the same XY shift without
  it, and the other way about. Without that, getting one right carries the other through
  with it, wrong sign included. A pair that matches best where the stage already put it
  is kept as a weak constraint, so tiles the stage has right hold each other in place.
  A shift that would leave less than `min_overlap` of overlap is not scored at all.
- **How tiles are placed.** One weighted least-squares fit over every accepted pair, so
  error does not build up across the mosaic. Pairs that are plainly wrong are left out
  and the fit redone: a pair more than 3.5 px, or 1.5 planes, from where the other pairs
  put its two tiles, provided those tiles are still joined by two separate routes without
  it, and it is at least four times as far out as the typical pair. The last condition
  keeps a mosaic whose pairs all disagree a little, as when the tissue moved between
  tiles, from being whittled down to the few that happen to agree. Pairs left out are
  named in a `[warn]` line; look at those seams. (The 2D fit is plain least squares.)
- **What Z corrections are relative to.** XY is pinned at the first tile, as in 2D. Along
  Z the median tile keeps its own planes, so as few stacks as possible are moved.
- **The ends of the stack.** The output has the planes you selected and no others. A
  tile corrected by +2 planes has nothing to give the first two planes of the file, and
  is left out there rather than filled from a neighbouring plane; the report lists the
  planes affected. Tiles are taken from anywhere in the file, so a `z_range` that stops
  short of the ends by a few planes loses nothing.

```
scene 0: 20/20 pairs accepted, RMS residual 0.00 px, 0.00 planes in Z
  3D registration on 3 slab(s) of 8 planes (Z 0-7, 8-15, 16-23), Z offsets searched to +/-3 planes
  max |correction| 9 px; Z corrections -1 to +1 planes
  [note] 2 of 24 output plane(s) lack some tiles (Z 0, 23): their Z offset reaches past
  the end of the stack there, and no plane is invented to fill the gap.
  depth check, slab by slab:
    Z   0-7: 20/20 accepted pairs usable
    Z  8-15: 20/20 accepted pairs usable
    Z 16-23: 20/20 accepted pairs usable
    no depth fits a different shift, in any of the 20 pair(s) usable in more than one slab
```

The depth check reads as in 2D, slab by slab. It is still one translation per tile: a
tile tilted or sheared against its neighbour shows up there as a shift that changes
with depth, and is reported, not corrected.

**Cost.** Every mosaic tile is read once per plane of every slab (80 planes per tile at
the defaults, against 5 in 2D). Some are read again: where a pair's slabs proposed
shifts the others had not already scored, and where the overlap volumes do not all fit
`mosaic_cache_mb` at once, since pairs are then measured in batches and the tiles shared
between batches are read for each. Tiles are read, pairs measured and the volume
transform run on the run's read threads (`n_read_threads`) and no others; the record is
the same however many there are. The cap covers everything held at once: the volumes,
one transform (16 bytes a voxel of the largest overlap), a whole tile per thread while
reading, and two planes per thread while scoring. Batches shrink to leave room for
those, and very large tiles are read on fewer threads. The report states the figure:

```
  8 thread(s), up to 385 MB: 273 MB of overlap volumes + 112 MB working, within mosaic_cache_mb = 768
```

That is for six 1920 px tiles with 10% overlap, where a pair took 26 ms per slab on 8
threads (77 ms on one). A large overlap is scored on a lattice of at least a million
voxels rather than on all of them. On a slow disk the reads will dominate.
`.stitch.json` records every pair's shift in every slab, and corrections as
`[dy, dx, dz]`. With a GPU the transform's memory is on the device, and is not counted.

## Interrupting and restarting

Every stage can be interrupted and re-run; nothing that finished is redone.

- **Convert.** HDF5 resumes from its last checkpoint, and gzip runs in the read threads
  rather than on HDF5's single thread (about 7x faster to write in a synthetic benchmark;
  the file is the same, and `cfg.hdf5_parallel_compress = False` restores the old path).
  TIFF is sequential, so an
  interrupted TIFF starts again — it is written as `<name>.partial` and only renamed when
  complete, so a half-written file never carries the final name. Either format records
  the settings it was made with, and a finished output is reused only if they match;
  otherwise you are told which setting changed.
- **Classify.** Resumes per (scene, project). Finished segmentations are skipped; an
  ilastik export that completed before its TIFF conversion was interrupted is reused (it
  is marked complete, and tied to the project file and input it came from); an
  interrupted ilastik run starts again, since ilastik cannot resume. The HDF5 staged for
  ilastik from a TIFF is kept until every project has finished.
- **A run that exports nothing is an error**, whatever ilastik's exit code: the Windows
  launcher has returned 0 after ilastik stopped on an exception at start-up. The message
  names the output file, quotes the last error in ilastik's log and gives the log's path.
  Such a run is never marked complete, so the next `classify()` runs it again.
- **Classify in Z-chunks** for runs measured in hours: `cfg.ilastik.chunk_z = 50` runs
  ilastik on 50 planes at a time, each with a Z-halo sized from the project's largest
  feature sigma so there are no seams, and checkpoints after every chunk. An interruption
  then loses one chunk, and re-running continues from the next. A partial export started
  with different settings is refused with the setting that changed. The halo is not free:
  a 35-plane halo makes each 50-plane chunk classify 120 planes, so the log prints the
  cost at the current `chunk_z` and at larger ones. `chunk_z` can be changed between runs
  of a partial export; finished planes are kept.
- **Switching `chunk_z` on or off between runs never costs an export.** A finished export
  is used whichever way it was made. A partial chunked one cannot be continued by a run
  in one piece, so with `chunk_z` off that run stops and says how many planes are done,
  instead of removing them. When an export is in the way, the message says to delete that
  file: `overwrite = True` would also rebuild the conversion it was made from.
- **`"Probabilities"` in chunks** has one channel per class, read from the project. (It
  was sized from the input's channels, so it only worked when the two counts matched.)
- **ilastik's RAM cap** (`cfg.ilastik.ram_mb`) defaults to 75% of the installed RAM. Set a
  number of MB to choose it yourself; ilastik treats the cap as approximate, so lower it
  if the machine starts swapping or ilastik is killed for memory.
- While ilastik runs, a line every 5 minutes reports elapsed time and the CPU used by its
  whole process tree, and warns when that stops rising (needs `psutil`).
- **After a kernel restart**, run `inspect()`, `plan()` and `convert()` again before
  `classify()`. `convert()` only re-attaches finished outputs; `classify()` refuses to run
  with nothing attached rather than silently doing nothing.
- Stopping ilastik stops its whole process tree (process group on macOS/Linux,
  `taskkill /T` on Windows), and `ilastik.timeout_h` is enforced while it runs.

## Layout

| Module | Responsibility |
|---|---|
| `config.py` | `Config` and its validation — every user knob |
| `source.py` | CZI inspection, metadata parsing, thread-local readers |
| `geometry.py` | `Tiler` (the canonical tile ordering) and the ordered prefetch `stream` |
| `compute.py` | optional CuPy backend; bit-depth conversion; FFT |
| `flatfield.py` | shading estimation (measured / file / BaSiC) and application |
| `plan.py` | per-scene planning: ranges, corrections, compositor |
| `worker.py` | the per-tile callable, dispatching libCZI vs compositor |
| `writers.py` | tiled BigTIFF and resumable chunked HDF5 |
| `validate.py` | output verification and the ilastik HDF5 check |
| `ilastik.py` | headless classification and result conversion |
| `stitch/mosaic.py` | tile geometry and the memory-bounded compositor |
| `stitch/refine.py` | multi-plane phase correlation, the global least-squares solve, the depth check |
| `stitch/refine3d.py` | 3D registration: slab phase correlation, an XY and Z shift per tile |
| `stitch/bigstitcher.py` | BDV XML transform import |
| `pipeline.py` | `Pipeline`, the orchestration the notebook calls |
| `bench.py` | stage timing |

## Tests

```bash
python -m pytest                  # 348 tests
python tools/minipytest.py        # if pytest is unavailable
```

The suite runs against synthetic data with known-correct output and fakes for
pylibCZIrw, aicspylibczi and h5py, so it needs none of the microscopy stack (the
chunked-ilastik and parallel-compression tests use the real h5py, because the chunk
record lives in HDF5 attributes and only the real library can decode a directly written
chunk, and are skipped without it). It covers
tile ordering and padding, TIFF and HDF5 round-trips being bit-exact, crash/resume
semantics, registration recovering known offsets, and the guards: an inverted
phase-correlation sign, a featureless tile, a rotated BigStitcher view, a multi-angle
dataset, a mismatched XML, changed settings on resume, and malformed HDF5.

Multi-plane registration is checked on five-plane synthetic stacks: a featureless middle
plane no longer loses the registration; a tile that sits 6 px elsewhere in the two
deepest planes keeps the majority's shift and is reported at exactly those planes; a
plane of stripes, which cannot fix one axis, is not reported; and noise alone is not
reported either. Measured over 12 seeds, no pair is falsely flagged up to a noise sd of
800, one pair in one seed at 2000, and about one pair in every third seed at 4000, by
which point a single plane has lost a third of its pairs.

3D registration is checked on a synthetic volume whose tiles are cut at known XY and Z
offsets. It recovers every offset exactly, and over 6 seeds still does at a noise sd of
2000; at 4000, where the noise is as large as the signal, 52 of 54 tiles are exact and
none is out by more than one step. The composite then equals the truth at every plane,
seams included, while the same XY corrections without the Z ones do not. That the NCC
decides, and not the correlation peak, is tested by handing the measurement wrong
proposals on purpose (Z sign inverted, XY sign inverted, both, 40 px off): none is
accepted as proposed, and 17 to 19 of the 20 pairs still end exactly right. Also
covered: a Z offset beyond `max_shift_z` is rejected rather than clipped, including one
a single plane past the range; a walk that never settles; a sliver of overlap; a
featureless slab; an offset that changes with depth; batches under a small memory cap,
and any number of threads, giving the identical record; and an output's fingerprint
changing only when a tile actually moves along Z. That it keeps to the run's settings
is tested too: the transform is given `n_read_threads` and not the machine's CPU count,
the memory the report states fits `mosaic_cache_mb`, and what numpy actually allocates
stays under the stated figure. The fit is tested with pairs made wrong on purpose: one
or two among twenty are left out and the tiles land exactly; a tile's only link, three
tiles in a ring, and pairs that all disagree a little are all kept.

`tests/test_real_czi.py` is the exception to the fakes: it writes a mosaic CZI with
pylibCZIrw and takes it through the real `Source` and aicspylibczi, in 2D and in 3D, to
a verified TIFF that equals the truth at every plane. It is skipped where those
libraries are missing. The pixels are still synthetic.

Both runners take every test: `tools/minipytest.py` accepts one `parametrize` per test
and has no `importorskip`, so a test that needs two parameters lists their product, and
one that needs a library skips itself when the import fails.

Flat-field is checked against a known synthetic vignette: the estimator recovers it to
under 0.05 worst-pixel, correction reduces residual shading error by more than 3x, and
the split-half diagnostic separates healthy estimates (1.5-3% RMS) from ones tracking
sample structure (~98%) with a wide margin. The polynomial fit is tested against the
Gaussian alternative it replaced, which was biased by ~0.14 at the corners.

Verification is itself tested against deliberately corrupted output — a check that
cannot fail is worth nothing. The same applies to the regression tests below: each was
confirmed to fail with its fix reverted.

### Field-reported fixes

Three failures found on a real 74 GB Zeiss Z1 file, all now covered:

- **Non-ASCII CZI metadata.** tifffile enforces `value.encode("ascii")` for `str` values
  written with TIFF type `s`, and Zeiss writes a micro sign (U+00B5) in the scaling
  units, so embedding the XML raised *"TIFF strings must be 7-bit ASCII"* on essentially
  every real file. The XML is now passed as `bytes`, which skips that check and is
  written verbatim; reading the tag back still yields a `str` with the sign intact.
- **`background_pixel` rejected.** Some pylibCZIrw builds call `Color(*background_pixel)`
  unconditionally, so a scalar raises `TypeError` even for Gray pixel types, contrary to
  Zeiss's own API docs. The argument is no longer passed anywhere — the library default
  is already black. `FakeReader` in the test suite raises on it deliberately, so any
  reintroduction fails the suite rather than only failing on a microscope file.
- **A blanket `except` misattributing failures.** The scene-argument probe caught every
  exception and blamed `scene=`, so the `background_pixel` `TypeError` above was reported
  as *"scene= argument rejected"* and silently disabled scene-restricted reads for the
  whole run. The catch is now narrow and re-raises anything not plausibly about `scene`.

A fourth report was a skipped notebook cell rather than a defect, but it pointed at a
real fragility: a 16-bit passthrough was indexing the intensity-range table it never
uses, making an unrelated planning step a hard prerequisite of every tile. The lookup is
now made only when a rescale actually needs it, and the 8-bit path fails with a message
naming `Pipeline.plan()` instead of a bare `KeyError`.

From a 136 GB light-sheet file classified with ilastik 1.4.1rc2 on Windows:

- **Unreadable CZI metadata.** pylibCZIrw failed with *"Invalid MetadataSegment-magic"*
  where a deleted segment sat in place of the metadata, although the pixel data read
  fine. The conversion now continues with a warning (no pixel size, channel names
  `C0`/`C1`…, no embedded XML), and `cfg.manual_scaling = {"X": …, "Y": …, "Z": …}`
  supplies the pixel size; it also overrides a recorded value you know to be wrong.
- **`--cutout_subregion` does not work** with a trained project on that build
  (`SlotNotReadyError` before it starts), so chunked classification stages each chunk as
  its own small input instead.
- **ilastik can log a fatal error and still exit 0.** Each chunk's export is checked for
  the expected shape before it is accepted, and the error points at that chunk's log.

Found by review rather than in the field:

- **Flat-field correction was never applied to mosaics.** The profile was estimated,
  reported and written into `conversion.json`, but not handed to the compositor, so the
  pixels came out uncorrected. It is applied now. **A mosaic converted with
  `flatfield.mode` set by an earlier version is uncorrected**, and is no longer mistaken
  for a corrected one: re-running with the same settings refuses it and says why. Set
  `flatfield.mode = "none"` to keep using that file as the uncorrected output it is (it
  is not modified, so a chunked classification already under way on it continues), or
  `overwrite = True` to rebuild it corrected. A classifier trained on the uncorrected
  pixels should be checked against the corrected ones. The exception is such a mosaic
  converted with `stitch_mode = "metadata"`: it was blended by czipipe only because a
  flat-field was set, so with the flat-field off no setting describes it, and it has to
  be rebuilt.
- **Sidecar names lost everything after the first dot in the file name**, so
  `brain_0.5x_scene00` wrote `brain_0.conversion.json` and every scene overwrote it.
- **Chunked classification staged only timepoint 0**, so later timepoints were classified
  from zeros.
- **`verify()` reported success with no output attached** (after a kernel restart). It now
  says to run `convert()` first, as `classify()` does.

## Known limitations

- **Flat-field correction only works per field of view.** On a mosaic that means the
  per-tile path (`refine`, `bigstitcher`, or `metadata` with `flatfield.mode` set, which
  reroutes through the same compositor). Once libCZI has blended a mosaic the fields are
  gone. The `measured` estimator assumes sample content is uncorrelated with position
  within the field; the split-half diagnostic reports when it is not.
- **Registration is one translation per tile.** By default `refine` is 2D: it measures
  several Z planes but applies one XY shift per tile to all Z/C/T, and a Z offset between
  tiles is not measured. `refine.mode = "3d"` measures and corrects that too, in whole
  planes. In either mode an offset that changes with depth (a tilted or sheared tile) is
  reported by the depth check and not corrected; with only two usable planes that
  disagree, the tile is left between the two. No chromatic correction.
- **3D registration has not been run on a tiled acquisition.** The suite tests it end to
  end on a real CZI file written from a synthetic volume. Outside the suite it has been
  run on real two-photon z-stacks cut into overlapping tiles at known offsets: with
  every tile from one acquisition it recovered each offset exactly and the output
  equalled the original stack. With tiles drawn alternately from two acquisitions of one
  field, so that overlaps had independent noise and the tissue had moved between them,
  two of the twelve pairs spanning the acquisitions measured 20 px out. They are left
  out of the fit and named, and the tiles then land within 2 px and one plane of the
  best whole-field alignment of the two acquisitions, which is itself no better defined
  than that: the tissue was not rigidly related between them. Its GPU path has only been
  run with numpy standing in for CuPy, which checks the code and not the device.
  Look at `preview()` and the report before a long conversion. It does not interpolate
  between planes, so a true half-plane offset is corrected to the nearer whole plane,
  and it does not add planes: the first and last few output planes can lack the tiles
  whose offset reaches past the end of the stack. `bigstitcher` mode still applies XY
  only.
- **`bigstitcher` mode is translation-only**, and refuses rotation, shear, multi-angle and
  multi-illumination rather than dropping them silently.
- **TIFF cannot resume mid-file.** Sequential format; an interrupted TIFF starts again
  (a finished one is reused). Use `output_format="hdf5"` for very long conversions.
- **ilastik cannot resume mid-run.** An interrupted classification of one (scene,
  project) restarts from the beginning, unless `cfg.ilastik.chunk_z` is set. Chunking
  costs ilastik reloading the project (and retraining, if it has no saved classifier)
  once per chunk, plus re-reading the halo planes.
- **The ilastik invocation has only been run against ilastik 1.4.1rc2 on Windows.** Flags
  follow the current headless documentation, but that interface has shifted between
  versions. Use `cfg.ilastik.dry_run = True` first on any other version.
- **8-bit conversion is irreversible** and breaks the linear intensity relationship most
  downstream analysis assumes. Everything else is lossless: every TIFF codec offered
  (zlib, lzw, zstd, jpeg2000) is written losslessly, and `Config.validate()` round-trips
  a probe through the chosen codec so a lossy or unavailable one fails before the run.
- **Paths containing spaces are refused** for ilastik inputs and outputs (an ilastik CLI
  limitation). Common on Windows (`C:\Users\First Last`, OneDrive) — use another folder.
- **On Windows, set `cfg.ilastik.exe`** if autodetection misses your install; it searches
  `Program Files` for `ilastik*\ilastik.exe`.
