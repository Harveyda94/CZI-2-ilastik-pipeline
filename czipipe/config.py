"""Configuration for the CZI conversion pipeline.

Every user-facing knob lives here. Nothing else in the package reads globals, so a
Config instance fully determines a run.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Union

PathLike = Union[str, Path]


def physical_ram_mb() -> Optional[int]:
    """Installed RAM in MB, or None if it cannot be read on this machine."""
    try:
        import psutil
        return int(psutil.virtual_memory().total // 2**20)
    except Exception:
        pass
    try:                                    # POSIX without psutil
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") // 2**20)
    except (AttributeError, ValueError, OSError):
        return None


@dataclass
class RefineConfig:
    """Settings for `stitch_mode="refine"` (registration computed by this package)."""

    channel: Optional[int] = None       # correlation channel; None -> first selected
    z: Optional[int] = None             # correlation plane; None -> middle of selection
    t: Optional[int] = None             # correlation timepoint; None -> first selected
    min_overlap: int = 32               # px; smaller overlaps are not attempted
    max_shift: int = 100                # px; larger implied corrections are rejected
    min_ncc: float = 0.20               # minimum normalised cross-correlation to accept
    min_gain: float = 0.02              # NCC must beat the metadata NCC by this much


@dataclass
class BigStitcherConfig:
    """Settings for `stitch_mode="bigstitcher"` (transforms read from a BDV XML)."""

    xml: Optional[PathLike] = None
    timepoint: int = 0
    max_shift: int = 200                # px; larger means tiles were matched wrongly
    force: bool = False                 # bypass the guards (don't)


@dataclass
class FlatFieldConfig:
    """Shading correction. Only applicable where individual fields of view survive."""

    mode: str = "none"              # "none" | "measured" | "file" | "basic"
    flat_file: Optional[PathLike] = None    # 2D TIFF, or one plane per selected channel
    dark: float = 0.0               # camera offset subtracted before dividing
    n_samples: int = 48             # fields sampled per channel for "measured"/"basic"
    z_samples: int = 3              # distinct Z planes to draw those samples from
    percentile: float = 50.0        # across-field statistic; median resists bright objects
    smooth_mode: str = "poly"       # "poly" (unbiased at the edges) | "gaussian"
    poly_degree: int = 4            # for "poly"; 4 fits typical vignetting comfortably
    smooth_sigma: float = 30.0      # px at full field scale, for "gaussian"
    min_gain: float = 0.2           # floor on the profile, so dark corners are not amplified
    estimate_max_px: int = 512      # longest edge used during estimation (then upsampled)
    basic_darkfield: bool = True    # let BaSiC fit a darkfield term too
    save_profile: bool = True       # write <stem>.flatfield.tif for inspection
    max_disagreement_pct: float = 10.0  # split-half RMS warning threshold; healthy
                                        # estimates measure 1.5-3%


@dataclass
class IlastikConfig:
    run: bool = False
    projects: Sequence = field(default_factory=list)   # paths, or (tag, path) pairs
    exe: Optional[PathLike] = None                     # None -> autodetect
    export_source: str = "Simple Segmentation"         # or "Probabilities"
    export_dtype: str = "uint8"
    ram_mb: Optional[int] = None       # LAZYFLOW_TOTAL_RAM_MB. None -> half the installed
                                        # RAM: ilastik documents the cap as approximate and
                                        # the OS still needs room to cache the input file.
    threads: Optional[int] = None      # None -> os.cpu_count()
    timeout_h: float = 48.0
    seg_compression: Optional[str] = "zlib"
    keep_h5: bool = False
    dry_run: bool = False              # print the command and stop
    chunk_z: Optional[int] = None      # None -> one monolithic ilastik run (default).
                                        # Set to export N Z-planes per ilastik invocation,
                                        # checkpointed so a crash only loses one chunk.
                                        # Worth it once a run is long enough that redoing
                                        # the whole thing is expensive (hours, not minutes).


@dataclass
class Config:
    # ---- input / output ---------------------------------------------------
    czi_path: PathLike
    out_dir: PathLike
    out_stem: Optional[str] = None
    output_format: str = "tiff"        # "tiff" (OME-TIFF) | "hdf5" (chunked, resumable)

    # ---- what to convert --------------------------------------------------
    channels: Optional[Sequence[int]] = None
    z_range: Optional[Sequence[int]] = None     # (start, stop) half-open
    t_range: Optional[Sequence[int]] = None
    scenes: Optional[Sequence[int]] = None
    split_scenes: bool = True

    # ---- physical calibration ----------------------------------------------
    # Override per-axis pixel size in micrometres, e.g. {"Z": 2.0}. Normally read
    # straight from the CZI's own XML metadata; set this when that read fails (some
    # files trip a pylibCZIrw parsing limitation - Source logs a [warn] when it
    # happens) or when you know the true value differs from what's recorded. Any
    # axis given here overrides the auto-detected value for that axis only; axes
    # left out are unaffected.
    manual_scaling: dict = field(default_factory=dict)

    # ---- bit depth --------------------------------------------------------
    output_bit_depth: int = 16                  # 16 = lossless passthrough
    eight_bit_method: str = "percentile"        # "percentile" | "shift" | "manual"
    percentiles: Sequence[float] = (0.1, 99.9)
    manual_ranges: dict = field(default_factory=dict)
    stats_max_px: int = 4096
    stats_n_planes: int = 5

    # ---- tiling / codec ---------------------------------------------------
    tile: Sequence[int] = (1024, 1024)          # (height, width), multiples of 16
    compression: Optional[str] = "zlib"         # None | zlib | lzw | zstd | jpeg2000
                                                # (all written losslessly)
    compression_level: int = 5

    # ---- performance ------------------------------------------------------
    n_read_threads: int = 8
    n_write_workers: int = 8
    prefetch: int = 24                          # tiles in flight; peak RAM scales with this
    use_gpu: bool = True

    # ---- stitching --------------------------------------------------------
    stitch_mode: str = "metadata"               # metadata | refine | bigstitcher
    blend: str = "linear"                       # "linear" | "none"
    blend_width: Optional[int] = None
    mosaic_cache_mb: int = 768
    stitch_report: bool = True
    refine: RefineConfig = field(default_factory=RefineConfig)
    bigstitcher: BigStitcherConfig = field(default_factory=BigStitcherConfig)

    # ---- HDF5 -------------------------------------------------------------
    hdf5_dataset: str = "/data"
    hdf5_compression: Optional[str] = "gzip"    # None | "gzip" | "lzf"
    hdf5_level: int = 4
    hdf5_shuffle: bool = True
    hdf5_resume: bool = True
    hdf5_checkpoint_every: int = 200
    hdf5_parallel_compress: bool = True         # gzip in the read threads instead of
                                                # inside HDF5 (one thread). Same file;
                                                # set False if a build misbehaves.

    # ---- extras -----------------------------------------------------------
    write_pyramid: bool = False
    pyramid_levels: int = 4
    embed_czi_xml: bool = True
    write_sidecars: bool = True
    verify_samples: int = 12
    overwrite: bool = False

    # ---- shading ----------------------------------------------------------
    flatfield: FlatFieldConfig = field(default_factory=FlatFieldConfig)

    # ---- segmentation -----------------------------------------------------
    ilastik: IlastikConfig = field(default_factory=IlastikConfig)

    # -----------------------------------------------------------------------
    def __post_init__(self):
        self.czi_path = Path(self.czi_path)
        self.out_dir = Path(self.out_dir)
        self.tile = tuple(int(v) for v in self.tile)
        if self.bigstitcher.xml is not None:
            self.bigstitcher.xml = Path(self.bigstitcher.xml)
        if self.flatfield.flat_file is not None:
            self.flatfield.flat_file = Path(self.flatfield.flat_file)

    @property
    def stem(self) -> str:
        return self.out_stem or self.czi_path.stem

    @property
    def ilastik_threads(self) -> int:
        return int(self.ilastik.threads or os.cpu_count() or 4)

    @property
    def ilastik_ram_mb(self) -> int:
        """LAZYFLOW_TOTAL_RAM_MB: ilastik.ram_mb if set, otherwise half the installed RAM
        (at least 2 GB), or 4 GB when the installed RAM cannot be read."""
        if self.ilastik.ram_mb:
            return int(self.ilastik.ram_mb)
        total = physical_ram_mb()
        return max(2048, total // 2) if total else 4096

    def validate(self) -> "Config":
        """Fail fast on settings that would otherwise blow up deep in a long run."""
        if self.output_bit_depth not in (8, 16):
            raise ValueError(f"output_bit_depth must be 8 or 16, got {self.output_bit_depth}")
        if self.output_format not in ("tiff", "hdf5"):
            raise ValueError(f'output_format must be "tiff" or "hdf5", got {self.output_format!r}')
        if self.stitch_mode not in ("metadata", "refine", "bigstitcher"):
            raise ValueError('stitch_mode must be "metadata", "refine" or "bigstitcher", '
                             f"got {self.stitch_mode!r}")
        if self.eight_bit_method not in ("percentile", "shift", "manual"):
            raise ValueError(f"unknown eight_bit_method {self.eight_bit_method!r}")
        if set(self.manual_scaling) - set("XYZ"):
            raise ValueError(f'manual_scaling keys must be "X", "Y" and/or "Z", '
                             f"got {sorted(self.manual_scaling)}")
        if any(v <= 0 for v in self.manual_scaling.values()):
            raise ValueError(f"manual_scaling values must be positive (um/px), "
                             f"got {self.manual_scaling}")
        if self.blend not in ("linear", "none"):
            raise ValueError(f'blend must be "linear" or "none", got {self.blend!r}')
        if self.hdf5_compression not in (None, "none", "gzip", "lzf"):
            raise ValueError(f"unsupported hdf5_compression {self.hdf5_compression!r}")
        if any(v % 16 for v in self.tile):
            raise ValueError(f"TIFF requires tile dimensions to be multiples of 16, got {self.tile}")
        if self.stitch_mode == "bigstitcher" and not self.bigstitcher.xml:
            raise ValueError('stitch_mode="bigstitcher" requires bigstitcher.xml')
        if self.flatfield.mode not in ("none", "measured", "file", "basic"):
            raise ValueError('flatfield.mode must be "none", "measured", "file" or '
                             f'"basic", got {self.flatfield.mode!r}')
        if self.flatfield.mode == "file" and not self.flatfield.flat_file:
            raise ValueError('flatfield.mode="file" requires flatfield.flat_file')
        if self.flatfield.mode == "file" and not Path(self.flatfield.flat_file).is_file():
            raise FileNotFoundError(f"flat-field file not found: {self.flatfield.flat_file}")
        if self.flatfield.smooth_mode not in ("poly", "gaussian"):
            raise ValueError('flatfield.smooth_mode must be "poly" or "gaussian", '
                             f"got {self.flatfield.smooth_mode!r}")
        if not 0 < self.flatfield.min_gain <= 1:
            raise ValueError("flatfield.min_gain must be in (0, 1]")
        if self.ilastik.run and not 1 <= len(self.ilastik.projects) <= 3:
            raise ValueError("ilastik.projects must hold between 1 and 3 projects")
        if self.ilastik.ram_mb is not None and self.ilastik.ram_mb <= 0:
            raise ValueError("ilastik.ram_mb must be a positive number of MB, or None to "
                             f"size it from the installed RAM; got {self.ilastik.ram_mb!r}")
        if self.ilastik.chunk_z is not None and self.ilastik.chunk_z < 1:
            raise ValueError(f"ilastik.chunk_z must be a positive int, got {self.ilastik.chunk_z!r}")
        # Round-trip a probe through each TIFF codec that will be used, with the settings
        # the writer will use: catches a missing imagecodecs, or a lossy combination,
        # before a multi-hour run rather than after it.
        from .writers import check_tiff_codec
        if self.output_format == "tiff":
            check_tiff_codec(self.compression, self.compression_level, "compression")
        if self.ilastik.run:
            check_tiff_codec(self.ilastik.seg_compression, self.compression_level,
                             "ilastik.seg_compression")
        if not self.czi_path.is_file():
            raise FileNotFoundError(f"CZI not found: {self.czi_path}")
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["czi_path"] = str(self.czi_path)
        d["out_dir"] = str(self.out_dir)
        if d["bigstitcher"].get("xml"):
            d["bigstitcher"]["xml"] = str(d["bigstitcher"]["xml"])
        if d["flatfield"].get("flat_file"):
            d["flatfield"]["flat_file"] = str(d["flatfield"]["flat_file"])
        d["ilastik"]["projects"] = [str(p) for p in d["ilastik"]["projects"]]
        d["ilastik"]["exe"] = str(d["ilastik"]["exe"]) if d["ilastik"]["exe"] else None
        return d
