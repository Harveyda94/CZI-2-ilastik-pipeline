"""Streaming CZI -> OME-TIFF / HDF5 conversion with optional stitching and segmentation.

    from czipipe import Config, Pipeline

    cfg = Config(czi_path="scan.czi", out_dir="out")
    with Pipeline(cfg) as p:
        p.run()

Peak memory is set by `prefetch * tile` rather than by the size of the image, so a
150 GB CZI converts on a laptop.
"""

from .config import (Config, RefineConfig, BigStitcherConfig, IlastikConfig,
                     FlatFieldConfig)
from .pipeline import Pipeline, Result
from .compute import Backend
from . import flatfield
from .geometry import Tiler, stream
from .source import Source
from .plan import ScenePlan, plan_scenes, compute_ranges
from .validate import verify, check_h5_for_ilastik, report_h5_check

__version__ = "1.0.0"

__all__ = [
    "Config", "RefineConfig", "BigStitcherConfig", "IlastikConfig",
    "FlatFieldConfig", "flatfield",
    "Pipeline", "Result", "Backend", "Tiler", "stream", "Source",
    "ScenePlan", "plan_scenes", "compute_ranges",
    "verify", "check_h5_for_ilastik", "report_h5_check",
]
