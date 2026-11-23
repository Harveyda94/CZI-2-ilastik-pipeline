"""Reading side of the pipeline: CZI inspection, metadata, thread-local readers."""

from __future__ import annotations

import threading
import xml.etree.ElementTree as ET
from typing import Optional

import numpy as np

PIXEL_DTYPE = {
    "Gray8":       (np.uint8,   1),
    "Gray16":      (np.uint16,  1),
    "Gray32Float": (np.float32, 1),
    "Bgr24":       (np.uint8,   3),
    "Bgr48":       (np.uint16,  3),
}


def parse_scaling(root: ET.Element) -> dict:
    """Physical pixel size per axis in micrometres (CZI stores metres)."""
    out = {}
    for d in root.findall(".//Scaling/Items/Distance"):
        axis, val = d.get("Id"), d.findtext("Value")
        if axis and val:
            try:
                out[axis] = float(val) * 1e6
            except ValueError:
                pass
    return out


def parse_channels(root: ET.Element) -> list:
    info = []
    for ch in root.findall(".//Dimensions/Channels/Channel"):
        info.append({
            "name": ch.get("Name") or ch.get("Id") or f"C{len(info)}",
            "dye": ch.findtext("DyeName"),
            "ex": ch.findtext("ExcitationWavelength"),
            "em": ch.findtext("EmissionWavelength"),
        })
    return info


class Source:
    """An open CZI, its metadata, and the resolved subset to convert.

    Readers are thread-local: libCZI decodes in native code and releases the GIL, so
    read threads scale, but rather than assume one reader object is safe to hammer from
    many threads each worker gets its own handle. The OS page cache makes this free.
    """

    def __init__(self, cfg, log=print):
        from pylibCZIrw import czi as pyczi

        self.cfg = cfg
        self._pyczi = pyczi
        self._tls = threading.local()
        self._open = []
        self._lock = threading.Lock()
        self._aics_cls = None

        with pyczi.open_czi(str(cfg.czi_path)) as doc:
            self.total_bbox = dict(doc.total_bounding_box)
            self.total_rect = tuple(doc.total_bounding_rectangle)
            self.scene_rects = {int(k): tuple(v)
                                for k, v in doc.scenes_bounding_rectangle.items()}
            self.pixel_types = dict(doc.pixel_types)
            try:
                self.raw_xml = doc.raw_metadata
            except Exception as exc:
                # Seen on real files from at least one acquisition setup: pylibCZIrw's
                # GetXmlMetadata() chokes on a "DELETED" tombstone segment sitting where
                # it expects the metadata segment, even though everything else needed for
                # actual pixel data (scenes, pixel types, subblocks) reads back fine. Losing
                # physical scaling/channel names/embedded provenance is real, but it beats
                # refusing to convert a file whose pixel data is otherwise intact.
                log(f"[warn] could not read raw CZI XML metadata ({type(exc).__name__}: {exc}); "
                    "continuing without it. This means: no physical pixel size (scaling), no "
                    "channel names from the CZI (falling back to C0/C1/...), and no original "
                    "metadata embedded in the output. Pixel data conversion is unaffected.")
                self.raw_xml = None

        types = set(self.pixel_types.values())
        if len(types) != 1:
            raise NotImplementedError(f"Mixed pixel types across channels: {self.pixel_types}")
        self.pixel_type = types.pop()
        if self.pixel_type not in PIXEL_DTYPE:
            raise NotImplementedError(f"Unsupported CZI pixel type {self.pixel_type!r}")
        self.dtype, self.samples = PIXEL_DTYPE[self.pixel_type]

        if self.raw_xml:
            root = ET.fromstring(self.raw_xml)
            self.scaling = parse_scaling(root)
            self.channel_info = parse_channels(root)
        else:
            self.scaling = {}
            self.channel_info = []
        if cfg.manual_scaling:
            log(f"[info] overriding scaling {[a for a in cfg.manual_scaling]} from "
                f"cfg.manual_scaling: {cfg.manual_scaling}")
            self.scaling.update({k: float(v) for k, v in cfg.manual_scaling.items()})

        self.c0, c1 = self._span("C")
        self.z0, z1 = self._span("Z")
        self.t0, t1 = self._span("T")
        while len(self.channel_info) < c1 - self.c0:
            self.channel_info.append({"name": f"C{len(self.channel_info)}",
                                      "dye": None, "ex": None, "em": None})

        self.sel_c = self._resolve("C", cfg.channels, self.c0, c1, is_range=False)
        self.sel_z = self._resolve("Z", cfg.z_range, self.z0, z1, is_range=True)
        self.sel_t = self._resolve("T", cfg.t_range, self.t0, t1, is_range=True)

        self.out_dtype = np.uint8 if cfg.output_bit_depth == 8 else self.dtype
        self.h5_axes = "tzcyxs" if self.samples == 3 else "tzcyx"
        self.scene_arg_ok = True

    # -- construction helpers ----------------------------------------------
    def _span(self, key, default=(0, 1)):
        v = self.total_bbox.get(key, default)
        return int(v[0]), int(v[1])

    @staticmethod
    def _resolve(name, spec, lo, hi, is_range):
        if spec is None:
            sel = list(range(lo, hi))
        elif is_range:
            sel = list(range(*spec))
        else:
            sel = [int(v) for v in spec]
        bad = [i for i in sel if not lo <= i < hi]
        if bad:
            raise ValueError(f"{name} indices {bad} outside file range [{lo}, {hi})")
        if not sel:
            raise ValueError(f"{name} selection is empty")
        return sel

    # -- public API ---------------------------------------------------------
    @property
    def jobs(self):
        """[(scene_id, rect)] — one entry per output file."""
        if self.scene_rects and self.cfg.split_scenes:
            ids = sorted(self.scene_rects) if self.cfg.scenes is None else \
                [int(s) for s in self.cfg.scenes]
            missing = [s for s in ids if s not in self.scene_rects]
            if missing:
                raise ValueError(f"scenes {missing} not in file (have {sorted(self.scene_rects)})")
            return [(s, self.scene_rects[s]) for s in ids]
        return [(None, self.total_rect)]

    @property
    def channel_names(self):
        return [self.channel_info[c - self.c0]["name"] for c in self.sel_c]

    def reader(self):
        """This thread's CZI reader, opened on first use."""
        doc = getattr(self._tls, "doc", None)
        if doc is None:
            cm = self._pyczi.open_czi(str(self.cfg.czi_path))
            doc = cm.__enter__()
            self._tls.doc = doc
            with self._lock:
                self._open.append((cm, doc))
        return doc

    def aics(self):
        """Thread-local aicspylibczi handle, for per-mosaic-tile access."""
        if self._aics_cls is None:
            try:
                from aicspylibczi import CziFile
            except ImportError as exc:
                raise ImportError(
                    f'stitch_mode="{self.cfg.stitch_mode}" needs per-mosaic-tile access: '
                    "pip install aicspylibczi\n"
                    'Use stitch_mode="metadata" for libCZI stage-position compositing.'
                ) from exc
            self._aics_cls = CziFile
        h = getattr(self._tls, "aics", None)
        if h is None:
            h = self._aics_cls(str(self.cfg.czi_path))
            self._tls.aics = h
        return h

    def probe_scene_arg(self, jobs, plane):
        """Does this pylibCZIrw build accept roi and scene together?

        The catch is deliberately narrow. A blanket `except Exception` here blames
        `scene=` for any failure in the call, including ones caused by the other
        arguments -- which is exactly how a `background_pixel` TypeError once got
        misreported as "scene= argument rejected", silently disabling scene-restricted
        reads for the whole run. Anything that is not plausibly about `scene` is
        re-raised so it surfaces where it actually happened.
        """
        sid, rect = jobs[0]
        if sid is None:
            return True
        try:
            self.reader().read(roi=(rect[0], rect[1], 8, 8), plane=plane, scene=sid)
        except TypeError as exc:
            if "scene" not in str(exc).lower():
                raise
            self.scene_arg_ok = False
        except (ValueError, KeyError, RuntimeError) as exc:
            if "scene" not in str(exc).lower():
                raise
            self.scene_arg_ok = False
        return self.scene_arg_ok

    def read_kwargs(self, scene_id):
        return {"scene": scene_id} if (scene_id is not None and self.scene_arg_ok) else {}

    def close(self):
        with self._lock:
            while self._open:
                cm, _ = self._open.pop()
                try:
                    cm.__exit__(None, None, None)
                except Exception:
                    pass
        self._tls = threading.local()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    # -- reporting ----------------------------------------------------------
    def describe(self) -> str:
        lines = [
            f"pixel type   {self.pixel_type} -> numpy {np.dtype(self.dtype).name}"
            + (" x3 samples" if self.samples == 3 else ""),
            f"bounding box {self.total_bbox}",
            f"total rect   x={self.total_rect[0]} y={self.total_rect[1]} "
            f"w={self.total_rect[2]} h={self.total_rect[3]}",
            f"scenes       {len(self.scene_rects) or 'none'}",
            f"scaling um   {self.scaling or 'not present in metadata'}",
            f"channels     " + ", ".join(f"{c}:{n}" for c, n in
                                         zip(self.sel_c, self.channel_names)),
            f"selection    T={len(self.sel_t)}  Z={len(self.sel_z)}  C={len(self.sel_c)}",
        ]
        nbytes = np.dtype(self.out_dtype).itemsize * self.samples
        for sid, rect in self.jobs:
            n = rect[2] * rect[3] * len(self.sel_t) * len(self.sel_z) * len(self.sel_c)
            lines.append(f"  scene {sid if sid is not None else '-'}: {rect[2]} x {rect[3]} px"
                         f" -> {n * nbytes / 1e9:.1f} GB uncompressed")
        return "\n".join(lines)
