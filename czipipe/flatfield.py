"""Flat-field (shading) correction.

Feather blending turns a brightness step at a tile seam into a brightness gradient. It
does not remove vignetting, so uncorrected shading remains a spatial confound: the same
structure measures dimmer at a field edge than at a field centre. This module estimates
or loads a per-channel illumination profile and divides it out.

Model: `observed = profile * true + dark`, so `true = (observed - dark) / profile`.
The profile is normalised to a median of 1, which preserves the overall intensity scale
so corrected values stay comparable with uncorrected ones.

**Correction is per field of view, so it can only be applied where individual fields are
still visible.** On a mosaic that means the per-mosaic-tile path (`stitch_mode` of
`refine` or `bigstitcher`, or `metadata` with `flatfield.mode` set, which routes through
the same compositor). Once libCZI has blended the mosaic, the fields are gone and no
per-field correction is possible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------- smoothing
def _box1d(a, radius, axis):
    """Box filter via cumulative sums, with edge replication."""
    if radius < 1:
        return a
    a = np.moveaxis(a, axis, 0)
    pad = np.concatenate([np.repeat(a[:1], radius, 0), a,
                          np.repeat(a[-1:], radius, 0)], axis=0)
    cs = np.cumsum(pad, axis=0, dtype=np.float64)
    cs = np.concatenate([np.zeros((1,) + cs.shape[1:], cs.dtype), cs], axis=0)
    w = 2 * radius + 1
    out = (cs[w:] - cs[:-w]) / w
    return np.moveaxis(out.astype(np.float32), 0, axis)


def gaussian_blur(a, sigma, passes=3):
    """Gaussian approximation by repeated box filtering; no SciPy dependency.

    n passes of a box of width w give variance n*(w**2 - 1)/12.
    """
    if sigma <= 0:
        return a.astype(np.float32)
    w = max(3, int(round(np.sqrt(12.0 * sigma * sigma / passes + 1))))
    r = max(1, (w - 1) // 2)
    out = a.astype(np.float32)
    for _ in range(passes):
        out = _box1d(_box1d(out, r, 0), r, 1)
    return out


def resize_bilinear(a, shape):
    """Separable bilinear resize. The profile is smooth, so this is lossless in practice."""
    if a.shape == tuple(shape):
        return a.astype(np.float32)
    h, w = a.shape
    H, W = shape
    yi = np.linspace(0, h - 1, H)
    xi = np.linspace(0, w - 1, W)
    rows = np.empty((h, W), np.float32)
    src_x = np.arange(w)
    for i in range(h):
        rows[i] = np.interp(xi, src_x, a[i])
    out = np.empty((H, W), np.float32)
    src_y = np.arange(h)
    for j in range(W):
        out[:, j] = np.interp(yi, src_y, rows[:, j])
    return out


def poly_surface(a, degree, out_shape):
    """Least-squares 2D polynomial fit, evaluated on an arbitrary output grid.

    Preferred over blurring for illumination profiles. A local average of a curved
    surface is biased at the boundary — edge replication drags the corners of a vignette
    toward the brighter interior, which on a 45% vignette overestimated the corner gain
    by about 0.14. A global fit has no boundary to bias, denoises far harder, and can be
    evaluated directly at full resolution, so there is no resampling error either.
    """
    h, w = a.shape
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    terms = [(i, j) for d in range(degree + 1) for i in range(d + 1) for j in [d - i]]
    A = np.stack([(xx ** i * yy ** j).ravel() for i, j in terms], axis=1)
    coef, *_ = np.linalg.lstsq(A, np.asarray(a, np.float64).ravel(), rcond=None)

    H, W = out_shape
    oy, ox = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    out = np.zeros((H, W), np.float64)
    for c, (i, j) in zip(coef, terms):
        out += c * (ox ** i) * (oy ** j)
    return out.astype(np.float32)


# ---------------------------------------------------------------- model
@dataclass
class FlatField:
    """A per-channel correction: multiply by `gain` after subtracting `dark`."""

    profile: np.ndarray                  # normalised illumination profile (median 1)
    gain: np.ndarray                     # 1 / clipped profile, precomputed
    dark: Union[float, np.ndarray] = 0.0
    stats: dict = field(default_factory=dict)

    @property
    def shape(self):
        return self.profile.shape

    def apply(self, patch, ys=None, xs=None):
        """Correct a patch. `ys`/`xs` locate it within the field; None means the whole field."""
        g = self.gain if ys is None else self.gain[ys, xs]
        out = np.asarray(patch, np.float32)
        if isinstance(self.dark, np.ndarray):
            d = self.dark if ys is None else self.dark[ys, xs]
            out = out - d
        elif self.dark:
            out = out - np.float32(self.dark)
        else:
            out = out.copy()
        out *= g
        return out


def build(profile, dark=0.0, min_gain=0.2, stats=None):
    """Normalise a raw profile to median 1, floor it, and precompute the gain.

    The floor matters: an unclipped profile that approaches zero in a dark corner turns
    the correction into an arbitrarily large multiplier and amplifies read noise into
    something that looks like signal.
    """
    p = np.asarray(profile, np.float32)
    med = float(np.median(p))
    if not np.isfinite(med) or med <= 0:
        raise ValueError("flat-field profile has a non-positive median; check the input")
    p = p / med
    clipped = int(np.count_nonzero(p < min_gain))
    p = np.maximum(p, min_gain)
    s = {"min": float(p.min()), "max": float(p.max()),
         "vignetting_pct": float(100 * (1 - p.min() / p.max())),
         "pixels_clipped_to_min_gain": clipped}
    s.update(stats or {})
    return FlatField(profile=p, gain=(1.0 / p).astype(np.float32), dark=dark, stats=s)


# ---------------------------------------------------------------- estimation
def _profile_from_stack(stack, ffc, target_shape):
    """Per-pixel across-sample statistic, then a smooth surface at the field resolution."""
    raw = np.percentile(np.asarray(stack, np.float32), ffc.percentile, axis=0)
    if ffc.smooth_mode == "poly":
        return poly_surface(raw, ffc.poly_degree, target_shape)
    sigma = ffc.smooth_sigma * raw.shape[0] / max(target_shape[0], 1)
    return resize_bilinear(gaussian_blur(raw, sigma), target_shape)


def estimate_from_samples(samples, ffc, target_shape, dark=0.0):
    """Estimate a profile from a set of same-shaped fields.

    The median across many fields approximates the illumination profile provided sample
    content is uncorrelated with position *within* the field. That assumption is the
    weak point, so `split_half_disagreement_pct` reports how far two independent halves
    of the sample set disagree — a large value means the estimate is tracking sample
    structure rather than illumination, and should not be trusted.
    """
    if len(samples) < 4:
        raise ValueError(f"need at least 4 sample fields to estimate a profile, got "
                         f"{len(samples)}")
    stack = np.asarray(samples, np.float32)
    if dark:
        stack = stack - np.float32(dark) if not isinstance(dark, np.ndarray) else stack - dark

    prof = _profile_from_stack(stack, ffc, target_shape)

    a = _profile_from_stack(stack[0::2], ffc, target_shape)
    b = _profile_from_stack(stack[1::2], ffc, target_shape)
    # RMS, not max: the worst pixel is corner-dominated and noisy. Measured separation
    # on synthetic fields is 1.5-2.7% when content is uncorrelated with field position
    # (n=9 to 48) against ~98% when it is, so 10% discriminates with wide margin.
    a = a / np.median(a)
    b = b / np.median(b)
    disagree = float(100 * np.sqrt(((a - b) ** 2).mean()))

    return build(prof, dark=dark, min_gain=ffc.min_gain,
                 stats={"n_samples": len(samples), "method": "measured",
                        "percentile": ffc.percentile, "smooth": ffc.smooth_mode,
                        "split_half_disagreement_pct": disagree})


def load_from_file(path, channel_index, n_channels, target_shape, dark=0.0, min_gain=0.2):
    """Load a profile from a TIFF. Either one 2D image, or one plane per selected channel."""
    import tifffile

    arr = np.asarray(tifffile.imread(str(path)), np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 3:
        if arr.shape[0] != n_channels:
            raise ValueError(f"flat-field file {Path(path).name} has {arr.shape[0]} planes "
                             f"but {n_channels} channels are selected")
        arr = arr[channel_index]
    elif arr.ndim != 2:
        raise ValueError(f"flat-field file {Path(path).name} has shape {arr.shape}; "
                         "expected 2D, or 3D with one plane per channel")
    if arr.shape != tuple(target_shape):
        raise ValueError(f"flat-field file {Path(path).name} is {arr.shape} but the field "
                         f"is {tuple(target_shape)}; they must match exactly")
    return build(arr, dark=dark, min_gain=min_gain,
                 stats={"method": "file", "path": str(path)})


def estimate_basic(samples, ffc, target_shape):
    """Delegate to BaSiC, which also separates a darkfield term."""
    try:
        from basicpy import BaSiC
    except ImportError as exc:
        raise ImportError('flatfield.mode="basic" needs BaSiC: pip install basicpy\n'
                          'Use mode="measured" for the dependency-free estimator.') from exc
    stack = np.asarray(samples, np.float32)
    basic = BaSiC(get_darkfield=ffc.basic_darkfield)
    basic.fit(stack)
    prof = resize_bilinear(np.asarray(basic.flatfield, np.float32), target_shape)
    dark = 0.0
    if ffc.basic_darkfield and basic.darkfield is not None:
        dark = resize_bilinear(np.asarray(basic.darkfield, np.float32), target_shape)
    return build(prof, dark=dark, min_gain=ffc.min_gain,
                 stats={"method": "basic", "n_samples": len(samples),
                        "darkfield": bool(ffc.basic_darkfield)})


# ---------------------------------------------------------------- reporting
def format_report(channel, ff, warn_disagreement=10.0):
    s = ff.stats
    lines = [f"  C{channel}: {s.get('method')} profile, "
             f"gain range {1/ff.profile.max():.3f}-{1/ff.profile.min():.3f}, "
             f"vignetting {s['vignetting_pct']:.1f}%"]
    if s.get("pixels_clipped_to_min_gain"):
        lines.append(f"    [warn] {s['pixels_clipped_to_min_gain']} px hit min_gain; the "
                     "profile is darker than the floor there and stays under-corrected")
    d = s.get("split_half_disagreement_pct")
    if d is not None:
        note = "" if d < warn_disagreement else \
            "   [warn] halves disagree; the estimate may be tracking sample structure, " \
            "not illumination. Use more samples, or supply a measured flat-field file."
        lines.append(f"    split-half disagreement {d:.1f}% (RMS){note}")
    if s["vignetting_pct"] < 2.0:
        lines.append("    [note] under 2% shading; correction will change very little")
    return "\n".join(lines)
