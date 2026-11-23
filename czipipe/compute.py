"""Optional GPU backend and the bit-depth conversion.

Worth being clear about scope: this job is dominated by CZI subblock decode and TIFF
tile deflate, both CPU. The rescale here is a handful of elementwise ops, so the GPU
path typically buys single-digit percent. `czipipe.bench` measures the three stages
separately so the choice can be made on evidence rather than hope.
"""

from __future__ import annotations

import threading

import numpy as np


def cast_to(arr, dtype):
    """Round, clip and cast float pixel values back to an integer output dtype.

    `astype` alone truncates. After a blend or a flat-field division, a value that should
    be exactly v often comes out a hair under it in float32 (v*w/w with w = 7/240, say),
    and truncation turns that into v-1 — on roughly 6% of feathered pixels — while
    biasing every genuinely fractional value down by half a step.
    """
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        return np.asarray(arr).astype(dtype, copy=False)
    if np.asarray(arr).dtype == dtype:
        return arr
    info = np.iinfo(dtype)
    out = np.rint(np.asarray(arr, np.float64 if dtype.itemsize >= 4 else np.float32))
    np.clip(out, info.min, info.max, out=out)
    return out.astype(dtype)


class Backend:
    """Holds the CuPy handle, if there is one, plus the lock guarding it."""

    def __init__(self, use_gpu=True, verbose=True):
        self.gpu = False
        self.cp = None
        self.name = "CPU"
        self._lock = threading.Lock()
        if not use_gpu:
            return
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                cp.zeros(1)                      # force context creation
                self.cp, self.gpu = cp, True
                self.name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
        except Exception as exc:
            if verbose:
                print(f"[gpu] unavailable, using CPU ({type(exc).__name__}: {exc})")

    def __repr__(self):
        return f"<Backend {'GPU: ' + self.name if self.gpu else 'CPU'}>"

    # -- conversion ---------------------------------------------------------
    def to_out_dtype(self, arr, lo, hi, src_dtype, out_dtype, method):
        """Cast a source tile to the output dtype, rescaling if 8-bit was requested."""
        if np.dtype(out_dtype) == np.dtype(src_dtype):
            return arr
        if method == "shift":
            return (arr >> 8).astype(np.uint8)
        scale = 255.0 / (hi - lo)
        if self.gpu:
            cp = self.cp
            with self._lock:
                g = cp.asarray(arr, dtype=cp.float32)
                g -= np.float32(lo)
                g *= np.float32(scale)
                cp.clip(g, 0, 255, out=g)
                return cp.asnumpy(g.astype(cp.uint8))
        g = arr.astype(np.float32)
        g -= np.float32(lo)
        g *= np.float32(scale)
        np.clip(g, 0, 255, out=g)
        return g.astype(np.uint8)

    def fft_phase_correlation(self, a, b):
        """Normalised cross-power spectrum; returns the correlation surface."""
        a = np.asarray(a, np.float32)
        b = np.asarray(b, np.float32)
        a = a - a.mean()
        b = b - b.mean()
        win = np.hanning(a.shape[0])[:, None] * np.hanning(a.shape[1])[None, :]
        if self.gpu:
            cp = self.cp
            with self._lock:
                ga, gb, gw = cp.asarray(a), cp.asarray(b), cp.asarray(win)
                R = cp.fft.rfft2(ga * gw) * cp.conj(cp.fft.rfft2(gb * gw))
                R /= cp.maximum(cp.abs(R), 1e-12)
                return cp.asnumpy(cp.fft.irfft2(R, s=a.shape))
        R = np.fft.rfft2(a * win) * np.conj(np.fft.rfft2(b * win))
        R /= np.maximum(np.abs(R), 1e-12)
        return np.fft.irfft2(R, s=a.shape)
