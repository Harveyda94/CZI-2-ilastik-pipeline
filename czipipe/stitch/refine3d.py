"""3D translation registration: an XY and a Z correction per mosaic tile.

`refine` measures an XY shift per pair one plane at a time and never looks along Z, so a
tile whose stack sits a few planes above or below its neighbour's is blended against the
wrong planes however good its XY fit. This is the alternative, `refine.mode = "3d"`: each
overlapping pair is phase-correlated as a volume, over a run of consecutive Z planes (a
slab) of the overlap, which gives (dy, dx, dz) in one measurement.

Phase correlation only proposes. Its strongest peaks, and the stage position itself, are
scored by the NCC of the two overlap volumes, and the best of them is walked a step at a
time to where the match stops improving, so the shift kept is one that no neighbouring
step beats. A proposal of the wrong sign, or nowhere near, is thereby walked to the true
shift or rejected; it is never accepted as proposed.

The 2D guard applies here too: a shift is kept only if it matches measurably better than
the stage positions do. The XY and Z parts are each held to that standard on their own,
because getting one right raises the NCC enough to carry the other past a test against
the stage positions alone, wrong sign included: a Z offset is kept only if it matches
measurably better than the same XY shift without it, and an XY shift only if it beats
the same Z offset without it. A pair the stage already has right says so too: where the
best match is at no shift at all, that is kept as a (weak) constraint rather than thrown
away.

It keeps to the run's settings. A measurement needs only the overlap of its two tiles,
over one slab, and pairs are measured in batches sized so that their volumes and the
working memory together fit `mosaic_cache_mb`. Tiles are read, pairs measured and the
volume transform run on the run's read threads and no others, one transform at a time.

Tiles are placed by one least-squares fit over the accepted pairs, from which pairs that
plainly contradict the rest are left out, and named in the report.

It is still one translation per tile. Several slabs are measured so that one featureless
depth does not lose a pair, and the report says where slabs disagree, but a tile that is
tilted or sheared against its neighbour is reported, not corrected.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from ..compute import VOLUME_BYTES_PER_VOXEL
from .refine import choose, overlap_slices

PEAKS = 5             # distinct correlation peaks tried per measurement, strongest first
SUPPRESS = 2          # voxels either side of a peak in which the next is not sought
CLIMB = 16            # steps a proposal may be walked towards a better match
STEPS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
SCORE_VOXELS = 1_000_000    # an NCC over this many samples is settled; a larger overlap
                            # is scored on a lattice holding at least this many
AGREE = 1             # shifts are integers, so a true half-step reads as either side
OUTLIER_PX = 3.5      # a pair this far in XY from where the others put its two tiles,
OUTLIER_PLANES = 1.5  # or this far along Z, is a candidate for leaving out of the fit
OUTLIER_RATIO = 4.0   # if it is also this many times further out than the typical pair
MIN_DEPTH = 3         # planes in a slab; fewer cannot tell a Z offset from none
ZERO = (0, 0, 0)


# ---------------------------------------------------------------- where to measure
def sample_slabs(sel_z, n, depth, centres=None, bounds=None):
    """Runs of `depth` consecutive Z planes to correlate as volumes.

    `n` runs, each centred in one of `n` equal parts of the selection, so they never
    overlap and an odd `n` includes the middle of the stack. A shallow selection gets as
    many as fit, down to a single run holding all of it. Or, with `centres`, one run
    around each of those planes, pulled inside `bounds` (lo, hi: the file's own Z extent)
    rather than cut short.

    Consecutive means consecutive in the file, whatever step the selection takes: a Z
    offset is measured, and later applied, in file planes.
    """
    lo, hi = min(sel_z), max(sel_z) + 1
    if centres is not None:
        if bounds is not None:
            lo, hi = min(lo, bounds[0]), max(hi, bounds[1])
        depth = min(int(depth), hi - lo)
        starts = [min(max(int(c) - depth // 2, lo), hi - depth) for c in centres]
    else:
        depth = min(int(depth), hi - lo)
        n = max(1, min(int(n), (hi - lo) // max(depth, 1)))
        starts = [min(max(lo + int((k + 0.5) * (hi - lo) / n) - depth // 2, lo), hi - depth)
                  for k in range(n)]
    if depth < MIN_DEPTH:
        raise ValueError(
            f'refine.mode="3d" needs at least {MIN_DEPTH} consecutive Z planes to correlate '
            f"(refine.slab), and has {depth}. Widen the Z selection, or use "
            'refine.mode="2d".')
    return [list(range(s, s + depth)) for s in dict.fromkeys(starts)]


def candidates(rects, min_overlap, pad):
    """Overlapping pairs worth measuring, and the piece of each tile a measurement needs.

    The piece is the overlap grown by `pad` (max_shift) and clipped to the tile. A shift
    of up to `pad` can only extend the overlap into that margin, so scoring it on the two
    pieces gives exactly what scoring it on the whole tiles would, without holding whole
    tiles through a slab.
    """
    ms = sorted(rects)
    out = []
    for i, mi in enumerate(ms):
        for mj in ms[i + 1:]:
            ra, rb = rects[mi], rects[mj]
            x0, x1 = max(ra[0], rb[0]), min(ra[0] + ra[2], rb[0] + rb[2])
            y0, y1 = max(ra[1], rb[1]), min(ra[1] + ra[3], rb[1] + rb[3])
            if x1 - x0 < 1 or y1 - y0 < 1 or min(x1 - x0, y1 - y0) < min_overlap:
                continue
            kept, cuts = [], []
            for x, y, w, h in (ra, rb):
                kx0, kx1 = max(x, x0 - pad), min(x + w, x1 + pad)
                ky0, ky1 = max(y, y0 - pad), min(y + h, y1 + pad)
                kept.append((kx0, ky0, kx1 - kx0, ky1 - ky0))
                cuts.append((slice(ky0 - y, ky1 - y), slice(kx0 - x, kx1 - x)))
            out.append({"i": mi, "j": mj, "overlap": [int(y1 - y0), int(x1 - x0)],
                        "rects": kept, "cut": cuts, "least": min_overlap,
                        "core": overlap_slices(kept[0], kept[1]),   # the overlap itself
                        "area": sum(k[2] * k[3] for k in kept), "obs": []})
    return out


def batches(cands, depth, itemsize, budget):
    """Split the pairs into runs whose volumes fit `budget` bytes together.

    Never fewer than one pair to a run: a single pair larger than the budget is still
    measured, on its own.
    """
    out, run, used = [], [], 0
    for c in cands:
        need = c["area"] * depth * itemsize
        if run and used + need > budget:
            out.append(run)
            run, used = [], 0
        run.append(c)
        used += need
    if run:
        out.append(run)
    return out


def load(read, slab, batch, dtype, pool=None, progress=None):
    """[(va, vb)] per pair of `batch`: its two pieces as (Z, Y, X) volumes over `slab`.

    Each tile the batch touches is read once per plane and cut for every pair it is in,
    so a whole tile is held only while it is being cut. With `pool` the reads run on its
    threads; every (plane, tile) fills slots of its own, so they need no ordering.
    """
    uses, vols = {}, []
    for n, c in enumerate(batch):
        vols.append([np.empty((len(slab), r[3], r[2]), dtype) for r in c["rects"]])
        for side, m in enumerate((c["i"], c["j"])):
            uses.setdefault(m, []).append((n, side, c["cut"][side]))

    def fetch(job):
        k, z, m = job
        tile = read(z, m)
        for n, side, cut in uses[m]:
            vols[n][side][k] = tile[cut]

    todo = [(k, z, m) for k, z in enumerate(slab) for m in sorted(uses)]
    done = pool.map(fetch, todo) if pool else map(fetch, todo)
    for _ in (progress(done, len(todo)) if progress else done):
        pass
    return vols


# ---------------------------------------------------------------- one measurement
def peaks(backend, a, b, n=PEAKS, workers=1):
    """Up to `n` distinct phase-correlation peaks as shifts, strongest first, and a
    peak-sharpness score. The transform runs on `workers` threads.

    A shift s = (dy, dx, dz) means b[k, i, j] ~ a[k + dz, i + dy, j + dx]: it is how far
    tile b has to move, in pixels and planes, to line up with tile a. XY first, so that
    the first two components mean what a 2D correction's two do.

    Distinct, because each peak blanks the voxels within SUPPRESS of it before the next
    is sought: the list is `n` different hypotheses, not one peak and its shoulders.
    Where exactly a peak's best match lies is settled by NCC, in `measure`.
    """
    cc = backend.fft_phase_correlation(a, b, workers=workers)
    floor = float(np.abs(cc).mean())
    out, top = [], None
    for _ in range(n):
        idx = np.unravel_index(int(cc.argmax()), cc.shape)
        value = float(cc[idx])
        if not np.isfinite(value):
            break                                        # nothing left that is not blanked
        top = value if top is None else top
        sz, sy, sx = (int(v) - size if v > size // 2 else int(v)
                      for v, size in zip(idx, cc.shape))
        out.append((sy, sx, sz))
        cc[np.ix_(*(np.arange(v - SUPPRESS, v + SUPPRESS + 1) % size
                    for v, size in zip(idx, cc.shape)))] = -np.inf
    return out, top / (floor + 1e-12)


def score_step(voxels):
    """The XY stride an overlap of `voxels` is scored at.

    The largest power of two (8 at most) that still leaves SCORE_VOXELS samples. An NCC
    over a million samples does not move in the third decimal place when given four
    million, and the scoring is most of the work of a measurement.
    """
    step = 1
    while step < 8 and voxels // (4 * step * step) >= SCORE_VOXELS:
        step *= 2
    return step


def volume_ncc(a, b):
    """NCC of two volumes of one shape, summed a plane at a time.

    Copying both volumes whole to subtract their means, as `refine.ncc` does for a plane,
    costs tens of MB a call for a slab, and with a thread to a pair, per thread. Here the
    sums are built up plane by plane in double precision, so a call holds two planes of
    what it is scoring. No BLAS: its thread pool would be one more that no setting sizes.
    """
    n = 0
    sa = sb = saa = sbb = sab = 0.0
    for pa, pb in zip(a, b):
        pa = np.ascontiguousarray(pa, np.float64).ravel()
        pb = np.ascontiguousarray(pb, np.float64).ravel()
        n += pa.size
        sa += float(pa.sum())
        sb += float(pb.sum())
        saa += float(np.einsum("i,i->", pa, pa))
        sbb += float(np.einsum("i,i->", pb, pb))
        sab += float(np.einsum("i,i->", pa, pb))
    if not n:
        return 0.0
    var = (saa - sa * sa / n) * (sbb - sb * sb / n)
    return (sab - sa * sb / n) / var ** 0.5 if var > 1e-9 else 0.0


def score(c, va, vb, s):
    """NCC of the pair's volumes with tile j moved by s, over everything they then share.

    Zero where that is less than min_overlap across: an overlap too small to attempt is
    too small to score, and a sliver of one can match anything.
    """
    sl = overlap_slices(c["rects"][0], c["rects"][1], s[:2])
    lo, hi = max(0, -s[2]), va.shape[0] - max(0, s[2])
    if sl is None or hi <= lo:
        return 0.0
    (ya, xa), (yb, xb) = sl
    if min(ya.stop - ya.start, xa.stop - xa.start) < c.get("least", 1):
        return 0.0
    step = c.get("step", 1)
    if step > 1:
        # A lattice fixed in tile i, so that two shifts are compared on the same voxels
        # of it wherever their overlaps coincide.
        oy, ox = -ya.start % step, -xa.start % step
        if ya.start + oy >= ya.stop or xa.start + ox >= xa.stop:
            return 0.0
        ya, yb = slice(ya.start + oy, ya.stop, step), slice(yb.start + oy, yb.stop, step)
        xa, xb = slice(xa.start + ox, xa.stop, step), slice(xb.start + ox, xb.stop, step)
    return volume_ncc(va[slice(lo + s[2], hi + s[2]), ya, xa], vb[slice(lo, hi), yb, xb])


def measure(backend, rc, max_z, c, va, vb, workers=1):
    """One pair in one slab: the shift it supports, and how that scores against metadata.

    Phase correlation only proposes, and NCC decides. The best-matching proposal (one of
    the correlation peaks, or the stage position) is walked a step at a time to where the
    match stops improving, so the shift kept is one that no neighbouring step beats. That
    settles a peak split between neighbouring pixels or planes, which under noise phase
    correlation gets wrong more often than not. It also means an offset beyond the search
    range walks out of it and is rejected, where taking the strongest peak inside the
    range would accept it a step short, at the boundary. A walk still improving after
    CLIMB steps started nowhere near a match, and is rejected too rather than kept
    wherever it had got to.

    `reason` is empty when the slab is usable: either that shift matches well enough and
    measurably better than the stage position, or the walk ends on the stage position
    (or a step from it, for no measurable gain) and that itself matches well: a shift of
    zero. `at` holds this slab's NCC at every shift scored on it, and `dropped` the part
    of the shift (its Z offset, or its XY part) that did not match measurably better
    than none.
    """
    whole = (slice(None),)
    found, sharp = peaks(backend, va[whole + c["core"][0]], vb[whole + c["core"][1]],
                         workers=workers)
    at = {}

    def ncc_at(p):
        if p not in at:
            at[p] = score(c, va, vb, p)
        return at[p]

    def within(p):
        return max(abs(p[0]), abs(p[1])) <= rc.max_shift and abs(p[2]) <= max_z

    def beyond(p):
        return (f"shift exceeds max_shift ({rc.max_shift} px)"
                if max(abs(p[0]), abs(p[1])) > rc.max_shift
                else f"Z shift exceeds the +/-{max_z} plane(s) searched "
                     "(refine.max_shift_z, and no more than a slab can vouch for)")

    # The stage position is a proposal too. Where it already matches better than any
    # peak, the walk starts from it, and finds a small real offset without the peaks'
    # help; where the walk stays on it, the pair confirms the stage.
    base = ncc_at(ZERO)
    s = max([p for p in found if within(p)] + [ZERO], key=ncc_at)     # ties: peaks first
    settled = False
    for _ in range(CLIMB):
        step = max(((s[0] + d[0], s[1] + d[1], s[2] + d[2]) for d in STEPS), key=ncc_at)
        if ncc_at(step) <= ncc_at(s):
            settled = True
            break
        s = step
        if not within(s):
            break

    dropped, got, best = None, None, s
    if not within(s):
        s, reason = None, beyond(s)
    elif not settled:
        got = at[s]                              # still improving: nowhere near a match
        reason = f"no settled match within {CLIMB} steps of any proposal"
    else:
        if s[2] and (s[0] or s[1]):
            # Each part of the shift has to earn its place against the same shift without
            # it. Scored against the stage positions alone, getting XY right would carry
            # any Z offset through with it, the wrong way included, and the other way
            # about.
            flat, lift = (s[0], s[1], 0), (0, 0, s[2])
            if ncc_at(s) < max(ncc_at(flat), ncc_at(lift)) + rc.min_gain:
                dropped = ("z", s[2]) if at[flat] >= at[lift] else ("xy", list(s[:2]))
                s = flat if at[flat] >= at[lift] else lift
        got = at[s]
        if s != ZERO and got >= rc.min_ncc and got >= base + rc.min_gain:
            reason = ""
        elif base >= rc.min_ncc and _apart(best, ZERO) <= AGREE:
            s, got, reason = ZERO, base, ""      # the best match is where the stage put it
        elif not within(found[0]):
            reason = beyond(found[0])            # the peak that mattered was out of range
        elif got < rc.min_ncc:
            reason = f"ncc {got:.3f} < min_ncc"
        else:
            reason = f"no gain over metadata ({got:.3f} vs {base:.3f})"
    return {"shift": s, "sharpness": sharp, "base": base, "got": got, "at": at,
            "dropped": dropped, "reason": reason, "status": "unusable"}


# ---------------------------------------------------------------- combining
def _apart(a, b):
    return max(abs(p - q) for p, q in zip(a, b))


def _groups(ms, edges):
    """Connected groups of tiles, as lists of indices into `ms`."""
    idx = {m: k for k, m in enumerate(ms)}
    parent = list(range(len(ms)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    for mi, mj, _, _ in edges:
        parent[find(idx[mi])] = find(idx[mj])
    groups = {}
    for k in range(len(ms)):
        groups.setdefault(find(k), []).append(k)
    return [g for g in groups.values() if len(g) > 1]


def _lower_median(values):
    return sorted(int(v) for v in values)[(len(values) - 1) // 2]


def _fit(A, rhs, wts, keep):
    """Weighted least squares on the pairs in `keep`: the solution, every pair's residual,
    and how far each kept pair is from where the *other* pairs put its two tiles, in
    units of the outlier thresholds.

    That last is the residual divided by (1 - leverage): what the residual would be had
    the pair not been in the fit pulling it towards itself. A pair that is a tile's only
    link has leverage 1 and nothing to be compared with, so it counts as no distance.
    """
    w = (wts * keep)[:, None]
    pinv = np.linalg.pinv(A * w)
    x = pinv @ (rhs * w)
    r = A @ x - rhs
    free = 1.0 - np.einsum("ij,ji->i", A * w, pinv)
    apart = np.abs(r) / np.where(free > 1e-9, free, np.inf)[:, None]
    far = np.maximum(apart[:, :2].max(axis=1) / OUTLIER_PX, apart[:, 2] / OUTLIER_PLANES)
    return x, r, np.where(keep, far, 0.0)


def _two_ways(links, a, b):
    """Are tiles `a` and `b` joined by two routes through `links` that share no pair?

    If they are, no single pair holds them together, and (Menger) the converse: so it
    is enough to find one route and see whether losing any one pair on it parts them.
    """
    def route(without=None):
        seen, queue = {a: None}, [a]
        for node in queue:
            for k, (i, j) in enumerate(links):
                if k == without or node not in (i, j):
                    continue
                other = j if node == i else i
                if other not in seen:
                    seen[other] = (node, k)
                    queue.append(other)
        if b not in seen:
            return None
        path, node = [], b
        while seen[node] is not None:
            node, k = seen[node]
            path.append(k)
        return path

    first = route()
    return first is not None and all(route(k) is not None for k in first)


def solve(ms, edges):
    """Weighted least squares over the tile graph: (dy, dx, dz) per tile.

    All constraints at once, so error does not compound across the mosaic. Least squares
    on its own shares a wrong measurement out among every tile it touches, however many
    other pairs agree with each other, so pairs that are plainly wrong are left out and
    the fit redone. A pair is left out, the furthest first, when all of these hold:

    - it is more than OUTLIER_PX, or OUTLIER_PLANES along Z, from where the other pairs
      put its two tiles;
    - without it those two tiles are still joined by two separate routes, so that what
      contradicts it is more than one other measurement's word against its own;
    - it is more than OUTLIER_RATIO times as far out as the typical pair is once it has
      gone. That is what stops a mosaic whose pairs all disagree a little, as when the
      tissue moved between tiles, from being whittled down to the few that happen to
      agree.

    What no measurement can fix is where each connected group of tiles sits as a whole:

    - In XY the group holding the first tile is pinned there, as in 2D. Any other group
      keeps its median tile where the stage put it.
    - Along Z every group keeps its median tile on its own planes. The canvas grows to
      hold an XY correction, but the output has the planes that were selected and no
      others, so the choice that moves the fewest tiles along Z also loses the least.

    Tiles with no accepted pair keep their metadata position. Returns the corrections,
    the RMS residual of the pairs in the fit in XY (px) and in Z (planes), the
    unconstrained tiles, every pair's distance from the fit as (dy, dx, dz), and the
    indices of the pairs left out of it.
    """
    if not edges:
        return {m: (0, 0, 0) for m in ms}, None, None, list(ms), [], []
    idx = {m: k for k, m in enumerate(ms)}
    A = np.zeros((len(edges), len(ms)), np.float64)
    rhs = np.zeros((len(edges), 3), np.float64)
    wts = np.ones(len(edges))
    for k, (mi, mj, s, w) in enumerate(edges):
        A[k, idx[mi]], A[k, idx[mj]] = -1.0, 1.0
        rhs[k] = s
        wts[k] = w

    keep = np.ones(len(edges), bool)
    x, r, far = _fit(A, rhs, wts, keep)
    while True:
        for worst in np.argsort(-far):
            if far[worst] <= 1.0:
                worst = None                     # nothing left that is far out
                break
            rest = keep.copy()
            rest[worst] = False
            links = [e[:2] for e, kept in zip(edges, rest) if kept]
            if not _two_ways(links, *edges[worst][:2]):
                continue
            trial = _fit(A, rhs, wts, rest)
            if far[worst] > OUTLIER_RATIO * float(np.median(trial[2][rest])):
                break
        else:
            worst = None
        if worst is None:
            break
        keep, (x, r, far) = rest, trial

    sol = np.zeros((len(ms), 3), int)
    for g in _groups(ms, edges):
        d = np.rint(x[g] - x[g[0]]).astype(int)
        if 0 in g:
            d[:, :2] -= d[g.index(0), :2].copy()
        else:
            d[:, :2] -= [_lower_median(d[:, 0]), _lower_median(d[:, 1])]
        d[:, 2] -= _lower_median(d[:, 2])
        sol[g] = d

    r = A @ sol - rhs
    linked = {m for e in edges for m in e[:2]}
    return ({m: tuple(int(v) for v in sol[idx[m]]) for m in ms},
            float(np.sqrt((r[keep, :2] ** 2).sum(axis=1).mean())),
            float(np.sqrt((r[keep, 2] ** 2).mean())),
            [m for m in ms if m not in linked],
            [tuple(int(v) for v in row) for row in np.rint(r)],
            [int(k) for k in np.flatnonzero(~keep)])


def incomplete_planes(sel_z, corr, z_range):
    """Selected planes at which some tile has nothing to give.

    A tile corrected by dz is taken from plane z - dz. Where the file has no such plane
    the compositor leaves the tile out, so those output planes have tile-shaped gaps.
    """
    shifts = {c[2] for c in corr.values() if c[2]}
    if z_range is None or not shifts:
        return []
    return [int(z) for z in sel_z
            if any(not z_range[0] <= z - dz < z_range[1] for dz in shifts)]


# ---------------------------------------------------------------- threads and memory
def working_plan(cands, tile_bytes, depth, itemsize, threads, budget):
    """How many threads to use, and how many bytes that leaves for overlap volumes.

    Registration has `budget` bytes in all (`mosaic_cache_mb`: the compositor's cache,
    which is empty until registration is over). Three things draw on it besides the
    volumes themselves:

    - whole tiles while they are read and cut, one per thread;
    - one volume transform, VOLUME_BYTES_PER_VOXEL for each voxel of the largest overlap;
    - two planes of what is being scored, in double precision, per thread.

    The first is capped at half the budget by using fewer threads than the run has, which
    only matters for very large tiles. Reading and measuring do not happen at once, so
    the working memory is the larger of the two, and the volumes get what remains. If
    that is less than one pair's volumes the pair is still measured, on its own: then the
    registration needs more than the budget, and the report says how much.
    """
    threads = max(1, min(int(threads), int(budget / 2 // max(tile_bytes, 1)) or 1))
    core = max((c["overlap"][0] * c["overlap"][1] for c in cands), default=0) * depth
    plane = max((max(r[2] * r[3] for r in c["rects"]) // c.get("step", 1) ** 2
                 for c in cands), default=0)
    working = max(threads * tile_bytes,
                  VOLUME_BYTES_PER_VOXEL * core + threads * 2 * 8 * plane)
    room = max(budget - working, 0)
    largest = max((sum(c["area"] for c in run) * depth * itemsize
                   for run in batches(cands, depth, itemsize, room)), default=0)
    return threads, room, {"budget_mb": round(budget / 1e6), "threads": threads,
                           "working_mb": round(working / 1e6),
                           "volumes_mb": round(largest / 1e6),
                           "peak_mb": round((working + largest) / 1e6),
                           "batches_per_slab": len(batches(cands, depth, itemsize, room))}


# ---------------------------------------------------------------- the registration
def register(source, cfg, backend, scene_id, rects, ref_plane, progress=None, slabs=None):
    """Per-tile integer corrections {m: (dy, dx, dz)} plus a diagnostics record.

    Each overlapping pair is measured independently in every slab (`slabs`, or those
    `refine.planes` / `refine.z` and `refine.slab` ask for), and the slabs are then
    combined per pair, so a pair needs only one slab with usable structure. Slabs that
    fit a different shift from the one chosen are reported rather than acted on.
    """
    rc = cfg.refine
    czi = source.aics()
    dims = czi.get_dims_shape()[0]
    if slabs is None:
        slabs = sample_slabs(source.sel_z, rc.planes, rc.slab, rc.z_planes(), dims.get("Z"))
    slabs = [[int(z) for z in s] for s in slabs]
    depth = min(len(s) for s in slabs)
    if depth < MIN_DEPTH:
        raise ValueError(f"3D registration needs slabs of at least {MIN_DEPTH} planes, "
                         f"got {depth}")
    max_z = min(int(rc.max_shift_z), (depth - 1) // 2)

    kw = {k: v for k, v in ref_plane.items() if k != "Z"}
    if scene_id is not None and "S" in dims:
        kw["S"] = scene_id

    def read(z, m):
        # source.aics() is per thread, which is what lets the reads run on several
        return np.squeeze(source.aics().read_image(M=m, Z=z, **kw)[0])

    ms = sorted(rects)
    cands = candidates(rects, rc.min_overlap, rc.max_shift)
    dtype = np.dtype(source.dtype)
    for c in cands:
        c["step"] = score_step(c["overlap"][0] * c["overlap"][1] * depth)
        c["proposed"] = []                       # distinct shifts, in the order proposed
    tile_bytes = max(r[2] * r[3] for r in rects.values()) * dtype.itemsize
    threads, room, memory = working_plan(cands, tile_bytes, depth, dtype.itemsize,
                                         cfg.read_threads, cfg.mosaic_cache_mb * 1e6)

    def fill(c, o, va, vb):
        """Score, in this slab, every shift the pair's slabs have proposed so far."""
        for s in c["proposed"]:
            if s not in o["at"]:
                o["at"][s] = score(c, va, vb, s)

    def first_look(k):
        def job(c, va, vb):
            o = measure(backend, rc, max_z, c, va, vb, workers=threads)
            o["slab"] = k
            c["obs"].append(o)
            if not o["reason"]:
                if o["shift"] not in c["proposed"]:
                    c["proposed"].append(o["shift"])
                fill(c, o, va, vb)
        return job

    # Tiles are read, and pairs measured, on the run's read threads (fewer if their
    # tiles would not fit: see working_plan), and the transform is given the same number.
    # A pair is only ever touched by the one thread measuring it, and slabs are taken in
    # turn, so the result does not depend on how the threads are scheduled.
    with ThreadPoolExecutor(threads) as pool:
        def each(k, group, job):
            """job(pair, va, vb) for every pair of `group` on its volumes in slab k,
            a batch that fits the memory budget at a time."""
            bar = None
            if progress:
                def bar(jobs, total):
                    return progress(jobs, total=total, leave=False,
                                    desc=f"register Z {slabs[k][0]}-{slabs[k][-1]}")
            for batch in batches(group, len(slabs[k]), dtype.itemsize, room):
                vols = load(read, slabs[k], batch, dtype, pool, bar)
                list(pool.map(lambda cv: job(cv[0], *cv[1]), zip(batch, vols)))

        for k in range(len(slabs)):
            each(k, cands, first_look(k))

        # The choice between a pair's proposals rests on how well each fits every usable
        # slab, so each proposal has to be scored in each of them. A slab has already
        # scored the steps around its own shift and whatever earlier slabs proposed,
        # which usually covers it: slabs mostly disagree by a step. Only what is still
        # missing costs a second read.
        missing = {}
        for c in cands:
            for o in c["obs"]:
                if not o["reason"] and any(s not in o["at"] for s in c["proposed"]):
                    missing.setdefault(o["slab"], []).append((c, o))
        for k, items in sorted(missing.items()):
            waiting = {id(c): o for c, o in items}
            each(k, [c for c, _ in items], lambda c, va, vb: fill(c, waiting[id(c)], va, vb))

    spans = [[s[0], s[-1]] for s in slabs]
    edges, pairs, in_fit = [], [], []
    for c in cands:
        obs = c["obs"]
        good = [o for o in obs if not o["reason"]]
        shift = choose(obs)
        if shift is None:
            # A rejected pair reports the slab that came closest, so the reason given is
            # the one that would have to change for the pair to be accepted.
            shown = max(obs, key=lambda o: -1.0 if o["got"] is None else o["got"])
        else:
            shown = max((o for o in good if o["shift"] == shift), key=lambda o: o["got"])
            for o in good:
                # A slab that proposed another shift is only evidence of an offset that
                # changes with depth if the chosen shift is measurably worse *in that
                # slab*. Where it scores about as well, the slab cannot tell the two
                # apart, and calling that a difference would be a false alarm.
                if _apart(o["shift"], shift) <= AGREE:
                    o["status"] = "agrees"
                elif o["got"] - o["at"][shift] >= rc.min_gain:
                    o["status"] = "differs"
                else:
                    o["status"] = "ambiguous"
            edges.append((c["i"], c["j"], shift,
                          max(sum(o["at"][shift] - o["base"] for o in good), 1e-3)))
            in_fit.append(len(pairs))            # the record this pair is about to get
        reason = shown["reason"]
        if reason and len(slabs) > 1:
            z0, z1 = spans[shown["slab"]]
            reason += f" (best of {len(slabs)} slabs, Z {z0}-{z1})"
        differing = [_apart(o["shift"], shift) for o in obs if o["status"] == "differs"]
        pairs.append({
            "i": c["i"], "j": c["j"], "overlap_px": c["overlap"],
            "shift": None if shown["shift"] is None else list(shown["shift"]),
            "sharpness": round(shown["sharpness"], 1),
            "ncc_metadata": round(shown["base"], 4),
            "ncc_shifted": None if shown["got"] is None else round(shown["got"], 4),
            "accepted": shift is not None, "reason": reason,
            "confirms_metadata": shift == ZERO,
            "z": spans[shown["slab"]],
            "slabs_usable": len(good),
            "depth_difference": max(differing, default=0),
            "slabs": [_slab_record(o, spans, shift) for o in obs]})

    corr, residual, residual_z, orphans, apart, dropped = solve(ms, edges)
    for k, (n, d) in enumerate(zip(in_fit, apart)):
        pairs[n]["from_fit"] = list(d)           # how far the pair is from the fit
        pairs[n]["left_out_of_fit"] = k in dropped
    left_out = [pairs[in_fit[k]] for k in dropped]
    acc = [p for p in pairs if p["accepted"]]
    log = {"scene": scene_id, "source": "refine", "mode": "3d",
           "reference_plane": ref_plane, "z_slabs": spans, "max_shift_z": max_z,
           "n_pairs": len(pairs), "n_accepted": len(acc),
           "n_confirming_metadata": sum(p["confirms_metadata"] for p in acc),
           "rms_residual_px": residual, "rms_residual_z": residual_z,
           "left_out_of_fit": [[p["i"], p["j"]] for p in left_out],
           "left_out_by": [max((max(abs(p["from_fit"][0]), abs(p["from_fit"][1]))
                                for p in left_out), default=0),
                           max((abs(p["from_fit"][2]) for p in left_out), default=0)],
           "unconstrained_tiles": orphans,
           "max_correction_px": max((abs(v[0]) + abs(v[1]) for v in corr.values()),
                                    default=0),
           "z_correction_range": [min((v[2] for v in corr.values()), default=0),
                                  max((v[2] for v in corr.values()), default=0)],
           "incomplete_planes": incomplete_planes(source.sel_z, corr, dims.get("Z")),
           "n_planes": len(source.sel_z), "memory": memory,
           "depth": depth_summary(spans, pairs) if len(slabs) > 1 else None,
           "pairs": pairs, "corrections": {str(m): list(v) for m, v in corr.items()}}
    return corr, log


def _slab_record(o, spans, chosen):
    rec = {"z": spans[o["slab"]], "shift": None if o["shift"] is None else list(o["shift"]),
           "sharpness": round(o["sharpness"], 1), "ncc_metadata": round(o["base"], 4),
           "ncc_shifted": None if o["got"] is None else round(o["got"], 4),
           "status": o["status"]}
    if o["dropped"]:
        rec[{"z": "z_offset_not_supported", "xy": "xy_shift_not_supported"}[o["dropped"][0]]] \
            = o["dropped"][1]
    if o["reason"]:
        rec["reason"] = o["reason"]
    elif chosen is not None and o["shift"] != chosen:
        rec["ncc_at_chosen_shift"] = round(o["at"][chosen], 4)
    return rec


def depth_summary(spans, pairs):
    """How far the accepted pairs' shifts agree between the slabs they were measured in.

    Per slab, of the accepted pairs: how many were usable there, and how many of those
    fit a shift more than AGREE from the one chosen *and* fit it measurably better. A
    tile tilted or sheared against its neighbour shows up here, as a shift that changes
    with depth, because one translation per tile cannot follow it.
    """
    acc = [p for p in pairs if p["accepted"]]
    out = []
    for k, span in enumerate(spans):
        obs = [(p, p["slabs"][k]) for p in acc]
        differ = [_apart(o["shift"], p["shift"]) for p, o in obs if o["status"] == "differs"]
        out.append({"z": span,
                    "usable": sum(o["status"] != "unusable" for _, o in obs),
                    "agree": sum(o["status"] == "agrees" for _, o in obs),
                    "ambiguous": sum(o["status"] == "ambiguous" for _, o in obs),
                    "differ": len(differ), "max_difference": max(differ, default=0)})
    return {"slabs": out,
            "pairs_compared": sum(p["slabs_usable"] > 1 for p in acc),
            "pairs_single_slab": sum(p["slabs_usable"] == 1 for p in acc),
            "pairs_differing": sum(p["depth_difference"] > 0 for p in acc),
            "max_difference": max((p["depth_difference"] for p in acc), default=0)}


# ---------------------------------------------------------------- reporting
def _runs(zs):
    """[0, 1, 2, 7, 8] -> "0-2, 7-8"."""
    out, zs = [], sorted(zs)
    for z in zs:
        if out and z == out[-1][1] + 1:
            out[-1][1] = z
        else:
            out.append([z, z])
    return ", ".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def format_report(log):
    sid = log["scene"]
    n, spans = log["n_accepted"], log["z_slabs"]
    res = (f"RMS residual {log['rms_residual_px']:.2f} px, "
           f"{log['rms_residual_z']:.2f} planes in Z"
           if log["rms_residual_px"] is not None else "no global fit")
    if log.get("left_out_of_fit"):
        res += f" over the {n - len(log['left_out_of_fit'])} in the fit"
    out = [f"scene {sid if sid is not None else '-'}: "
           f"{n}/{log['n_pairs']} pairs accepted, {res}"]
    depth = spans[0][1] - spans[0][0] + 1
    out.append(f"  3D registration on {len(spans)} slab(s) of {depth} planes "
               f"(Z {', '.join(f'{a}-{b}' for a, b in spans)}), "
               f"Z offsets searched to +/-{log['max_shift_z']} planes")
    out.extend(_format_memory(log.get("memory")))
    if log["n_confirming_metadata"]:
        out.append(f"  {log['n_confirming_metadata']} of the accepted pair(s) match best "
                   "where the stage put them")
    orphans = log["unconstrained_tiles"]
    if orphans:
        out.append(f"  [warn] {len(orphans)} tile(s) unconstrained {orphans[:8]}"
                   f"{' ...' if len(orphans) > 8 else ''} - left at metadata positions")
    gone = log.get("left_out_of_fit") or []
    if gone:
        px, planes = log["left_out_by"]
        names = ", ".join(f"{i}-{j}" for i, j in gone[:8]) + (" ..." if len(gone) > 8 else "")
        out.append(f"  [warn] {len(gone)} accepted pair(s) left out of the fit ({names}): "
                   f"up to {px} px / {planes} planes from where the other pairs put their "
                   "tiles. Look at those seams in the preview.")
    z0, z1 = log["z_correction_range"]
    out.append(f"  max |correction| {log['max_correction_px']} px; Z corrections "
               + ("none" if z0 == z1 == 0 else f"{z0:+d} to {z1:+d} planes"))
    if n == 0 and log["n_pairs"]:
        out.append("  [warn] nothing accepted. Check refine.min_ncc / refine.max_shift / "
                   "refine.max_shift_z, or the overlaps may be featureless in "
                   f"{'this slab' if len(spans) == 1 else 'these slabs'}. "
                   "Using metadata positions.")
    elif log["rms_residual_px"] is not None and (log["rms_residual_px"] > 2.0
                                                 or log["rms_residual_z"] > 1.0):
        out.append(f"  [warn] residual {log['rms_residual_px']:.2f} px / "
                   f"{log['rms_residual_z']:.2f} planes is high for pure translation; "
                   "tiles may need rotation/affine (see BigStitcher).")
    gaps = log["incomplete_planes"]
    if gaps:
        out.append(f"  [note] {len(gaps)} of {log['n_planes']} output plane(s) lack some "
                   f"tiles (Z {_runs(gaps)}): their Z offset reaches past the end of the "
                   "stack there, and no plane is invented to fill the gap.")
    out.extend(_format_depth(log, spans))
    return "\n".join(out)


def _format_memory(m):
    """What registration used of the threads and memory the settings allow."""
    if not m:
        return []
    line = (f"  {m['threads']} thread(s), up to {m['peak_mb']} MB: {m['volumes_mb']} MB of "
            f"overlap volumes"
            + (f" at a time ({m['batches_per_slab']} batches a slab)"
               if m["batches_per_slab"] > 1 else "")
            + f" + {m['working_mb']} MB working")
    if m["peak_mb"] <= m["budget_mb"]:
        return [line + f", within mosaic_cache_mb = {m['budget_mb']}"]
    return [line, f"  [warn] that is over mosaic_cache_mb = {m['budget_mb']}: one pair's "
                  "volumes and one transform are the least it can work with. Raise "
                  "mosaic_cache_mb to cover it, or lower refine.slab to need less."]


def _format_depth(log, spans):
    d = log.get("depth")
    n = log["n_accepted"]
    if not d or not n:
        return []
    names = [f"{a}-{b}" for a, b in spans]
    width = max(len(s) for s in names)
    out = ["  depth check, slab by slab:"]
    for name, p in zip(names, d["slabs"]):
        line = f"    Z {name:>{width}}: {p['usable']}/{n} accepted pairs usable"
        if p["differ"]:
            line += (f", {p['differ']} fit a different shift "
                     f"(up to {p['max_difference']} px/planes away)")
        if p["ambiguous"]:
            line += f", {p['ambiguous']} undecided between shifts"
        out.append(line)
    if d["pairs_differing"]:
        out.append(f"  [warn] {d['pairs_differing']} pair(s) fit a different shift at some "
                   f"depths, by up to {d['max_difference']} px/planes. One translation per "
                   "tile cannot fit every depth: expect seams offset by about that much "
                   "in the slabs marked above.")
    elif d["pairs_compared"]:
        out.append(f"    no depth fits a different shift, in any of the "
                   f"{d['pairs_compared']} pair(s) usable in more than one slab")
    if d["pairs_single_slab"]:
        out.append(f"    {d['pairs_single_slab']} pair(s) were usable in one slab only, so "
                   "were not compared between depths")
    return out
