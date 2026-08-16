import heapq
import math
import os

import numpy as np
from scipy.sparse import csr_matrix

ACTIVE_CUT = -8.0


class HexGraph:
    def __init__(self, labels, succ, excl):
        self.labels = labels
        self.n = len(labels)
        self.succ = succ
        self.strict = succ & ~np.eye(self.n, dtype=bool)
        self.excl = excl
        self.strict_sp = csr_matrix(self.strict.astype(np.float64))
        two_hop = (self.strict_sp @ self.strict_sp).toarray() > 0
        red = self.strict & ~two_hop
        self.parent = np.where(red.any(1), red.argmax(1), -1)
        height = np.zeros(self.n, dtype=np.int64)
        for v in np.argsort(-self.strict.sum(1)):
            p = self.parent[v]
            if p >= 0:
                height[p] = max(height[p], height[v] + 1)
        self.forest_order = np.argsort(height, kind="stable")


def graph_from_edges(n, hierarchy, exclusion, labels=None):
    succ = np.eye(n, dtype=bool)
    changed = True
    while changed:
        changed = False
        for p, c in hierarchy:
            new = succ[c] | succ[p]
            new[p] = True
            if not np.array_equal(new, succ[c]):
                succ[c] = new
                changed = True
    excl = np.zeros((n, n), dtype=bool)
    for i, j in exclusion:
        sub_i = succ[:, i]
        sub_j = succ[:, j]
        block = sub_i[:, None] & sub_j[None, :]
        excl |= block | block.T
    np.fill_diagonal(excl, False)
    return HexGraph(labels or [str(i) for i in range(n)], succ, excl)


def brute_force_marginals(g, f):
    n = g.n
    logw, mass = [], []
    for code in range(1 << n):
        y = np.array([(code >> i) & 1 for i in range(n)], dtype=bool)
        if (y[:, None] & ~y[None, :] & g.strict).any():
            continue
        if (y[:, None] & y[None, :] & g.excl).any():
            continue
        logw.append(f[y].sum())
        mass.append(y)
    logw = np.array(logw)
    mass = np.array(mass)
    z = logsumexp(logw)
    out = np.zeros(n)
    for i in range(n):
        sel = mass[:, i]
        out[i] = np.exp(logsumexp(logw[sel]) - z) if sel.any() else 0.0
    return out, len(logw)


def logsumexp(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return -np.inf
    m = a.max()
    if not np.isfinite(m):
        return m
    return m + np.log(np.exp(a - m).sum())


def propagate(g, on, off, seed_on=None, seed_off=None):
    on = on.copy()
    off = off.copy()
    if seed_on is None and seed_off is None:
        q_on = np.nonzero(on)[0].tolist()
        q_off = np.nonzero(off)[0].tolist()
    else:
        q_on = [seed_on] if seed_on is not None else []
        q_off = [seed_off] if seed_off is not None else []
    while q_on or q_off:
        if q_on:
            idx = np.asarray(q_on, dtype=np.int64)
            q_on = []
            anc = g.succ[idx].any(0)
            on |= anc
            if (on & off).any():
                return None, None, False
            ex = g.excl[np.nonzero(anc)[0]].any(0)
            new_off = ex & ~off
            if new_off.any():
                off |= new_off
                if (on & off).any():
                    return None, None, False
                q_off.extend(np.nonzero(new_off)[0].tolist())
        else:
            idx = np.asarray(q_off, dtype=np.int64)
            q_off = []
            off |= g.succ[:, idx].any(1)
            if (on & off).any():
                return None, None, False
    return on, off, True


def _lse0(a):
    mx = a.max(axis=0)
    ok = np.isfinite(mx)
    out = np.full(a.shape[1], -np.inf)
    if ok.any():
        sub = a[:, ok] - mx[ok]
        out[ok] = mx[ok] + np.log(np.exp(sub).sum(axis=0))
    return out


def _slack_log(lo, hi):
    if hi <= lo:
        return -np.inf
    return hi + np.log1p(-np.exp(lo - hi))


def bounded_marginals(
    g,
    f,
    budget=2000,
    return_items=False,
    threshold=None,
    check_every=25,
    warmup=50,
):
    f = np.asarray(f, dtype=float)
    softplus = np.logaddexp(0.0, f)
    log_sig = -np.logaddexp(0.0, -f)
    log_sig_neg = -np.logaddexp(0.0, f)
    anc_all = g.strict_sp @ f

    active = f > ACTIVE_CUT
    for v in g.forest_order:
        p = g.parent[v]
        if p >= 0 and active[v]:
            active[p] = True
    tree_order = [int(v) for v in g.forest_order if active[v]]
    tree_f = [float(f[v]) for v in tree_order]
    tree_parent = [
        int(g.parent[v]) if g.parent[v] >= 0 and active[g.parent[v]] else -1
        for v in tree_order
    ]
    softplus_flat = np.where(active, 0.0, softplus)

    on0 = np.zeros(g.n, dtype=bool)
    off0 = np.zeros(g.n, dtype=bool)
    on0, off0, ok = propagate(g, on0, off0)
    assert ok

    def forest_slack(free):
        acc = {}
        total = 0.0
        for v, fv, p in zip(tree_order, tree_f, tree_parent):
            if not free[v]:
                continue
            a = fv + acc.pop(v, 0.0)
            s = a if a > 30.0 else math.log1p(math.exp(a))
            if p >= 0 and free[p]:
                acc[p] = acc.get(p, 0.0) + s
            else:
                total += s
        return total

    def bounds(on, off):
        free = ~on & ~off
        lo = f[on].sum()
        hi_flat = lo + softplus[free].sum()
        hi = lo + softplus_flat[free].sum() + forest_slack(free)
        return lo, hi, hi_flat, free

    def aggregate(items, chunk=512):
        NEG = -np.inf
        Nlo = np.full(g.n, NEG)
        Nhi = np.full(g.n, NEG)
        Mlo = np.full(g.n, NEG)
        Mhi = np.full(g.n, NEG)
        Zlo, Zhi = NEG, NEG
        for start in range(0, len(items), chunk):
            block = items[start : start + chunk]
            ON = np.stack([it[0] for it in block])
            OFF = np.stack([it[1] for it in block])
            FREE = ~ON & ~OFF
            LO = np.array([it[2] for it in block])
            HI = np.array([it[3] for it in block])
            ANC = anc_all[None, :] - (g.strict_sp @ (ON * f).T).T
            lo_c = LO[:, None]
            hi_c = HI[:, None]
            flat_c = np.array([it[4] for it in block])[:, None]
            Zlo = np.logaddexp(Zlo, logsumexp(LO))
            Zhi = np.logaddexp(Zhi, logsumexp(HI))
            Nlo = np.logaddexp(
                Nlo,
                _lse0(np.where(ON, lo_c, np.where(FREE, lo_c + f[None, :] + ANC, NEG))),
            )
            Nhi = np.logaddexp(
                Nhi,
                _lse0(
                    np.where(
                        ON,
                        hi_c,
                        np.where(
                            FREE, np.minimum(hi_c, flat_c + log_sig[None, :]), NEG
                        ),
                    )
                ),
            )
            Mlo = np.logaddexp(Mlo, _lse0(np.where(ON, NEG, lo_c)))
            Mhi = np.logaddexp(
                Mhi,
                _lse0(
                    np.where(
                        OFF,
                        hi_c,
                        np.where(
                            FREE, np.minimum(hi_c, flat_c + log_sig_neg[None, :]), NEG
                        ),
                    )
                ),
            )
        return (
            np.exp(Nlo - np.logaddexp(Nlo, Mhi)),
            np.exp(Nhi - np.logaddexp(Nhi, Mlo)),
            Zlo,
            Zhi,
        )

    counter = 0
    lo0, hi0, flat0, _ = bounds(on0, off0)
    heap = [(-_slack_log(lo0, hi0), counter, on0, off0, lo0, hi0, flat0)]
    closed = []
    expansions = 0
    while heap and expansions < budget:
        if (
            threshold is not None
            and expansions >= warmup
            and (expansions - warmup) % check_every == 0
        ):
            lb, ub = aggregate([it[2:] for it in heap] + closed)[:2]
            if ((ub <= threshold) | (lb > threshold)).all():
                break

        negslack, _, on, off, lo, hi, flat = heapq.heappop(heap)
        free = ~on & ~off
        if not free.any() or not np.isfinite(negslack):
            closed.append((on, off, lo, hi, flat))
            continue
        k = int(np.argmax(np.where(free, softplus, -np.inf)))
        expansions += 1
        for bit in (True, False):
            n_on = on.copy()
            n_off = off.copy()
            if bit:
                n_on[k] = True
            else:
                n_off[k] = True
            a, b, good = propagate(
                g, n_on, n_off, seed_on=k if bit else None, seed_off=None if bit else k
            )
            if not good:
                continue
            l2, h2, flat2, _ = bounds(a, b)
            counter += 1
            heapq.heappush(heap, (-_slack_log(l2, h2), counter, a, b, l2, h2, flat2))

    items = [it[2:] for it in heap] + closed
    lb, ub, Zlo, Zhi = aggregate(items)
    out = {
        "lb": lb,
        "ub": ub,
        "z_lo": Zlo,
        "z_hi": Zhi,
        "n_items": len(items),
        "expansions": expansions,
        "exhausted": len(heap) == 0,
    }
    if threshold is not None:
        out["undecided"] = np.nonzero(~((ub <= threshold) | (lb > threshold)))[0]
    if return_items:
        out["items"] = items
    return out


def has_violation(g, y):
    pos = np.nonzero(y)[0]
    if pos.size == 0:
        return False
    if (g.strict[pos].any(0) & ~y).any():
        return True
    return bool(pos.size > 1 and g.excl[np.ix_(pos, pos)].any())


_WORKER_GRAPH = None


def _init_worker(g):
    global _WORKER_GRAPH
    _WORKER_GRAPH = g


def _run_one(args):
    f, budget, threshold, check_every = args
    b = bounded_marginals(
        _WORKER_GRAPH,
        f,
        budget=budget,
        threshold=threshold,
        check_every=check_every,
    )
    return (
        b["lb"].astype(np.float32),
        b["ub"].astype(np.float32),
        b["expansions"],
        b["exhausted"],
    )


def _single_thread_blas():
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"


def bounded_marginals_many(
    g,
    F,
    budget=400,
    threshold=None,
    check_every=1000,
    processes=None,
    chunksize=8,
    pool=None,
):
    import multiprocessing as mp

    if processes is None:
        processes = min(8, mp.cpu_count())
    _single_thread_blas()
    tasks = [(F[i], budget, threshold, check_every) for i in range(len(F))]
    if pool is not None:
        out = pool.map(_run_one, tasks, chunksize=chunksize)
    elif processes == 1:
        _init_worker(g)
        out = [_run_one(t) for t in tasks]
    else:
        with mp.Pool(processes, initializer=_init_worker, initargs=(g,)) as own_pool:
            out = own_pool.map(_run_one, tasks, chunksize=chunksize)
    return {
        "lb": np.stack([o[0] for o in out]),
        "ub": np.stack([o[1] for o in out]),
        "expansions": np.array([o[2] for o in out]),
        "exhausted": np.array([o[3] for o in out]),
    }


class BoundedHexSmoother:
    def __init__(
        self,
        chebi_graph,
        label_names=None,
        disjoint_files=None,
        verbose=False,
        budget=2000,
        threshold=0.5,
        processes=None,
    ):
        from chebifier.inconsistency_resolution import ScoreBasedPredictionSmoother

        self.verbose = verbose
        self.budget = budget
        self.threshold = threshold
        self.processes = processes
        self.n_uncertified = 0
        self._pool = None
        self.graph = None
        self._base = ScoreBasedPredictionSmoother(
            chebi_graph, None, disjoint_files, verbose
        )
        self.set_label_names(label_names)

    def set_label_names(self, label_names):
        from chebifier.inconsistency_resolution import densified_exclusion_matrix

        self.label_names = label_names
        self.close()
        self.graph = None
        if label_names is None:
            return
        base = self._base
        base.set_label_names(label_names)
        self.graph = HexGraph(
            label_names,
            base.label_successors[0].numpy(),
            densified_exclusion_matrix(
                label_names, base.label_successors, base.disjoint_groups
            ).numpy(),
        )

    def _worker_pool(self):
        import multiprocessing as mp

        if self.processes == 1:
            return None
        if self._pool is None:
            _single_thread_blas()
            self._pool = mp.Pool(
                self.processes or min(8, mp.cpu_count()),
                initializer=_init_worker,
                initargs=(self.graph,),
            )
        return self._pool

    def close(self):
        if getattr(self, "_pool", None) is not None:
            self._pool.terminate()
            self._pool = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __call__(self, preds, valid_mask=None):
        import torch

        from chebifier.ensemble.level1 import (
            POSITIVE_THRESHOLD,
            rescale_from_threshold,
            rescale_to_threshold,
        )
        from chebifier.inconsistency_resolution import seed_uncovered

        # the marginals are taken over legal states only, so they already satisfy the constraints
        # as inequalities: disjoint classes cannot both exceed half the mass, and a subclass never
        # carries more than its superclass. Thresholding turns those inequalities into a consistent
        # assignment only at the neutral point, so the ensemble's operating point is rescaled onto
        # it on the way in and the marginals are rescaled back on the way out.
        preds = seed_uncovered(preds, valid_mask, self.threshold)
        p = rescale_to_threshold(preds.detach().cpu().numpy(), self.threshold)
        p = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)
        f = np.log(p) - np.log1p(-p)
        res = bounded_marginals_many(
            self.graph,
            f,
            budget=self.budget,
            threshold=POSITIVE_THRESHOLD,
            processes=self.processes,
            pool=self._worker_pool(),
        )
        lb, ub = res["lb"], res["ub"]
        certified = (lb > POSITIVE_THRESHOLD) | (ub <= POSITIVE_THRESHOLD)
        self.n_uncertified += int((~certified).sum())
        # reporting the lower bound sends every uncertified label negative, which keeps the
        # assignment consistent: lb > 1/2 for two disjoint classes would put their true marginals
        # over the total mass
        out = rescale_from_threshold(lb, self.threshold)
        out = np.where(
            lb > POSITIVE_THRESHOLD,
            np.maximum(out, np.nextafter(np.float32(self.threshold), np.float32(1))),
            np.minimum(out, np.float32(self.threshold)),
        )
        if self.verbose:
            print(f"BoundedHex: {int((~certified).sum())} uncertified label decisions")
        return torch.tensor(out, dtype=preds.dtype, device=preds.device)
