import heapq
import os

import numpy as np

LABEL_FILE = os.path.join(
    "data", "chebi_v252", "ensemble_classes_dl_symbolic_3star.txt"
)
GAT_PREDS = os.path.join(
    "ensemble_08-26",
    "base_learner_cache-3star",
    "gat_chebi25-3star_v252_test_predictions.npz",
)


class HexGraph:
    def __init__(self, labels, succ, excl):
        self.labels = labels
        self.n = len(labels)
        self.succ = succ
        self.strict = succ & ~np.eye(self.n, dtype=bool)
        self.excl = excl
        f = self.strict.astype(np.float32)
        hier_sparse = self.strict & ~((f @ f) > 0)
        a = succ.astype(np.float32)
        excl_sparse = excl & ((a @ excl.astype(np.float32) @ a.T) < 2)
        self.hier_sparse = hier_sparse
        self.excl_sparse = excl_sparse
        adj_bool = hier_sparse | hier_sparse.T | excl_sparse
        self.adj_sparse = [
            set(np.nonzero(adj_bool[i])[0].tolist()) for i in range(self.n)
        ]

    def order_nodes(self, nodes):
        nodes = sorted(nodes)
        s = set(nodes)
        sup = {c: set(np.nonzero(self.strict[c])[0].tolist()) & s for c in nodes}
        return sorted(nodes, key=lambda c: len(sup[c])), sup

    def local_constraints(self, nodes):
        order, sup = self.order_nodes(nodes)
        pos = {c: k for k, c in enumerate(order)}
        exc = {c: set(np.nonzero(self.excl[c])[0].tolist()) & set(nodes) for c in order}
        sup_l = [[pos[x] for x in sup[c] if pos[x] < k] for k, c in enumerate(order)]
        exc_l = [[pos[x] for x in exc[c] if pos[x] < k] for k, c in enumerate(order)]
        return order, sup_l, exc_l

    def enumerate_states(self, nodes, cap=10**6):
        order, sup_l, exc_l = self.local_constraints(nodes)
        out, cur, over = [], [], [False]

        def rec(k):
            if over[0]:
                return
            if k == len(order):
                if len(out) >= cap:
                    over[0] = True
                    return
                out.append(cur.copy())
                return
            for bit in (0, 1):
                if bit == 1:
                    if any(cur[j] == 0 for j in sup_l[k]):
                        continue
                    if any(cur[j] == 1 for j in exc_l[k]):
                        continue
                cur.append(bit)
                rec(k + 1)
                cur.pop()
                if over[0]:
                    return

        rec(0)
        if over[0]:
            return order, None
        return order, np.array(out, dtype=bool).reshape(len(out), len(order))

    def count_states(self, nodes, cap=10**6):
        order, sup_l, exc_l = self.local_constraints(nodes)
        cur, total = [], [0]

        def rec(k):
            if total[0] > cap:
                return
            if k == len(order):
                total[0] += 1
                return
            for bit in (0, 1):
                if bit == 1:
                    if any(cur[j] == 0 for j in sup_l[k]):
                        continue
                    if any(cur[j] == 1 for j in exc_l[k]):
                        continue
                cur.append(bit)
                rec(k + 1)
                cur.pop()
                if total[0] > cap:
                    return

        rec(0)
        return total[0] if total[0] <= cap else -1


def build_graph(restrict_to_gat=False, labels=None):
    from chebifier.inconsistency_resolution import (
        PredictionSmoother,
        densified_exclusion_matrix,
    )
    from chebifier.utils import get_disjoint_files, load_chebi_graph

    if labels is None:
        with open(LABEL_FILE) as fh:
            labels = [line.strip() for line in fh if line.strip()]
        if restrict_to_gat:
            cl = set(np.load(GAT_PREDS, allow_pickle=True)["classes"].tolist())
            labels = [x for x in labels if x in cl]
    sm = PredictionSmoother(load_chebi_graph(), labels, get_disjoint_files())
    succ = sm.label_successors[0].numpy()
    excl = densified_exclusion_matrix(
        labels, sm.label_successors, sm.disjoint_groups
    ).numpy()
    return HexGraph(labels, succ, excl)


def triangulate(adj):
    n = len(adj)
    work = [set(s) for s in adj]
    alive = np.ones(n, dtype=bool)

    def fill_of(v):
        nb = list(work[v])
        return sum(
            1
            for i in range(len(nb))
            for j in range(i + 1, len(nb))
            if nb[j] not in work[nb[i]]
        )

    heap = [(fill_of(v), len(work[v]), v) for v in range(n)]
    heapq.heapify(heap)
    cliques = []
    while heap:
        fill, dg, v = heapq.heappop(heap)
        if not alive[v]:
            continue
        if (fill_of(v), len(work[v])) != (fill, dg):
            heapq.heappush(heap, (fill_of(v), len(work[v]), v))
            continue
        nb = list(work[v])
        cliques.append(frozenset(nb + [v]))
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                work[nb[i]].add(nb[j])
                work[nb[j]].add(nb[i])
        for u in nb:
            work[u].discard(v)
        alive[v] = False
        for u in nb:
            heapq.heappush(heap, (fill_of(u), len(work[u]), u))
    return list(dict.fromkeys(c for c in cliques if not any(c < d for d in cliques)))


def max_weight_tree(cliques):
    m = len(cliques)
    masks = []
    for c in cliques:
        b = 0
        for v in c:
            b |= 1 << v
        masks.append(b)
    in_tree = np.zeros(m, dtype=bool)
    in_tree[0] = True
    best_w = np.array([(masks[0] & masks[j]).bit_count() for j in range(m)])
    best_i = np.zeros(m, dtype=np.int64)
    edges = []
    for _ in range(m - 1):
        j = int(np.where(in_tree, -1, best_w).argmax())
        edges.append((int(best_i[j]), j))
        in_tree[j] = True
        mj = masks[j]
        for t in np.nonzero(~in_tree)[0]:
            w = (mj & masks[t]).bit_count()
            if w > best_w[t]:
                best_w[t] = w
                best_i[t] = j
    return edges


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


def exact_marginals_jtree(g, f, cap=10**6):
    cliques = triangulate(g.adj_sparse)
    edges = max_weight_tree(cliques)
    states, orders = [], []
    for c in cliques:
        o, s = g.enumerate_states(c, cap=cap)
        if s is None:
            raise RuntimeError(f"clique of size {len(c)} exceeds cap")
        orders.append(o)
        states.append(s)
    pot = []
    assigned_unary = set()
    for k, c in enumerate(cliques):
        o = np.array(orders[k])
        contrib = np.zeros(states[k].shape[0])
        for pos, v in enumerate(o):
            if v not in assigned_unary:
                assigned_unary.add(v)
                contrib = contrib + states[k][:, pos] * f[v]
        pot.append(contrib)
    assert assigned_unary == set(range(g.n))

    adjm = {k: [] for k in range(len(cliques))}
    for i, j in edges:
        adjm[i].append(j)
        adjm[j].append(i)
    root = 0
    order_dfs, parent, seen = [], {root: None}, {root}
    stack = [root]
    while stack:
        v = stack.pop()
        order_dfs.append(v)
        for u in adjm[v]:
            if u not in seen:
                seen.add(u)
                parent[u] = v
                stack.append(u)

    def sep_key(a, b):
        sep = sorted(set(cliques[a]) & set(cliques[b]))
        idx = [orders[a].index(v) for v in sep]
        return sep, idx

    msg = {}
    for v in reversed(order_dfs):
        p = parent[v]
        if p is None:
            continue
        acc = pot[v].copy()
        for u in adjm[v]:
            if u == p:
                continue
            sep, idx = sep_key(v, u)
            m_in = msg[(u, v)]
            keys = (
                _pack(states[v][:, idx])
                if idx
                else np.zeros(states[v].shape[0], dtype=np.int64)
            )
            acc = acc + np.array([m_in.get(int(kk), -np.inf) for kk in keys])
        sep, idx = sep_key(v, p)
        keys = (
            _pack(states[v][:, idx])
            if idx
            else np.zeros(states[v].shape[0], dtype=np.int64)
        )
        out = {}
        for kk, val in zip(keys, acc):
            kk = int(kk)
            out[kk] = np.logaddexp(out.get(kk, -np.inf), val)
        msg[(v, p)] = out
    for v in order_dfs:
        for u in adjm[v]:
            if u == parent[v]:
                continue
            acc = pot[v].copy()
            for w in adjm[v]:
                if w == u:
                    continue
                sep, idx = sep_key(v, w)
                m_in = msg[(w, v)]
                keys = (
                    _pack(states[v][:, idx])
                    if idx
                    else np.zeros(states[v].shape[0], dtype=np.int64)
                )
                acc = acc + np.array([m_in.get(int(kk), -np.inf) for kk in keys])
            sep, idx = sep_key(v, u)
            keys = (
                _pack(states[v][:, idx])
                if idx
                else np.zeros(states[v].shape[0], dtype=np.int64)
            )
            out = {}
            for kk, val in zip(keys, acc):
                kk = int(kk)
                out[kk] = np.logaddexp(out.get(kk, -np.inf), val)
            msg[(v, u)] = out

    marg = np.full(g.n, np.nan)
    for k in range(len(cliques)):
        belief = pot[k].copy()
        for u in adjm[k]:
            sep, idx = sep_key(k, u)
            m_in = msg[(u, k)]
            keys = (
                _pack(states[k][:, idx])
                if idx
                else np.zeros(states[k].shape[0], dtype=np.int64)
            )
            belief = belief + np.array([m_in.get(int(kk), -np.inf) for kk in keys])
        z = logsumexp(belief)
        for pos, v in enumerate(orders[k]):
            sel = states[k][:, pos]
            val = np.exp(logsumexp(belief[sel]) - z) if sel.any() else 0.0
            if not np.isnan(marg[v]):
                assert abs(marg[v] - val) < 1e-8, (v, marg[v], val)
            marg[v] = val
    return marg


def _pack(bits):
    if bits.shape[1] == 0:
        return np.zeros(bits.shape[0], dtype=np.int64)
    w = 1 << np.arange(bits.shape[1], dtype=np.int64)
    return (bits.astype(np.int64) * w).sum(axis=1)


def propagate(g, on, off):
    for _ in range(64):
        new_on = on.copy()
        new_off = off.copy()
        if on.any():
            new_on = new_on | g.succ[on].any(0)
            new_off = new_off | g.excl[on].any(0)
        if off.any():
            new_off = new_off | g.succ[:, off].any(1)
        if (new_on & new_off).any():
            return None, None, False
        if np.array_equal(new_on, on) and np.array_equal(new_off, off):
            return on, off, True
        on, off = new_on, new_off
    return on, off, True


def _slack_log(lo, hi):
    if hi <= lo:
        return -np.inf
    return hi + np.log1p(-np.exp(lo - hi))


def bounded_marginals(g, f, budget=2000, return_items=False):
    f = np.asarray(f, dtype=float)
    softplus = np.logaddexp(0.0, f)
    log_sig = -np.logaddexp(0.0, -f)
    log_sig_neg = -np.logaddexp(0.0, f)
    strict_f = g.strict.astype(np.float64)
    anc_all = strict_f @ f

    on0 = np.zeros(g.n, dtype=bool)
    off0 = np.zeros(g.n, dtype=bool)
    on0, off0, ok = propagate(g, on0, off0)
    assert ok

    def bounds(on, off):
        free = ~on & ~off
        lo = f[on].sum()
        hi = lo + softplus[free].sum()
        return lo, hi, free

    counter = 0
    lo0, hi0, _ = bounds(on0, off0)
    heap = [(-_slack_log(lo0, hi0), counter, on0, off0)]
    closed = []
    expansions = 0
    while heap and expansions < budget:
        negslack, _, on, off = heapq.heappop(heap)
        lo, hi, free = bounds(on, off)
        if not free.any() or not np.isfinite(negslack):
            closed.append((on, off))
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
            a, b, good = propagate(g, n_on, n_off)
            if not good:
                continue
            l2, h2, _ = bounds(a, b)
            counter += 1
            heapq.heappush(heap, (-_slack_log(l2, h2), counter, a, b))
    items = [(on, off) for (_, _, on, off) in heap] + closed

    NEG = -np.inf
    Nlo = np.full(g.n, NEG)
    Nhi = np.full(g.n, NEG)
    Mlo = np.full(g.n, NEG)
    Mhi = np.full(g.n, NEG)
    Zlo, Zhi = NEG, NEG
    for on, off in items:
        lo, hi, free = bounds(on, off)
        Zlo = np.logaddexp(Zlo, lo)
        Zhi = np.logaddexp(Zhi, hi)
        Nlo[on] = np.logaddexp(Nlo[on], lo)
        Nhi[on] = np.logaddexp(Nhi[on], hi)
        Mlo[off] = np.logaddexp(Mlo[off], lo)
        Mhi[off] = np.logaddexp(Mhi[off], hi)
        if free.any():
            anc = anc_all - (strict_f[:, on] @ f[on] if on.any() else 0.0)
            Nlo[free] = np.logaddexp(Nlo[free], lo + f[free] + anc[free])
            Nhi[free] = np.logaddexp(Nhi[free], hi + log_sig[free])
            Mlo[free] = np.logaddexp(Mlo[free], lo)
            Mhi[free] = np.logaddexp(Mhi[free], hi + log_sig_neg[free])

    lb = np.exp(Nlo - np.logaddexp(Nlo, Mhi))
    ub = np.exp(Nhi - np.logaddexp(Nhi, Mlo))
    out = {
        "lb": lb,
        "ub": ub,
        "z_lo": Zlo,
        "z_hi": Zhi,
        "n_items": len(items),
        "expansions": expansions,
        "exhausted": len(heap) == 0,
    }
    if return_items:
        out["items"] = items
    return out
