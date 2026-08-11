import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from chebifier.inconsistency_resolution import (
    NEUTRAL,
    ScoreBasedPredictionSmoother,
    densified_exclusion_matrix,
    to_logit,
)


class HexSmoother(ScoreBasedPredictionSmoother):
    def __init__(
        self,
        chebi_graph,
        label_names=None,
        disjoint_files=None,
        verbose=False,
        delta=0.0,
        max_states=2**20,
        max_component_size=40,
    ):
        self.delta = delta
        self.max_states = max_states
        self.max_component_size = max_component_size
        self.n_fallbacks = 0
        self.max_component = 0
        self.dead_labels = []
        self._state_cache = {}
        super().__init__(chebi_graph, label_names, disjoint_files, verbose)
        self._build()

    def set_label_names(self, label_names):
        super().set_label_names(label_names)
        self._build()

    def _build(self):
        if getattr(self, "label_names", None) is None:
            return
        if getattr(self, "disjoint_groups", None) is None:
            return
        self._state_cache = {}
        succ = self.label_successors[0]
        n = succ.shape[0]
        excl = densified_exclusion_matrix(
            self.label_names, self.label_successors, self.disjoint_groups
        )
        self.excl_matrix = excl
        self.excl_pairs = torch.nonzero(torch.triu(excl), as_tuple=False)
        strict = succ & ~torch.eye(n, dtype=torch.bool)
        self.sup_sets = [
            set(torch.nonzero(strict[i]).flatten().tolist()) for i in range(n)
        ]
        self.excl_sets = [
            set(torch.nonzero(excl[i]).flatten().tolist()) for i in range(n)
        ]
        self.dead_labels = self._find_dead(succ)
        if self.verbose and self.dead_labels:
            print(f"HEX: {len(self.dead_labels)} dead labels (inconsistent graph)")

    def _find_dead(self, succ):
        index = {label: i for i, label in enumerate(self.label_names)}
        dead = set()
        for group in self.disjoint_groups:
            members = [index[g] for g in group if g in index]
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    both = succ[:, members[i]] & succ[:, members[j]]
                    dead.update(torch.nonzero(both).flatten().tolist())
        return sorted(dead)

    def _legal_states(self, comp):
        key = frozenset(comp)
        if key in self._state_cache:
            return self._state_cache[key]
        comp_set = set(comp)
        order = sorted(comp, key=lambda c: len(self.sup_sets[c] & comp_set))
        local = {c: i for i, c in enumerate(order)}
        sup_local = [
            [local[s] for s in self.sup_sets[c] & comp_set if local[s] < i]
            for i, c in enumerate(order)
        ]
        excl_local = [
            [local[e] for e in self.excl_sets[c] & comp_set if local[e] < i]
            for i, c in enumerate(order)
        ]
        states = []
        cur = []
        overflow = [False]

        def rec(i):
            if overflow[0]:
                return
            if i == len(order):
                if len(states) >= self.max_states:
                    overflow[0] = True
                    return
                states.append(cur.copy())
                return
            for bit in (0, 1):
                if bit == 1:
                    if any(cur[j] == 0 for j in sup_local[i]):
                        continue
                    if any(cur[j] == 1 for j in excl_local[i]):
                        continue
                cur.append(bit)
                rec(i + 1)
                cur.pop()
                if overflow[0]:
                    return

        rec(0)
        result = (
            None
            if overflow[0]
            else (np.array(order, dtype=np.int64), np.array(states, dtype=bool))
        )
        self._state_cache[key] = result
        return result

    def _violating(self, pos):
        bad = self.label_successors[0] & pos.unsqueeze(1) & ~pos.unsqueeze(0)
        viol = bad.any(dim=1) | bad.any(dim=0)
        if self.excl_pairs.shape[0]:
            a, b = self.excl_pairs[:, 0], self.excl_pairs[:, 1]
            both = pos[a] & pos[b]
            if both.any():
                viol = viol.clone()
                viol[a[both]] = True
                viol[b[both]] = True
        return viol

    def _resolve_row(self, f, scores, valid):
        pos = scores > NEUTRAL
        known = torch.ones_like(pos) if valid is None else valid
        if valid is not None:
            pos = pos & valid
        active = (
            ((scores - NEUTRAL).abs() < self.delta) | self._violating(pos)
        ) & known
        idx = torch.nonzero(active).flatten()
        if idx.numel() == 0:
            return scores
        succ = self.label_successors[0]
        sup = succ[idx]
        subs = succ[:, idx].T
        exc = self.excl_matrix[idx]
        value = pos.clone()
        free = torch.ones(idx.numel(), dtype=torch.bool)
        out = scores.clone()
        for _ in range(20):
            settled = (~active) & known
            clamped_zero = settled & ~value
            clamped_one = settled & value
            forced_zero = (
                (sup & clamped_zero.unsqueeze(0)).any(dim=1)
                | (exc & clamped_one.unsqueeze(0)).any(dim=1)
            ) & free
            forced_one = (subs & clamped_one.unsqueeze(0)).any(dim=1) & free
            if bool((forced_zero & forced_one).any()):
                return None
            newly = forced_zero | forced_one
            if not bool(newly.any()):
                break
            value[idx[forced_one]] = True
            value[idx[forced_zero]] = False
            out[idx[forced_one]] = 1.0
            out[idx[forced_zero]] = 0.0
            free = free & ~newly
            active = active.clone()
            active[idx[newly]] = False
        idx = idx[free]
        if idx.numel() == 0:
            return out
        sub = succ[idx][:, idx]
        adj = (sub | sub.T | self.excl_matrix[idx][:, idx]).numpy()
        n_comp, labels = connected_components(
            csr_matrix(adj), directed=False, return_labels=True
        )
        for c in range(n_comp):
            members = np.nonzero(labels == c)[0]
            if members.size < 2:
                continue
            self.max_component = max(self.max_component, int(members.size))
            if members.size > self.max_component_size:
                return None
            enumerated = self._legal_states(tuple(sorted(int(idx[m]) for m in members)))
            if enumerated is None:
                return None
            order, states = enumerated
            st = torch.from_numpy(states).to(f.dtype)
            weights = torch.softmax(st @ f[torch.from_numpy(order)], dim=0)
            out[torch.from_numpy(order)] = weights @ st
        return out

    def __call__(self, preds, valid_mask=None):
        if preds.shape[1] == 0:
            return preds
        out = preds.clone()
        # the softmax over legal states is a log-linear model, P(state) proportional to
        # exp(sum of the logits that are on), so this is the one step that needs log-odds
        f = to_logit(preds)
        for row in range(preds.shape[0]):
            valid = valid_mask[row] if valid_mask is not None else None
            resolved = self._resolve_row(f[row], preds[row], valid)
            if resolved is None:
                self.n_fallbacks += 1
                out[row] = super().__call__(preds[row : row + 1])[0]
            else:
                out[row] = resolved
        return out
