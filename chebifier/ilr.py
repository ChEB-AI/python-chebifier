import torch

from chebifier.inconsistency_resolution import (
    NEUTRAL,
    ScoreBasedPredictionSmoother,
    densified_exclusion_pairs,
    seed_uncovered,
)


class ILRSmoother(ScoreBasedPredictionSmoother):
    def __init__(
        self,
        chebi_graph,
        label_names=None,
        disjoint_files=None,
        verbose=False,
        alpha=1.0,
        max_iter=10,
        tol=1e-4,
        threshold=NEUTRAL,
    ):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.excl_pairs = torch.zeros((0, 2), dtype=torch.long)
        self.last_iterations = 0
        self.max_iterations = 0
        self._valid = None
        super().__init__(chebi_graph, label_names, disjoint_files, verbose, threshold)
        self._build_exclusions()

    def set_label_names(self, label_names):
        super().set_label_names(label_names)
        self._build_exclusions()

    def _build_exclusions(self):
        if getattr(self, "label_names", None) is None:
            return
        if getattr(self, "disjoint_groups", None) is None:
            return
        self.excl_pairs = densified_exclusion_pairs(
            self.label_names, self.label_successors, self.disjoint_groups
        )

    def _source(self, p, fill):
        if self._valid is None:
            return p
        return torch.where(self._valid, p, torch.full_like(p, fill))

    def _up(self, p):
        masked = torch.where(
            torch.transpose(self.label_successors, 1, 2),
            self._source(p, 0.0).unsqueeze(1),
            -torch.inf,
        )
        return masked.max(dim=2).values

    def _down(self, p):
        masked = torch.where(
            self.label_successors, self._source(p, 1.0).unsqueeze(1), torch.inf
        )
        return masked.min(dim=2).values

    def _disjointness_deviation(self, p):
        raise NotImplementedError

    def _subsumption_deviation(self, p):
        raise NotImplementedError

    def _step(self, p):
        sub_dev = self._subsumption_deviation(p)
        disj_dev = self._disjointness_deviation(p)
        dev = torch.where(sub_dev.abs() >= disj_dev.abs(), sub_dev, disj_dev)
        return (p + self.alpha * dev).clamp(0.0, 1.0)

    def __call__(self, preds, valid_mask=None):
        if preds.shape[1] == 0:
            return preds
        self._valid = valid_mask
        p = seed_uncovered(preds, valid_mask, self.threshold).clamp(0.0, 1.0)
        self.last_iterations = 0
        for _ in range(self.max_iter):
            p_new = self._step(p)
            self.last_iterations += 1
            if torch.max((p_new - p).abs()) < self.tol:
                p = p_new
                break
            p = p_new
        self.max_iterations = max(self.max_iterations, self.last_iterations)
        self._valid = None
        return p


class GodelILRSmoother(ILRSmoother):
    def _subsumption_deviation(self, p):
        return self._up(p) - p

    def _disjointness_deviation(self, p):
        if self.excl_pairs.shape[0] == 0:
            return torch.zeros_like(p)
        a, b = self.excl_pairs[:, 0], self.excl_pairs[:, 1]
        src = self._source(p, 0.0)
        pa, pb = src[:, a], src[:, b]
        violated = torch.minimum(pa, pb) > 0
        a_is_min = pa <= pb
        acc = torch.zeros_like(p)
        n = p.shape[0]
        acc.scatter_add_(
            1, a.unsqueeze(0).expand(n, -1), (violated & a_is_min).to(p.dtype)
        )
        acc.scatter_add_(
            1, b.unsqueeze(0).expand(n, -1), (violated & ~a_is_min).to(p.dtype)
        )
        return torch.where(acc > 0, -p, torch.zeros_like(p))


class LukasiewiczILRSmoother(ILRSmoother):
    def _subsumption_deviation(self, p):
        raise_ = (self._up(p) - p).clamp(min=0.0) / 2
        lower = (p - self._down(p)).clamp(min=0.0) / 2
        return torch.where(raise_ >= lower, raise_, -lower)

    def _disjointness_deviation(self, p):
        if self.excl_pairs.shape[0] == 0:
            return torch.zeros_like(p)
        a, b = self.excl_pairs[:, 0], self.excl_pairs[:, 1]
        src = self._source(p, 0.0)
        excess = (src[:, a] + src[:, b] - 2.0 * self.threshold).clamp(min=0.0) / 2
        acc = torch.zeros_like(p)
        n = p.shape[0]
        acc.scatter_reduce_(
            1, a.unsqueeze(0).expand(n, -1), excess, reduce="amax", include_self=True
        )
        acc.scatter_reduce_(
            1, b.unsqueeze(0).expand(n, -1), excess, reduce="amax", include_self=True
        )
        return -acc
