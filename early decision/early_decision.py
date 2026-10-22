"""Input-space B0 prototype. Row convention: X @ W, W has shape [D, d].

The selector never computes dense QK. Float64 intervals implement analytical
real-arithmetic bounds plus a configurable guard, NOT directed-rounding proofs.
No per-token RoPE, bias, or post-projection QDQ is silently assumed.
"""
from dataclasses import dataclass
import math
import torch


def _matrix(x, name):
    if not isinstance(x, torch.Tensor) or x.ndim != 2:
        raise ValueError(f"{name} must be a 2-D tensor")
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"{name} must be finite floating point")
    if 0 in x.shape:
        raise ValueError(f"{name} must have nonempty dimensions")
    return x.detach().to(dtype=torch.float64)


@dataclass(frozen=True)
class ScoreBounds:
    estimate: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor


@dataclass(frozen=True)
class Decision:
    mask: torch.Tensor
    mass_lower_bound: float | None


@dataclass(frozen=True)
class BilinearModel:
    left: torch.Tensor
    right: torch.Tensor
    left_tail: torch.Tensor
    right_tail: torch.Tensor
    singular_values: torch.Tensor
    head_dim: int
    guard: float

    @property
    def input_dim(self):
        return self.left.shape[0]

    @property
    def rank(self):
        return self.left.shape[1]

    @property
    def residual_sigma(self):
        return (float(self.singular_values[self.rank])
                if self.rank < self.singular_values.numel() else 0.0)

    @property
    def storage_bytes(self):
        tensors = (self.left, self.right, self.left_tail,
                   self.right_tail, self.singular_values)
        return sum(t.numel() * t.element_size() for t in tensors)

    def new_cache(self, bound_mode="tail"):
        return InputCache(self, bound_mode)


@torch.no_grad()
def fit_bilinear(w_q, w_k, rank, *, score_transform=None, guard=1e-10):
    """Factor Wq @ T @ Wk.T without constructing the [D,D] matrix.

    T must be ONE shared [d,d] score transform for this entire index. Identity
    is the default. For RoPE T=R_t.T @ R_j depends on the token pair: fitting
    T=I is NOT generally valid for a real RoPE model.
    """
    w_q, w_k = _matrix(w_q, "w_q"), _matrix(w_k, "w_k")
    if w_q.shape != w_k.shape or w_q.device != w_k.device:
        raise ValueError("w_q and w_k must share [D,d] shape and device")
    if not math.isfinite(guard) or guard < 0:
        raise ValueError("guard must be finite and nonnegative")
    max_rank = min(w_q.shape)
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank <= max_rank:
        raise ValueError("rank must be an integer in [0,min(D,d)]")
    if score_transform is not None:
        transform = _matrix(score_transform, "score_transform")
        if transform.shape != (w_q.shape[1], w_q.shape[1]):
            raise ValueError("score_transform must be [d,d], not a per-token array")
        if transform.device != w_q.device:
            raise ValueError("score_transform must share weight device")
        w_q = w_q @ transform
    uq, rq = torch.linalg.qr(w_q, mode="reduced")
    uk, rk = torch.linalg.qr(w_k, mode="reduced")
    u, sigma, vh = torch.linalg.svd(rq @ rk.T, full_matrices=False)
    left_basis, right_basis = uq @ u, uk @ vh.T
    root = sigma[:rank].sqrt()
    return BilinearModel(
        (left_basis[:, :rank] * root).contiguous(),
        (right_basis[:, :rank] * root).contiguous(),
        left_basis[:, rank:].contiguous(),
        right_basis[:, rank:].contiguous(), sigma, w_q.shape[1], float(guard),
    )


class InputCache:
    """Persistent history: low-rank input features and one norm per token.

    append() does not store historical X or generate full K. Tail mode spends
    extra input projections during append/query to tighten the bound. Norm
    mode uses ||x|| and needs no tail projections. Metadata is not free.
    """
    def __init__(self, model, bound_mode):
        if bound_mode not in ("tail", "norm"):
            raise ValueError("bound_mode must be tail or norm")
        self.model, self.bound_mode = model, bound_mode
        self.features = torch.empty((0, model.rank), dtype=torch.float64,
                                    device=model.left.device)
        self.norms = torch.empty(0, dtype=torch.float64, device=model.left.device)

    @property
    def storage_bytes(self):
        return (self.features.numel() + self.norms.numel()) * 8

    @torch.no_grad()
    def append(self, x_keys):
        x_keys = _matrix(x_keys, "x_keys")
        if x_keys.shape[1] != self.model.input_dim or x_keys.device != self.features.device:
            raise ValueError("x_keys must match model input dimension and device")
        features = x_keys @ self.model.right
        if self.bound_mode == "tail":
            norms = torch.linalg.vector_norm(x_keys @ self.model.right_tail, dim=-1)
        else:
            norms = torch.linalg.vector_norm(x_keys, dim=-1)
        if not torch.isfinite(features).all() or not torch.isfinite(norms).all():
            raise ValueError("input feature projection overflow")
        self.features = torch.cat((self.features, features))
        self.norms = torch.cat((self.norms, norms))

    @torch.no_grad()
    def bounds(self, x_query):
        if not isinstance(x_query, torch.Tensor) or x_query.ndim != 1:
            raise ValueError("x_query must be one [D] vector")
        x_query = _matrix(x_query[None], "x_query")[0]
        model = self.model
        if x_query.numel() != model.input_dim or x_query.device != self.features.device:
            raise ValueError("query must match model input dimension and device")
        if not self.norms.numel():
            raise ValueError("cache is empty")
        scale = math.sqrt(model.head_dim)
        estimate = self.features @ (x_query @ model.left) / scale
        if self.bound_mode == "tail":
            qnorm = torch.linalg.vector_norm(x_query @ model.left_tail)
        else:
            qnorm = torch.linalg.vector_norm(x_query)
        error = model.residual_sigma * qnorm * self.norms / scale
        error = error + model.guard * (1.0 + estimate.abs() + error)
        if not torch.isfinite(estimate).all() or not torch.isfinite(error).all():
            raise ValueError("score bound overflow")
        return ScoreBounds(estimate, estimate - error, estimate + error)


def _validate_bounds(bounds):
    if not isinstance(bounds, ScoreBounds):
        raise ValueError("expected ScoreBounds")
    est, lo, hi = bounds.estimate, bounds.lower, bounds.upper
    if any(t.ndim != 1 for t in (est, lo, hi)) or not est.numel():
        raise ValueError("bounds must be nonempty vectors")
    if any(t.shape != est.shape or t.device != est.device for t in (lo, hi)):
        raise ValueError("bounds must share shape and device")
    if not all(torch.isfinite(t).all() for t in (est, lo, hi)):
        raise ValueError("bounds must be finite")
    if not ((lo <= est).all() and (est <= hi).all()):
        raise ValueError("bounds must enclose estimate")


@torch.no_grad()
def topk_candidates(bounds, k):
    """Keep a superset of true top-k in exact arithmetic; ties are retained."""
    _validate_bounds(bounds)
    n = bounds.estimate.numel()
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= n:
        raise ValueError("k must be in [1,N]")
    threshold = torch.topk(bounds.lower, k).values[-1]
    return Decision(bounds.upper >= threshold, None)


@torch.no_grad()
def mass_lower_bound(bounds, mask):
    """exp(L_inside)/(exp(L_inside)+exp(U_outside)), stably evaluated."""
    _validate_bounds(bounds)
    if mask.dtype != torch.bool or mask.shape != bounds.lower.shape or mask.device != bounds.lower.device:
        raise ValueError("mask must be a matching boolean vector")
    if not mask.any():
        return 0.0
    if mask.all():
        return 1.0
    a = torch.logsumexp(bounds.lower[mask], dim=0)
    b = torch.logsumexp(bounds.upper[~mask], dim=0)
    return float(torch.exp(a - torch.logaddexp(a, b)))


@torch.no_grad()
def mass_candidates(bounds, p):
    """Choose an estimated-score prefix whose analytical mass bound >= p.

    This returns B0, NOT the exact top-p set, and is not minimum-cardinality
    over arbitrary subsets. No dense scores or full Q/K are consulted.
    """
    _validate_bounds(bounds)
    if isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p) or not 0 < p <= 1:
        raise ValueError("p must be in (0,1]")
    order = torch.argsort(bounds.estimate, descending=True, stable=True)
    if p == 1:
        return Decision(torch.ones_like(bounds.estimate, dtype=torch.bool), 1.0)
    lo, hi = bounds.lower[order], bounds.upper[order]
    prefix = torch.logcumsumexp(lo, dim=0)
    suffix = torch.logcumsumexp(hi.flip(0), dim=0).flip(0)
    outside = torch.cat((suffix[1:], hi.new_tensor([-torch.inf])))
    lower_mass = torch.exp(prefix - torch.logaddexp(prefix, outside))
    count = int(torch.nonzero(lower_mass >= p, as_tuple=False)[0, 0]) + 1
    mask = torch.zeros_like(bounds.estimate, dtype=torch.bool)
    mask[order[:count]] = True
    return Decision(mask, mass_lower_bound(bounds, mask))


@torch.no_grad()
def quantize_projection(weight, bits):
    """Per OUTPUT-column symmetric QDQ for row-layout [D,d] weights.

    Used only for a frozen numerical reference; NOT native integer execution.
    """
    weight = _matrix(weight, "weight")
    if isinstance(bits, bool) or not isinstance(bits, int) or not 2 <= bits <= 8:
        raise ValueError("bits must be in [2,8]")
    limit = 2 ** (bits - 1) - 1
    scale = weight.abs().amax(dim=0, keepdim=True).clamp_min(1e-12) / limit
    codes = torch.round(weight / scale).clamp(-limit, limit)
    return codes * scale
