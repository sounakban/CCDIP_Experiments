"""
inference_torch.py -- the forward recursion as torch, matching the numpy version exactly.

Why this exists (Design Log Entry 13): the numpy recursion is ~36% of a likelihood
evaluation and its cost is per-step dispatch overhead, not arithmetic. A torch port
removes that overhead (on CPU via compilation, on GPU via batching), runs on either
device unchanged, and -- because the recursion is a smooth function of the choice
tables -- opens the exact-gradient path for locating the posterior peak: automatic
differentiation through this module has no finite-difference probing step, so the
measured flat-at-1e-8 artifact cannot occur.

The transition bookkeeping is NOT re-derived here. It is converted from the numpy
:class:`~human_modeling.inference.inference._KeepPlan`, the same object the reference
recursion uses, so the two implementations cannot disagree about which subtasks survive
a step or where their labels move. Only the arithmetic is reimplemented.

Public API: :class:`TorchPlan`, :func:`forward_pass_torch`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .inference import _keep_plan
from .model.kernel import EpisodeKernel

NEG_INF = float("-inf")


class TorchPlan:
    """
    The episode's transition bookkeeping, converted once to tensors on one device.

    Built FROM the numpy keep-plan rather than from the kernel, so the survival and
    re-label maps are byte-identical to what the reference recursion uses.

    :param kernel: Episode kernel; hands its geometry to the numpy plan builder.
    :type kernel: EpisodeKernel
    :param shapes: Per-step table shapes ``(G, M_t)``.
    :type shapes: list[tuple[int, int]]
    :param device: Where the tensors live.
    :type device: torch.device | str
    :ivar device: That device, so the recursion can bring its tables to it.
    :vartype device: torch.device
    :ivar dead: Per step, indices of goal rows that END at this step (long tensor).
    :vartype dead: list[torch.Tensor]
    :ivar src_g: Per step, indices of surviving goal rows.
    :vartype src_g: list[torch.Tensor]
    :ivar flat_tgt: Per step, flat scatter targets for survivors, or None.
    :vartype flat_tgt: list[torch.Tensor | None]
    :ivar injective: Per step, whether the scatter is one-to-one.
    :vartype injective: list[bool]
    :ivar next_size: Per step, flattened size of the next table.
    :vartype next_size: list[int]
    """

    def __init__(self, kernel: EpisodeKernel, shapes: list[tuple[int, int]],
                 device: "torch.device | str" = "cpu") -> None:
        plan = _keep_plan(kernel, shapes)
        dev = torch.device(device)
        self.device = dev
        self.dead:     list[torch.Tensor] = []
        self.src_g:    list[torch.Tensor] = []
        self.flat_tgt: list[Optional[torch.Tensor]] = []
        self.injective = list(plan.injective)
        self.next_size = list(plan.next_size)
        for t in range(len(plan.alive)):
            alive = plan.alive[t]
            self.dead.append(torch.as_tensor(np.nonzero(~alive)[0],
                                             dtype=torch.long, device=dev))
            self.src_g.append(torch.as_tensor(plan.src_g[t],
                                              dtype=torch.long, device=dev))
            ft = plan.flat_tgt[t]
            self.flat_tgt.append(None if ft is None else
                                 torch.as_tensor(ft, dtype=torch.long, device=dev))


def _safe_logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    ``logaddexp`` with exact forward values and a finite backward pass.

    Plain ``torch.logaddexp(-inf, -inf)`` returns the correct ``-inf`` but its gradient
    is ``exp(-inf - (-inf)) = NaN``, and one NaN cell contaminates every earlier step
    through the shared freed-mass term. Cells where BOTH inputs are impossible carry no
    gradient by definition, so they are routed through a dummy finite value and the
    result forced back to ``-inf``: forward unchanged, backward zero at those cells.

    :param a: Log-space tensor.
    :type a: torch.Tensor
    :param b: Log-space tensor, broadcastable against ``a``.
    :type b: torch.Tensor
    :return: ``log(exp(a) + exp(b))`` with the same shape broadcasting gives.
    :rtype: torch.Tensor
    """
    both_dead = torch.isneginf(a) & torch.isneginf(b)
    a_safe = torch.where(both_dead, torch.zeros_like(a), a)
    b_safe = torch.where(both_dead, torch.zeros_like(b), b)
    out = torch.logaddexp(a_safe, b_safe)
    return torch.where(both_dead, torch.full_like(out, NEG_INF), out)


def forward_pass_torch(
    log_rho: list,
    log_b: list,
    kernel: EpisodeKernel,
    plan: TorchPlan,
) -> torch.Tensor:
    """
    Episode log-likelihood, on CPU or GPU.

    Mirrors :func:`human_modeling.inference.inference.forward_pass` operation for
    operation; the correctness gate is numerical equality with it, checked by
    ``tests/test_forward_torch.py`` on synthetic geometry covering every branch and by
    the real-episode gate before any adoption.

    :param log_rho: Per-step choice tables, each ``(G, M_t)``; arrays or tensors.
    :type log_rho: list
    :param log_b: Per-step emission tables, same shapes as ``log_rho``.
    :type log_b: list
    :param kernel: Episode kernel; supplies ``N``.
    :type kernel: EpisodeKernel
    :param plan: The episode's prebuilt :class:`TorchPlan`; its device is where the
        recursion runs. float64 throughout, matching the numpy reference to full precision.
    :type plan: TorchPlan
    :return: The log-likelihood, a scalar tensor.
    :rtype: torch.Tensor
    """
    dev, dtype = plan.device, torch.float64
    rho = [torch.as_tensor(a, dtype=dtype, device=dev) for a in log_rho]
    emis = [torch.as_tensor(a, dtype=dtype, device=dev) for a in log_b]
    N = kernel.N

    consts = torch.empty(N, dtype=dtype, device=dev)
    alpha = rho[0] + emis[0]
    c = torch.logsumexp(alpha.reshape(-1), dim=0)
    alpha = alpha - c
    consts[0] = c

    for t in range(N - 1):
        nxt = rho[t + 1]
        g1, m1 = nxt.shape
        dead = plan.dead[t]
        if dead.numel():
            freed = torch.logsumexp(alpha.index_select(0, dead).reshape(-1), dim=0)
        else:
            freed = torch.tensor(NEG_INF, dtype=dtype, device=dev)
        ft = plan.flat_tgt[t]
        if ft is None:
            # | Nothing survives this step: the keep route is empty.
            keep = torch.full((g1, m1), NEG_INF, dtype=dtype, device=dev)
        else:
            vals = alpha.index_select(0, plan.src_g[t]).reshape(-1)
            kflat = torch.full((plan.next_size[t],), NEG_INF, dtype=dtype, device=dev)
            if plan.injective[t]:
                kflat[ft] = vals
            else:
                # | Merged cells must COMBINE in log space. Shift by the max so the
                # | exponentials stay in range; a table whose max is -inf carries no mass
                # | and contributes nothing (weights forced to zero).
                vmax = vals.max()
                finite = torch.isfinite(vmax)
                safe = torch.where(finite, vmax, torch.zeros_like(vmax))
                w = torch.exp(vals - safe) * finite.to(dtype)
                lin = torch.zeros(plan.next_size[t], dtype=dtype, device=dev)
                lin.index_add_(0, ft, w)
                # | log(0) must be -inf in the FORWARD value but must not send its
                # | infinite slope (1/0) into the backward pass: clamp the log's
                # | input away from zero (identical values for every positive cell)
                # | and force the zero cells to -inf through a where, whose gradient
                # | to the unselected branch is zero.
                lin_floor = lin.clamp_min(torch.finfo(dtype).tiny)
                kflat = torch.where(lin > 0, torch.log(lin_floor) + safe,
                                    torch.full_like(lin, NEG_INF))
            keep = kflat.view(g1, m1)

        alpha = nxt + freed
        alpha = _safe_logaddexp(keep, alpha)
        alpha = alpha + emis[t + 1]
        c = torch.logsumexp(alpha.reshape(-1), dim=0)
        alpha = alpha - c
        consts[t + 1] = c
    return consts.sum()
