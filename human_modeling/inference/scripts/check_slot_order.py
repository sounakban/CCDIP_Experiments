"""
Does `hlp_extract_car_rows` put the real cars in the first `num_cars` slots, on every step of
every human episode, on this device? A gate for the slot-order defect found 2026-09-23
(SSP implementation.md Design Log 24; memory note topk-unsorted-slot-prefix-bug).

THE DEFECT, AND ITS FIX
    `hlp_extract_car_rows` (modules/base_models/SSP/hlp_core.py) picks each step's K
    nearest cars with `torch.topk(scores, K, sorted=False)`, scoring every non-car row -inf so
    it can never win a slot. With fewer than K cars visible, some returned entries are filler;
    the function points every filler slot at observation row 0 -- the driver's own car -- and
    returns only a COUNT. Every caller then marks the FIRST `count` slots as the real ones
    (`arange(K) < count`). `sorted=False` promises which entries come back, not their order,
    and the NYU cluster's CPU build returned filler first: phantom cars at the driver's position,
    real cars hidden, the canary fixture's log-likelihood 7.2 nats off. Since Design Log 24 the
    function stably moves the real cars to the front itself, which is the identity wherever
    topk already put them first.

WHAT THIS CHECKS
    The function's CONTRACT, on its actual output: on every step, slots ``< num_cars`` hold a
    non-ego row and slots ``>= num_cars`` hold row 0. Row 0 is the ego and can never be a car,
    so that test is exact. Two call shapes, because a topk implementation's order can depend
    on the batch it is given:
        episode   one call per episode with every step at once -- the likelihood's path
                  (`model/kernel.py`, `ChoiceModel.log_rho`)
        single    one call per step -- the rollout path (`estimation/validate.py`)
    Before counting anything it runs a SELF-TEST on the first episode, both directions:
      - the fix must ABSORB a forced filler-first topk order (output unchanged, 0 flags), and
      - with the fix bypassed (a stand-in that skips the reorder), that same forced order must
        be FLAGGED -- so a pass here is known to be a working check, not a blind one.

OUTPUT AND EXIT CODE
    One line per (device, shape): rows checked, rows violating the contract, episodes affected,
    the first few offending (subject, game, step). Exits 1 when the GATE device has any
    violation in either shape. Other devices are reported, not gated.

Typical usage (defaults run locally in about two minutes)::

    uv run python -m human_modeling.inference.scripts.check_slot_order \\
        --devices cpu cuda --gate-device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import numpy as np
import torch

from human_modeling.inference.inputs import build_episode_state_action_arrays, load_pilot_episodes
from human_modeling.inference.model.kernel import load_hlp_metadata
from modules.base_models.SSP import hlp_core

N_EXAMPLES = 5                 # offending (subject, game, step) triples printed per line
EGO_ROW = 0                    # the observation row filler slots are clamped to; never a car


def contract_violations(car_row_indices: torch.Tensor, num_cars: torch.Tensor) -> torch.Tensor:
    """
    Which steps break `hlp_extract_car_rows`' contract: real cars first, then row-0 filler.

    Sample reading: rows ``[[30, 45, 0, 0]]`` with count 2 -> ``[False]``;
    ``[[0, 0, 30, 45]]`` with count 2 -> ``[True]``.

    :param car_row_indices: ``(B, K)`` observation row per slot.
    :type car_row_indices: torch.Tensor
    :param num_cars: ``(B,)`` real-car count per step.
    :type num_cars: torch.Tensor
    :return: ``(B,)`` bool, True where a leading slot holds the ego row or a trailing one does not.
    :rtype: torch.Tensor
    """
    k = car_row_indices.shape[1]
    leading = torch.arange(k, device=car_row_indices.device).unsqueeze(0) < num_cars.unsqueeze(1)
    is_ego = car_row_indices == EGO_ROW
    return ((leading & is_ego) | (~leading & ~is_ego)).any(dim=1)


class FillerFirstTopk:
    """
    Context manager: make every 2-D ``torch.topk(..., sorted=False)`` return filler first.

    The order the cluster's CPU build produced and the one the fix exists to absorb. A LEGAL
    ``sorted=False`` answer: the same entries, reordered.
    """

    def __enter__(self) -> "FillerFirstTopk":
        self._original: Any = torch.topk
        torch.topk = self._wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        torch.topk = self._original

    def _wrapped(self, input: torch.Tensor, k: int, dim: int = -1, largest: bool = True,
                 sorted: bool = True, **kwargs: Any):
        result = self._original(input, k, dim=dim, largest=largest, sorted=sorted, **kwargs)
        if sorted or result[0].dim() != 2 or dim not in (1, -1):
            return result
        values, indices = result[0], result[1]
        order = torch.argsort(torch.isfinite(values).to(torch.int8), dim=1, stable=True)
        return values.gather(1, order), indices.gather(1, order)


def unfixed_extract_car_rows(raw_obs_flat: torch.Tensor, n_slots: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    The pre-Design-Log-24 function: topk's raw order trusted as-is. The self-test's known-bad arm.

    :param raw_obs_flat: ``(B, OBS_SIZE * OBS_FEATURES)`` observations.
    :type raw_obs_flat: torch.Tensor
    :param n_slots: K.
    :type n_slots: int
    :return: ``(car_row_indices, num_cars)`` as the old code computed them.
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    b = raw_obs_flat.shape[0]
    obs3 = raw_obs_flat.view(b, hlp_core.OBS_SIZE, hlp_core.OBS_FEATURES)
    kind = obs3[:, :, 6].long()
    is_car = torch.zeros_like(kind, dtype=torch.bool)
    for cid in hlp_core.CAR_KIND_IDS:
        is_car |= kind == cid
    scores = torch.where(is_car, -torch.linalg.vector_norm(obs3[:, :, :2], dim=2),
                         torch.full_like(kind, float("-inf"), dtype=torch.float32))
    vals, idx = torch.topk(scores, k=min(n_slots, hlp_core.OBS_SIZE), dim=1, largest=True, sorted=False)
    valid = vals.isfinite()
    rows = idx.long()
    rows[~valid] = EGO_ROW
    return rows, valid.sum(dim=1)


def audit_episode(obs: np.ndarray, n_slots: int, device: str, shape: str,
                  extract=hlp_core.hlp_extract_car_rows) -> np.ndarray:
    """
    Run the extractor over one episode and return each step's contract-violation flag.

    :param obs: Flat observations, ``(N, obs_dim)``.
    :type obs: np.ndarray
    :param n_slots: K, the checkpoint's car-slot count.
    :type n_slots: int
    :param device: ``"cpu"`` or ``"cuda"``.
    :type device: str
    :param shape: ``"episode"`` (one call, all steps) or ``"single"`` (one call per step).
    :type shape: str
    :param extract: The extractor under test; the real one by default, the unfixed stand-in in
        the self-test's known-bad arm.
    :type extract: Callable
    :return: ``(N,)`` bool, True where that step's slots break the contract.
    :rtype: np.ndarray
    """
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    with torch.no_grad():
        if shape == "episode":
            rows, counts = extract(obs_t, n_slots)
            return contract_violations(rows, counts).cpu().numpy()
        flags = []
        for t in range(obs_t.shape[0]):
            rows, counts = extract(obs_t[t:t + 1], n_slots)
            flags.append(contract_violations(rows, counts).cpu().numpy())
        return np.concatenate(flags)


def self_test(obs: np.ndarray, n_slots: int, device: str) -> None:
    """
    Prove the check fires on the defect and that the fix absorbs it, on one real episode.

    :raises SystemExit: If either direction fails; the gate must not be trusted then.
    """
    with FillerFirstTopk():
        absorbed = audit_episode(obs, n_slots, device, "episode")
        exposed = audit_episode(obs, n_slots, device, "episode", extract=unfixed_extract_car_rows)
    if not exposed.any():
        sys.exit(f"SELF-TEST FAILED on {device}: a forced filler-first order was not flagged on "
                 f"the unfixed extractor -- the check is blind")
    if absorbed.any():
        sys.exit(f"SELF-TEST FAILED on {device}: hlp_extract_car_rows let a forced filler-first "
                 f"order through on {int(absorbed.sum())} steps -- is the Design Log 24 fix in place?")
    print(f"  self-test on {device}: forced filler-first order flagged on {int(exposed.sum())}/"
          f"{exposed.size} steps of the unfixed extractor, 0/{absorbed.size} of the real one",
          flush=True)


DATA_PATH = "human_modeling/data/exp1_filtered.json"


def main() -> None:
    """1. load episodes and K; 2. self-test per device; 3. audit both shapes; 4. report, gate."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hlp-checkpoint-dir", default="human_modeling/inference/checkpoint")
    ap.add_argument("--devices", nargs="+", default=None,
                    help="devices to audit; default cpu, plus cuda when visible")
    ap.add_argument("--gate-device", default=None,
                    help="the device whose violation count sets the exit code -- the one "
                         "production runs on. Default cuda when visible, else cpu")
    args = ap.parse_args()
    cuda = torch.cuda.is_available()
    devices = args.devices or (["cpu", "cuda"] if cuda else ["cpu"])
    gate = args.gate_device or ("cuda" if cuda else "cpu")
    if "cuda" in devices and not cuda:
        sys.exit("cuda requested but no GPU is visible")
    if gate not in devices:
        sys.exit(f"gate device {gate} is not among the audited devices {devices}")

    started = time.time()
    n_slots = int(load_hlp_metadata(args.hlp_checkpoint_dir)["hlp_num_car_slots"])
    episodes = load_pilot_episodes(DATA_PATH)
    print(f"=== slot-order audit: {len(episodes)} episodes, K = {n_slots}, devices {devices}, "
          f"gate {gate}, torch {torch.__version__} ===", flush=True)
    keys, observations, counters = [], [], {}
    for ep in episodes:
        counters[ep.subject_id] = counters.get(ep.subject_id, 0) + 1
        keys.append((ep.subject_id, counters[ep.subject_id]))
        observations.append(build_episode_state_action_arrays(ep)[0])

    for device in devices:
        self_test(observations[0], n_slots, device)

    gate_failed = False
    for device in devices:
        for shape in ("episode", "single"):
            rows = bad = affected = 0
            examples: list[str] = []
            for key, obs in zip(keys, observations):
                flags = audit_episode(obs, n_slots, device, shape)
                rows += flags.size
                bad += int(flags.sum())
                if flags.any():
                    affected += 1
                    for step in np.nonzero(flags)[0][:max(0, N_EXAMPLES - len(examples))]:
                        examples.append(f"{key[0][:8]} g{key[1]} step {int(step)}")
            verdict = "CLEAN" if bad == 0 else "VIOLATION"
            gated = device == gate
            gate_failed |= gated and bad > 0
            print(f"  {verdict:<10} {device:<4} {shape:<7} {bad}/{rows} steps break the contract in "
                  f"{affected}/{len(keys)} episodes{' [GATE]' if gated else ''}"
                  + (f"; e.g. {', '.join(examples)}" if examples else ""), flush=True)
    print(f"  done in {time.time() - started:.0f}s", flush=True)
    if gate_failed:
        print(f"GATE FAILED: on {gate}, hlp_extract_car_rows returned filler ahead of a car on some "
              f"steps despite the Design Log 24 reorder. Do not run inference here until understood.",
              flush=True)
        sys.exit(1)
    print(f"GATE PASSED: on {gate}, every step of every episode has its real cars first.", flush=True)


if __name__ == "__main__":
    main()
