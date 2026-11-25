"""
Fit every human episode under one or more MODEL VARIANTS, in the schema the behavioural
validation consumes. Resumable per (episode, variant); chunkable for a job array.

THE FITTING METHOD
    Gradient ascent to the mode, curvature at the mode, one 81-point box over the free theta
    (`fit_posterior`).

    The grid "lattice" search this script once offered as a second arm was removed once the
    comparison it existed for was settled -- V4 reaches the same accuracy for a fraction of
    the cost. The comparison itself is recorded in `version_history.md`; nothing here runs it.

THE VARIANTS (`variants.py`, implementation.md Entry 35 and its addendum 2)
    Each variant names the coefficients it fits and pins the rest; the estimator sees only
    the free vector and the likelihood only the full one. Within one episode the requested
    variants are fitted in `variants.FIT_ORDER`, the nested chain C -> B -> A' -> A: each
    model is WARM-STARTED from the one it contains, so more parameters never fit worse. B, the
    shipped fit, is the one model with two sensible starts and runs BOTH -- the midpoint (the
    shipped start) and C's solution -- keeping the better; the `warm_start` column records
    which won. The default is `B_no_heuristics` alone, whose output is then bit-identical to
    what this script wrote before variants existed (gated: tests/test_variants.py).

    `C_full_visibility` (Entry 36) sits outside the chain: it starts from the midpoint, nothing
    starts from it, and it is fitted on a SECOND prepared episode whose tables have every car
    visible. That episode is built only when the variant is requested, so it costs nothing
    otherwise; when it is, the log line's scaffolding time covers both builds.

WHAT HUMAN DATA MAY AND MAY NOT ANSWER (hlp_v3/running.md, the standing objective)
    Human episodes have no true coefficients, so nothing here is a recovery claim. What Test 4
    asks is: rolled out from its fitted coefficients, how well does each variant's driver
    reproduce the HUMAN'S collisions, lane changes, median speed and the two evasive-driving
    shares? That is `estimation/validate.py`'s question, and this script only produces the
    fit file it reads. The MODEL-COMPARISON question -- which variant earns its parameters --
    is `scripts/plot_model_comparison.py`'s, read off the `logL_mode`, `n_free_params` and
    `n_frames` columns written here.

OUTPUT: one `fit_variant_<name>.csv` per variant
    `subject_id, game_index` (a per-subject counter from 1, the convention `validate.py` and
    `load_episode_maps` both use), `variant`, `n_free_params` (the BIC ``k``), `n_frames`
    (the BIC ``n``), `seconds`, `iters`, then per theta coefficient under its RAW name
    (`hlp_collision_penalty`, ...) the POSTERIOR MEAN -- because that is the value and the
    column name `validate.py` reads -- or the PINNED value when the variant does not fit it;
    `mode_<coef>` (the raw gradient-search PEAK, or again the pinned value); and
    `<coef>_q50` / `_q05` / `_q95` / `_edge_mass` for the interval and railing read-outs,
    BLANK for a pinned coefficient because no posterior was computed over it. The heuristic
    weights are `w_<name>`: POINT estimates at the peak (no interval -- the posterior is
    described over theta alone, Entry 34), 0 for a weight the variant pins, and blank for a
    variant with no heuristics at all so one reader handles every file.

    `logL_mean` and `logL_mode` (Entry 24) are the episode log-likelihood at the vector
    `validate.py` will roll out (theta means, weights at the peak) and at the raw peak. The
    peak value is the MAXIMUM likelihood, so it is the one BIC is defined on.

    `no_progress` (1 when the ascent never beat its own start: either a stall, NOT an
    estimate, or a smaller model's solution kept as this variant's boundary optimum, which
    IS one -- `boundary_coefficients` tells them apart and the log line says which),
    `rescued_by_retry`, `lr_used`, and `warm_start` (the parent variant the ascent started
    from, or `midpoint`).

Typical usage::

    uv run python -m human_modeling.inference.scripts.run_human_inference \\
        --hlp-checkpoint-dir human_modeling/inference/checkpoint --results-dir <out> \\
        --variants B_no_heuristics A_full A_no_mean_distance C_no_cognitive_cost \\
        --max-episodes 0 --chunk-index 3 --n-chunks 10
"""

from __future__ import annotations

import argparse
import csv
import io
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from human_modeling.inference.fit_gradient import (DEFAULT_LR_JOINT,
                                                      DEFAULT_MAX_ITERS_JOINT,
                                                      curvature_half_widths, fit_mode,
                                                      fit_mode_with_retry, fit_posterior,
                                                      hessian_diag)
from human_modeling.inference.inputs import load_pilot_episodes
from human_modeling.inference.likelihood_grad import EpisodeGradContext
from human_modeling.inference.model.choice import HEURISTIC_NAMES, THETA_NAMES
from human_modeling.inference.posterior.likelihood import (load_models, log_likelihood,
                                                              prepare_episode)
from human_modeling.inference.variants import (DEFAULT_VARIANT, FIT_ORDER, VARIANTS,
                                                  FitVariant, VariantGradContext, as_eight,
                                                  check_pins_inside, ordered)

MIN_FRAMES = 100
DATA_PATH = "human_modeling/data/exp1_filtered.json"
POSTERIOR_TAGS = ("q50", "q05", "q95", "edge_mass")
FIELDS = (["subject_id", "game_index", "variant", "n_free_params", "n_frames", "seconds",
           "iters", "logL_mean", "logL_mode"]
          + list(THETA_NAMES)                                   # the posterior MEAN
          + [f"mode_{n}" for n in THETA_NAMES]                  # the gradient PEAK
          + [f"{n}_{t}" for n in THETA_NAMES for t in POSTERIOR_TAGS]
          + [f"w_{n}" for n in HEURISTIC_NAMES]
          + ["no_progress", "rescued_by_retry", "lr_used", "warm_start"])
_LOG_PATH: Path | None = None


def log(msg: str) -> None:
    """One timestamped line, also appended to the run log."""
    stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    if _LOG_PATH is not None:
        with io.open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


# ===========================================================================
# 1. One fit
# ===========================================================================

def fit_variant(
    stack: Any, inp: Any, variant: FitVariant,
    start_full: np.ndarray | None = None, start_label: str = "midpoint",
    alternative_starts: Sequence[tuple[str, np.ndarray]] = (),
) -> tuple[dict[str, float], dict[str, Any], np.ndarray]:
    """
    Gradient mode over the variant's free vector, curvature, one box over its free theta, on
    one episode.

    The MODE is fitted over every free coefficient (theta and weights together for a
    heuristic variant) while the posterior box stays over the free THETA with the weights
    pinned at the values the mode found (implementation.md Entry 34). Two reasons, and the
    second is binding: an 8-dimensional lattice is 6,561 evaluations instead of 81, and
    describing the weights' uncertainty is a design question nobody has settled. So theta's
    intervals are CONDITIONAL on the weights, and the weights come back as point estimates.

    :param stack: Loaded models and metadata.
    :type stack: ModelStack
    :param inp: Prepared episode.
    :type inp: EpisodeInputs
    :param variant: Which coefficients are free and what the rest are pinned at.
    :type variant: FitVariant
    :param start_full: A parent variant's FULL solution to warm-start from, or None for the
        box midpoint. The values are clipped into this variant's box first.
    :type start_full: np.ndarray | None
    :param start_label: How the info dict names that start (``"midpoint"`` or the parent's
        name).
    :type start_label: str
    :param alternative_starts: Further ``(label, full solution)`` pairs to try; one ascent
        each, and the start whose ascent reaches the highest log-likelihood wins. The
        posterior box is built once, at the winner.
    :type alternative_starts: Sequence[tuple[str, np.ndarray]]
    :return: ``(posterior summary over the free theta, the ascent's info dict, the FULL peak
        vector in the variant's full_names order)``. The info dict carries ``start`` (which
        start won) and ``no_progress``, which the caller MUST surface: a fit that made no
        progress returns exactly its start and is otherwise indistinguishable from a fit that
        found nothing to change.
    :rtype: tuple[dict[str, float], dict[str, Any], np.ndarray]
    """
    check_pins_inside(variant, stack)
    lo, hi = variant.lo_hi(stack)
    ctx = VariantGradContext(EpisodeGradContext(stack, inp, with_heuristics=variant.uses_heuristics),
                             variant)

    def ascend(full: np.ndarray | None) -> tuple[np.ndarray, dict[str, Any]]:
        start = None if full is None else np.clip(variant.start_from(full), lo, hi)
        if variant.uses_heuristics:
            # | The joint fit takes a smaller step and more of them, and is retried smaller
            # | still if it does not move at all -- that is where the stall was MEASURED
            # | (Entry 34). A rescued episode is still flagged; see the caller.
            return fit_mode_with_retry(ctx, lo, hi, start=start, lr=DEFAULT_LR_JOINT,
                                       max_iters=DEFAULT_MAX_ITERS_JOINT)
        return fit_mode(ctx, lo, hi, start=start)

    # | One ascent per start; the best log-likelihood wins. The first start is the default
    # | one, so with no alternatives this is exactly the single ascent it always was.
    mode_free, info = ascend(start_full)
    info = dict(info, start=start_label, attempts_by_start={start_label: info["logL"]})
    for label, full in alternative_starts:
        cand_mode, cand_info = ascend(full)
        info["attempts_by_start"][label] = cand_info["logL"]
        if cand_info["logL"] > info["logL"]:
            mode_free = cand_mode
            info = dict(cand_info, start=label, attempts_by_start=info["attempts_by_start"])

    # | The posterior is described over the free THETA only: the leading n_free_theta entries
    # | of the free vector. Curvature is computed on those axes alone (the weights' would be
    # | thrown away), and every lattice point is embedded into the full vector with the
    # | weights held at the peak.
    k = variant.n_free_theta
    lo_theta, hi_theta = lo[:k], hi[:k]
    tail = mode_free[k:].copy()
    hd = hessian_diag(ctx, mode_free, lo, hi, axes=range(k))[:k]
    est = fit_posterior(
        stack, inp, mode_free[:k], lo_theta, hi_theta,
        half=curvature_half_widths(hd, hi_theta - lo_theta), n_per_axis=3,
        embed=lambda point: variant.embed(np.concatenate([point, tail])),
        names=variant.free_theta_names)
    full_mode = variant.embed(mode_free)
    del ctx
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return est, info, full_mode


def boundary_coefficients(variant: FitVariant, info: dict[str, Any]) -> tuple[str, ...]:
    """
    The coefficients that make a no-progress fit a boundary optimum, or ``()`` for a stall.

    A fit that never beat its start (``no_progress``) is one of two things, and the ascent
    alone cannot tell them apart (fit_gradient.fit_mode, at its return):

    - a STALL: the step overshot and never recovered. The row is its start, NOT an estimate.
      MEASURED in Entry 34, where the weights stayed at exactly the zeros they began at.
    - a BOUNDARY OPTIMUM: the start was a smaller model's solution, and every coefficient
      this variant adds to it (:meth:`FitVariant.added_over`) sits at a bound of the box, so
      no move inside the box improves on it. The row IS the estimate and ties the smaller
      model's likelihood. MEASURED in Entry 35 addendum 3: all ten no-progress B fits
      started from C with cognitive cost at its lower bound 0.

    The second case needs a bound to sit on. A heuristic weight's box is symmetric about 0,
    so a weight started at 0 is never at a bound, and a heuristic variant's no-progress fit
    always reads as a stall here, which is Entry 34's case.

    :param variant: The fitted variant.
    :type variant: FitVariant
    :param info: :func:`fit_variant`'s info dict (``no_progress``, ``start``, ``at_bound``).
    :type info: dict[str, Any]
    :return: The added coefficients, all at a bound, when the fit is a boundary optimum;
        ``()`` when it made progress, started from the midpoint, or is a stall.
    :rtype: tuple[str, ...]
    """
    if not info["no_progress"] or info["start"] not in VARIANTS:
        return ()
    added = variant.added_over(VARIANTS[info["start"]])
    at_bound = dict(zip(variant.free_names, info["at_bound"]))
    return added if added and all(at_bound[n] for n in added) else ()


def rollout_vector(variant: FitVariant, est: dict[str, float], full_mode: np.ndarray) -> np.ndarray:
    """
    The full vector `validate.py` will roll out: theta at its posterior MEAN, everything
    else (pinned theta, heuristic weights) at the peak.

    :param variant: The fitted variant.
    :type variant: FitVariant
    :param est: :func:`fit_variant`'s posterior summary.
    :type est: dict[str, float]
    :param full_mode: :func:`fit_variant`'s full peak vector.
    :type full_mode: np.ndarray
    :return: Same shape and order as ``full_mode``.
    :rtype: np.ndarray
    """
    out = full_mode.copy()
    for j, name in enumerate(variant.full_names):
        if name in variant.free_theta_names:
            out[j] = est[f"{name}_mean"]
    return out


def make_row(key: tuple[str, int], n_frames: int, variant: FitVariant,
             est: dict[str, float], info: dict[str, Any], full_mode: np.ndarray,
             logl_mean: float, logl_mode: float, seconds: float,
             warm_start: str) -> dict[str, Any]:
    """
    One CSV row in :data:`FIELDS`, from one fit. See the module docstring for what each
    column holds and which are blank for a pinned coefficient.

    :param key: ``(subject_id, game_index)``.
    :type key: tuple[str, int]
    :param n_frames: Trajectory steps in the episode (the BIC ``n``).
    :type n_frames: int
    :param variant: The fitted variant.
    :type variant: FitVariant
    :param est: Posterior summary over the free theta.
    :type est: dict[str, float]
    :param info: The ascent's info dict.
    :type info: dict[str, Any]
    :param full_mode: The full peak vector.
    :type full_mode: np.ndarray
    :param logl_mean: Log-likelihood at :func:`rollout_vector`.
    :type logl_mean: float
    :param logl_mode: Log-likelihood at the peak.
    :type logl_mode: float
    :param seconds: Wall time of the fit.
    :type seconds: float
    :param warm_start: Parent variant name or ``"midpoint"``.
    :type warm_start: str
    :return: The row, every :data:`FIELDS` key present (blank where not applicable).
    :rtype: dict[str, Any]
    """
    row: dict[str, Any] = {k: "" for k in FIELDS}
    row.update({"subject_id": key[0], "game_index": key[1], "variant": variant.name,
                "n_free_params": variant.n_free, "n_frames": n_frames,
                "seconds": round(seconds, 1), "iters": int(info["iters"]),
                "logL_mean": logl_mean, "logL_mode": logl_mode,
                "no_progress": int(bool(info["no_progress"])),
                "rescued_by_retry": int(bool(info.get("rescued_by_retry", False))),
                "lr_used": info.get("lr_used", ""), "warm_start": warm_start})
    full = dict(zip(variant.full_names, full_mode))
    for name in THETA_NAMES:
        if name in variant.free_theta_names:
            row[name] = est[f"{name}_mean"]
            for tag in POSTERIOR_TAGS:
                row[f"{name}_{tag}"] = est[f"{name}_{tag}"]
        else:
            row[name] = variant.pinned_value(name)
        row[f"mode_{name}"] = float(full[name])          # every theta is in the full vector
    if variant.uses_heuristics:
        for name in HEURISTIC_NAMES:
            row[f"w_{name}"] = float(full[name])
    return row


def full_vector_from_row(row: dict[str, str]) -> np.ndarray:
    """
    The full 8-vector peak recorded in a row, for warm-starting a child variant on resume.

    :param row: A row of a `fit_variant_*.csv`.
    :type row: dict[str, str]
    :return: ``(8,)`` in ``FIT_NAMES`` order; blank weight cells read as 0.
    :rtype: np.ndarray
    """
    theta = [float(row[f"mode_{n}"]) for n in THETA_NAMES]
    weights = [float(row[f"w_{n}"]) if row.get(f"w_{n}", "") != "" else 0.0
               for n in HEURISTIC_NAMES]
    return as_eight(np.array(theta + weights, dtype=np.float64))


# ===========================================================================
# 2. Resume bookkeeping
# ===========================================================================

class VariantSink:
    """
    One variant's CSV: what is already on disk, and an appender for new rows.

    :param path: The CSV file.
    :type path: Path
    :ivar done: ``{(subject_id, game_index): full peak vector}`` for every row on disk.
    :vartype done: dict[tuple[str, int], np.ndarray]
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.done: dict[tuple[str, int], np.ndarray] = {}
        # | Resume safety, two checks (independent review, 2026-09-23). The header is written
        # | only when the file is ABSENT OR EMPTY -- keying it on "no rows yet" appended a
        # | second header to a file a killed run had left header-only, and the third run then
        # | read that header as a row and crashed on int("game_index"). And an existing
        # | header must equal FIELDS: appending 42-column rows under a pre-variant 27-column
        # | header would land every value under the wrong name, which validate.py reads
        # | silently.
        has_rows = path.exists() and path.stat().st_size > 0
        if has_rows:
            with io.open(path, encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                if reader.fieldnames is not None and list(reader.fieldnames) != FIELDS:
                    raise ValueError(
                        f"{path} has a different column layout from this script's FIELDS; it "
                        f"was written by another version (the pre-variant human_fits.csv?). "
                        f"Point --results-dir at a fresh directory rather than appending to it")
                for r in reader:
                    self.done[(r["subject_id"], int(r["game_index"]))] = full_vector_from_row(r)
        self._fh = io.open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._fh, fieldnames=FIELDS)
        if not has_rows:
            self._writer.writeheader()
            self._fh.flush()
        self.n_fit = self.n_no_progress = self.n_rescued = self.n_failed = 0
        self.n_boundary = 0    # [boundary-start] no-progress fits that are boundary optima

    def write(self, key: tuple[str, int], row: dict[str, Any], full_mode: np.ndarray,
              boundary: bool = False) -> None:
        """
        Append one row, flush, and remember its peak for warm starts.

        :param key: ``(subject_id, game_index)``.
        :type key: tuple[str, int]
        :param row: The :data:`FIELDS` row.
        :type row: dict[str, Any]
        :param full_mode: The full peak vector, kept for the next variant's warm start.
        :type full_mode: np.ndarray
        :param boundary: True when :func:`boundary_coefficients` classed this no-progress fit
            as a boundary optimum; counted apart from the stalls in :func:`summarize_run`.
        :type boundary: bool
        """
        self._writer.writerow({k: row.get(k, "") for k in FIELDS})
        self._fh.flush()
        self.done[key] = as_eight(full_mode)
        self.n_fit += 1
        self.n_no_progress += int(row["no_progress"])
        self.n_boundary += int(boundary)
        self.n_rescued += int(row["rescued_by_retry"])

    def close(self) -> None:
        """Close the file handle."""
        self._fh.close()


def summarize_run(sinks: dict[str, VariantSink]) -> None:
    """
    The headline check on the whole run, per variant. One stalled episode is a bad row; a
    handful is a badly chosen step size, and the fits are not usable until it is changed.
    Printed even when zero, so a run that did not stall says so rather than staying silent.
    A no-progress fit that is a boundary optimum (:func:`boundary_coefficients`) is counted
    and reported apart: it is an estimate, and a smaller step would not change it.

    :param sinks: Per-variant sinks after the run.
    :type sinks: dict[str, VariantSink]
    """
    for name, sink in sinks.items():
        log(f"  {name}: fitted {sink.n_fit} this run, {len(sink.done)} on disk, "
            f"{sink.n_failed} failed")
        if not sink.n_fit:
            continue
        n_stalled = sink.n_no_progress - sink.n_boundary
        log(f"    ascents that never improved on their start: {sink.n_no_progress}/{sink.n_fit} "
            f"({sink.n_no_progress / sink.n_fit:.0%}): {n_stalled} stalled, "
            f"{sink.n_boundary} at a smaller model's boundary optimum")
        log(f"    episodes rescued by a smaller-step retry: {sink.n_rescued}/{sink.n_fit} "
            f"({sink.n_rescued / sink.n_fit:.0%})")
        if sink.n_rescued:
            log("    ^^ these fits ARE usable -- the retry found the optimum the first step "
                "overshot -- but the default step was wrong for them. A large share here means "
                "DEFAULT_LR_JOINT should come down rather than the retry carrying the run.")
        # | [boundary-start] The two kinds of no-progress fit get opposite advice: a boundary
        # | optimum is kept as it is, a stall is re-fitted (boundary_coefficients).
        if sink.n_boundary:
            log(f"    ^^ the {sink.n_boundary} boundary fits ARE usable. Each started from a "
                f"smaller model's solution with every coefficient it adds at a bound, so no "
                f"move inside the box improves on it: the row is the estimate and ties the "
                f"smaller model's likelihood. A smaller step would not change it (Entry 35 "
                f"addendum 3: B started from C with cognitive cost at 0).")
        if n_stalled:
            log(f"    ^^ THE {n_stalled} STALLED FITS ARE NOT USABLE. Their coefficients are the "
                "starting values, not estimates. Lower the learning rate "
                "(fit_gradient.DEFAULT_LR_JOINT for the joint fit) and re-run them; the "
                "`no_progress` column marks which, and the per-episode log line says which "
                "kind each one is. MEASURED precedent: implementation.md Entry 34, where "
                "lr=0.08 stalled 8 of 21 joint fits at exactly w=0 and lr=0.02 stalled none.")


# ===========================================================================
# 3. Entry point
# ===========================================================================

def main() -> None:
    """
    1. load episodes with the (subject, per-subject index from 1) identity `validate.py` joins
    on, keeping this chunk's share; 2. per episode, skip variants already on disk; 3. fit the
    rest in FIT_ORDER, warm-starting children from their parent's peak; 4. append per fit;
    5. write DONE when every assigned episode is present under every requested variant.
    """
    global _LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hlp-checkpoint-dir", required=True)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--variants", nargs="+", default=None, choices=list(FIT_ORDER),
                    metavar="NAME",
                    help=f"model variants to fit per episode, any order (fitted in "
                         f"{list(FIT_ORDER)}). Default: {DEFAULT_VARIANT}, the shipped "
                         f"4-coefficient fit")
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="fit only the first N assigned episodes; 0 (the default) = every "
                         "assigned episode")
    ap.add_argument("--chunk-index", type=int, default=0,
                    help="which share of the episodes this process fits: episode i is "
                         "assigned to chunk i mod --n-chunks (round-robin, so chunks are "
                         "balanced across participants). For a SLURM job array")
    ap.add_argument("--n-chunks", type=int, default=1)
    args = ap.parse_args()
    variants = ordered(args.variants or [DEFAULT_VARIANT])
    if not 0 <= args.chunk_index < args.n_chunks:
        ap.error(f"--chunk-index {args.chunk_index} outside [0, {args.n_chunks})")

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = out_dir / "human_inference_log.txt"
    log(f"=== human inference | variants {[v.name for v in variants]} | "
        f"chunk {args.chunk_index + 1}/{args.n_chunks} ===")

    stack = load_models(args.hlp_checkpoint_dir)
    lo, hi = stack.lo_hi
    log("  ranges: " + ", ".join(f"{n.replace('hlp_', '')}=[{lo[j]:g},{hi[j]:g}]"
                                 for j, n in enumerate(THETA_NAMES)))

    # | Resume: keys already on disk are skipped per variant, so a walltime kill loses at most
    # | one (episode, variant) fit, and a child variant can still warm-start from a parent
    # | that finished in an earlier attempt.
    sinks = {v.name: VariantSink(out_dir / f"fit_variant_{v.name}.csv") for v in variants}
    for name, sink in sinks.items():
        if sink.done:
            log(f"  RESUMING {name}: {len(sink.done)} episodes already fitted")

    counters: dict[str, int] = {}
    n_assigned = n_short = 0
    for index, ep in enumerate(load_pilot_episodes(DATA_PATH)):
        counters[ep.subject_id] = counters.get(ep.subject_id, 0) + 1
        key = (ep.subject_id, counters[ep.subject_id])
        if index % args.n_chunks != args.chunk_index:
            continue
        n_assigned += 1
        if args.max_episodes and n_assigned > args.max_episodes:
            n_assigned -= 1
            break
        if ep.num_steps < MIN_FRAMES:
            n_short += 1
            log(f"  {key[0][:8]} g{key[1]}: only {ep.num_steps} frames, skipped")
            continue
        missing = [v for v in variants if key not in sinks[v.name].done]
        if not missing:
            continue
        try:
            t_prep = time.time()
            # | [full-visibility] One prepared episode per visibility mode the missing variants
            # | need: the full-visibility model's tables have one construal column per step,
            # | so it cannot share the ordinary ones (VariantGradContext refuses a mismatch).
            # | A run without that model builds exactly the one episode it always did.
            label = f"{key[0][:8]}g{key[1]}"
            inputs_by_mode = {mode: prepare_episode(stack, ep, label=label,
                                                    full_visibility=mode)
                              for mode in sorted({v.full_visibility for v in missing})}
            modes_note = {(False,): "", (True,): " (full-visibility tables only)",
                          (False, True): " (+ full-visibility tables)"}[tuple(inputs_by_mode)]
            log(f"  [{index + 1}] {key[0][:8]} g{key[1]}: {ep.num_steps} frames, scaffolding "
                f"{time.time() - t_prep:.0f}s{modes_note}, fitting {[v.name for v in missing]}")
        except Exception as exc:                     # noqa: BLE001 - long-run isolation
            for v in missing:
                sinks[v.name].n_failed += 1
            log(f"  {key[0][:8]} g{key[1]} FAILED to prepare: {type(exc).__name__}: {exc}\n"
                f"{traceback.format_exc()}")
            continue
        # | Peaks of this episode's parents, from this run or from disk, for warm starts.
        peaks = {v.name: sinks[v.name].done[key] for v in variants
                 if key in sinks[v.name].done}
        for variant in missing:
            t0 = time.time()
            inp_ = inputs_by_mode[variant.full_visibility]
            parent = variant.warm_start_from
            start_full = peaks.get(parent) if parent else None
            start_label = parent if (parent and start_full is not None) else "midpoint"
            alternatives = [(n, peaks[n]) for n in variant.extra_starts if n in peaks]
            try:
                est, info, full_mode = fit_variant(stack, inp_, variant,
                                                   start_full=start_full,
                                                   start_label=start_label,
                                                   alternative_starts=alternatives)
                warm_start = info["start"]
                # | Score the two point summaries actually used downstream under the SAME
                # | model the fit used (weights included): validate.py rolls out the mean;
                # | BIC reads the peak. ~0.15 s each, negligible beside the fit (Entry 24).
                logl_mean = log_likelihood(stack, inp_, rollout_vector(variant, est, full_mode))
                logl_mode = log_likelihood(stack, inp_, full_mode)
                row = make_row(key, ep.num_steps, variant, est, info, full_mode,
                               logl_mean, logl_mode, time.time() - t0, warm_start)
                boundary = boundary_coefficients(variant, info)
                sinks[variant.name].write(key, row, full_mode, boundary=bool(boundary))
                peaks[variant.name] = as_eight(full_mode)
                railed = sum(est[f"{n}_edge_mass"] >= 0.30 for n in variant.free_theta_names)
                tried = info.get("attempts_by_start", {})
                starts_note = (" (tried " + ", ".join(f"{k}={v:.1f}" for k, v in tried.items()) + ")"
                               if len(tried) > 1 else "")
                log(f"    {variant.name}: railed {railed}/{variant.n_free_theta}, "
                    f"logL_mode={logl_mode:.1f}, {info['iters']} iters, "
                    f"{time.time() - t0:.0f}s, start={warm_start}{starts_note}")
                if info.get("rescued_by_retry"):
                    log(f"    NOTE: the first ascent did not move; re-fitted at "
                        f"lr={info['lr_used']:g} after {info['attempts']} attempts "
                        f"(gain {info['improvement']:+.3f}). Usable, but the default step "
                        f"was {DEFAULT_LR_JOINT / info['lr_used']:.0f}x too large here.")
                # | [boundary-start] A no-progress fit from a smaller model's solution, with
                # | every added coefficient at a bound, is that model's optimum kept on the
                # | boundary -- an estimate. Anything else that did not move is a stall.
                if boundary:
                    log(f"    NOTE: no step improved on {warm_start}'s solution, and every "
                        f"coefficient {variant.name} adds to it ({', '.join(boundary)}) sits at "
                        f"a bound: that solution is this variant's optimum, on the boundary. "
                        f"The row IS an estimate and ties {warm_start}'s likelihood "
                        f"(Entry 35 addendum 3).")
                elif info["no_progress"]:
                    log(f"    WARNING: the ascent never improved on its starting point "
                        f"({info['iters']} iterations, gain {info['improvement']:+.4f}). The "
                        f"coefficients reported are the ones it started with -- NOT an "
                        f"estimate (Entry 34's step-size stall). Re-fit at a smaller lr.")
            except Exception as exc:                 # noqa: BLE001 - long-run isolation
                sinks[variant.name].n_failed += 1
                log(f"    {variant.name} FAILED: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}")
        # | Both names, or the last prepared episode stays alive through the next preparation.
        del inputs_by_mode, inp_
    for sink in sinks.values():
        sink.close()

    # | 5. Read-out and the completion marker. DONE means every assigned episode that was
    # | long enough is on disk under every requested variant; the merge step downstream
    # | requires it, so a chunk that lost an episode to a failure is re-run rather than
    # | silently merged short.
    summarize_run(sinks)
    expected = n_assigned - n_short
    complete = all(len(sink.done) >= expected for sink in sinks.values())
    n_failed = sum(sink.n_failed for sink in sinks.values())
    if args.max_episodes == 0 and complete and n_failed == 0:
        (out_dir / "DONE").write_text(
            "\n".join(f"{name} {len(sink.done)}" for name, sink in sinks.items()) + "\n",
            encoding="utf-8")
        log(f"  all {expected} assigned episodes present under every variant -> DONE")
    elif args.max_episodes == 0:
        log(f"  NOT complete: {expected} episodes assigned, on disk per variant "
            f"{ {n: len(s.done) for n, s in sinks.items()} }, {n_failed} failures this run")
    log(f"wrote {[str(s.path) for s in sinks.values()]}")


if __name__ == "__main__":
    main()
