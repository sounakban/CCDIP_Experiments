"""
The parameter-recovery study designed in implementation.md Entry 31, extended in Entry 35 to
any model variant: draw agents with known coefficients, simulate each on every human map,
fit every episode, aggregate per agent.

WHAT MAKES THIS DIFFERENT FROM THE STUDY IT REPLACES (Entry 29's stratified fits)
    Entry 29 drew truths in four STRATA -- middle-of-range, within 5% of a random end, beyond
    a random end, and a per-coefficient mixture. Two of those are bimodal by construction, so
    their correlations measured a two-group separation rather than recovery across a range;
    each stratum held only 6 agents; and the narrowest cell spanned 21% of its coefficient's
    range, which depresses a correlation whatever the estimator does. Here every coefficient
    is drawn INDEPENDENTLY and UNIFORMLY across its whole range, with no strata at all, which
    removes those three problems together. Independence between coefficients also makes the
    confusion matrix readable: any correlation between one coefficient's truth and another's
    estimate is the estimator trading them off, not the design putting them in step.

    What it costs is MISSPECIFICATION, which is narrower than "the boundary". Near-bound
    truths are still produced, and continuously rather than as Entry 29's discrete contrast:
    COMPUTED on the seeded 100-agent draw, 39 of the 400 coefficient values land within 5% of
    a bound and the closest sits 0.0014 of range from one. What is absent is (a) any truth
    BEYOND a bound -- and railing is diagnosed in this project as a symptom of
    misspecification, not of near-bound truth -- and (b) the human pattern's intensity, since
    humans rail a mean of 2.79 coefficients of 4 at once while only 4 of these 100 agents
    have even two near a bound. That exclusion is a decision (Entry 31), not an oversight,
    and this study must not be quoted on the regime the human fits occupy.

THE VARIANT, AND THE PAIRED ARM (Entry 35)
    `--variant` names the model that GENERATES the agents and is fitted to them; its free
    coefficients are the truths (`variants.py`). The default is the shipped theta-only fit,
    reproducing the Entry 31 study exactly. For `A_full` the truths are eight -- theta plus
    the four heuristic weights, the weights uniform over the calibrated +-bounds of Entry 33.
    A second variant (`--paired-arm`, default `B_no_heuristics`, the shipped theta-only
    fit) is ALSO fitted to every episode, first, and the variant is warm-started
    from it; the paired arm's estimates are recorded beside the variant's so theta recovery
    WITH and WITHOUT the weights is read off the same episodes. NOTE (Entry 35 addendum 2):
    the human fit chain now warm-starts `A_full` from `A_no_mean_distance`; this study keeps
    B as A_full's start on purpose, so its chunk 0 (run before the chain) stays comparable and
    the paired arm stays the theta-only fit the user asked to compare against.
    The parent's fit is the misspecified arm here: the data carry heuristics it cannot
    represent, which is the direction the human fits are suspected of.

EXTRA ARMS (Entry 35 addendum 6)
    `--extra-arms` names further variants fitted to every episode, each from the box
    MIDPOINT. The model comparison adds `C_no_cognitive_cost`, putting all three reported
    models' likelihoods on the same synthetic episodes. Its columns carry its name as a
    prefix, like the paired arm's, plus `<arm>_seconds`. An extra arm the paired arm lists in
    its `extra_starts` (C, for B) is fitted FIRST and its solution offered to the paired arm as
    a second start, exactly as `run_human_inference` fits B; `<paired>_start` records which
    start won. That is what keeps "B contains C" true in the fits: from the midpoint alone,
    B ended below C on 1 of 6 episodes tried locally (by 0.31 nats).

WHY EVERY AGENT DRIVES THE SAME MAPS
    Not a design choice. MEASURED over `human_modeling/data/exp1_filtered.json`: its 230
    episodes contain only SIX distinct maps, each played by 37-39 participants -- the human
    study is 39 participants x 6 routes, not 230 routes. There is no second set of routes to
    give a different agent, and no held-out route, so route generalisation cannot be measured
    on this data at all and every result here is conditional on these six layouts.

CHUNKING FOR A JOB ARRAY
    `--agents lo:hi` restricts one process to agents ``lo <= i < hi``; truths are seeded from
    the agent index so every chunk draws the same agent the whole study would. Each chunk
    writes its own directory, and `--merge-chunks '<glob>'` afterwards concatenates their
    `episode_fits.csv` into this directory and produces every aggregate file from the merged
    rows, refusing (exit 1) unless every (agent, map) is present.

OUTPUTS
    `agent_truths.csv`      one row per agent: the drawn coefficients.
    `episode_fits.csv`      one row per (agent, map): the fitted coefficients, resumable.
    `agent_aggregates.csv`  one row per agent: truth beside the mean and median of its fits,
                            for the variant and (when fitted) the paired arm.
    `recovery_rows_{mean,median}.csv`             the variant's results in
                            `analyze_param_recovery.py`'s input schema (`true_<coef>` /
                            `fit_<coef>` over every free coefficient), so the read-out needs
                            no adapter. One file per aggregator because that script reads a
                            single `fit_<coef>` column.
    `recovery_rows_<arm>_{mean,median}.csv`      the same for the paired arm and each extra
                            arm, over that arm's free coefficients.
    `maps.json`             the six extracted maps, so a restart need not re-parse the 900 MB
                            source and so the map set is auditable after the fact.

Typical usage (defaults are a LOCAL SMOKE: 2 agents x 2 maps; the sbatch passes the real size)::

    uv run python -m human_modeling.inference.scripts.run_synthetic_recovery \\
        --hlp-checkpoint-dir <ckpt> --results-dir <out> --variant A_full \\
        --n-agents 100 --n-maps 6 --agents 0:10
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import io
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from human_modeling.inference.estimation.validate import (RecordingEnv, behaviour_metrics,
                                                             rollout_episode)
from human_modeling.inference.inputs import HumanEpisode, load_episode_maps
from human_modeling.inference.model.choice import THETA_NAMES
from human_modeling.inference.posterior.likelihood import (load_models, log_likelihood,
                                                              prepare_episode, theta_dict,
                                                              weight_dict)
from human_modeling.inference.scripts.merge_chunk_csvs import merge_rows
from human_modeling.inference.scripts.run_human_inference import (POSTERIOR_TAGS,
                                                                     boundary_coefficients,
                                                                     fit_variant,
                                                                     rollout_vector)
from human_modeling.inference.variants import (ATTENTION_CHOOSING_VARIANTS, DEFAULT_VARIANT,
                                                  FitVariant, get_variant)

MIN_FRAMES = 100           # an episode shorter than this is discarded, as in run_human_inference
N_HUMAN_MAPS = 6           # MEASURED: exp1_filtered.json contains exactly this many distinct maps
_LOG_PATH: Path | None = None


def log(msg: str) -> None:
    """One timestamped line, also appended to the run log.

    :param msg: Message text; ASCII only, since it reaches a console (Rule 12).
    :type msg: str
    """
    stamped = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(stamped, flush=True)
    if _LOG_PATH is not None:
        with io.open(_LOG_PATH, "a", encoding="utf-8") as fh:
            fh.write(stamped + "\n")


def stable_seed(*parts: int) -> int:
    """One reproducible 32-bit seed from any number of integers.

    Uses SHA-256 rather than `hash`, whose per-process randomisation would make a resumed run
    draw different numbers from the run it resumes.

    :param parts: Components identifying what is being seeded.
    :type parts: int
    :return: Seed in ``[0, 2**32)``.
    :rtype: int
    """
    payload = ":".join(str(part) for part in parts).encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2 ** 32)


# ===========================================================================
# 1. The six human maps
# ===========================================================================


def map_id_of(map_payload: Any) -> str:
    """The route label of a map: the leading 8 hex digits of its content hash.

    Two episodes on the same road get the same label whatever their file order, so the label
    identifies a route across the human data, the recovery study and the variance read-out.

    :param map_payload: The episode's map as stored in the telemetry (JSON-serialisable).
    :type map_payload: Any
    :return: 8 hex characters.
    :rtype: str
    """
    return hashlib.sha1(json.dumps(map_payload, sort_keys=True).encode()).hexdigest()[:8]


def human_routes(data_path: Path) -> dict[tuple[str, int], str]:
    """The route label of every human episode, from the telemetry itself.

    :param data_path: The human data JSON.
    :type data_path: Path
    :return: ``(subject_id, game_index) -> map_id`` for every episode in the file.
    :rtype: dict[tuple[str, int], str]
    """
    return {(e["subject_id"], int(e["game_index"])): map_id_of(e["map"]) for e in load_episode_maps(data_path)}


def extract_human_maps(data_path: Path, n_maps: int) -> list[dict[str, Any]]:
    """The distinct maps in the human telemetry, deduplicated and deterministically ordered.

    Numbered stages:
    1. Read every episode's map payload and group episodes by the map's content hash.
    2. Order the distinct maps by that hash, so the selection never depends on file order.
    3. Keep the leading `n_maps`, recording which episode each came from.

    :param data_path: The human data JSON.
    :type data_path: Path
    :param n_maps: How many distinct maps to keep; fewer than all six is a smoke setting.
    :type n_maps: int
    :return: One ``{"map_id", "map", "n_episodes", "first_subject", "first_game_index"}`` per
        distinct map.
    :rtype: list[dict[str, Any]]
    :raises ValueError: If the data holds fewer distinct maps than requested.
    """
    by_hash: dict[str, dict[str, Any]] = {}
    for entry in load_episode_maps(data_path):
        map_id = map_id_of(entry["map"])
        record = by_hash.setdefault(map_id, {"map_id": map_id, "map": entry["map"],
                                             "n_episodes": 0,
                                             "first_subject": entry["subject_id"],
                                             "first_game_index": entry["game_index"]})
        record["n_episodes"] += 1
    if len(by_hash) < n_maps:
        raise ValueError(f"{data_path} holds {len(by_hash)} distinct maps, {n_maps} requested")
    return [by_hash[k] for k in sorted(by_hash)][:n_maps]


def load_or_extract_maps(results_dir: Path, data_path: Path, n_maps: int) -> list[dict[str, Any]]:
    """The map set for this study, cached beside the results after the first extraction.

    Parsing the 900 MB source takes minutes, and a resumed run would pay it again on every
    restart, so the extracted maps are written once and reloaded thereafter.

    :param results_dir: Study output directory, holding `maps.json` once written.
    :type results_dir: Path
    :param data_path: The human data JSON, read only when the cache is absent.
    :type data_path: Path
    :param n_maps: How many distinct maps the study uses.
    :type n_maps: int
    :return: The map records, as :func:`extract_human_maps` returns them.
    :rtype: list[dict[str, Any]]
    :raises ValueError: If a cached file holds a different number of maps than requested.
    """
    cache_path = results_dir / "maps.json"
    if cache_path.exists():
        with io.open(cache_path, encoding="utf-8") as fh:
            cached = json.load(fh)
        if len(cached) != n_maps:
            raise ValueError(f"{cache_path} holds {len(cached)} maps, {n_maps} requested; "
                             f"delete it to re-extract")
        log(f"  reusing {len(cached)} cached maps from {cache_path.name}")
        return cached
    log(f"  extracting distinct maps from {data_path} (minutes: the file is ~900 MB)")
    maps = extract_human_maps(data_path, n_maps)
    with io.open(cache_path, "w", encoding="utf-8") as fh:
        json.dump(maps, fh, sort_keys=True)
    log(f"  extracted {len(maps)} distinct maps, played by "
        + ", ".join(str(m["n_episodes"]) for m in maps) + " episodes respectively")
    return maps


# ===========================================================================
# 2. Agents and episodes
# ===========================================================================


def agent_truth(study_seed: int, agent_index: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """One agent's true coefficients, each drawn independently and uniformly over its range.

    Seeded from the agent's index rather than from a running generator, so agent 57's truth is
    the same whether the run reached it directly or after three restarts -- and the same in
    every chunk of a job array.

    :param study_seed: Seed identifying this study.
    :type study_seed: int
    :param agent_index: Which agent, counting from 0.
    :type agent_index: int
    :param lo: Lower bound per FREE coefficient of the generating variant.
    :type lo: np.ndarray
    :param hi: Upper bound per free coefficient.
    :type hi: np.ndarray
    :return: The drawn coefficients, ``(n_free,)`` in the variant's free order.
    :rtype: np.ndarray
    """
    return np.random.default_rng(stable_seed(study_seed, agent_index, 17)).uniform(lo, hi)


def simulate_episode(
    stack: Any, map_data: dict[str, Any], full_truth: np.ndarray,
    agent_index: int, map_index: int, study_seed: int,
) -> tuple[HumanEpisode, dict[str, Any], dict[str, float]]:
    """Simulate one agent on one map and package it as an episode the fitter accepts.

    Actions are SAMPLED rather than taken greedily, matching every earlier synthetic test in
    this package and giving the episode-to-episode variation the reliability ceiling measures.

    :param stack: Loaded models.
    :type stack: Any
    :param map_data: One map payload.
    :type map_data: dict[str, Any]
    :param full_truth: The agent's FULL coefficient vector -- ``(4,)`` theta only, or ``(8,)``
        with the heuristic weights that then drive its construal choices.
    :type full_truth: np.ndarray
    :param agent_index: Which agent, for seeding and for the episode's identity.
    :type agent_index: int
    :param map_index: Which map, for seeding.
    :type map_index: int
    :param study_seed: Seed identifying this study.
    :type study_seed: int
    :return: ``(episode, rollout summary, behavioural metrics)``.
    :rtype: tuple[HumanEpisode, dict[str, Any], dict[str, float]]
    :raises RuntimeError: If the rollout produced too few frames to fit.
    """
    env = RecordingEnv(fixed_map=map_data, training_mode=False)
    rng = np.random.default_rng(stable_seed(study_seed, agent_index, map_index, 31))
    rollout = rollout_episode(env, stack.choice, stack.llp, stack.meta,
                              theta_dict(full_truth), weight_dict(full_truth), rng)
    frames = env.frames
    if len(frames) < MIN_FRAMES:
        raise RuntimeError(f"episode has only {len(frames)} recorded frames")
    episode = HumanEpisode(
        subject_id=f"synthetic_agent_{agent_index:04d}",
        map_base=map_data["base"],
        steps=frames,
        actions=np.asarray([f["action"] for f in frames], dtype=np.int64),
    )
    return episode, rollout, behaviour_metrics(frames)


def fit_columns(variant: FitVariant, arm_prefix: str = "") -> dict[str, str]:
    """The episode-row column holding each free coefficient's point estimate.

    The variant's own theta sit under their bare names (the posterior MEAN, the value the
    behavioural validation rolls out) and its weights under ``w_<name>`` (the peak -- they
    have no posterior, Entry 34), mirroring `run_human_inference.py`'s schema. A paired arm's
    columns carry its name as a prefix.

    :param variant: The fitted variant.
    :type variant: FitVariant
    :param arm_prefix: ``""`` for the study's own variant, ``"<arm>_"`` for a paired arm.
    :type arm_prefix: str
    :return: ``{coefficient: column}`` over the variant's free coefficients.
    :rtype: dict[str, str]
    """
    return {n: f"{arm_prefix}{n}" if n in THETA_NAMES else f"{arm_prefix}w_{n}"
            for n in variant.free_names}


def fit_record(variant: FitVariant, est: dict[str, float], info: dict[str, Any],
               full_mode: np.ndarray, logl_mode: float, arm_prefix: str = "") -> dict[str, Any]:
    """One arm's contribution to an episode row.

    :param variant: The fitted variant.
    :type variant: FitVariant
    :param est: Posterior summary over the free theta.
    :type est: dict[str, float]
    :param info: The ascent's info dict.
    :type info: dict[str, Any]
    :param full_mode: The full peak vector.
    :type full_mode: np.ndarray
    :param logl_mode: Log-likelihood at the peak.
    :type logl_mode: float
    :param arm_prefix: ``""`` or ``"<arm>_"``.
    :type arm_prefix: str
    :return: The columns, point estimates per :func:`fit_columns` plus `mode_`, quantiles
        (theta only), `iters`, `logL_mode` and `no_progress`.
    :rtype: dict[str, Any]
    """
    full = dict(zip(variant.full_names, full_mode))
    out: dict[str, Any] = {f"{arm_prefix}iters": int(info["iters"]),
                           f"{arm_prefix}logL_mode": logl_mode,
                           f"{arm_prefix}no_progress": int(bool(info["no_progress"]))}
    for name, column in fit_columns(variant, arm_prefix).items():
        if name in THETA_NAMES:
            out[column] = est[f"{name}_mean"]
            out[f"{arm_prefix}mode_{name}"] = float(full[name])
            for tag in POSTERIOR_TAGS:
                out[f"{arm_prefix}{name}_{tag}"] = est[f"{name}_{tag}"]
        else:
            out[column] = float(full[name])
    return out


def has_arm(row: dict[str, Any], arm: FitVariant) -> bool:
    """True when an episode row already holds a fit of the secondary `arm` (its logL is filled)."""
    return row.get(f"{arm.name}_logL_mode") not in (None, "")


def starter_arms(paired: FitVariant, extras: Sequence[FitVariant]) -> list[FitVariant]:
    """The extra arms whose solutions the paired arm ALSO starts from (its `extra_starts`)."""
    return [arm for arm in extras if arm.name in paired.extra_starts]


def fit_into(
    row: dict[str, Any], stack: Any, inp: Any, arm: FitVariant, prefix: str,
    start_full: np.ndarray | None = None, start_label: str = "midpoint",
    alternatives: Sequence[tuple[str, np.ndarray]] = (),
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """
    Fit one arm to one episode and write its columns into `row`.

    :param row: The episode row, updated in place.
    :type row: dict[str, Any]
    :param stack: Loaded models.
    :type stack: Any
    :param inp: The prepared episode.
    :type inp: Any
    :param arm: The arm to fit.
    :type arm: FitVariant
    :param prefix: Its column prefix, ``""`` for the study's own variant.
    :type prefix: str
    :param start_full: A solution to start from, or None for the midpoint.
    :type start_full: np.ndarray | None
    :param start_label: How that start is named.
    :type start_label: str
    :param alternatives: Further ``(label, full solution)`` starts; the best ascent wins.
    :type alternatives: Sequence[tuple[str, np.ndarray]]
    :return: ``(posterior summary, ascent info, full peak vector)`` from `fit_variant`.
    :rtype: tuple[dict[str, Any], dict[str, Any], np.ndarray]
    """
    est, info, mode = fit_variant(stack, inp, arm, start_full=start_full,
                                  start_label=start_label, alternative_starts=alternatives)
    row.update(fit_record(arm, est, info, mode, log_likelihood(stack, inp, mode), prefix))
    return est, info, mode


def fit_extra_arm(row: dict[str, Any], stack: Any, inp: Any, arm: FitVariant) -> np.ndarray:
    """
    Fit one extra arm from the midpoint into `row`, with its time and a stall warning.

    :param row: The episode row, updated in place.
    :type row: dict[str, Any]
    :param stack: Loaded models.
    :type stack: Any
    :param inp: The prepared episode.
    :type inp: Any
    :param arm: The extra arm.
    :type arm: FitVariant
    :return: Its full peak vector, for use as another arm's start.
    :rtype: np.ndarray
    """
    t_arm = time.time()
    _, info, mode = fit_into(row, stack, inp, arm, f"{arm.name}_")
    row[f"{arm.name}_seconds"] = round(time.time() - t_arm, 1)
    log(f"    {arm.name}: logL_mode={row[f'{arm.name}_logL_mode']:.1f}, "
        f"{row[f'{arm.name}_seconds']:.0f}s")
    if info["no_progress"]:
        log(f"    WARNING: {arm.name}'s ascent never improved on its midpoint start "
            f"(no_progress = 1); that arm's row is NOT an estimate -- see implementation.md Entry 34")
    return mode


def fit_new_episode(row: dict[str, Any], stack: Any, inp: Any, variant: FitVariant,
                    paired: FitVariant, extras: Sequence[FitVariant],
                    ) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Every arm on a newly simulated episode, in the human fit's chain order.

    1. The extra arms the paired arm starts from (C), each from the midpoint.
    2. The paired arm (B) from the midpoint AND from those solutions, keeping the best ascent,
       as `run_human_inference` fits B. Its winning start goes in `<paired>_start`.
    3. The variant (A) warm-started from the paired arm's solution.
    4. Every other extra arm, from the midpoint.

    :param row: The episode row, updated in place.
    :type row: dict[str, Any]
    :param stack: Loaded models.
    :type stack: Any
    :param inp: The prepared episode.
    :type inp: Any
    :param variant: The study's variant.
    :type variant: FitVariant
    :param paired: The paired arm.
    :type paired: FitVariant
    :param extras: The extra arms.
    :type extras: Sequence[FitVariant]
    :return: The variant's ``(posterior summary, ascent info)``.
    :rtype: tuple[dict[str, Any], dict[str, Any]]
    """
    starters = starter_arms(paired, extras)
    modes = {arm.name: fit_extra_arm(row, stack, inp, arm) for arm in starters}
    alternatives = [(name, mode) for name, mode in modes.items()]
    _, info_p, start_full = fit_into(row, stack, inp, paired, f"{paired.name}_",
                                     alternatives=alternatives)
    if alternatives:
        row[f"{paired.name}_start"] = info_p["start"]
    # | [boundary-start] The start is named so boundary_coefficients can tell a boundary
    # | optimum from a stall; unnamed, a warm start read as "midpoint".
    est, info, full_mode = fit_into(row, stack, inp, variant, "", start_full=start_full,
                                    start_label=paired.name)
    row["logL_mean"] = log_likelihood(stack, inp, rollout_vector(variant, est, full_mode))
    for arm in extras:
        if arm not in starters:
            fit_extra_arm(row, stack, inp, arm)
    return est, info


# ===========================================================================
# 3. Aggregation and output
# ===========================================================================


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write rows with a stable column order.

    :param path: Destination.
    :type path: Path
    :param rows: Records to write.
    :type rows: list[dict[str, Any]]
    :param fields: Column order; keys outside it are dropped.
    :type fields: list[str]
    """
    # | Written to a sibling file and swapped in, so a kill mid-write leaves the previous
    # | complete file rather than a truncated one -- the episode file is rewritten whole after
    # | every episode, and losing a chunk's whole history to a millisecond window is the one
    # | failure a resumable study must not have (independent review, 2026-09-23).
    tmp = path.with_name(path.name + ".tmp")
    with io.open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def aggregate_rows(
    episode_rows: list[dict[str, Any]], truth_rows: list[dict[str, Any]],
    columns: dict[str, str] | None = None, prefix: str = "",
) -> list[dict[str, Any]]:
    """One row per agent: its truth beside the mean and median of its episode fits.

    Sample reading: ``{"agent_index": 7, "n_successful_episodes": 6,
    "true_hlp_collision_penalty": 4.12, "mean_hlp_collision_penalty": 3.88,
    "median_hlp_collision_penalty": 3.91, "mean_hlp_collision_penalty_error": -0.24, ...}``.

    :param episode_rows: Every completed episode fit.
    :type episode_rows: list[dict[str, Any]]
    :param truth_rows: One row per agent carrying its drawn coefficients as `true_<coef>`.
    :type truth_rows: list[dict[str, Any]]
    :param columns: ``{coefficient: episode column}`` to aggregate; the four theta under their
        bare names by default (the pre-variant schema).
    :type columns: dict[str, str] | None
    :param prefix: Prepended to the `mean_`/`median_` keys, to keep a paired arm's aggregates
        apart from the variant's in one record.
    :type prefix: str
    :return: One record per agent that has at least one successful fit.
    :rtype: list[dict[str, Any]]
    """
    columns = columns or {n: n for n in THETA_NAMES}
    truth_by_agent = {int(r["agent_index"]): r for r in truth_rows}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in episode_rows:
        grouped.setdefault(int(row["agent_index"]), []).append(row)
    out: list[dict[str, Any]] = []
    for agent_index, rows in sorted(grouped.items()):
        truth = truth_by_agent[agent_index]
        record: dict[str, Any] = {"agent_index": agent_index, "n_successful_episodes": len(rows)}
        for name, column in columns.items():
            values = np.asarray([float(r[column]) for r in rows])          # (n_episodes,)
            record[f"true_{name}"] = float(truth[f"true_{name}"])
            record[f"{prefix}mean_{name}"] = float(np.mean(values))
            record[f"{prefix}median_{name}"] = float(np.median(values))
            for how in ("mean", "median"):
                record[f"{prefix}{how}_{name}_error"] = (record[f"{prefix}{how}_{name}"]
                                                         - record[f"true_{name}"])
        out.append(record)
    return out


def write_recovery_rows(
    path: Path, episode_rows: list[dict[str, Any]], aggregate: list[dict[str, Any]], how: str,
    columns: dict[str, str] | None = None, prefix: str = "",
) -> None:
    """The study's results in `analyze_param_recovery.py`'s own input schema.

    That script reads `level`, `stratum`, `person`, `episode` and a `true_<coef>` /
    `fit_<coef>` pair per coefficient, so writing this file means the read-out needs no
    separate adapter and the two studies are analysed by identical code. Every row carries
    ``stratum="all"``, since this design has no strata.

    :param path: Destination CSV.
    :type path: Path
    :param episode_rows: Every completed episode fit, becoming the ``level="episode"`` rows.
    :type episode_rows: list[dict[str, Any]]
    :param aggregate: Per-agent records, becoming the ``level="person"`` rows.
    :type aggregate: list[dict[str, Any]]
    :param how: Which aggregator fills `fit_<coef>` at agent level, ``"mean"`` or ``"median"``.
    :type how: str
    :param columns: ``{coefficient: episode column}``; the four theta by default.
    :type columns: dict[str, str] | None
    :param prefix: The aggregate keys' prefix, as given to :func:`aggregate_rows`.
    :type prefix: str
    """
    columns = columns or {n: n for n in THETA_NAMES}
    names = list(columns)
    rows: list[dict[str, Any]] = []
    for row in episode_rows:
        rows.append({"level": "episode", "stratum": "all", "person": row["agent_index"],
                     "episode": row["map_index"], "map": row["map_id"],
                     **{f"true_{n}": row[f"true_{n}"] for n in names},
                     **{f"fit_{n}": row[columns[n]] for n in names}})
    for record in aggregate:
        rows.append({"level": "person", "stratum": "all", "person": record["agent_index"],
                     "episode": "", "map": "",
                     **{f"true_{n}": record[f"true_{n}"] for n in names},
                     **{f"fit_{n}": record[f"{prefix}{how}_{n}"] for n in names}})
    fields = (["level", "stratum", "person", "episode", "map"]
              + [f"true_{n}" for n in names] + [f"fit_{n}" for n in names])
    write_csv(path, rows, fields)


def truth_rows_for(variant: FitVariant, lo: np.ndarray, hi: np.ndarray, n_agents: int,
                   study_seed: int) -> list[dict[str, Any]]:
    """Every agent's drawn truth, ``true_<coef>`` over the variant's free coefficients.

    :param variant: The generating variant.
    :type variant: FitVariant
    :param lo: Lower bound per free coefficient.
    :type lo: np.ndarray
    :param hi: Upper bound per free coefficient.
    :type hi: np.ndarray
    :param n_agents: Agents in the whole study (not just this chunk).
    :type n_agents: int
    :param study_seed: Seed identifying this study.
    :type study_seed: int
    :return: One record per agent.
    :rtype: list[dict[str, Any]]
    """
    return [{"agent_index": i,
             **{f"true_{n}": float(v) for n, v in
                zip(variant.free_names, agent_truth(study_seed, i, lo, hi))}}
            for i in range(n_agents)]


def write_outputs(out_dir: Path, episode_rows: list[dict[str, Any]],
                  truth_rows: list[dict[str, Any]], variant: FitVariant,
                  paired: FitVariant, n_agents: int, n_maps: int, n_fit: int,
                  extras: Sequence[FitVariant] = ()) -> bool:
    """Stage 5: aggregate per agent and emit every read-out-ready file, plus DONE.

    :param out_dir: The study directory.
    :type out_dir: Path
    :param episode_rows: Every completed episode fit.
    :type episode_rows: list[dict[str, Any]]
    :param truth_rows: One row per agent in the whole study.
    :type truth_rows: list[dict[str, Any]]
    :param variant: The study's variant.
    :type variant: FitVariant
    :param paired: The paired arm.
    :type paired: FitVariant
    :param n_agents: Agents expected.
    :type n_agents: int
    :param n_maps: Maps per agent.
    :type n_maps: int
    :param n_fit: Episodes fitted in this process, for the log line.
    :type n_fit: int
    :param extras: The extra arms (module docstring, EXTRA ARMS).
    :type extras: Sequence[FitVariant]
    :return: True when every (agent, map) is present with every extra arm fitted.
    :rtype: bool
    """
    main_columns = fit_columns(variant)
    aggregate = aggregate_rows(episode_rows, truth_rows, main_columns)
    # | 1. Each secondary arm is aggregated over the rows that hold it, so chunks merged with
    # |    differing arm lists do not break the files of the arms every row has.
    secondary = [paired] + list(extras)
    per_arm: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    for arm in secondary:
        prefix = f"{arm.name}_"
        arm_rows = [r for r in episode_rows if has_arm(r, arm)]
        arm_aggregate = aggregate_rows(arm_rows, truth_rows, fit_columns(arm, prefix),
                                       prefix=prefix)
        per_arm[arm.name] = (arm_rows, arm_aggregate)
        by_agent = {r["agent_index"]: r for r in arm_aggregate}
        for record in aggregate:
            record.update({k: v for k, v in by_agent.get(record["agent_index"], {}).items()
                           if k.startswith(arm.name)})
    aggregate_fields = ["agent_index", "n_successful_episodes"] + sorted(
        {k for r in aggregate for k in r} - {"agent_index", "n_successful_episodes"})
    write_csv(out_dir / "agent_aggregates.csv", aggregate, aggregate_fields)
    for how in ("mean", "median"):
        write_recovery_rows(out_dir / f"recovery_rows_{how}.csv", episode_rows, aggregate, how,
                            main_columns)
        for arm in secondary:
            arm_rows, arm_aggregate = per_arm[arm.name]
            write_recovery_rows(out_dir / f"recovery_rows_{arm.name}_{how}.csv",
                                arm_rows, arm_aggregate, how,
                                fit_columns(arm, f"{arm.name}_"), prefix=f"{arm.name}_")

    # | 2. The summary, and DONE only when every episode carries every extra arm too.
    missing = {arm.name: len(episode_rows) - len(per_arm[arm.name][0]) for arm in extras}
    complete_agents = sum(int(r["n_successful_episodes"]) == n_maps for r in aggregate)
    with io.open(out_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump({"variant": variant.name, "paired_arm": paired.name,
                   "extra_arms": [arm.name for arm in extras],
                   "episodes_missing_extra_arm": missing,
                   "n_agents_requested": n_agents,
                   "n_agents_with_results": len(aggregate),
                   "n_episode_fits": len(episode_rows),
                   "expected_episode_fits": n_agents * n_maps,
                   "complete_agents": complete_agents,
                   "n_stalled_fits": int(sum(int(float(r.get("no_progress", 0) or 0))
                                             for r in episode_rows))},
                  fh, indent=2, sort_keys=True)
    log(f"  fitted {n_fit} episodes this run; {len(episode_rows)} on disk; "
        f"{complete_agents}/{n_agents} agents complete"
        + "".join(f"; {name} missing on {n}" for name, n in missing.items()))
    complete = len(episode_rows) == n_agents * n_maps and not any(missing.values())
    if complete:
        (out_dir / "DONE").write_text(f"{len(episode_rows)}\n", encoding="utf-8")
        log("  every episode present -> DONE")
    return complete


def episode_fields(variant: FitVariant, paired: FitVariant,
                   extras: Sequence[FitVariant] = ()) -> list[str]:
    """The `episode_fits.csv` column order for this study's arms.

    :param variant: The study's variant.
    :type variant: FitVariant
    :param paired: The paired arm.
    :type paired: FitVariant
    :param extras: The extra arms, each also given a `<arm>_seconds` column.
    :type extras: Sequence[FitVariant]
    :return: Column names.
    :rtype: list[str]
    """
    fields = ["agent_index", "map_index", "map_id", "n_frames", "seconds", "variant",
              "sim_collisions", "sim_median_speed", "sim_episode_length", "logL_mean"]
    fields += [f"true_{n}" for n in variant.free_names]
    arms = [(variant, "")] + [(arm, f"{arm.name}_") for arm in [paired] + list(extras)]
    for arm, prefix in arms:
        fields += [f"{prefix}iters", f"{prefix}logL_mode", f"{prefix}no_progress"]
        fields += list(fit_columns(arm, prefix).values())
        fields += [f"{prefix}mode_{n}" for n in arm.free_theta_names]
        fields += [f"{prefix}{n}_{t}" for n in arm.free_theta_names for t in POSTERIOR_TAGS]
    fields += [f"{arm.name}_seconds" for arm in extras]
    if starter_arms(paired, extras):
        fields.append(f"{paired.name}_start")      # which start won: "midpoint" or an extra arm
    return fields


# ===========================================================================
# 4. Entry point
# ===========================================================================


def parse_agent_range(spec: str | None, n_agents: int) -> tuple[int, int]:
    """``"lo:hi"`` -> ``(lo, hi)`` clipped to ``[0, n_agents]``; None means every agent.

    :param spec: The `--agents` argument.
    :type spec: str | None
    :param n_agents: Agents in the whole study.
    :type n_agents: int
    :return: Half-open range.
    :rtype: tuple[int, int]
    :raises ValueError: On a malformed or empty range.
    """
    if spec is None:
        return 0, n_agents
    lo_s, _, hi_s = spec.partition(":")
    lo, hi = int(lo_s or 0), min(int(hi_s or n_agents), n_agents)
    if not 0 <= lo < hi:
        raise ValueError(f"--agents {spec!r} is empty or outside [0, {n_agents})")
    return lo, hi


def merge_chunks(out_dir: Path, pattern: str, variant: FitVariant, paired: FitVariant,
                 lo: np.ndarray, hi: np.ndarray, n_agents: int, n_maps: int,
                 study_seed: int, extras: Sequence[FitVariant] = ()) -> bool:
    """`--merge-chunks`: concatenate chunk directories' episode fits and write every aggregate.

    :param out_dir: Where the merged study lands.
    :type out_dir: Path
    :param pattern: Glob matching the chunk directories.
    :type pattern: str
    :param variant: The study's variant.
    :type variant: FitVariant
    :param paired: The paired arm.
    :type paired: FitVariant
    :param extras: The extra arms.
    :type extras: Sequence[FitVariant]
    :param lo: Lower bound per free coefficient, for regenerating the truths.
    :type lo: np.ndarray
    :param hi: Upper bound per free coefficient.
    :type hi: np.ndarray
    :param n_agents: Agents in the whole study.
    :type n_agents: int
    :param n_maps: Maps per agent.
    :type n_maps: int
    :param study_seed: Seed identifying this study.
    :type study_seed: int
    :return: True when the merge is complete.
    :rtype: bool
    """
    paths = sorted(Path(d) / "episode_fits.csv" for d in glob.glob(pattern)
                   if (Path(d) / "episode_fits.csv").exists())
    if not paths:
        raise SystemExit(f"no episode_fits.csv under any directory matching {pattern}")
    header, rows = merge_rows(paths, ["agent_index", "map_index"])
    log(f"  merged {len(rows)} episode fits from {len(paths)} chunks")
    write_csv(out_dir / "episode_fits.csv", rows, header)
    truth_rows = truth_rows_for(variant, lo, hi, n_agents, study_seed)
    write_csv(out_dir / "agent_truths.csv", truth_rows,
              ["agent_index"] + [f"true_{n}" for n in variant.free_names])
    return write_outputs(out_dir, rows, truth_rows, variant, paired, n_agents, n_maps, 0, extras)


def main() -> None:
    """CLI: draw agents, simulate and fit every (agent, map) in this chunk, aggregate, write.

    Numbered stages, each announced in the log (Rule 46):
    1. Load the models, the variant's box and the map set.
    2. Write the manifest, so the run is reconstructible from its own output.
    3. Resume: skip (agent, map) pairs already on disk; a recorded pair that lacks an extra
       arm stops the run, since this study cannot add an arm to a finished episode.
    4. Simulate and fit each remaining pair -- any extra arm the paired arm starts from, the
       paired arm, then the variant warm-started from it, then the other extra arms
       (:func:`fit_new_episode`) -- flushing after every one.
    5. Aggregate per agent and write the read-out-ready files.
    """
    global _LOG_PATH
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hlp-checkpoint-dir", required=True)
    ap.add_argument("--hlp-model-filename", default="hlp_model_final.pt")
    ap.add_argument("--data-path", default="human_modeling/data/exp1_filtered.json",
                    help="source of the maps; parsed once, then cached as maps.json")
    ap.add_argument("--results-dir", required=True)
    # | [full-visibility] Only the attention-choosing variants: this study prepares and
    # | simulates ordinary episodes, so a full-visibility arm would fail on every one (Entry 36).
    ap.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(ATTENTION_CHOOSING_VARIANTS),
                    help=f"the model that generates the agents and is fitted to them; its "
                         f"free coefficients are the truths. Default {DEFAULT_VARIANT} "
                         f"reproduces the Entry 31 study")
    ap.add_argument("--paired-arm", default="B_no_heuristics",
                    choices=list(ATTENTION_CHOOSING_VARIANTS),
                    help="a second variant fitted to every episode first (the variant is "
                         "then warm-started from it). Default B_no_heuristics, the theta-only "
                         "fit -- deliberately NOT the human chain's parent of A_full, which is "
                         "A_no_mean_distance since Entry 35 addendum 2")
    ap.add_argument("--extra-arms", nargs="*", default=[],
                    choices=list(ATTENTION_CHOOSING_VARIANTS),
                    metavar="NAME",
                    help="further variants fitted to every episode from the box midpoint, "
                         "independently of the other arms. Ex: --extra-arms C_no_cognitive_cost")
    ap.add_argument("--n-agents", type=int, default=2,
                    help="agents in the WHOLE study. Default 2 completes locally in minutes; "
                         "Entry 31 fixes the study size at 100, passed by the sbatch")
    ap.add_argument("--agents", default=None, metavar="LO:HI",
                    help="only agents lo <= i < hi in this process (a job-array chunk); "
                         "default every agent")
    ap.add_argument("--n-maps", type=int, default=2,
                    help=f"distinct human maps per agent, at most {N_HUMAN_MAPS}. Default 2 is "
                         f"the local smoke; the study uses all {N_HUMAN_MAPS}")
    ap.add_argument("--merge-chunks", default=None, metavar="GLOB",
                    help="merge mode: concatenate these chunk directories' episode fits into "
                         "--results-dir and write every aggregate file; fits nothing. Exits 1 "
                         "unless every (agent, map) is present, so a chain gated on it stops")
    ap.add_argument("--study-seed", type=int, default=20260921,
                    help="seeds the truths and the simulations; fixed so a rerun reproduces them")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if args.n_agents < 1:
        raise ValueError(f"--n-agents must be positive, got {args.n_agents}")
    if not 1 <= args.n_maps <= N_HUMAN_MAPS:
        raise ValueError(f"--n-maps must be between 1 and {N_HUMAN_MAPS}, got {args.n_maps}")
    variant = get_variant(args.variant)
    paired = get_variant(args.paired_arm)
    if paired.name == variant.name:
        raise ValueError("the paired arm must differ from the variant")
    extras = [get_variant(n) for n in args.extra_arms]
    taken = {variant.name, paired.name}
    if len(set(args.extra_arms)) != len(extras) or taken & set(args.extra_arms):
        raise ValueError(f"--extra-arms {args.extra_arms} must be distinct and differ from the "
                         f"variant and the paired arm ({sorted(taken)})")
    agent_lo, agent_hi = parse_agent_range(args.agents, args.n_agents)

    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _LOG_PATH = out_dir / "study_log.txt"
    log(f"=== synthetic parameter recovery: variant {variant.name} + paired {paired.name}"
        f"{' + extra ' + ', '.join(a.name for a in extras) if extras else ''}, "
        f"{args.n_agents} agents x {args.n_maps} maps, agents [{agent_lo}, {agent_hi}) ===")

    # | 1. Models, the variant's box, and the maps.
    stack = load_models(args.hlp_checkpoint_dir, args.hlp_model_filename, args.device)
    lo, hi = variant.lo_hi(stack)
    log("  ranges: " + ", ".join(f"{n.replace('hlp_', '')}=[{lo[j]:g},{hi[j]:g}]"
                                 for j, n in enumerate(variant.free_names)))
    if args.merge_chunks:
        complete = merge_chunks(out_dir, args.merge_chunks, variant, paired, lo, hi,
                                args.n_agents, args.n_maps, args.study_seed, extras)
        raise SystemExit(0 if complete else 1)
    maps = load_or_extract_maps(out_dir, Path(args.data_path), args.n_maps)

    # | 2. Manifest: everything needed to reconstruct the run, per Rule 51.
    manifest = {"study_seed": args.study_seed, "n_agents": args.n_agents,
                "agents_in_this_process": [agent_lo, agent_hi],
                "n_maps": args.n_maps, "checkpoint_dir": str(args.hlp_checkpoint_dir),
                "model_filename": args.hlp_model_filename, "data_path": str(args.data_path),
                "variant": variant.name, "paired_arm": paired.name,
                "extra_arms": [arm.name for arm in extras],
                "coefficient_names": list(variant.free_names),
                "coefficient_bounds": {n: [float(lo[j]), float(hi[j])]
                                       for j, n in enumerate(variant.free_names)},
                "map_ids": [m["map_id"] for m in maps],
                "truth_sampling": "independent uniform over each coefficient's range "
                                  "(trained range for theta, calibrated +-bound for a weight)",
                "strata": None}
    with io.open(out_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    # | 3. Resume. A walltime kill loses at most the episode in flight.
    episode_path = out_dir / "episode_fits.csv"
    episode_rows: list[dict[str, Any]] = []
    if episode_path.exists():
        with io.open(episode_path, encoding="utf-8") as fh:
            episode_rows = list(csv.DictReader(fh))
        log(f"  RESUMING: {len(episode_rows)} episode fits already on disk")
    done = {(int(r["agent_index"]), int(r["map_index"])): r for r in episode_rows}
    fields = episode_fields(variant, paired, extras)
    truth_rows = truth_rows_for(variant, lo, hi, args.n_agents, args.study_seed)
    write_csv(out_dir / "agent_truths.csv", truth_rows,
              ["agent_index"] + [f"true_{n}" for n in variant.free_names])

    # | 4. The work. Every pair is independent, so a failure is logged and the run continues.
    n_fit = 0
    for truth_row in truth_rows[agent_lo:agent_hi]:
        agent_index = int(truth_row["agent_index"])
        free_truth = np.array([truth_row[f"true_{n}"] for n in variant.free_names])
        full_truth = variant.embed(free_truth)
        for map_index, map_record in enumerate(maps):
            # | A pair on disk is done. A recorded row lacking an extra arm can only come from
            # | a run with a different --extra-arms list; this study does not add an arm to a
            # | finished episode, so it stops rather than merge the two silently.
            recorded = done.get((agent_index, map_index))
            if recorded is not None:
                lacking = [arm.name for arm in extras if not has_arm(recorded, arm)]
                if lacking:
                    raise SystemExit(f"agent {agent_index} map {map_index} is on disk without "
                                     f"{lacking}; fit the study with these arms in a fresh "
                                     f"--results-dir")
                continue
            t0 = time.time()
            try:
                episode, rollout, metrics = simulate_episode(
                    stack, map_record["map"], full_truth, agent_index, map_index, args.study_seed)
                label = f"a{agent_index:04d}m{map_index}"
                inp = prepare_episode(stack, episode, label=label)
                row: dict[str, Any] = {
                    "agent_index": agent_index, "map_index": map_index,
                    "map_id": map_record["map_id"], "n_frames": episode.num_steps,
                    "variant": variant.name,
                    "sim_collisions": rollout["collisions"],
                    "sim_median_speed": metrics["median_speed"],
                    "sim_episode_length": rollout["episode_length"],
                    **{f"true_{n}": truth_row[f"true_{n}"] for n in variant.free_names}}
                # | The chain the human fit runs (variants.py, WARM STARTS): C, then B from the
                # | midpoint and from C, then A from B, then any other extra arm.
                est, info = fit_new_episode(row, stack, inp, variant, paired, extras)
                row["seconds"] = round(time.time() - t0, 1)
                episode_rows.append(row)
                done[(agent_index, map_index)] = row
                n_fit += 1
                railed = sum(est[f"{n}_edge_mass"] >= 0.30 for n in variant.free_theta_names)
                log(f"  agent {agent_index + 1}/{args.n_agents} map {map_index + 1}/"
                    f"{args.n_maps}: {episode.num_steps} frames, railed {railed}/"
                    f"{variant.n_free_theta}, logL_mode={row['logL_mode']:.1f}"
                    f" (paired {row[f'{paired.name}_logL_mode']:.1f}), {time.time() - t0:.0f}s")
                # | [boundary-start] The paired arm starts at the midpoint, so its no-progress
                # | fit is always a stall; the variant's may be the paired arm's boundary optimum.
                boundary = boundary_coefficients(variant, info)
                if boundary:
                    log(f"    NOTE: {variant.name} kept {paired.name}'s solution with "
                        f"{', '.join(boundary)} at a bound -- a boundary optimum, which IS an "
                        f"estimate (run_human_inference.boundary_coefficients)")
                if (info["no_progress"] and not boundary) or row[f"{paired.name}_no_progress"]:
                    log("    WARNING: an ascent never improved on its start (no_progress = 1); "
                        "that arm's row is NOT an estimate -- see implementation.md Entry 34")
            except Exception as exc:              # noqa: BLE001 - long-run isolation
                with io.open(out_dir / "failures.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"agent_index": agent_index, "map_index": map_index,
                                         "error": repr(exc),
                                         "traceback": traceback.format_exc()}) + "\n")
                log(f"  agent {agent_index} map {map_index} FAILED: "
                    f"{type(exc).__name__}: {exc}")
            write_csv(episode_path, episode_rows, fields)
    # | Rewritten once more so the header carries every arm's columns even when nothing here
    # | needed fitting -- the merge refuses chunk files whose headers differ.
    write_csv(episode_path, episode_rows, fields)

    # | 5. Aggregate and emit the read-out-ready files. In a chunked run the expected count is
    # |    this chunk's, and the merge step produces the whole-study files.
    n_agents_here = agent_hi - agent_lo if args.agents else args.n_agents
    write_outputs(out_dir, episode_rows, truth_rows, variant, paired, n_agents_here,
                  args.n_maps, n_fit, extras)
    log(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
