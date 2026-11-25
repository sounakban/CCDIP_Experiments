# CCDIP: Cognition-Conditioned Deep Inverse Planning

Code, data and trained policies for *Inferring Cognitive Profiles with Cognition-Conditioned
Deep Inverse Planning*. The repository holds exactly what the paper's experiments need: the
driving environment, the two-level policy hierarchy, the inference package that fits cognitive
profiles to human driving, the human data, the trained checkpoint, and the cluster job files
that produced the reported numbers.

## The pipeline

| stage | package | what it does |
|---|---|---|
| environment | `drive_env/` | 2D top-down driving game: physics, procedural maps, flags, traffic; a Gymnasium interface for training and a Pyodide build for the browser experiment |
| subtask policy (SP) | `modules/base_models/SP/` | goal-conditioned low-level driving policy, trained with PPO against randomly sampled goals; acts on a *construed* state (masked cars are encoded as absent) |
| subtask-selection policy (SSP) | `modules/base_models/SSP/` | chooses, once per subtask, a goal (lane + forward distance) and a scope (which cars the SP sees), conditioned on a cognitive-profile vector θ; trained with PPO on top of the frozen SP |
| inference | `human_modeling/inference/` | fits θ and the heuristic weights to each human episode by maximum likelihood with exact gradients, builds a local posterior, compares model variants, validates fitted agents by rollout, and runs the synthetic recovery study |

Two names differ between the paper and the code. The code calls the SP the *LLP* (low-level
policy, prefix `llp_`) and the SSP the *HLP* (high-level policy, prefix `hlp_`). The package
directories carry the paper's names; the module, function and metadata names inside carry the
code's.

## Layout

```
drive_env/                     the game (core/mdp.py is the simulator; core/maps/ generates roads)
modules/base_models/SP/        SP training, rollout video, tests
modules/base_models/SSP/       SSP training, policy loader, evaluation harness, Figure 2b script, tests
human_modeling/data/           exp1_filtered.zip: the human telemetry (230 episodes, 39 participants, 6 routes)
human_modeling/inference/      the inference package, its tests, the checkpoint and the stored results
extras/server_instructions/    NYU HPC setup notes and the Slurm job files of every cluster run
```

Each package has a `CONTEXT.md` describing its files and the data layouts they share.

## Setup

Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # everything in pyproject.toml
uv pip install torch                      # CPU build; see pytorch.org for a CUDA build
uv pip install pufferlib==3.0.0           # optional, Unix only: the vectorised training backend
```

torch is deliberately absent from `pyproject.toml` because the right build depends on the
machine. Anything installed with `uv pip install` is outside `uv.lock`, so repeat those two
lines after any later `uv sync`.

Unzip the human data once; every inference script reads the JSON:

```bash
python -c "import zipfile; zipfile.ZipFile('human_modeling/data/exp1_filtered.zip').extractall('human_modeling/data')"
```

Run the tests from the repo root:

```bash
uv run python -m pytest modules/base_models/SP/tests modules/base_models/SSP/tests human_modeling/inference/tests -q
```

The cluster setup (Singularity image, conda overlay, pufferlib, CUDA torch) is in
`extras/server_instructions/NYU_HPC.md`. Every job file is submitted from the clone's root.

## The checkpoint

`human_modeling/inference/checkpoint/` holds the SP and SSP used for every result in the
paper: `llp_model_final.pt` + `hrl_llp_metadata.json` (the SP and its training settings) and
`hlp_model_final.pt` + `hlp_metadata.json` (the SSP, its settings, and the trained range of
each θ coefficient). Every script below takes it through `--run-dir` or
`--hlp-checkpoint-dir`. Coefficients outside their trained range are clamped by the policy
loader, so sweeps and fits stay inside the ranges recorded in `hlp_metadata.json`.

## Reproducing the paper

| paper item | command or job | output |
|---|---|---|
| train the SP | `sbatch extras/server_instructions/scripts/train_hlr_llp.sbatch` | a run directory with `model_final.pt` and `hrl_llp_metadata.json` |
| train the SSP on a frozen SP | `sbatch extras/server_instructions/scripts/train_hlr_hlp.sbatch` (point `--llp_checkpoint_dir` at the SP run) | `hlp_model_final.pt`, `hlp_metadata.json`, copies of the SP files |
| Figure 1c, rollout videos | `uv run python modules/base_models/SSP/simulate_hlp.py --run-dir human_modeling/inference/checkpoint` | `rollout.mp4` and a per-step JSONL trace |
| Figure 2b, cost-performance trade-off | `uv run python modules/base_models/SSP/plot_cost_performance.py --sims 40 --points 9` | `human_modeling/inference/results/cost_performance/cost_performance_hlp_cognitive_cost_coef.{csv,png,pdf}` and the pilot table `_pilot.csv` |
| heuristic-weight bounds (Appendix) | `uv run python -m human_modeling.inference.scripts.calibrate_heuristic_bounds --hlp-checkpoint-dir human_modeling/inference/checkpoint --output-json <path>` | the bounds; the adopted values are `HEURISTIC_BOUNDS` in `human_modeling/inference/model/choice.py` |
| model comparison, validation rollouts, synthetic recovery | `bash extras/server_instructions/scripts/inference_model_comparison.sbatch` (submits three chained jobs) | `fit/`, `validate/`, `recovery/` under the job's results root |
| per-participant BIC tests (Wilcoxon, Holm) | `uv run python -m human_modeling.inference.scripts.participant_bic_tests --results-dir <fit dir> --variants C_no_cognitive_cost B_no_heuristics A_full` | `participant_bic_tests.{csv,txt}` |
| between-participant share of each coefficient (ICC) | `uv run python -m human_modeling.inference.scripts.parameter_variance --fit-csv <fit dir>/fit_variant_A_full.csv --out-dir <fit dir>` | `parameter_variance_A_full.{csv,txt}` |
| full-visibility sanity check | `sbatch extras/server_instructions/scripts/inference_full_visibility_sanity.sbatch` | `delta_bic_full_visibility.{csv,txt}` and the refit check |

The stored outputs of the cluster runs behind the paper are in
`human_modeling/inference/results/model_comparison/`: the four fitted variants
(`fit/fit_variant_*.csv`), the comparison tables, the participant tests, the variance
decomposition, the validation rollouts per variant, the recovery study, and the full-visibility
sanity check. The scripts above regenerate every table and figure from those CSVs.

Every inference script runs locally at a smoke size (`--max-episodes 3`, a 2x2 recovery) so a
change can be checked before a cluster submission. For example, fitting three human episodes
under all four variants:

```bash
uv run python -m human_modeling.inference.scripts.run_human_inference \
    --hlp-checkpoint-dir human_modeling/inference/checkpoint --results-dir temp/smoke_fit \
    --variants C_no_cognitive_cost B_no_heuristics A_no_mean_distance A_full --max-episodes 3
```

## Playing the game

Open `index.html` in a desktop browser; Pyodide loads the environment. If the browser blocks
`file://` origins, serve the folder with `python -m http.server 8000`. Keys: W/S accelerate
and brake, A/D steer (or I/K/J/L).

## License

See `LICENSE`.
