# CCDIP — repository context

What the repository is now: structure, the data that flows between its parts, and the
dependencies. Each package's `CONTEXT.md` describes that package's files. `README.md` has the
setup and the commands that reproduce the paper.

## Top level

```
drive_env/                  the driving game, its Gymnasium interface and the browser build
modules/base_models/SP/     subtask policy (code name LLP): goal-conditioned low-level driving
modules/base_models/SSP/    subtask-selection policy (code name HLP): goal + scope, conditioned on θ
human_modeling/data/        exp1_filtered.zip -> exp1_filtered.json, the human telemetry
human_modeling/inference/   cognitive-profile inference, model comparison, recovery, validation
extras/server_instructions/ NYU HPC notes, Slurm job files, the GPU heartbeat helper
index.html                  browser entry point (Pyodide)
pyproject.toml, uv.lock     dependencies (Python >= 3.13); torch and pufferlib are installed separately
```

`CLAUDE.md`, `ContextRules/`, `.claude/` and `pyrightconfig.json` are local assistant and
editor tooling, gitignored. `to_delete/` (gitignored) parks files cut from the repository
until the paper's results have been reproduced from what remains. `temp/` (gitignored) takes
scratch output: smoke runs, and cluster job results before they are copied into the tracked
`human_modeling/inference/results/`.

## drive_env/ — the game

```
drive_env/
├── __init__.py             registers CarDrivingEnv-v0 with Gymnasium
├── interface_gym.py        Gymnasium wrapper around the MDP, plus the MultiEncoder network input layout
├── interface_web.py        the browser interface (Pyodide / JS)
├── video_generator.py      headless Pygame rendering of rollouts to MP4
└── core/
    ├── config.py           shared constants: frame rate, tile size, speeds, visible window, map layout
    ├── mdp.py              the simulator: physics, objects, observations, rewards; no rendering dependency
    ├── utils.py            map and observation-window helpers
    ├── telemetry.py        the per-frame record format the human experiment stores
    ├── data_converter.py   human telemetry <-> RL observation conversion
    ├── assets/             PNG sprites
    └── maps/
        ├── mapgenerator.py procedural road generation from a construction seed
        └── objects.py      tile and object registry (sizes, speeds, crash and reward flags, sprites)
```

The MDP exposes two observations: `observationRL()`, a flat vector of up to 200 objects x 7
features `(dx, dy, vx, vy, crash_flag, interact_flag, type_id)` for the networks, and
`observationHuman()`, a dict for data collection. Physics is pure NumPy, the same code in the
browser and in training. A map is fully determined by its
construction seed, which is how the inference package rebuilds a human's route and how the
evaluation harness pairs maps across conditions.

## The hierarchy

The SSP observes the processed state, up to K = 6 car slots, the nearest flag and θ (scaled to
-1..1), and emits a goal (lane x forward-distance bin) and a car mask. The SP then drives
toward that goal for one sub-episode, seeing only the unmasked cars (masked cars are encoded
exactly like absent ones). θ enters the SSP's reward as per-coefficient penalties (collision,
attentional cost per attended car per step, goal proximity, goal-flag distance); each
coefficient is sampled per episode from a trained range recorded in `hlp_metadata.json`, and
the policy loader clamps values outside it.

The pair used by every paper result is committed in `human_modeling/inference/checkpoint/`.

## The human data

`exp1_filtered.json`: 230 episodes from 39 participants on 6 distinct routes, each episode a
list of telemetry frames (`telemetry.py` format) with the map's construction seed. The
inference package reads the JSON directly; the zip is the tracked copy.

## Dependencies

`pyproject.toml`: gymnasium, numpy, scipy, statsmodels (participant tests), matplotlib,
loguru, tqdm, tyro (trainer CLIs), tensorboard (training curves), pygame and imageio[pyav]
(videos), pytest. Installed separately: torch (CPU or CUDA build per machine) and, for
cluster training only, pufferlib (vectorised environments).
