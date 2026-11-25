# inference — cognitive-profile inference from human driving

Fits the cognitive-profile vector θ of the subtask-selection policy (SSP, `modules/base_models/SSP/`)
and the four heuristic weights to each human episode, by maximum likelihood with exact
gradients, then describes a local posterior around the peak. Around that estimator: the model
variants the paper compares, the comparison statistics, validation of fitted agents by rollout,
and the synthetic recovery study. The package is self-contained: it imports the SSP and SP
packages and `drive_env`, nothing else in `human_modeling`.

## How an episode is scored

1. `inputs.py` rebuilds the episode's map from its construction seed and turns the telemetry
   frames into per-step world states.
2. `model/emission.py` builds the emission table: for every step, every candidate subtask
   (goal x construal mask), the SP's log-probability of the human's action. The observation
   encoder runs once per (step, mask) and its output is shared across the 96 goals, since the
   encoder never sees the goal.
3. `model/choice.py` gives the SSP's subtask log-probabilities at θ, plus the heuristic
   values (lead-vehicle inverse time-to-collision, cars in the goal path, closest and mean car
   distance) weighted by the four fitted weights, bounded by `HEURISTIC_BOUNDS`.
4. `model/kernel.py` builds the episode kernel: the candidate subtasks at each step, which
   of them end there, and how the rest carry over to the next step.
5. `inference.py` (numpy) and `inference_torch.py` (torch, bit-matched) run the forward
   recursion of this hidden-Markov model over subtasks and return the episode log-likelihood.
6. `likelihood_grad.py` wraps 2-5 into one differentiable torch function of the fitted vector,
   4 coefficients (θ alone, weights pinned at 0) or 8 (θ then the weights). An episode prepared
   with `full_visibility=True` has one construal column per step (every car visible).
7. `fit_gradient.py`: `fit_mode` finds the peak by projected Adam ascent inside the trained
   box; `fit_posterior` tabulates an 81-point lattice around it, sized by the curvature at the
   peak (`hessian_diag`, `curvature_half_widths`), and reports the mean, median, interval and
   edge mass per coefficient.
8. `posterior/` holds the lattice machinery: `grid.py` (lattice construction and zoom),
   `likelihood.py` (the expensive table and the arithmetic on it), `update.py` (posterior
   weights and summaries).

## Model variants (`variants.py`)

A `FitVariant` names its free coefficients and pins the rest. `C_no_cognitive_cost` (3 free θ,
cognitive cost pinned at 0), `B_no_heuristics` (4 θ), `A_no_mean_distance` (7), `A_full` (8).
They are a nested chain fitted smallest first (`FIT_ORDER`): each warm-starts from the one it
contains, and B also tries C's solution beside the midpoint start, so a larger model never
fits worse. `NESTED_PAIRS` names the chain links the comparison checks. `C_full_visibility`
(C's coefficients with every car always visible) sits outside the chain as a sanity check.
`embed`/`embed_torch` turn a variant's free vector into the 8-vector the likelihood reads.

## Files

| file | what it is |
|---|---|
| `checkpoint/` | the SP + SSP pair used for every paper result: `hlp_model_final.pt`, `hlp_metadata.json`, `llp_model_final.pt`, `hrl_llp_metadata.json`; pass as `--hlp-checkpoint-dir` |
| `scripts/run_human_inference.py` | fits every human episode under the requested `--variants` in chain order; one `fit_variant_<name>.csv` per variant with the posterior mean per coefficient, the peak (`mode_<coef>`), median, interval, edge mass, the weights `w_<name>`, `n_free_params`, `n_frames`, `logL_mean`, `logL_mode` and the ascent diagnostics; resumable per (episode, variant); `--chunk-index/--n-chunks` splits the 230 episodes for a job array |
| `scripts/merge_chunk_csvs.py` | concatenates a job array's per-chunk CSVs into one and exits 1 unless the row count matches `--expect` |
| `scripts/plot_model_comparison.py` | per-episode NLL and BIC (`k ln n - 2 logL` at the peak), the paired BIC difference against `B_no_heuristics` with a participant bootstrap, best-fit counts at the BIC-margin-10 rule, nesting-violation counts per chain link; `model_comparison.csv`, `model_comparison_summary.txt` and the figures |
| `scripts/participant_bic_tests.py` | the paper's test: per pair of variants, each participant's mean per-episode BIC difference, a participant bootstrap, a Wilcoxon signed-rank test, Holm across pairs; `participant_bic_tests.{csv,txt}` |
| `scripts/parameter_variance.py` | between- and within-participant variance of each fitted coefficient of one variant (route means removed, one-way random effects, participant-bootstrap intervals); each episode's route is read from the human data (the map's content hash); `parameter_variance_<variant>.{csv,txt}` |
| `estimation/validate.py` | rolls every row of a fit CSV out on its own human map and regresses the behavioural metrics (collisions, lane changes per 1,000 steps, median speed, evasive shares) against the human's; resumable per episode, `--workers N` parallel |
| `estimation/markers.py` | `evasion_markers`, the collision-specific behavioural shares `validate.py` reports |
| `scripts/plot_by_person.py` | the validation regression plot with one point per participant instead of per episode |
| `scripts/run_synthetic_recovery.py` | the recovery study: agents with known coefficients drawn uniformly over their ranges, simulated on every human map, each episode fitted; the warm-start parent is fitted as the paired arm and `--extra-arms` adds further variants; `--agents lo:hi` chunks for a job array, `--merge-chunks` reassembles; writes `recovery_rows_{mean,median}.csv` |
| `scripts/analyze_param_recovery.py` | recovery read-out per coefficient: Spearman rho with bootstrap CI and reliability ceiling, RMSE and mean bias, raw and as a share of the range; `param_recovery_metrics.csv` and the true-vs-estimated figures |
| `scripts/full_visibility_sanity.py` | per-episode delta BIC between `C_full_visibility` and `C_no_cognitive_cost`, its mean, SE and participant-bootstrap interval, an equivalence verdict, and a refit check against the stored C |
| `scripts/calibrate_heuristic_bounds.py` | the heuristic-weight bounds: one pass over sampled human decisions yields the scale the weights compete against and a bound per heuristic; run once per checkpoint; the adopted values are `HEURISTIC_BOUNDS` in `model/choice.py` |
| `scripts/check_slot_order.py` | gate on `hlp_extract_car_rows` over every step of all 230 episodes, per device: real cars must occupy the first `num_cars` slots; exits 1 on a violation |
| `scripts/bootstrap_stats.py` | `bootstrap_correlation` (Pearson) and `bootstrap_spearman`, each with a percentile-bootstrap interval |
| `tests/` | checkpoint-free and checkpoint-backed gates on every module above; the two cluster fit jobs run the suite before fitting. The checkpoint-backed tests read `human_modeling/data/exp1_filtered.json` |

Run the tests with `uv run python -m pytest human_modeling/inference/tests -q`.

## Results

`results/model_comparison/` is the stored output of the cluster run behind the paper:

```
fit/          fit_variant_{C_no_cognitive_cost,B_no_heuristics,A_no_mean_distance,A_full}.csv,
              model_comparison.csv + _summary.txt, participant_bic_tests.*, parameter_variance_A_full.*
validate/     validate_simulated_<variant>.csv for C, B and A (20 rollouts per episode)
recovery/     agent_truths.csv, episode_fits.csv, agent_aggregates.csv, recovery_rows_*.csv,
              metrics_<variant>_{mean,median}/param_recovery_metrics.csv, summary.json
sanity_full_visibility/   fit_variant_C_full_visibility.csv, the refit C, delta_bic_full_visibility.*
```

`results/cost_performance/` is Figure 2b, run locally (not on the cluster) by
`modules/base_models/SSP/plot_cost_performance.py --sims 40 --points 9` on `checkpoint/`: the
sweep table, the pilot table that chose the swept costs, and the figure as PNG and PDF.

## Cluster jobs (`extras/server_instructions/scripts/`)

`inference_model_comparison.sbatch` (`bash <file>` submits three jobs): `fit` (GPU: the test
suite and the slot-order gate, then all four variants over 10 round-robin chunks, merge,
comparison plots for C, B, A), `validate` (CPU, after `fit`: the reported variants at 20
rollouts per episode, then the person-level plots) and `recovery` (GPU, independent: 100
agents x 6 maps in 10 chunks with B, A and C, merge, read-out). `inference_full_visibility_sanity.sbatch`:
the suite and the gate, then C and `C_full_visibility` on all 230 episodes, then the sanity
read-out against the stored C. Both read the committed `checkpoint/` and write under the
clone's results root.
