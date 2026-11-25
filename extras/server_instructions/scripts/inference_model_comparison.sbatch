#!/bin/bash

# ============================================================================
# HOW TO RUN
# ----------------------------------------------------------------------------
#   1. Go to the ROOT of the clone the jobs should run from (from the login node) and update
#      it. No path is written into this script: it runs from the clone it sits in, and finds
#      the overlay and slurm_output/ in that clone's parent directory.
#        cd <the clone>
#        git pull
#   2. Check the two inputs exist. The checkpoint arrives with the pull; the human data is
#      NOT in git (untracked), so it must have been copied into this clone once:
#        ls human_modeling/inference/checkpoint/hlp_metadata.json \
#           human_modeling/data/exp1_filtered.json
#   3. Submit the whole chain -- with `bash`, NOT `sbatch`:
#        bash extras/server_instructions/scripts/inference_model_comparison.sbatch
#      It prints three job IDs and exits within seconds.
#   4. Watch it:
#        squeue --me
#        tail -f ../slurm_output/inference_mc_fit_<JOBID>.out
#   5. If a job times out or is left incomplete, run step 3 again. Every stage skips work
#      already on disk (per episode, per variant, per synthetic agent). The merges refuse a
#      short result, so an episode that fails on every re-run has to be understood, not skipped.
#   6. Results, in the clone's gitignored temp/ (see RESULTS_ROOT below):
#      temp/reproduction/inference/model_comparison/
#        fit/model_comparison_summary.txt, fit/plots/         model comparison
#        fit/participant_bic_tests.*, fit/parameter_variance_A_full.*   the paper's tests and ICC
#        validate/<variant>/plots/                             five behavioural plots each
#        recovery/metrics_{A_full,B_no_heuristics,C_no_cognitive_cost}_{mean,median}/
#                                                              parameter recovery
#   7. Copy that folder into the LOCAL repo as human_modeling/inference/results/model_comparison/
#      and commit it there -- never from the server clone.
#
# WHY `bash` AND WHAT RUNS WHERE
#   Run with `bash`, this file is a DRIVER: on the login node it only checks that three
#   files exist, creates the log and results directories, and calls `sbatch` three times.
#   No Python, no container, no computation runs on the login node. Every stage is its own
#   SLURM job on a compute node. Submitted with `sbatch` directly, the file would run inside
#   a job with no STAGE set and exit immediately with an error.
# ============================================================================

# ============================================================================
# inference -- MODEL COMPARISON WITH HEURISTICS on all 230 human episodes, plus the
#                 parameter-recovery study with the heuristic weights included
#                 (inference implementation.md Entry 35 and its addenda)
# ----------------------------------------------------------------------------
# THE QUESTION
#   Fitted to each of the 230 human episodes, which of four models earns its parameters?
#   The four form a NESTED CHAIN and are fitted in this order (Entry 35 addendum 2), each
#   warm-started from the model it contains, so more parameters can never fit worse:
#     C_no_cognitive_cost  theta with the cognitive cost pinned at 0, its trained lower
#                          bound (3 free), no heuristics. NOT the AAAI "no construals"
#                          variant: the policy still chooses construals, only the attention
#                          cost is off (user decision 2026-09-23). Starts at the midpoint.
#     B_no_heuristics      theta only (4 free) -- the shipped V4 fit, the reference. Started
#                          TWICE, from the midpoint and from C's solution, keeping the better.
#     A_no_mean_distance   theta + three weights, mean_vehicle_distance pinned at 0 (7);
#                          started from B. FITTED ONLY AS A's WARM START, NOT REPORTED
#                          (user decision 2026-09-24, Entry 35 addendum 3): it tied A within
#                          10 BIC on 130 of 230 episodes, since the two distance heuristics
#                          are collinear at r = 0.987.
#     A_full               theta + all four heuristic weights (8); started from A'.
#   Read out as the AAAI pipeline did, over REPORTED_VARIANTS: per-episode NLL, BIC =
#   k ln(n) - 2 logL at the peak, the paired BIC difference against B with a participant
#   bootstrap, the best-fit count (BIC margin >= 10), and -- new -- the count of nesting
#   violations per reported link, which should read zero. Then, per reported variant, the
#   five behavioural recoverability plots
#   estimation/validate.py draws (collisions, lane changes per 1k frames, median speed,
#   evasive lane-change share, car-entry longitudinal share), each rolled out from that
#   variant's own fit.
#
#   And separately: with the weights among the fitted parameters, does the estimator
#   recover them? 100 synthetic agents with 8 known coefficients (theta uniform over its
#   trained ranges, weights uniform over the Entry 33 bounds) x the 6 human maps, each
#   episode fitted with B first and then A_full warm-started from B, so theta's recovery is
#   read WITH and WITHOUT the weights on the same episodes. (A starts from B here, not from
#   A': the study fits only A and its paired B arm -- user decision 2026-09-24.) C is fitted
#   to every episode too (RECOVERY_EXTRA_ARMS, user 2026-09-25), FIRST and from the midpoint,
#   and B then starts from the midpoint and from C's solution, keeping the better -- the human
#   fit's chain -- so all three reported models' likelihoods sit on the same synthetic episodes
#   with C nested in B in A (Entry 35 addendum 6).
#
# WHAT IT DOES NOT ANSWER
#   Anything about a full-visibility model. Anything about which CHECKPOINT is better (one
#   checkpoint throughout: the one committed with the package). Whether the fitted weights
#   are right on HUMAN data -- humans have no true weights; only the synthetic study speaks
#   to recovery, and only inside the trained ranges (Entry 31's caveat on misspecification
#   stands). Whether the variants differ in behavioural recovery is READ OFF the validate
#   plots by eye or by a later analysis; nothing here tests that difference formally.
#
# THE THREE JOBS (each a single SLURM job; --dependency links them)
#   fit       GPU. Checks, then the chained fit over all N_FIT_CHUNKS chunks in one process
#             (each chunk resumable per episode and variant), then the merge and the
#             model-comparison plots.
#   validate  CPU, 50 cores, no GPU. Waits for fit to SUCCEED. The reported variants, 40 worker
#             processes, HUMAN_ROLLOUTS sampled rollouts per episode, the seed and flags of every
#             V4 validation so far; resumable per episode. Then the person-level plots.
#             CPU because rollouts step the simulator on the CPU whatever device the networks
#             sit on: Entry 32 MEASURED 8 CPU workers at 6.8 s per rollout per worker against
#             11 s serially on a GPU, reproducing the GPU rollouts to the printed digit.
#   recovery  GPU, independent of fit. Checks, then all N_AGENTS agents in N_REC_CHUNKS
#             chunks, then the merge and the read-out.
#
#   "Checks" in a GPU job = the package suite on that GPU, the slot-order audit gating on
#   that GPU, and -- fit job only -- the four stall canaries and the audit on the node's CPU
#   with the GPU hidden. Those four tests are where the slot-order defect (SSP Design
#   Log 24) was observed, on this cluster's CPU, before the fix; passing there is the fix's
#   end-to-end verdict, and the job stops if they do not.
#
# COSTS, MEASURED on this cluster (jobs 18411129/32, 2026-09-24), and they depend on the GPU
#   SLURM assigns. Fit, the whole chain (C, B from two starts, A', A): ~218 s per human
#   episode on an A100 -> 230 episodes ~14 h, inside FIT_TIME's 15 h (an L40S ran the
#   unchained four at 112 s per episode, which would put the chain near 9 h). Recovery: ~65 s
#   per synthetic episode on an H100 (B + A_full, incl. simulation) -> 600 episodes ~11 h;
#   C and B's second start add a PROJECTED ~50-70 s per episode (C took 19-30 s per human
#   episode on the A100, B's two starts 45-84 s; C 40-60 s per synthetic episode on a laptop
#   GPU, MEASURED 2026-09-25) -> ~19-22 h.
#   Validation: 230 x 20 rollouts / 40 workers at 6.8-17 s per rollout (Entry 32's cluster CPU;
#   the laptop at 12 workers) -> ~15-35 min per variant, under 2 h for the three reported.
#   A job that times out is finished by re-running the one command.
# ============================================================================

# --- Default allocation; the driver overrides per job -----------------------------------
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_1094_tandon_priority
# | Email notification is given on the command line, never written in here:
# |   MAIL_USER=<address> bash <this file>     (the driver passes it to the three jobs)

# --- Configuration -- edit here, never as sbatch arguments -----------------------------
# | The checkpoint committed with the package (CONTEXT.md): hlp_model_final.pt + its paired
# | LLP, run CarDrivingEnv-v0__hlp_minspd15_extendParams__158__310553__1788309855. It travels
# | with `git pull`, unlike modules/base_models/trained_policies, so no search is needed.
CHECKPOINT_DIR="human_modeling/inference/checkpoint"
# | [server-output] Written under the clone's temp/, which .gitignore ignores, NOT under
# | human_modeling/inference/results/: that folder is tracked, so output written there on the
# | server would block the clone's next pull once the same files are committed from the local
# | repo (user, 2026-11-11; implementation.md Entry 36 addendum). Copy back by name:
# | temp/reproduction/inference/<X> -> human_modeling/inference/results/<X>.
RESULTS_ROOT="temp/reproduction/inference/model_comparison"
VARIANTS="C_no_cognitive_cost B_no_heuristics A_no_mean_distance A_full"   # the chain, in order
REPORTED_VARIANTS="C_no_cognitive_cost B_no_heuristics A_full"   # compared and validated; A' is A's start only
N_HUMAN_EPISODES=230       # MEASURED: exp1_filtered.json, 39 participants x 6 routes
N_FIT_CHUNKS=10            # round-robin: episode i goes to chunk i mod 10, balanced across people
# | 20, not the earlier 5 (user decision 2026-09-24): at 5 the rollout noise alone moved B's
# | person-level collision R2 by up to +/-0.095 (Entry 35 addendum 3); 4x the rollouts halves it.
HUMAN_ROLLOUTS=20
VALIDATE_SEED=20260902     # every V4 validation's seed: rollout r of fit row i is 20260902 + 1000*i + r
VALIDATE_WORKERS=40        # of 50 cores: the rest for the OS, the parent and torch's own threads
RECOVERY_VARIANT="A_full"  # run_synthetic_recovery's default --paired-arm (B) is fitted first
RECOVERY_EXTRA_ARMS="C_no_cognitive_cost"   # also fitted to every synthetic episode, from the midpoint
N_AGENTS=100               # Entry 31's power check: the smallest count answering both questions
N_MAPS=6                   # every distinct map in the human data -- there are exactly six
N_REC_CHUNKS=10            # 10 agents x 6 maps = 60 episodes each
STUDY_SEED=20260923        # a NEW seed: these truths are 8-dimensional, not Entry 31's draw

FIT_TIME="15:00:00"        # ~14 h measured on an A100 (see COSTS)
VALIDATE_TIME="04:00:00"   # under 2 h projected at 20 rollouts
RECOVERY_TIME="24:00:00"   # ~11 h measured on an H100 for B + A, ~19-22 h projected with C and B's second start

# --- Where things are: FOUND, never written in (user, 2026-11-11) ------------------------
# | [no-hardcoded-paths] The driver (`bash <this file>`) finds the clone from this file's own
# | location, three directories up, and submits the stage jobs from the clone's root; so a
# | stage job's SLURM_SUBMIT_DIR is that clone. The project folder holding the overlay and
# | slurm_output/ is the clone's parent.
if [ -n "${SLURM_JOB_ID:-}" ]; then
  REPO_DIR="${SLURM_SUBMIT_DIR}"
else
  REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
fi
PROJECT_DIR="$(dirname "${REPO_DIR}")"
SELF="${REPO_DIR}/extras/server_instructions/scripts/inference_model_comparison.sbatch"
LOG_DIR="${PROJECT_DIR}/slurm_output"
ROOT="${REPO_DIR}/${RESULTS_ROOT}"
[ -f "${SELF}" ] || { echo "ERROR: ${REPO_DIR} is not a clone holding this script"; exit 1; }

# ===========================================================================
# DRIVER MODE: `bash <this file>` submits the three jobs and exits
# ===========================================================================
if [ -z "${SLURM_JOB_ID:-}" ]; then
  [ -f "${REPO_DIR}/${CHECKPOINT_DIR}/hlp_metadata.json" ] \
    || { echo "ERROR: no checkpoint at ${REPO_DIR}/${CHECKPOINT_DIR}"; exit 1; }
  [ -f "${REPO_DIR}/human_modeling/data/exp1_filtered.json" ] \
    || { echo "ERROR: the human data is missing"; exit 1; }
  mkdir -p "${LOG_DIR}" "${ROOT}"
  cd "${REPO_DIR}"     # [no-hardcoded-paths] so each stage job's SLURM_SUBMIT_DIR is this clone
  MAIL_ARGS=""; [ -n "${MAIL_USER:-}" ] && MAIL_ARGS="--mail-type=END --mail-user=${MAIL_USER}"
  submit() {  # submit <stage> <time> <extra sbatch args...> ; prints the job id
    local stage="$1" time="$2"; shift 2
    sbatch --parsable --job-name="v4mc_${stage}" --time="${time}" ${MAIL_ARGS} \
           --output="${LOG_DIR}/inference_mc_${stage}_%j.out" \
           --export=ALL,STAGE="${stage}" "$@" "${SELF}" | cut -d';' -f1
  }
  FIT=$(submit fit "${FIT_TIME}")
  VALIDATE=$(submit validate "${VALIDATE_TIME}" --gres=gpu:0 --cpus-per-task=50 --mem=64GB \
             --dependency=afterok:${FIT})
  RECOVERY=$(submit recovery "${RECOVERY_TIME}")
  echo "submitted: fit ${FIT} -> validate ${VALIDATE} (afterok); recovery ${RECOVERY} (independent)"
  echo "results: ${RESULTS_ROOT}/{fit,validate,recovery}; logs: ${LOG_DIR}/inference_mc_*"
  echo "if anything is left incomplete after the jobs end, run this same command again."
  exit 0
fi

# ===========================================================================
# STAGE MODE: one SLURM job runs one stage
# ===========================================================================
STAGE="${STAGE:?STAGE must be set; submit through 'bash ${SELF}'}"
cd "${PROJECT_DIR}"
echo "=== inference model comparison | stage ${STAGE} | job ${SLURM_JOB_ID} | $(date) | $(hostname) ==="

if [ "${STAGE}" != "validate" ]; then
  ( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 300; done ) &
  MONITOR_PID=$!
fi

singularity exec --nv \
  --overlay ${PROJECT_DIR}/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash << OUTER
set -e
source /ext3/env.sh
conda activate puffer_env
cd ${REPO_DIR}
HEARTBEAT_PID=""
if [ "${STAGE}" != "validate" ]; then
  export HB_N="\${HB_N:-10240}"; export HB_TARGET="\${HB_TARGET:-75}"
  nice -n 19 python extras/server_instructions/gpu_heartbeat.py &
  HEARTBEAT_PID=\$!
fi
# | [heartbeat-trap] Stop the heartbeat however this script ends: with set -e, a failing step
# | exits at once, and a heartbeat still running holds the container open until a timeout.
# | The TERM/INT trap turns a signal (walltime, scancel) into an exit, so the EXIT trap runs.
trap '[ -n "\${HEARTBEAT_PID}" ] && kill \${HEARTBEAT_PID} 2>/dev/null; true' EXIT; trap 'exit 143' TERM INT

# --- loguru and statsmodels, which the V4 suite needs beyond puffer_env (user, 2026-09-25) --
# | The overlay is mounted read-only, so they go to the user site (~/.local), and pip adds only
# | what is missing. The package list is compared before and after the install: if any package
# | already there changed version or vanished, the job stops before anything runs on it. A
# | lock keeps the fit and recovery jobs, which can start together, from installing at once.
ensure_packages() {
  python -c "import loguru, statsmodels" 2>/dev/null && return 0
  exec 9> "\${HOME}/.inference_pip.lock"; flock 9
  if ! python -c "import loguru, statsmodels" 2>/dev/null; then
    python -m pip freeze --all | sort > /tmp/pkgs_before_${SLURM_JOB_ID}.txt
    python -m pip install --user loguru statsmodels
    python -m pip freeze --all | sort > /tmp/pkgs_after_${SLURM_JOB_ID}.txt
    echo "packages added:"; comm -13 /tmp/pkgs_before_${SLURM_JOB_ID}.txt /tmp/pkgs_after_${SLURM_JOB_ID}.txt
    CHANGED=\$(comm -23 /tmp/pkgs_before_${SLURM_JOB_ID}.txt /tmp/pkgs_after_${SLURM_JOB_ID}.txt)
    if [ -n "\${CHANGED}" ]; then
      echo "ERROR: the install changed or removed packages that were already installed:"
      echo "\${CHANGED}"; exit 1
    fi
  fi
  flock -u 9
}
ensure_packages
python -c "import torch, loguru, statsmodels; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# --- Checks shared by the GPU jobs: the suite and the slot-order gate on THIS GPU --------
gpu_checks() {
  python -m pytest human_modeling/inference/tests -q --import-mode=importlib
  python -m human_modeling.inference.scripts.check_slot_order \
    --hlp-checkpoint-dir "${CHECKPOINT_DIR}" --devices cpu cuda --gate-device cuda
}

case "${STAGE}" in

  fit)
    gpu_checks
    # | The fix's end-to-end verdict (SSP Design Log 24): these four failed on this
    # | cluster's CPU before the fix, and the slot-order audit gated on CPU must be clean now.
    echo "--- CPU verdict on the slot-order fix (GPU hidden) ---"
    CUDA_VISIBLE_DEVICES="" python -m pytest -q --import-mode=importlib \
      human_modeling/inference/tests/test_heuristic_fitting.py::StallDetection \
      human_modeling/inference/tests/test_heuristic_fitting.py::StallRetry
    CUDA_VISIBLE_DEVICES="" python -m human_modeling.inference.scripts.check_slot_order \
      --hlp-checkpoint-dir "${CHECKPOINT_DIR}" --devices cpu --gate-device cpu

    for i in \$(seq 0 $((N_FIT_CHUNKS-1))); do
      python -m human_modeling.inference.scripts.run_human_inference \
        --hlp-checkpoint-dir "${CHECKPOINT_DIR}" \
        --results-dir "${RESULTS_ROOT}/fit/chunk_\$(printf '%02d' \$i)" \
        --variants ${VARIANTS} --max-episodes 0 \
        --chunk-index \$i --n-chunks ${N_FIT_CHUNKS}
    done

    MISSING=0
    for i in \$(seq 0 $((N_FIT_CHUNKS-1))); do
      [ -f "${RESULTS_ROOT}/fit/chunk_\$(printf '%02d' \$i)/DONE" ] || { echo "chunk \$i not DONE"; MISSING=1; }
    done
    if [ "\${MISSING}" -ne 0 ]; then
      echo "fit incomplete -- re-run 'bash ${SELF}' to resume"; exit 1
    fi
    for v in ${VARIANTS}; do
      EXPECT=\$(cat ${RESULTS_ROOT}/fit/chunk_*/DONE | awk -v v="\$v" '\$1 == v {s += \$2} END {print s}')
      python -m human_modeling.inference.scripts.merge_chunk_csvs \
        --inputs "${RESULTS_ROOT}/fit/chunk_*/fit_variant_\${v}.csv" \
        --out "${RESULTS_ROOT}/fit/fit_variant_\${v}.csv" --key subject_id game_index --expect \${EXPECT}
    done
    python -m human_modeling.inference.scripts.plot_model_comparison --results-dir "${RESULTS_ROOT}/fit" \
      --variants ${REPORTED_VARIANTS}
    # | The two read-outs the paper's text quotes beyond the comparison table: the
    # | per-participant BIC tests and the between-participant variance share (ICC).
    python -m human_modeling.inference.scripts.participant_bic_tests --results-dir "${RESULTS_ROOT}/fit" \
      --variants ${REPORTED_VARIANTS}
    python -m human_modeling.inference.scripts.parameter_variance \
      --fit-csv "${RESULTS_ROOT}/fit/fit_variant_${RECOVERY_VARIANT}.csv" --variant ${RECOVERY_VARIANT} \
      --out-dir "${RESULTS_ROOT}/fit"
    ;;

  validate)
    for v in ${REPORTED_VARIANTS}; do
      OUT="${RESULTS_ROOT}/validate/\${v}"
      python -m human_modeling.inference.estimation.validate \
        --hlp-checkpoint-dir "${CHECKPOINT_DIR}" \
        --fit-csv "${RESULTS_ROOT}/fit/fit_variant_\${v}.csv" \
        --results-dir "\${OUT}" \
        --n-rollouts ${HUMAN_ROLLOUTS} --seed ${VALIDATE_SEED} --tag "\${v}" \
        --workers ${VALIDATE_WORKERS}
      python -m human_modeling.inference.scripts.plot_by_person \
        --validate-csv "\${OUT}/validate_simulated_\${v}.csv" \
        --out-dir "\${OUT}/plots" --tag "\${v}"
    done
    ;;

  recovery)
    gpu_checks
    for j in \$(seq 0 $((N_REC_CHUNKS-1))); do
      LO=\$(( j * ${N_AGENTS} / ${N_REC_CHUNKS} )); HI=\$(( (j + 1) * ${N_AGENTS} / ${N_REC_CHUNKS} ))
      [ "\$j" -eq $((N_REC_CHUNKS-1)) ] && HI=${N_AGENTS}
      python -m human_modeling.inference.scripts.run_synthetic_recovery \
        --hlp-checkpoint-dir "${CHECKPOINT_DIR}" \
        --results-dir "${RESULTS_ROOT}/recovery/chunk_\$(printf '%02d' \$j)" \
        --variant ${RECOVERY_VARIANT} --n-agents ${N_AGENTS} --n-maps ${N_MAPS} \
        --agents \${LO}:\${HI} --study-seed ${STUDY_SEED} \
        --extra-arms ${RECOVERY_EXTRA_ARMS}
    done
    # | The merge exits 1 unless every (agent, map) is present with every extra arm -- then
    # | nothing below runs and the fix is to re-run the one command, which resumes.
    python -m human_modeling.inference.scripts.run_synthetic_recovery \
      --hlp-checkpoint-dir "${CHECKPOINT_DIR}" --results-dir "${RESULTS_ROOT}/recovery" \
      --variant ${RECOVERY_VARIANT} --n-agents ${N_AGENTS} --n-maps ${N_MAPS} \
      --study-seed ${STUDY_SEED} --merge-chunks "${RESULTS_ROOT}/recovery/chunk_*" \
      --extra-arms ${RECOVERY_EXTRA_ARMS}
    for HOW in mean median; do
      python -m human_modeling.inference.scripts.analyze_param_recovery \
        --recovery-csv "${RESULTS_ROOT}/recovery/recovery_rows_\${HOW}.csv" \
        --hlp-checkpoint-dir "${CHECKPOINT_DIR}" \
        --out-dir "${RESULTS_ROOT}/recovery/metrics_${RECOVERY_VARIANT}_\${HOW}"
      for ARM in B_no_heuristics ${RECOVERY_EXTRA_ARMS}; do
        python -m human_modeling.inference.scripts.analyze_param_recovery \
          --recovery-csv "${RESULTS_ROOT}/recovery/recovery_rows_\${ARM}_\${HOW}.csv" \
          --hlp-checkpoint-dir "${CHECKPOINT_DIR}" \
          --out-dir "${RESULTS_ROOT}/recovery/metrics_\${ARM}_\${HOW}"
      done
    done
    ;;

  *)
    echo "unknown STAGE ${STAGE}"; exit 1 ;;
esac
OUTER
RC=$?
[ -n "${MONITOR_PID:-}" ] && kill ${MONITOR_PID} 2>/dev/null
echo "=== stage ${STAGE} finished with code ${RC} at $(date) ==="
exit ${RC}
