#!/bin/bash

# ============================================================================
# HOW TO RUN
# ----------------------------------------------------------------------------
#   1. Go to the ROOT of the clone the job should run from (from the login node) and update it.
#      No path is written into this script: the job runs whichever clone it is submitted
#      from, and finds the overlay and slurm_output/ in that clone's parent directory.
#        cd <the clone>
#        git pull
#   2. Check the three inputs exist. The checkpoint and the stored no-cognitive-cost fit
#      arrive with the pull; the human data is NOT in git (untracked), so it must have been
#      copied into this clone once:
#        ls human_modeling/inference/checkpoint/hlp_metadata.json \
#           human_modeling/inference/results/model_comparison/fit/fit_variant_C_no_cognitive_cost.csv \
#           human_modeling/data/exp1_filtered.json
#   3. Submit from that same directory (one job, so plain sbatch):
#        sbatch extras/server_instructions/scripts/inference_full_visibility_sanity.sbatch
#   4. Watch it:
#        squeue --me
#        tail -f ../slurm_output/inference_fullvis_<JOBID>.out
#   5. If it times out, submit step 3 again: the fit skips every (episode, model) on disk.
#   6. Results land in the clone's gitignored temp/ (see WHY below), in
#        temp/reproduction/inference/model_comparison/sanity_full_visibility/
#          delta_bic_full_visibility.txt      the statistics, the verdict and the refit check
#          delta_bic_full_visibility.csv      the same statistics, one row per reference fit
#          delta_bic_full_visibility_episodes.csv, refit_check_C_no_cognitive_cost.csv
#          fit_variant_{C_no_cognitive_cost,C_full_visibility}.csv, human_inference_log.txt
#          commit.txt                         the commit the job ran, and any local changes
#   7. Copy that folder into the LOCAL repo as
#        human_modeling/inference/results/model_comparison/sanity_full_visibility/
#      and commit it there. Never commit it from the server clone. Do not run `git clean -x`
#      or `-X` in the server clone before copying: it deletes ignored files, these included.
#
# WHY THE RESULTS ARE WRITTEN UNDER temp/
#   results/model_comparison/ is tracked by git. Files the job wrote there in the server's
#   clone would sit untracked; once the same files are committed from the local repo, the
#   server's next `git pull` or `git checkout` refuses ("untracked working tree files would be
#   overwritten"). temp/ is ignored (.gitignore), so the clone stays clean and pulls never
#   conflict (user, 2026-11-11). The job only READS from the tracked tree: the code, the
#   checkpoint, the data and the stored fit.
# ============================================================================

# ============================================================================
# inference -- FULL-VISIBILITY SANITY CHECK on all 230 human episodes
#                 (inference implementation.md Entry 36)
# ----------------------------------------------------------------------------
# THE QUESTION (user, 2026-11-10)
#   Does forcing every car visible change the fit of the no-cognitive-cost model? Two models
#   are fitted to every human episode in ONE process, so they share code, device and run:
#     C_no_cognitive_cost  the reported model, refitted here: 3 free coefficients, cognitive
#                          cost pinned at 0, the policy still chooses which cars to attend.
#     C_full_visibility    the same 3 coefficients and pin, with the 6 nearest cars always
#                          visible (no attention choice).
#   Read-out (scripts/full_visibility_sanity.py): per-episode delta BIC = BIC(full) - BIC(C),
#   its mean, SD, SE (naive and participant-bootstrap) and 95% participant-bootstrap
#   interval, and an equivalence verdict at +-10 BIC per episode -- against the refit, and
#   again against the stored October fit. Plus the refit check: the refitted C against the
#   stored C, episode by episode.
#
# WHAT IT DOES NOT ANSWER
#   Whether the two models DRIVE alike (no rollouts / validate stage). Parameter recovery.
#   Anything about the heuristic models. It is a sanity check, not a paper finding.
#
# COSTS (projected, not measured on the cluster)
#   C took 19-30 s per human episode on an A100 and 8-22 s on an L40S in the October run;
#   preparation 1-9 s. The full-visibility model measured ~5x cheaper per likelihood-and-
#   gradient evaluation than C on a local CPU (0.99 s vs 5.1 s on a 1,046-step episode), its
#   whole fit 3-6x faster (45-52 s vs 158-319 s on two episodes), its preparation 1.4 s vs
#   5.2 s. So ~30-55 s per episode -> 230 episodes ~2-3.5 h, plus the test suite and the
#   slot-order audit before the fit. TIME below allows for a slower GPU.
# ============================================================================

#SBATCH --job-name=v4_fullvis
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=torch_pr_1094_tandon_priority
# | Relative to the job's working directory, which is the directory sbatch ran in -- the
# | clone's root -- so the log lands in slurm_output/ beside the clone.
#SBATCH --output=../slurm_output/inference_fullvis_%j.out

# --- Where things are: FOUND, never written in (user, 2026-11-11) ------------------------
# | [no-hardcoded-paths] The clone is the directory sbatch was run from (SLURM_SUBMIT_DIR);
# | the project folder holding the overlay and slurm_output/ is its parent. The check below
# | stops a job submitted from anywhere else, before anything runs.
REPO_DIR="${SLURM_SUBMIT_DIR:?submit with sbatch, from the root of the clone}"
PROJECT_DIR="$(dirname "${REPO_DIR}")"
[ -f "${REPO_DIR}/extras/server_instructions/scripts/inference_full_visibility_sanity.sbatch" ] \
  || { echo "ERROR: ${REPO_DIR} is not the root of a clone holding this script; cd to the clone's root and submit again"; exit 1; }

# --- Configuration -- edit here, never as sbatch arguments -----------------------------
CHECKPOINT_DIR="human_modeling/inference/checkpoint"                    # read, in the clone
STORED_FIT="human_modeling/inference/results/model_comparison/fit/fit_variant_C_no_cognitive_cost.csv"   # read, in the clone
# | [server-output] Written, relative to the clone, under its gitignored temp/ (see WHY above).
# | Copy back by name: temp/reproduction/inference/<X> -> human_modeling/inference/results/<X>.
OUT_DIR="temp/reproduction/inference/model_comparison/sanity_full_visibility"
VARIANTS="C_no_cognitive_cost C_full_visibility"

cd "${PROJECT_DIR}"
echo "=== inference full-visibility sanity check | job ${SLURM_JOB_ID} | $(date) | $(hostname) ==="
for f in "${CHECKPOINT_DIR}/hlp_metadata.json" "${STORED_FIT}" "human_modeling/data/exp1_filtered.json"; do
  [ -f "${REPO_DIR}/${f}" ] || { echo "ERROR: missing ${REPO_DIR}/${f}"; exit 1; }
done

# | Provenance: the output no longer sits beside the code that made it, so record the commit
# | and any uncommitted changes in the clone. Appended, so a resumed run keeps every entry.
# | (This runs before the container, in PROJECT_DIR, hence the REPO_DIR prefix.)
mkdir -p "${REPO_DIR}/${OUT_DIR}"
{ echo "job ${SLURM_JOB_ID} $(date)"; git -C "${REPO_DIR}" rev-parse HEAD
  git -C "${REPO_DIR}" status --short; echo; } >> "${REPO_DIR}/${OUT_DIR}/commit.txt"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 300; done ) &
MONITOR_PID=$!

singularity exec --nv \
  --overlay ${PROJECT_DIR}/overlay-25GB-500K.ext3:ro \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash << OUTER
set -e
source /ext3/env.sh
conda activate puffer_env
cd ${REPO_DIR}

# | loguru and statsmodels are not in puffer_env; the model-comparison job installed them to
# | ~/.local. This job does not install anything: it stops and says so if they are missing.
python -c "import loguru, statsmodels" || { echo "ERROR: loguru/statsmodels missing from ~/.local"; exit 1; }
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# | Keeps GPU utilization above the cluster's minimum while the fit is CPU-bound in places.
export HB_N="\${HB_N:-10240}"; export HB_TARGET="\${HB_TARGET:-75}"
nice -n 19 python extras/server_instructions/gpu_heartbeat.py &
HEARTBEAT_PID=\$!
# | [heartbeat-trap] Stop the heartbeat however this script ends. With set -e, a failing step
# | exits at once, and a heartbeat still running then holds the container open until it is
# | killed on a timeout (job 22755397's "Terminating squashfuse_ll after timeout"). The TERM/INT
# | trap turns a signal (walltime, scancel, timeout) into an exit, so the EXIT trap runs too.
trap 'kill \$HEARTBEAT_PID 2>/dev/null || true' EXIT; trap 'exit 143' TERM INT

# | 1. Checks on THIS GPU: the package suite (it includes the full-visibility gates, which
# | compare the torch and numpy likelihoods on a real episode) and the slot-order audit.
python -m pytest human_modeling/inference/tests -q --import-mode=importlib
python -m human_modeling.inference.scripts.check_slot_order \
  --hlp-checkpoint-dir "${CHECKPOINT_DIR}" --devices cpu cuda --gate-device cuda

# | 2. Both models on every episode, one process, resumable per (episode, model).
python -m human_modeling.inference.scripts.run_human_inference \
  --hlp-checkpoint-dir "${CHECKPOINT_DIR}" --results-dir "${OUT_DIR}" \
  --variants ${VARIANTS} --max-episodes 0
# | The read-out runs only once every episode is fitted under both models (DONE).
if [ ! -f "${OUT_DIR}/DONE" ]; then
  echo "fit incomplete -- submit again to resume"; exit 1
fi

# | 3. The read-out.
python -m human_modeling.inference.scripts.full_visibility_sanity \
  --results-dir "${OUT_DIR}" --stored-fit "${STORED_FIT}"
OUTER
RC=$?
kill ${MONITOR_PID} 2>/dev/null
echo "=== finished with code ${RC} at $(date) ==="
exit ${RC}
