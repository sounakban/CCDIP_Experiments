#!/bin/bash
# ============================================================================
# Train the subtask policy (SP, code name LLP) with the settings of the committed checkpoint
# ----------------------------------------------------------------------------
# HOW TO RUN
#   1. From the login node, go to the ROOT of the clone and update it:
#        cd <the clone>
#        git pull
#   2. Submit from that directory (the job finds the clone through SLURM_SUBMIT_DIR, the
#      overlay and slurm_output/ in the clone's parent directory; no path is written in here):
#        sbatch extras/server_instructions/scripts/train_hlr_llp.sbatch
#   3. Watch it:
#        squeue --me
#        tail -f ../slurm_output/train_SP_<JOBID>.out
#   4. The run lands in the clone's gitignored temp/ (see OUT_PATH below):
#        temp/reproduction/training/SP_reproduction__<date>__<timestamp>/
#          model_final.pt, hrl_llp_metadata.json, training plots, one rollout video
#      Pass that directory to train_hlr_hlp.sbatch (LLP_DIR) to train an SSP on it, and copy
#      it down for the local behavioural comparison against human_modeling/inference/checkpoint.
#
# SETTINGS
#   Everything below the PPO block reproduces human_modeling/inference/checkpoint/
#   hrl_llp_metadata.json (SP run llp_missTerm_varDensity__260814_2313__1786763626): 800M
#   steps, seed 83, 432 envs x 256 steps, hidden 256, goal range 0.2-12 m, miss penalty 0.3
#   ending the sub-episode, env-reward coefficient drawn from [0, 1], speed and action-switch
#   coefficients 0.005 / 0.007, traffic density buffer 1-3, no reaction delay.
#   The PPO block (learning rate, annealing, entropy, minibatches) is not in that metadata;
#   it was CONFIRMED against the original run's full argument table, the "hyperparameters"
#   text in the TensorBoard event file of its run directory (2026-11-23): every value below
#   is the original's.
# ============================================================================
#SBATCH --job-name=train_SP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50              # ~48 puffer workers + overhead
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm_output/train_SP_%j.out
#SBATCH --account=torch_pr_1094_tandon_priority

# --- Configuration -- edit here, never as sbatch arguments -----------------------------
OUT_PATH="temp/reproduction/training"     # under the clone, gitignored
EXP_NAME="SP_reproduction"              # run directory = OUT_PATH/<EXP_NAME>__<date>__<timestamp>
TOTAL_TIMESTEPS=800_000_000
SEED=83

# --- Where things are: FOUND, never written in ------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_DIR="$(dirname "${REPO_DIR}")"
[ -f "${REPO_DIR}/modules/base_models/SP/train_llp_PPO.py" ] \
  || { echo "ERROR: submit this job from the clone's root (got ${REPO_DIR})"; exit 1; }
mkdir -p "${PROJECT_DIR}/slurm_output" "${REPO_DIR}/${OUT_PATH}"
cd "${PROJECT_DIR}"
echo "=== train SP | job ${SLURM_JOB_ID} | $(date) | $(hostname) | clone ${REPO_DIR} ==="

# --- Utilisation monitor (every 5 minutes, to the job log) ------------------------------
( while true; do
    echo "--- [$(date +%H:%M:%S)] CPU $(top -bn1 | grep "Cpu(s)" | awk '{printf "%.1f", 100-$8}')%  $(free -h | grep Mem | awk '{print "RAM " $3 "/" $2}')"
    command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
    sleep 300
  done ) &
MONITOR_PID=$!

module purge
# | Clear POSIX semaphores and shared memory a killed job may have left on this node;
# | otherwise PufferLib's worker start-up can block on a semaphore held by a dead process.
ipcrm -a 2>/dev/null || true

singularity exec --nv \
  --overlay "${PROJECT_DIR}/overlay-25GB-500K.ext3:ro" \
  /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
  /bin/bash << OUTER
set -e
source /ext3/env.sh
conda activate puffer_env
cd "${REPO_DIR}"
nice -n 19 python extras/server_instructions/gpu_heartbeat.py &
HEARTBEAT_PID=\$!
trap 'kill \${HEARTBEAT_PID} 2>/dev/null; true' EXIT; trap 'exit 143' TERM INT

python modules/base_models/SP/train_llp_PPO.py \
  --total_timesteps           ${TOTAL_TIMESTEPS} \
  --seed                      ${SEED} \
  --exp_name                  ${EXP_NAME} \
  --out_path                  ${OUT_PATH} \
  --num_envs                  432 \
  --num_steps                 256 \
  --hidden_size               256 \
  --learning_rate             1e-3 \
  --anneal_lr \
  --ent_coef                  0.01 \
  --num_minibatches           216 \
  --goal_forward_m_min        0.2 \
  --goal_forward_m_max        12.0 \
  --goal_radius_m             1.0 \
  --sub_episode_horizon       100 \
  --llp_reach_reward          0.0 \
  --llp_miss_penalty          0.3 \
  --llp_miss_ends_subepisode \
  --llp_env_reward_coef_min   0.0 \
  --llp_env_reward_coef_max   1.0 \
  --llp_speed_coef            0.005 \
  --llp_action_switch_coef    0.007 \
  --llp_truncation_penalty    0.0 \
  --car_buffer_y_min          1 \
  --car_buffer_y_max          3 \
  --vectorization_mode        puffer \
  --puffer_num_workers        48 \
  --save_frequency            600
OUTER
RC=$?
kill ${MONITOR_PID} 2>/dev/null
echo "=== train SP finished with code ${RC} at $(date) ==="
exit ${RC}
