#!/bin/bash
# ============================================================================
# Train the subtask-selection policy (SSP, code name HLP) on a frozen SP, with the settings
# of the committed checkpoint
# ----------------------------------------------------------------------------
# HOW TO RUN
#   1. From the login node, go to the ROOT of the clone and update it:
#        cd <the clone>
#        git pull
#   2. Choose the SP in the configuration block below. LLP_DIR is any directory holding
#      hrl_llp_metadata.json and the SP weights (model_final.pt as the SP trainer writes
#      them, or llp_model_final.pt as an SSP run directory and the committed checkpoint
#      carry them):
#        human_modeling/inference/checkpoint                        the committed SP
#        temp/reproduction/training/SP_reproduction__<date>__<ts>     a fresh SP from train_hlr_llp.sbatch
#   3. Submit from the clone's root (the job finds the clone through SLURM_SUBMIT_DIR, the
#      overlay and slurm_output/ in the clone's parent directory; no path is written in here):
#        sbatch extras/server_instructions/scripts/train_hlr_hlp.sbatch
#      For an email when it ends, give it on the command line (never written in here):
#        sbatch --mail-type=END --mail-user=<address> extras/server_instructions/scripts/train_hlr_hlp.sbatch
#   4. Watch it:
#        squeue --me
#        tail -f ../slurm_output/train_SSP_<JOBID>.out
#   5. The run lands in the clone's gitignored temp/ (see OUT_PATH below):
#        temp/reproduction/training/CarDrivingEnv-v0__<EXP_NAME>__<seed>__<pid>__<timestamp>/
#          hlp_model_final.pt, hlp_metadata.json, llp_model_final.pt, hrl_llp_metadata.json,
#          training plots, one rollout
#      That directory is a complete checkpoint: copy it down and compare it with
#      human_modeling/inference/checkpoint through the local behavioural check.
#
# SETTINGS
#   The argument block reproduces human_modeling/inference/checkpoint/hlp_metadata.json (SSP
#   run CarDrivingEnv-v0__hlp_minspd15_extendParams__158__310553__1788309855): 1.5B env
#   steps, seed 158, 432 envs x 200 SSP steps, learning rate 2.5e-4 annealed, gamma 0.5, 16
#   minibatches x 4 epochs, 6 car slots, pooled mask head, hidden 128 with a 2x critic,
#   attentional cost per step, the theta ranges collision [0, 10], attention [0, 0.15],
#   goal proximity [0, 6], flag terms fixed at 0, goal-flag distance [0, 1.5].
#   SP-side settings (goal range, sub-episode horizon, miss penalty) are read
#   from the SP checkpoint's hrl_llp_metadata.json by the trainer, never from here. A PPO
#   constant equal to the trainer's default (train_hlp_PPO.py Args) is not passed; the default
#   supplies it.
# ============================================================================
#SBATCH --job-name=train_SSP
#SBATCH --nodes=1
#SBATCH --cpus-per-task=50              # ~48 puffer workers + overhead
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm_output/train_SSP_%j.out
#SBATCH --account=torch_pr_1094_tandon_priority

# --- Configuration -- edit here, never as sbatch arguments -----------------------------
LLP_DIR="human_modeling/inference/checkpoint"   # the SP to train on (see HOW TO RUN, step 2)
OUT_PATH="temp/reproduction/training"             # under the clone, gitignored
EXP_NAME="SSP_reproduction"                     # name the SP in it when it is not the committed one
TOTAL_TIMESTEPS=1_500_000_000
SEED=158

# --- Where things are: FOUND, never written in ------------------------------------------
REPO_DIR="${SLURM_SUBMIT_DIR}"
PROJECT_DIR="$(dirname "${REPO_DIR}")"
[ -f "${REPO_DIR}/modules/base_models/SSP/train_hlp_PPO.py" ] \
  || { echo "ERROR: submit this job from the clone's root (got ${REPO_DIR})"; exit 1; }
[ -f "${REPO_DIR}/${LLP_DIR}/hrl_llp_metadata.json" ] \
  || { echo "ERROR: no SP checkpoint at ${REPO_DIR}/${LLP_DIR}"; exit 1; }
mkdir -p "${PROJECT_DIR}/slurm_output" "${REPO_DIR}/${OUT_PATH}"
cd "${PROJECT_DIR}"
echo "=== train SSP on ${LLP_DIR} | job ${SLURM_JOB_ID} | $(date) | $(hostname) | clone ${REPO_DIR} ==="

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

python modules/base_models/SSP/train_hlp_PPO.py \
  --llp_checkpoint_dir      ${LLP_DIR} \
  --total_timesteps         ${TOTAL_TIMESTEPS} \
  --seed                    ${SEED} \
  --exp_name                ${EXP_NAME} \
  --out_path                ${OUT_PATH} \
  --num_envs                432 \
  --num_hlp_steps           200 \
  --anneal_lr \
  --gamma                   0.5 \
  --num_minibatches         16 \
  --hlp_num_car_slots       6 \
  --hlp_hidden_size         128 \
  --hlp_critic_hidden_size_multiplier  2 \
  --hlp_reach_reward        0.0 \
  --hlp_env_reward_coef     0.0 \
  --llp_env_reward_coef     0.1 \
  --hlp_goal_flag_hinge_m   0.5 \
  --hlp_collision_penalty_min        0.0 \
  --hlp_collision_penalty_max        10.0 \
  --hlp_cognitive_cost_coef_min      0.0 \
  --hlp_cognitive_cost_coef_max      0.15 \
  --hlp_goal_proximity_cost_coef_min 0.0 \
  --hlp_goal_proximity_cost_coef_max 6.0 \
  --hlp_flag_miss_penalty_min        0.0 \
  --hlp_flag_miss_penalty_max        0.0 \
  --hlp_flag_collect_reward_min      0.0 \
  --hlp_flag_collect_reward_max      0.0 \
  --hlp_goal_flag_dist_penalty_min   0.0 \
  --hlp_goal_flag_dist_penalty_max   1.5 \
  --car_buffer_y_min        1 \
  --car_buffer_y_max        3 \
  --vectorization_mode      puffer \
  --puffer_num_workers      48 \
  --save_frequency          100
OUTER
RC=$?
kill ${MONITOR_PID} 2>/dev/null
echo "=== train SSP finished with code ${RC} at $(date) ==="
exit ${RC}
