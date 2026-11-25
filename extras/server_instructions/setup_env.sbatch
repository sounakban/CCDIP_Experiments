#!/bin/bash

#SBATCH --job-name=setup_env         # The name of the job
#SBATCH --nodes=1                           # Request 1 compute node per job instance
#SBATCH --cpus-per-task=5                  # Request 50 CPUs for 512 parallel envs
#SBATCH --mem=10GB                          # Request 64GB of RAM
#SBATCH --time=04:00:00                     # Request 24 hours
#SBATCH --gres=gpu:1                        # Request 1 L40S GPU
#SBATCH --output=/scratch/%u/psychdrive-files/slurm_output/out_%A.out
#SBATCH --mail-type=END                     # Send an email when the job is completed
#SBATCH --mail-user=<NetID>@nyu.edu         # Email address

# NetID variable
NETID=$(whoami)

# Directory setup
PROJECT_DIR="/scratch/${NETID}/psychdrive-files"
cd ${PROJECT_DIR}

# Mount container
singularity exec --nv --fakeroot --overlay \
        overlay-25GB-500K.ext3:rw \
        /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
        /bin/bash -c "
source /ext3/env.sh
conda create -n puffer_env python=3.13 -y
conda activate puffer_env
git clone https://github.com/sounakban/PsychDrive.git
cd PsychDrive
pip install -e ."
