# Using, once environment is set up

To check what to do to set up environment look at next section

Always deploy from project directory for it to be trated as root
```console
cd /scratch/${USER}/psychdrive-files
```
## Interactive terminal

Get compute resources allocated
```console
srun --cpus-per-task=4 --mem=10GB --time=01:00:00 --pty /bin/bash
```

Start singularity container and load environment
```console
singularity exec --fakeroot --overlay overlay-25GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
source /ext3/env.sh
conda activate puffer_env
```

For GPU requests:
```console
cd /scratch/${USER}/psychdrive-files/
srun --cpus-per-task=50 --mem=60GB --gres=gpu:1 --time=01:30:00 --pty /bin/bash
singularity exec --nv --fakeroot --overlay overlay-25GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
source /ext3/env.sh
conda activate puffer_env
cd PsychDrive
```


## Other useful commands

Call an sbatch script
```console
sbatch extras/server_instructions/train_ppo_hpc2.sbatch
```

Check for currently active jobs
```console
squeue --me
```

Kill a currently running job
```console
scancel 4705823
```

Activate console for an active job
```console
srun --jobid=4705823 --pty bash
```

Get CPU usage
```console
htop -u ${USER}
```

Get GPU usage
```console
 watch -n 1 nvidia-smi
```



# First time setup

Detailed guide: https://github.com/curlsloth/NYU-HPC-4-newbies

## Set up environment

Create a directory for the project (where everything will be kept):
```console
mkdir /scratch/${USER}/psychdrive-files
cd /scratch/${USER}/psychdrive-files
```

Copy an appropriate gzipped overlay images from the overlay directory. overlay-15GB-500K.ext3.gz is good enough for this project.
```console
cp -rp /share/apps/overlay-fs-ext3/overlay-25GB-500K.ext3.gz .
gunzip overlay-15GB-500K.ext3.gz
```

You can browse available images to see available options.
```console
ls /share/apps/overlay-fs-ext3
```

Choose a corresponding Singularity image. For this example we will use the following image.
```console
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif
```

For Singularity image available on nyu HPC Torch, please check the singularity images folder.
```console
ls /share/apps/images/
```

Launch the appropriate Singularity container in read/write mode (with the :rw flag): start a bash shell inside the referenced Singularity Container overlaid with the 15GB 500K set up earlier.
```console
singularity exec --fakeroot --overlay overlay-25GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
```

Inside the container, download and install miniforge to /ext3/miniforge3.
```console
wget --no-check-certificate https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p /ext3/miniforge3
# rm Miniforge3-Linux-x86_64.sh # if you don't need this file any longer
```

Create a wrapper script /ext3/env.sh using a text editor, like nano:
```console
touch /ext3/env.sh
nano /ext3/env.sh
```

The wrapper script will activate your conda environment, to which you will be installing your packages and dependencies. The script should contain the following.

```console
#!/bin/bash

unset -f which

source /ext3/miniforge3/etc/profile.d/conda.sh
export PATH=/ext3/miniforge3/bin:$PATH
export PYTHONPATH=/ext3/miniforge3/bin:$PATH
```

Activate your conda environment with the following:
```console
source /ext3/env.sh
```

If you have the "defaults" channel enabled, please disable it with:
```console
conda config --remove channels defaults
```

Now that your environment is activated, you can update and install packages:
```console
conda update -n base conda -y
conda clean --all --yes
conda install pip ipykernel -y # Note: ipykernel is required to run as a kernel 
```

To confirm that your environment is appropriately referencing your Miniforge installation, try out the following:
```console
unset -f which
which conda
# output: /ext3/miniforge3/bin/conda

which python
# output: /ext3/miniforge3/bin/python

python --version
# output: Python 3.11.10

which pip
# output: /ext3/miniforge3/bin/pip

exit
# exit Singularity
```

## Interactive job

Start an interactive job (get assigned a compute node), with adequate compute and memory resources to install packages:
```console
srun --cpus-per-task=2 --mem=10GB --time=01:00:00 --pty /bin/bash
```

with a gpu:
```console
srun --cpus-per-task=2 --mem=10GB --gres=gpu:1 --time=01:00:00 --pty /bin/bash
```

After it is running, you’ll be redirected to a compute node. From there, run singularity to setup on conda environment, same as you were doing on login node.
```console
# Your prompt should now look something like this once your jobs starts: [NetID@cm001 psychdrive-files]$

singularity exec --fakeroot --overlay overlay-25GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
# Singularity>

# For GPU Passthrough, use:
singularity exec --nv --fakeroot --overlay overlay-25GB-500K.ext3:rw /share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif /bin/bash
# Singularity>

source /ext3/env.sh
```

## Set up PsychDrive

Ensure you have requested a GPU node. Confirm with:
```console
nvidia-smi
```

Clone the repo from github 
```console
git clone https://github.com/sounakban/PsychDrive.git
cd PsychDrive
```

Create a new conda environment and activate it
```console
conda create -n puffer_env python=3.13 -y
conda activate puffer_env
```

Install pufferlib first, since installing it later will replace existing packages with older versions
NOTE: pufferlib is only available for unix-based systems
```console
pip install pufferlib==3.0.0
```

Then install all other dependencies (replaces old pufferlib dependencies with new versions)
```console
pip install -e .
```

Install Pytorch at the end (if correct version not already installed)
Check https://pytorch.org/get-started/locally/ for CUDA X.X compatible installation command
```console
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```



EXTRA:
```console
#!/bin/bash
#
#SBATCH --job-name=adv_puffer
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00
#SBATCH --account torch_pr_355_tandon_priority
#SBATCH --mail-type=END
#SBATCH --mail-user=<NetID>@nyu.edu

module purge
singularity exec --nv /scratch/<NetID>/singularity_images/puffer.sif bash << EOF
set -e
cd /scratch/<NetID>/PufferDrive/
source .venv/bin/activate
python setup.py build_ext --inplace --force
nice -n 19 python extras\server_instructions\gpu_heartbeat.py &
HEARTBEAT_PID=$!
echo "Started Smart Heartbeat with PID: $HEARTBEAT_PID"
puffer train puffer_drive --wandb --wandb-project adv_exp9
kill $HEARTBEAT_PID
EOF
```