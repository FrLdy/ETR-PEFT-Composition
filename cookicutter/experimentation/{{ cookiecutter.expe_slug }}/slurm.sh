#!/bin/bash
#SBATCH --job-name={{ cookiecutter.expe_slug }}
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --mail-type ALL
#SBATCH --mail-user {{ cookiecutter.author_mail }}

#SBATCH --time={{ cookiecutter.time }} 

#SBATCH --nodes={{ cookiecutter.nodes }}
#SBATCH --ntasks-per-node=1 

#SBATCH --partition={{ cookiecutter.partition }}
#SBATCH --cpus-per-gpu=8
#SBATCH --gpus-per-node={{ cookiecutter.gpus }}
#SBATCH --gres=gpu:{{ cookiecutter.gpus }}

job_dir=/dlocal/run/$SLURM_JOB_ID

# Set TMPDIR to work in /dlocal/run instead of lustre
tmpdir=$job_dir/tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir

# Set results location
results_dir=$job_dir/results
mkdir -p $results_dir
ln -s $results_dir results

module load aidl/pytorch/2.0.0-cuda11.7
srun python ./run.py \
    --storage_path=$results_dir \
    --expe_name=$SLURM_JOB_NAME \
    --ressources_config.use_gpu=true \
    --ressources_config.num_workers=$SLURM_GPUS_ON_NODE \
    --ressources_config.cpus_per_worker=$SLURM_CPUS_PER_GPU

