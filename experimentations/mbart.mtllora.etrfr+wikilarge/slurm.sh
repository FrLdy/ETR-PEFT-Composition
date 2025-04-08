#!/bin/bash
#SBATCH --job-name=mbart.mtllora.etrfr+wikilarge
#SBATCH --output=logs/%x-%j.out
#SBATCH --err=logs/%x-%j.err
#SBATCH --mail-type ALL
#SBATCH --mail-user francois.ledoyen@unicaen.fr

#SBATCH --time=48:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 


#SBATCH --partition=gpu_h200

#SBATCH --reservation=c23meso

#SBATCH --cpus-per-gpu=20
#SBATCH --gpus-per-node=8
#SBATCH --gres=gpu:8

job_dir=/dlocal/run/$SLURM_JOB_ID

# Set TMPDIR to work in /dlocal/run instead of lustre
tmpdir=$job_dir/tmp
mkdir -p $tmpdir
export TMPDIR=$tmpdir

# Set results location
results_dir=$job_dir/results
mkdir -p $results_dir

# Parse arguments
overwrite=false
continue_run=false
for arg in "$@"; do
    case $arg in
        --overwrite)
            overwrite=true
            ;;
        --continue)
            continue_run=true
            ;;
    esac
done

# Handle overwrite option
if [ "$overwrite" = true ]; then
    [ -L results ] && rm results
fi

# Handle continue option
if [ "$continue_run" = true ]; then
    if [ -d "./results" ]; then
        actual_result_path=$(readlink ./results)
        rm -rf $results_dir # remove /dlocal/run/newid/results
        ln -sf $actual_result_path $results_dir # link prev results dir in /dlocal/run/newid
        rm -rf $tmpdir # same for tmp dir
        ln -sf $(dirname $actual_result_path)/tmp $tmpdir
    fi
else
    # Create symbolic link if not overwritten
    ln -s $results_dir results
fi

export NCCL_DEBUG=INFO
module load aidl/pytorch/2.6.0-py3.12-cuda12.6
srun python ./run.py \
    --storage_path=$results_dir \
    --expe_name=$SLURM_JOB_NAME \
    --ressources_config.use_gpu=true \
    --ressources_config.num_workers=$SLURM_GPUS_ON_NODE \
    --ressources_config.cpus_per_worker=$SLURM_CPUS_PER_GPU

