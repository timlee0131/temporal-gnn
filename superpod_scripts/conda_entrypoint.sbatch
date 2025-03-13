#!/usr/bin/env zsh
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -J train
#SBATCH -o ./experiments/output/exp/DSTAN--jobID-%j.out
#SBATCH -c 32 --mem=156G
#SBATCH --nodes=1
#SBATCH --time=0-6:00:00
#SBATCH -G 1
#SBATCH -p batch

save_dir="/lustre/smuexa01/client/users/hunjael/results/temporal-gnn/${SLURM_JOB_ID}_${dataset}"
mkdir -p ${save_dir}
mkdir -p ${save_dir}/results
mkdir -p ${save_dir}/data

# set read/write permissions
chmod -R 777 ${save_dir}

module purge
module load conda
conda activate /users/hunjael/.conda/envs/torch-st

srun\
    bash -c "${HOME}/Projects/temporal-gnn/superpod_scripts/conda_entrypoint.sh"
