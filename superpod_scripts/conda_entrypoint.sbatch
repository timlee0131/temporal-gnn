#!/usr/bin/env zsh
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -J train
#SBATCH -o ./experiments/output/metrLA/MPDSTANv2--jobID-%j.out
#SBATCH -c 32 --mem=82G
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH -G 1


current_percentage=0.1

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
