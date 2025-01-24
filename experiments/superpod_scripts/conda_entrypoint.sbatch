#!/usr/bin/env zsh
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -J test
#SBATCH -o output/pemsbay/TTS-RNN_GCN--jobID-%j.out
#SBATCH -c 16 --mem=82G
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=NONE     


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
    bash -c "${HOME}/Projects/temporal-gnn/experiments/superpod_scripts/conda_entrypoint.sh"
