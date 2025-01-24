#!/usr/bin/env zsh
#SBATCH -A coreyc_coreyc_mp_jepa_0001
#SBATCH -J test
#SBATCH -p short
#SBATCH -o output/labn/GroupedCNN-jobID-%j.out
#SBATCH -c 16 --mem=82G
#SBATCH --nodes=1
#SBATCH -G 1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=NONE     


current_percentage=0.1

save_dir="/lustre/smuexa01/client/users/hunjael/results/wave-forecast/${SLURM_JOB_ID}_${dataset}"
mkdir -p ${save_dir}
mkdir -p ${save_dir}/results
mkdir -p ${save_dir}/data

# set read/write permissions
chmod -R 777 ${save_dir}

srun\
    --no-container-entrypoint\
    --container-image ${GWM_PROJECT_DIR}/containers/sqsh_images/nvidia-pyg.sqsh\
    --container-mounts="${HOME}"/Projects/wave-forecast:/wave-forecast,${save_dir}/data:/data\
    --container-workdir /wave-forecast\
    bash -c "${HOME}/Projects/wave-forecast/experiments/superpod_scripts/trainer_entrypoint.sh"
