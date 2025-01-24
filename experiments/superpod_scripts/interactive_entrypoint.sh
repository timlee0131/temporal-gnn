#!/usr/bin/env zsh

save_dir="/lustre/smuexa01/client/users/hunjael/results/temporal-gnn/${SLURM_JOB_ID}_${dataset}"

module load conda
conda activate /users/hunjael/.conda/envs/torch-st

srun\
    -N1\
    -G1\
    -p short\
    -A coreyc_coreyc_mp_jepa_0001\
    --pty $SHELL\