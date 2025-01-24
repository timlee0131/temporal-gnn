#!/bin/bash

# Function to handle signals and propagate them to child processes
sig_handler() {
    echo "Received signal, forwarding to child processes..."
    kill -SIGTERM $srun_pid
    wait $srun_pid
}

# Trap signals and call the handler function
trap 'sig_handler' SIGTERM SIGINT SIGCONT

cd /users/hunjael/Projects/temporal-gnn/
python -m pip install -r requirements.txt

python main.py tsl -d pemsbay
srun_pid=$!
wait $srun_pid
