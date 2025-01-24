#!/bin/bash

# Function to handle signals and propagate them to child processes
sig_handler() {
    echo "Received signal, forwarding to child processes..."
    kill -SIGTERM $srun_pid
    wait $srun_pid
}

# Trap signals and call the handler function
trap 'sig_handler' SIGTERM SIGINT SIGCONT

# install the graph_jepa module in editable mode
cd /users/hunjael/Projects/wave-forecast/
python -m pip install -r requirements.txt

python main.py gnn -d labn
srun_pid=$!
wait $srun_pid
