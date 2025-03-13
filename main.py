import argparse
from experiments import trainer

def get_args():
    """
    args
    --dataset
        TSL datasets
            metrLA: METR-LA dataset
            pemsBay: PEMS-BAY dataset
        benchmark datasets
            electricity
            exchange
            traffic
            solar
    --model
        tts_rnn_gcn: benchmark RNN + GCN model
        tan: Temporal Attention Network
        mp_dstan: Message Passing DSTAN
        transformer_dsta: Transformer using DSTAN
        dstan: flagship attention model
    --program
        train: train model
        tune: tune model
    --wandb
        enabled: wandb enabled
        disabled: wandb disabled
    """
    parser = argparse.ArgumentParser(description="Temporal GNN Research")
    
    parser.add_argument("-d", "--dataset", type=str, default="metrLA", choices=['metrLA', 'pemsbay', 'electricity', 'exchange', 'traffic', 'solar'], help="Dataset to use")
    parser.add_argument("-m", "--model", type=str, default="dstan", choices=['tts_rnn_gcn', 'tan', 'mp_dstan', 'transformer_dsta', 'dstan', 'dstan_experiments'], help="Model to use")
    parser.add_argument("-p", "--program", type=str, default="train", choices=['train', 'tune'], help="Program to run")
    parser.add_argument("-w", "--wandb", type=str, default="disabled", choices=['e', 'enabled', 'd', 'disabled'], help="Wandb mode")
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()

    if args.program == "train":
        trainer.driver(args.dataset, args.model, args.wandb)
    elif args.program == "tune":
        trainer.tuner(args.dataset, args.model, args.wandb)
    else:
        print(f"Invalid program: {args.program}")
    
    
if __name__ == "__main__":
    main()