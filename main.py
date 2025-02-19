import argparse
from experiments import trainer

def get_args():
    """
    args
    --dataset
        TSL datasets
            metrLA: METR-LA dataset
            pemsBay: PEMS-BAY dataset
        individual datasets
    --model
        tts_rnn_gcn: time then space model --  tRNN, sGCN
        tts_trf_gat: time then space model -- tTRF, sGAT
        tgat: temporal GAT
        travnet: traversenet
        
        dstan_v1: dynamic spatio-temporal attention network v1
        dstan_v2: dynamic spatio-temporal attention network
    --program
        train: train model
        tune: tune model
    --wandb
        enabled: wandb enabled
        disabled: wandb disabled
    """
    parser = argparse.ArgumentParser(description="Temporal GNN Research")
    
    parser.add_argument("-d", "--dataset", type=str, default="metrLA", choices=['metrLA', 'pemsbay'], help="Dataset to use")
    parser.add_argument("-m", "--model", type=str, default="mp_dstan", choices=['st_tran', 'travnet', 'tts_rnn_gcn', 'tts_trf_gat', 'tgat', 'dstan_v1', 'tan', 'mp_dstan', 'mp_dstanv2', 'transformer_dsta'], help="Model to use")
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