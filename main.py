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
    """
    parser = argparse.ArgumentParser(description="Temporal GNN Research")
    
    parser.add_argument("-d", "--dataset", type=str, default="metrLA", choices=['metrLA', 'pemsbay'], help="Dataset to use")
    parser.add_argument("-m", "--model", type=str, default="tgat", choices=['st_tran', 'travnet', 'tts_rnn_gcn', 'tts_trf_gat', 'tgat', 'dstan_v1', 'dstan_v2'], help="Model to use")
    parser.add_argument("-p", "--program", type=str, default="train", choices=['train', 'tune'], help="Program to run")
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()

    if args.program == "train":
        trainer.driver(args.dataset, args.model)
    elif args.program == "tune":
        print("Tuning not implemented yet")
    else:
        print(f"Invalid program: {args.program}")
    
    
if __name__ == "__main__":
    main()