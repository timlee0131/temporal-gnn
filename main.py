import argparse
from experiments import trainer

def get_args():
    """
    args
    --cat
        tsl: torch TSL datasets
        custom: custom datasets
    --dataset
        TSL datasets
            metrLA: METR-LA dataset
            pemsBay: PEMS-BAY dataset
        individual datasets
    --mode
        train: train model
        test: test frozen model on (unseen) data
        dat: data engineering (creating pt files, handling raw data, etc.)
    """
    parser = argparse.ArgumentParser(description="Temporal GNN Research")
    sub_parser = parser.add_subparsers(dest="cat")
    
    tsl_parser = sub_parser.add_parser("tsl")
    tsl_parser.add_argument("-d", "--dataset", choices=['metrLA', 'pemsbay'], type=str, default='metrLA')
    tsl_parser.add_argument("-m", "--mode", choices=['train', 'test', 'dat'], type=str, default='train')
    
    custom_parser = sub_parser.add_parser("custom")
    custom_parser.add_argument("-d", "--dataset", choices=[], type=str, default='nill')
    custom_parser.add_argument("-m", "--mode", choices=['train', 'test', 'dat'], type=str, default='train')
    
    args = parser.parse_args()
    
    return args
    
def main():
    args = get_args()
    
    if args.cat == "tsl":
        trainer.driver(args.dataset)
    elif args.cat == "custom":
        print("Custom datasets not yet supported")
    else:
        raise ValueError("Invalid category")
    
if __name__ == "__main__":
    main()