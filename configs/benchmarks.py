import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./data_pipeline/data/benchmarks"
        if config.computer == "local"
        else "./data_pipeline/data/benchmarks"
    )
    
    # Dataset
    config.dataset = 'benchmarks'
    config.connectivity_threshold = 0.1
    config.include_self = True
    config.normalize_axis = 1
    config.layout = 'edge_index'
    config.num_channels = 1
    config.d_ramanujan = 6
    
    config.horizon = 96
    config.window = 96
    config.stride = 1
    config.batch_size = 128
    
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    
    # Transformer Setup
    config.num_heads = 8
    config.attention_dropout = 0.0
    config.cross_dropout = 0.6
    config.ff_dropout = 0.1
    config.num_layers = 1
    
    # Space Model
    config.space_hidden = 32
    config.rnn_layers = 1
    
    # Temporal Model
    config.trf_hidden_dim = 4
    config.num_trf_layers = 2
    config.ff_dropout = 0.2
    config.tgat_hidden = 32
    
    # DSTAN
    config.dstan_hidden = 96
    config.conv_layers = 2
    config.negative_slope = 0.2
    
    # Training
    config.warmup_epochs = 5
    config.epochs = 100
    config.lr = 5e-3
    config.lr_min = 1e-3
    config.lr_step_size = 20
    config.lr_gamma = 0.8
    config.early_stopping = False
    config.es_patience = 10
    config.es_delta = 0.0
    
    # misc.
    config.runs = 1
    config.verbose = True
    config.time_verbose = True

    return config