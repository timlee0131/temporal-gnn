import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./data_pipeline/data/pemsbay"
        if config.computer == "local"
        else "./data_pipeline/data/pemsbay"
    )

    # Dataset
    config.dataset = 'pemsbay'
    config.connectivity_threshold = 0.1
    config.include_self = True
    config.normalize_axis = 1
    config.layout = 'edge_index'
    config.num_channels = 1
    config.num_nodes = 325
    
    config.horizon = 12
    config.window = 12
    config.stride = 1
    config.batch_size = 64
    
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    
    # Transformer Setup
    config.num_heads = 8
    config.attention_dropout = 0.6
    config.cross_dropout = 0.6
    config.ff_dropout = 0.1
    config.num_layers = 2
    
    # Space Model
    config.space_hidden = 24
    config.rnn_layers = 1
    
    # Temporal Model
    config.trf_hidden_dim = 4
    config.num_trf_layers = 1
    config.ff_dropout = 0.2
    
    # TGAT Model
    config.tgat_hidden = 32
    
    # DSTAN Models
    config.dstan_hidden = 32
    
    # Training
    config.warmup_epochs = 5
    config.epochs = 100
    config.lr = 1e-3
    config.lr_min = 1e-6
    config.lr_step_size = 20
    config.lr_gamma = 0.8
    config.early_stopping = False
    config.es_patience = 10
    config.es_delta = 0.0
    
    # misc.
    config.verbose = False
    config.time_verbose = True

    return config