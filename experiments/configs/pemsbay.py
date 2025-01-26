import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./data/pemsbay"
        if config.computer == "local"
        else "/data/pemsbay"
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
    config.window = 24
    config.stride = 1
    config.batch_size = 64
    
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    
    # Model (TTS_RNN_GCN)
    config.hidden_dim = 32
    config.rnn_layers = 1
    
    # Training
    config.epochs = 100
    config.lr = 1e-3
    
    # misc.
    config.verbose = False
    config.time_verbose = True

    return config