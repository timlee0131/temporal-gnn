import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./data_pipeline/data/metrLA"
        if config.computer == "local"
        else "./data_pipeline/data/metrLA"
    )

    # Description
    config.description = "\nCONFIG DESCRIPTION: temporal GAT; ep=100 lr=0.001"
    
    # Dataset
    config.dataset = 'metrLA'
    config.connectivity_threshold = 0.1
    config.include_self = True
    config.normalize_axis = 1
    config.layout = 'edge_index'
    config.num_channels = 1
    config.num_nodes = 207
    
    config.horizon = 12
    config.window = 12
    config.stride = 1
    config.batch_size = 128
    
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    
    # Space Model
    config.space_hidden = 32
    config.rnn_layers = 1
    
    # Temporal Model
    config.trf_hidden_dim = 4
    config.num_heads = 8
    config.num_trf_layers = 2
    config.attention_dropout = 0.6
    config.ff_dropout = 0.2
    
    # TGAT Model
    config.tgat_hidden = 32
    
    # Training
    config.epochs = 100
    config.lr = 0.001
    
    # misc.
    config.verbose = True
    config.time_verbose = True

    return config