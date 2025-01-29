import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    config.computer = "local"
    config.n_gpus = 1 if config.computer == "superpod" else 0
    config.data_dir = (
        "./data/metrLA"
        if config.computer == "local"
        else "/data/metrLA"
    )

    # Description
    config.description = "\ntemporal only model, 150 epochs, lr=0.005"
    
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
    
    # Model - TTS_RNN_GCN
    config.hidden_dim = 32
    config.rnn_layers = 1
    
    # Model - TTS_TRF_GAT
    config.trf_hidden_dim = 4
    config.num_heads = 6
    config.num_trf_layers = 2
    config.attention_dropout = 0.6
    config.ff_dropout = 0.2
    
    # Training
    config.epochs = 150
    config.lr = 0.001
    
    # misc.
    config.verbose = True
    config.time_verbose = True

    return config