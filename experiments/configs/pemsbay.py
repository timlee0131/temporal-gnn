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
    
    # Description
    config.description = "\nCONFIG DESCRIPTION: 8 heads, 100 epochs, 3 layer GAT"

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
    config.batch_size = 128
    
    config.val_ratio = 0.1
    config.test_ratio = 0.2
    
    # Model (TTS_RNN_GCN)
    config.hidden_dim = 24
    config.rnn_layers = 1
    
    # Model - TTS_TRF_GAT
    config.trf_hidden_dim = 4
    config.num_heads = 8
    config.num_trf_layers = 1
    config.attention_dropout = 0.6
    config.ff_dropout = 0.2
    
    # Training
    config.epochs = 100
    config.lr = 1e-3
    
    # misc.
    config.verbose = False
    config.time_verbose = True

    return config