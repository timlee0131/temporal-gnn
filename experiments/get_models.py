from models.tts_models import TTS_RNN_GCN, TTS_TRF_GAT
from models.transformers.temporal_gat_model import TemporalGAT
from models.transformersV2.dstan import DSTANv1, DSTANv2

def get_model(config, model_name):
    tts_rnn_gcn = TTS_RNN_GCN(
        input_size=config.num_channels,
        n_nodes=config.num_nodes,
        horizon=config.horizon,
        hidden_size=config.space_hidden,
        rnn_layers=config.rnn_layers
    )
    
    tts_transformer = TTS_TRF_GAT(
        config=config,
        input_size=config.num_channels,
        n_nodes=config.num_nodes,
        window=config.window,
        horizon=config.horizon,
        time_hidden=config.trf_hidden_dim,
        space_hidden=config.space_hidden,
        n_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        ff_dropout=config.ff_dropout,
        n_layers=config.num_trf_layers
    )
    
    tgat = TemporalGAT(
        config=config,
        input_size=config.num_channels,
        window=config.window,
        horizon=config.horizon,
        hidden_size=config.tgat_hidden,
        n_heads=config.num_heads,
        attention_dropout=config.attention_dropout,
        ff_dropout=config.ff_dropout,
        n_layers=config.num_trf_layers
    )
    
    dstan_v1 = DSTANv1(
        config=config,
        input_size=config.num_channels,
        hidden_size=config.trf_hidden_dim,
        num_nodes=config.num_nodes,
        window_size=config.window,
        horizon=config.horizon,
        num_heads=config.num_heads,
        dropout=config.attention_dropout
    )
    
    dstan_v2 = DSTANv2(
        config=config,
        input_size=config.num_channels,
        hidden_size=config.trf_hidden_dim,
        num_nodes=config.num_nodes,
        window_size=config.window,
        horizon=config.horizon,
        num_heads=config.num_heads,
        dropout=config.attention_dropout
    )
    
    if model_name == 'tts_rnn_gcn':
        return tts_rnn_gcn
    elif model_name == 'tts_trf_gat':
        return tts_transformer
    elif model_name == 'tgat':
        return tgat
    # TransformerV2 models
    elif model_name == 'dstan_v1':
        return dstan_v1
    elif model_name == 'dstan_v2':
        return dstan_v2
    else:
        raise ValueError(f"Invalid model name: {model_name}")