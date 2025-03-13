from models.tts_models import TTS_RNN_GCN, TTS_TRF_GAT
from models.transformers.temporal_gat_model import TemporalGAT
from models.dstan_model.dstan import TAN, MP_DSTAN, GraphTransformerDSTA, DSTAN, DSTANExperiments

def get_model(config, model_name):
    if model_name == 'tts_rnn_gcn':
        return TTS_RNN_GCN(
            input_size=config.num_channels,
            n_nodes=config.num_nodes,
            horizon=config.horizon,
            hidden_size=config.space_hidden,
            rnn_layers=config.rnn_layers
        )
    elif model_name == 'tan':
        return TAN(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            num_nodes=config.num_nodes,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
    elif model_name == 'mp_dstan':
        return MP_DSTAN(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
    elif model_name == 'transformer_dsta':
        return GraphTransformerDSTA(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            attention_dropout=config.attention_dropout,
            cross_dropout=config.cross_dropout,
            ff_dropout=config.ff_dropout
        )
    elif model_name == 'dstan':
        return DSTAN(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            negative_slope=config.negative_slope,
            num_layers=config.conv_layers
        )
    elif model_name == 'dstan_experiments':
        return DSTANExperiments(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout,
            negative_slope=config.negative_slope,
            num_layers=config.conv_layers
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")