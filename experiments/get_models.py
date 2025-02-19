from models.tts_models import TTS_RNN_GCN, TTS_TRF_GAT
from models.transformers.temporal_gat_model import TemporalGAT
from models.dstan_model.dstan import DSTANv1, TAN, MP_DSTAN, GraphTransformerDSTA, MP_DSTANv2

def get_model(config, model_name):
    if model_name == 'tts_rnn_gcn':
        tts_rnn_gcn = TTS_RNN_GCN(
            input_size=config.num_channels,
            n_nodes=config.num_nodes,
            horizon=config.horizon,
            hidden_size=config.space_hidden,
            rnn_layers=config.rnn_layers
        )
        return tts_rnn_gcn
    elif model_name == 'tts_trf_gat':
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
        return tts_transformer
    elif model_name == 'tgat':
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
        return tgat
    # TransformerV2 models
    elif model_name == 'dstan_v1':
        dstan_v1 = DSTANv1(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            num_nodes=config.num_nodes,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        return dstan_v1
    elif model_name == 'tan':
        tan = TAN(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            num_nodes=config.num_nodes,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        return tan
    elif model_name == 'mp_dstan':
        mp_dstan = MP_DSTAN(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        return mp_dstan
    elif model_name == 'mp_dstanv2':
        mp_dstanv2 = MP_DSTANv2(
            config=config,
            input_size=config.num_channels,
            hidden_size=config.dstan_hidden,
            window_size=config.window,
            horizon=config.horizon,
            num_heads=config.num_heads,
            dropout=config.attention_dropout
        )
        return mp_dstanv2
    elif model_name == 'transformer_dsta':
        transformer_dsta = GraphTransformerDSTA(
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
        return transformer_dsta
    else:
        raise ValueError(f"Invalid model name: {model_name}")