sweep_configuration = {
    'method': 'bayes',  # Bayesian optimization
    'metric': {
        'name': 'val_loss (mse)',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01, 0.005, 0.001, 0.0005]
        },
        'num_heads': {
            'values': [4, 6, 8]
        },
        'epochs': {
            'values': [50, 100, 150, 200]
        },
        'hidden': {
            'values': [2, 4, 8, 16]
        },
        'attention_dropout': {
            'values': [0.0, 0.2, 0.4, 0.6]
        }
    }
}
