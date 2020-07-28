config = {
    'ml_settings': {
        'train_learning_rate': 0.01,
        'opt_learning_rate': 0.01,
        'training_size': 100,
        'neurons': (16, 8),
        'train_epochs': 100,
        'opt_epochs': 100,
        'early_stop_patience': 10,
        'early_stop_delta': 0.1,
        'validation_split': 0.2
    }
}

# user-defined cost function from xmds output variable
# eg return minimum of vector
def cost(y):
    return min(y)