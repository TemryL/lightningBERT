# Model:
bert_config = {
    'vocab_size': 30522,
    'context_size': 512,
    'n_seg_types': 2,
    'n_layers': 12,
    'n_heads': 12,
    'h_dim': 768,
    'ln_eps': 1e-12,
    'dropout': 0.1
}


# Data:
batch_size = 256
train_val_split = 0.998
mlm_probability = 0.15


# Optimization:
learning_rate = 1e-4
adamw_epsilon = 1e-6
adamw_betas = (0.9, 0.98)
warmup_steps = 10000
weight_decay = 0.01