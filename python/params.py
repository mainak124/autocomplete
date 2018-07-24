## Hyperparameters

class Params():
    # Model
    char_embed_dim = 30
    hidden_dim = 64

    # Train
    train_batch_size = 32
    dropout_val = 0.2
    max_grad_norm = 5
    train_lr= 0.01
    num_epochs = 100
    model_dir = 'char-pred-model'
    save_summary_steps = 5

    # Data
    base_dir = '../data/'
    train_fname = base_dir + 'history.txt'
    vocab_fname = base_dir + 'vocab.txt'
    buffer_size = 1000
    num_oov_buckets = 1
    train_data_size= 874
    
    # Additional Tokens
    start_token = '<s>'
    end_token = '</s>'
    pad_token = ''

    # Inference
    beam_width = 5
