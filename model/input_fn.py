import tensorflow as tf

def extract_char(token, default_value='<pad>'):
    # Split characters
    out = tf.string_split(token, delimiter='')
    # Convert to Dense tensor, filling with default value
    out = tf.sparse_tensor_to_dense(out, default_value=default_value)
    return out

def load_dataset_from_file(fname, vocab, params):        
    dataset = tf.data.TextLineDataset(fname)
    # Dataset yields characters
    dataset = dataset.map(lambda token: (extract_char([token])[0]))
    # Lookup tokens to return their ids
    dataset = dataset.map(lambda tokens: (tokens, vocab.lookup(tokens)))
    print(dataset)
    dataset = dataset.map(lambda token0, token1: (
                                          token0,
                                          tf.concat([[params.start_token_id], token1], axis=0),
                                          tf.concat([token1, [params.end_token_id]], axis=0)
                                         ))
    dataset = dataset.map(lambda token0, token1, token2: (token0, token1, token2, tf.size(token1)))
    print(dataset)
    return dataset

def input_fn(mode, dataset, params):
    
    # Create batches and pad the sequences of different length
    padded_shapes = (tf.TensorShape([None]), tf.TensorShape([None]),
                     tf.TensorShape([None]), tf.TensorShape([]))   # sequence of unknown length
    padded_values = (params.pad_token, params.pad_token_id, params.pad_token_id, 0)

    # Shuffle the dataset and then create the padded batches
    dataset = (dataset
                .shuffle(buffer_size=params.buffer_size)
                #.repeat(count = num_epochs)
                .padded_batch(params.train_batch_size, padded_shapes=padded_shapes)#, padding_values=padded_values)
                .prefetch(1)
              )
    
    # Create initializable iterator from this dataset so that we can reset at each epoch
    iterator = dataset.make_initializable_iterator()

    # Query the output of the iterator for input to the model
    (input_seq_chars, src_sequence, tgt_sequence, src_sequence_length) = iterator.get_next()
    data_init_op = iterator.initializer

    lr = tf.placeholder(tf.float32, [], 'learning-rate')
    dropout = tf.placeholder(tf.float32, [], 'dropout')
    
    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'src_sequence': src_sequence,
        'tgt_sequence': tgt_sequence,
        'src_sequence_length': src_sequence_length,
        'lr': lr,
        'dropout': dropout,
        'iterator_init_op': data_init_op
    }
    return inputs

def test_input_fn(mode, vocab, params):
    text_tf_till_t = tf.placeholder(tf.string, shape=[None])
    text_tf_at_t = tf.placeholder(tf.string, shape=[None])
    src_sequence_length = tf.placeholder(tf.int32, shape=[None])
    input_chars_till_t = extract_char(text_tf_till_t)
    input_till_t = vocab.lookup(input_chars_till_t)
    input_at_t = vocab.lookup(text_tf_at_t)
    input_at_t = tf.cast(input_at_t, tf.int32)
    batch_size = tf.shape(input_till_t)[0]
    start_tokens = tf.fill([batch_size, 1], params.start_token_id)
    print('start ', start_tokens)
    src_sequence = tf.concat([start_tokens, input_till_t], axis=-1)
    print(src_sequence)
    
    # Build and return a dictionnary containing the nodes / ops
    inputs = {
        'text_tf_till_t': text_tf_till_t,
        'text_tf_at_t': text_tf_at_t,
        'src_sequence': src_sequence,
        'src_sequence_length': src_sequence_length,
        'infer_start_tokens': input_at_t,
        'dropout': tf.constant(0.0)
    }
    return inputs
