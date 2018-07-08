import tensorflow as tf

def build_model(mode, inputs, params):
    src_sequence, src_sequence_length = inputs['src_sequence'], inputs['src_sequence_length']  
    dropout = inputs['dropout']
    # Model Definition
    # Embedding
    with tf.variable_scope('embedding'):
        char_embedding = tf.get_variable('char-embedding', [params.vocab_size_val, params.char_embed_dim])
                                    #initializer=tf.random_normal_initializer(0, 0.01))
        src_embed = tf.nn.embedding_lookup(char_embedding, src_sequence)
        #src_embed = tf.contrib.layers.embed_sequence(
        #    src_sequence, vocab_size=params.vocab_size_val, embed_dim=params.char_embed_dim, scope='char-embedding')

    # Encoder
    with tf.variable_scope('encoder'):
        # Build RNN cell
        encoder_cell = tf.contrib.rnn.DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(params.hidden_dim), output_keep_prob=1-dropout)

        # Run Dynamic RNN
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            encoder_cell, src_embed,
            sequence_length=src_sequence_length,
            dtype=tf.float32)
    
    with tf.variable_scope('output'):
        seq_len = tf.shape(encoder_outputs)[1]
        encoder_outputs_token_major = tf.reshape(encoder_outputs, [-1, params.hidden_dim]) # (batch_size * seq_len) x encoder_dim

        projection_layer = tf.layers.Dense(units = params.vocab_size_val, use_bias=False)
        output_logits_token_major = projection_layer(encoder_outputs_token_major)
        
        #output_logits_token_major = tf.layers.dense(encoder_outputs_token_major, units = params.vocab_size_val, use_bias=False)
        output_logits = tf.reshape(output_logits_token_major, [-1, seq_len, params.vocab_size_val])

    outputs = {'logits': output_logits}

    if mode == 'test':
        with tf.variable_scope('output', reuse=True):
            start_tokens = inputs['infer_start_tokens']
            end_token_id = tf.cast(params.end_token_id, tf.int32)
            # Helper
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                char_embedding,
                start_tokens, end_token_id)

            # Decoder
            decoder = tf.contrib.seq2seq.BasicDecoder(
                encoder_cell, helper, encoder_state,
                output_layer=projection_layer)
            # Dynamic decoding
            decoder_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder, maximum_iterations=5)
            translations = decoder_outputs.sample_id
            outputs['inference'] = translations
    return outputs

def model_fn(mode, inputs, params, reuse=False):    

    model_spec = inputs

    is_training = (mode=='train')
    src_sequence_length = inputs['src_sequence_length']
    
    with tf.variable_scope('model', reuse=reuse):
        # Compute the output distribution of the model and the predictions
        model_output = build_model(mode, inputs, params)
        logits = model_output['logits']
        probs = tf.nn.softmax(logits, axis=-1)
        top_k_probs, top_k_preds = tf.nn.top_k(probs, k=3, sorted=True)
        top_k_preds = tf.cast(top_k_preds, tf.int64)
        print(top_k_preds)
        if mode == 'test':
            model_spec['inference'] = model_output['inference']

    if mode in ['train', 'eval']:
        tgt_sequence = inputs['tgt_sequence']        
        # Loss function - weighted softmax cross entropy
        tgt_mask = tf.sequence_mask(src_sequence_length, dtype=tf.float32, name='masks')
        loss = tf.contrib.seq2seq.sequence_loss(
                logits,
                tgt_sequence,
                tgt_mask)

        tf.summary.scalar('loss', loss)
        model_spec['loss'] = loss

    if is_training:
        lr = inputs['lr']
        # Optimizer
        opt = tf.train.AdamOptimizer(lr)
        tvars = tf.trainable_variables()
        for v in tvars: print(v)
    
        # Gradient Clipping
        grads = tf.gradients(loss, tvars)
        grads, global_norm = tf.clip_by_global_norm(grads, params.max_grad_norm)
        tf.summary.scalar('global_norm', global_norm)
        train_op = opt.apply_gradients(zip(grads, tvars), name='train_step')
        model_spec['train_op'] = train_op
        
    #model_spec = inputs if mode in ['train', 'eval'] else {}
        
    variable_init_op = tf.group(*[tf.global_variables_initializer(), tf.tables_initializer()])
    # variable_init_op = tf.global_variables_initializer()
    model_spec['variable_init_op'] = variable_init_op
    model_spec['predictions'] = top_k_preds
    model_spec['prediction_probs'] = top_k_probs

    return model_spec
