import tensorflow as tf
import numpy as np
import os

def train_sess(sess, model_spec, num_steps, writer, params):
    """
    Train the model on `num_steps` batches
    """
    lr, dropout = model_spec['lr'], model_spec['dropout']
    loss, train_op = model_spec['loss'], model_spec['train_op']
    summary_op = model_spec['summary_op']
    global_step = tf.train.get_global_step()
    
    # Load data from tf.Dataset
    sess.run(model_spec['iterator_init_op'])
    sess_loss = []
    for step_id in range(num_steps):
        if step_id % params.save_summary_steps == 0:
            to_return = {'train_op': train_op, 'loss': loss, 'summary': summary_op, 'global_step_val': global_step}
            results = sess.run(to_return, feed_dict={lr: params.train_lr, dropout: params.dropout_val})
            # Write summaries for tensorboard
            writer.add_summary(results['summary'], results['global_step_val'])
        else:
            to_return = {'train_op': train_op, 'loss': loss}
            results = sess.run(to_return, feed_dict={lr: params.train_lr, dropout: params.dropout_val})
        sess_loss.append(results['loss'])
    return np.mean(sess_loss)

def train_test(train_model_spec, test_model_spec, params, vocab_rev, restore_from=None):
    """
    Train the model
    """
    # Initialize tf.Saver instances to save weights during training
    last_saver = tf.train.Saver() # will keep last 5 epochs
    begin_at_epoch = 0    

    with tf.Session() as sess:
        # Initialize model variables
        sess.run(train_model_spec['variable_init_op'])
        
        # Reload weights from directory if specified
        if restore_from is not None:
            #logging.info("Restoring parameters from {}".format(restore_from))
            if os.path.isdir(restore_from):
                restore_from = tf.train.latest_checkpoint(restore_from)
                begin_at_epoch = int(restore_from.split('-')[-1])
            last_saver.restore(sess, restore_from)

        # For tensorboard (takes care of writing summaries to files)
        train_writer = tf.summary.FileWriter(os.path.join(params.model_dir, 'train_summaries'), sess.graph)

        train_num_steps = (params.train_data_size + params.train_batch_size - 1) // params.train_batch_size
        for epoch_id in range(begin_at_epoch, begin_at_epoch + params.num_epochs):
            #sess.run(data_init_op)
            batch_loss = train_sess(sess, train_model_spec, train_num_steps, train_writer, params)
            print('Epoch: {}, Train Loss: {:0.4f}'.format(epoch_id, np.mean(batch_loss)))
            
            # Save weights
            last_save_path = os.path.join(params.model_dir, 'last_weights', 'after-epoch')
            if not os.path.exists(last_save_path):
                os.makedirs(last_save_path)
            last_saver.save(sess, last_save_path, global_step=epoch_id + 1)
            # tf.train.write_graph(sess.graph_def, params.model_dir, 'input_graph.pb', as_text=False)
            
    """
            if epoch_id%20 == 0:
                preds_string = test_sess('ffm', test_model_spec, params, vocab_rev, sess=sess)                
                print(preds_string)
    """

def save_model_and_graph(inputs, outputs, sess):
    export_path = '../saved_model/'
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    text_till_t = inputs['text_till_t']
    text_at_t = inputs['text_at_t']
    src_sequence_length = inputs['src_sequence_length']

    predictions = outputs['predictions']
    prediction_logits = outputs['prediction_logits']

    text_till_t_tensor_info = tf.saved_model.utils.build_tensor_info(text_till_t)
    text_at_t_tensor_info = tf.saved_model.utils.build_tensor_info(text_at_t)
    src_sequence_length_tensor_info = tf.saved_model.utils.build_tensor_info(src_sequence_length)

    signature_inputs = {
        'text_till_t': text_till_t_tensor_info,
        'text_at_t': text_at_t_tensor_info,
        'src_sequence_length': src_sequence_length_tensor_info
    }

    predictions_tensor_info = tf.saved_model.utils.build_tensor_info(predictions)
    prediction_logits_tensor_info = tf.saved_model.utils.build_tensor_info(prediction_logits)

    signature_outputs = {
        'predictions': predictions_tensor_info,
        'prediction_logits': prediction_logits_tensor_info
    }

    exec_signature = tf.saved_model.signature_def_utils.build_signature_def(
                         inputs=signature_inputs,
                         outputs=signature_outputs,
                         method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME
                     )

    builder.add_meta_graph_and_variables(
        sess, [tf.saved_model.tag_constants.SERVING],
        signature_def_map={
            'run': exec_signature,
        },
        legacy_init_op = tf.tables_initializer()
    )

    builder.save()


def test_sess(texts, model_spec, params, vocab_rev, sess=None, restore_from=None):
    curr_sess = sess if sess else tf.Session()
    curr_sess.run(model_spec['variable_init_op'])
    assert not(not sess and not restore_from)
    # Reload weights from directory if specified
    if restore_from is not None:
        #logging.info("Restoring parameters from {}".format(restore_from))
        saver = tf.train.Saver()
        if os.path.isdir(restore_from):
            restore_from = tf.train.latest_checkpoint(restore_from)
        saver.restore(curr_sess, restore_from)
            
    text_tf_till_t = model_spec['text_tf_till_t']
    text_tf_at_t = model_spec['text_tf_at_t']
    src_sequence_length = model_spec['src_sequence_length']
    text_till_t = [text[:-1] for text in texts]
    text_at_t = [text[-1] for text in texts]
    text_len = [len(text) for text in texts]
    feed_dict = {text_tf_till_t: text_till_t, text_tf_at_t: text_at_t, src_sequence_length: text_len}
    print(model_spec['inference-logits'])

    predictions = model_spec['inference']
    predictions = tf.cast(predictions, tf.int64)
    predicted_chars = vocab_rev.lookup(predictions, name='predictions')
    prediction_logits = tf.identity(model_spec['inference'], name='prediction_logits')
    print('eta: ', predicted_chars)
    to_return = {'predictions': predicted_chars, 'prediction_logits': model_spec['inference-logits']}
    results = curr_sess.run(to_return, feed_dict=feed_dict)

    # tf.train.write_graph(curr_sess.graph_def, params.model_dir, 'input_graph.pb', as_text=False)
    inputs = {'text_till_t': text_tf_till_t, 'text_at_t': text_tf_at_t, 'src_sequence_length': src_sequence_length}
    outputs = {'predictions': predicted_chars, 'prediction_logits': model_spec['inference-logits']}
    save_model_and_graph(inputs, outputs, curr_sess)
    if not sess: curr_sess.close()

    preds = results['predictions'] # batch_size x seq_len x beam_width
    preds_per_beam = np.transpose(preds, [0,2,1])
    pred_logits = results['prediction_logits']
    pred_strings = [[''.join([c.decode('UTF-8') for c in chars]) for chars in beam_preds] for beam_preds in preds_per_beam]
    full_pred_strings = [[text+chars for chars in beam_preds] for text, beam_preds in zip(texts, pred_strings)]
    return full_pred_strings, pred_logits
