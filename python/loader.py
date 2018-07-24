import numpy as np
import tensorflow as tf
from tensorflow.contrib.seq2seq.python.ops import beam_search_ops

export_dir = '../saved_model/'

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], export_dir)
    graph = tf.get_default_graph()
    # for op in graph.get_operations(): print(op)

    texts = ['ff', 'av']
    text_till_t = np.array([text[:-1] for text in texts])
    text_at_t = np.array([text[-1] for text in texts])
    src_sequence_length = np.array([len(text) for text in texts])

    to_return = {'predictions': 'predictions:0', 'prediction_logits': 'prediction_logits:0'}
    feed_dict = {'text_till_t:0': text_till_t, 'text_at_t:0': text_at_t, 'src_sequence_length:0': src_sequence_length}
    results = sess.run(to_return, feed_dict)

    preds = results['predictions'] # batch_size x seq_len x beam_width
    preds_per_beam = np.transpose(preds, [0,2,1])
    pred_logits = results['prediction_logits']
    pred_strings = [[''.join([c.decode('UTF-8') for c in chars]) for chars in beam_preds] for beam_preds in preds_per_beam]
    full_pred_strings = [[text+chars for chars in beam_preds] for text, beam_preds in zip(texts, pred_strings)]
    beam_width = preds_per_beam.shape[1]

    # print(preds.shape, preds)

    for text, t_preds, t_pred_logits in zip(texts, preds, pred_logits):
        print('Input: {}'.format(text))
        print('Top {} predictions:'.format(beam_width))
        for pred, pred_logit in zip(t_preds, t_pred_logits):
            print('{}: {}'.format(pred, pred_logit))
        print('\n')
        print(full_pred_strings, pred_logits)
