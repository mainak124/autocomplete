import os
import tensorflow as tf

from params import Params
from model.input_fn import input_fn, load_dataset_from_file
from model.input_fn import test_input_fn
from model.model_fn import model_fn
from model.training import train_test
from model.training import test_sess

params = Params()

# Create character vocabulary
char_vocab = tf.contrib.lookup.index_table_from_file(params.vocab_fname, num_oov_buckets=params.num_oov_buckets)
char_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(params.vocab_fname, default_value='<unk>')
vocab_size = char_vocab.size()

params.pad_token_id = char_vocab.lookup(tf.constant(params.pad_token))
params.start_token_id = char_vocab.lookup(tf.constant(params.start_token))
params.end_token_id = char_vocab.lookup(tf.constant(params.end_token))

with tf.Session() as sess:
    sess.run(tf.tables_initializer())
    params.vocab_size_val = vocab_size.eval()

train_dataset = load_dataset_from_file(params.train_fname, char_vocab, params)
train_inputs = input_fn('train', train_dataset, params)
train_model_spec = model_fn('train', train_inputs, params, reuse=False)

test_inputs = test_input_fn('test', char_vocab, params)
test_model_spec = model_fn('test', test_inputs, params, reuse=True)

train_test(train_model_spec, test_model_spec, params, char_vocab_rev)

texts = ['ff', 'av']
preds, pred_logits = test_sess(texts, test_model_spec, params, char_vocab_rev, 
          restore_from=os.path.join(params.model_dir, 'last_weights'))

for text, t_preds, t_pred_logits in zip(texts, preds, pred_logits):
    print('Input: {}'.format(text))
    print('Top {} predictions:'.format(params.beam_width))
    for pred, pred_logit in zip(t_preds, t_pred_logits):
        print('{}: {}'.format(pred, pred_logit))
    print('\n')
