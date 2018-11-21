import tensorflow as tf
import os

tf.app.flags.DEFINE_string('mode','train', 'train / val / test / generate')
tf.app.flags.DEFINE_string('model_dir','model', 'output model dir')
tf.app.flags.DEFINE_string('data_dir','data', 'data dir')
#tf.app.flags.DEFINE_string('data_name','NLPCC', 'data dir')
tf.app.flags.DEFINE_string('data_name','BG', 'data dir')
tf.app.flags.DEFINE_integer('embedding_dim', 300, 'embedding dimension')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_integer('unit_size', 256, 'unit size')
#tf.app.flags.DEFINE_integer('max_length', 25, 'max sentence length')
tf.app.flags.DEFINE_integer('max_length', 10, 'max sentence length')

"""
tf.app.flags.DEFINE_integer('printing_step', 1, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 2, 'saving step')
tf.app.flags.DEFINE_integer('num_step', 6, 'number of steps')
"""
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 10000, 'saving step')
tf.app.flags.DEFINE_integer('num_step', 50000, 'number of steps')

FLAGS = tf.app.flags.FLAGS

FLAGS.data_dir = os.path.join(FLAGS.data_dir, 'data_{}'.format(FLAGS.data_name))
FLAGS.model_dir = os.path.join(FLAGS.model_dir, 'model_{}'.format(FLAGS.data_name))
FLAGS.max_length = FLAGS.max_length + 2

if not os.path.exists(FLAGS.model_dir):
  os.mkdir(FLAGS.model_dir)
  print ('Create model dir : {}'.format(FLAGS.model_dir))

BOS = 0
EOS = 1
UNK = 2
DROPOUT = 3

