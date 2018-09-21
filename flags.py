import tensorflow as tf

tf.app.flags.DEFINE_string('mode','train', 'train / test / stdin')
tf.app.flags.DEFINE_string('model_dir','model', 'output model dir')
tf.app.flags.DEFINE_string('data_dir','data', 'data dir')
tf.app.flags.DEFINE_integer('embedding_dim', 256, 'embedding dimension')
tf.app.flags.DEFINE_integer('batch_size', 256, 'batch size')
tf.app.flags.DEFINE_integer('unit_size', 128, 'unit size')
tf.app.flags.DEFINE_integer('max_length', 12, 'max sentence length')
tf.app.flags.DEFINE_integer('printing_step', 1000, 'printing step')
tf.app.flags.DEFINE_integer('saving_step', 20000, 'saving step')
tf.app.flags.DEFINE_integer('num_step', 100000, 'number of steps')

FLAGS = tf.app.flags.FLAGS
BOS = 0
EOS = 1
UNK = 2

