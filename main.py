import collections
import os
import tensorflow as tf 
import numpy as np
import model
import argparse
from flags import FLAGS

embedding_dim = 128
max_length = 30
unit_size = 128
batch_size = 1024
load_model = False


def run():
    with tf.device('/gpu:1'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        classifier =  model.sentiment_classifier(sess, FLAGS)
        if FLAGS.mode == 'train':
            classifier.train()
        elif FLAGS.mode == 'test':
            classifier.test()

if __name__ == '__main__':
	run()

