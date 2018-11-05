import collections
import os, re, json
import tensorflow as tf 
import numpy as np

class sentiment_classifier(object):
  def __init__(self, sess, args ):
    self.model_dir = args.model_dir
    try: 
      print ('creating the dicectory %s...' %(self.model_dir))
      os.mkdir(self.model_dir)
    except:
      print ('%s has created...' %(self.model_dir))
    self.model_path = os.path.join(self.model_dir,'model_{m_type}'.format(m_type='sent'))
    self.data_dir = args.data_dir
    self.dict_file = os.path.join(self.data_dir, 'dict')
    self.sess = sess
    self.training_epochs = 50
    self.learning_rate = 0.001
    self.printing_step = args.printing_step
    self.saving_step = args.saving_step
    self.num_step = args.num_step
    self.embedding_dim = args.embedding_dim
    self.max_length = args.max_length
    self.unit_size = args.unit_size
    self.batch_size = args.batch_size
    self.dictionary = self.read_json(self.dict_file)
    self.num_words = len(self.dictionary)
    self.bos = self.dictionary['__BOS__']
    self.eos = self.dictionary['__EOS__']
    self.unk = self.dictionary['__UNK__']
    self.build_model()
    self.saver = tf.train.Saver(tf.all_variables(), max_to_keep=5)
  
  def build_model(self):
    #placeholder
    print ('placeholding...')
    self.encoder_inputs = tf.placeholder(tf.int32, shape=[None, self.max_length])
    self.target = tf.placeholder(tf.float32, shape=[None, 1])
    #variable
    weights = {
      'w2v' : tf.Variable(tf.random_uniform([self.num_words, self.embedding_dim], -0.1, 0.1, dtype=tf.float32), name='w2v'),
      'out_1' : tf.Variable(tf.random_normal([self.unit_size*2, 1]), name='w_out_1'),
    }
    biases = {
        'out_1' : tf.Variable(tf.random_normal([1]), name='b_out_1'),
    }
    ###############structure###############
    print ('building structure...')
    def BiRNN(x):
      x = tf.unstack(x, self.max_length, 1)
      lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(self.unit_size, forget_bias=1.0)
      lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(self.unit_size, forget_bias=1.0)
      outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype = tf.float32 )
      return outputs[-1]

    embed_layer = tf.nn.embedding_lookup(weights['w2v'], self.encoder_inputs )
    layer_1 = BiRNN(embed_layer)
    pred = tf.matmul(layer_1, weights['out_1']) + biases['out_1']
    self.cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.target) )
    self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    ###testing part
    self.score = tf.sigmoid(pred)
    pred_sig = tf.cast(tf.greater(tf.sigmoid(pred), 0.5), tf.float32)
    correct_ped = tf.equal(pred_sig,self.target)
    self.accuracy = tf.reduce_mean(tf.cast(correct_ped, tf.float32))

  def step(self, x_batch, y_batch, predict):
    feed_dict = { self.encoder_inputs: x_batch, self.target: y_batch }
    output_feed = [self.cost, self.accuracy]
    if predict:
      outputs = self.sess.run(output_feed, feed_dict=feed_dict)
    else:
      output_feed.append(self.optimizer)
      outputs = self.sess.run(output_feed, feed_dict=feed_dict)

    return outputs[0], outputs[1]

  def get_score(self, input_idlist):
    feed_dict = { self.encoder_inputs: input_idlist }
    output_feed = [self.score]
    outputs = self.sess.run(output_feed, feed_dict=feed_dict)
    return outputs[0]

  def read_json(self, filename):
    with open(filename, 'r') as fp:
        data = json.load(fp)
    return data

  def tokenizer(self, input_sentence):
    data = [self.eos]*self.max_length
    data[0] = self.bos
    data[self.max_length-1] = self.eos
    word_count = 1
    for word in input_sentence.split():
      word = word.decode('utf8')
      if word_count>=self.max_length-1:
        break
      if word in self.dictionary:
        data[word_count] = self.dictionary[word]
      else:
        data[word_count] = self.unk
      word_count += 1
    return data

  def build_dataset(self, f='source_train'):
    fp = os.path.join(self.data_dir,f)
    if os.path.exists(fp):
      print ('building dataset...')
      x_data = []
      y_data = []
      sen = []
      with open(fp, 'r') as file:
        un_parse = file.readlines()
        for line in un_parse:
          line = line.strip('\n').split(' +++$+++ ')
          y_data.append([int(line[0])])
          x_data.append(self.tokenizer(line[1]))
          sen.append(line[1])
      return x_data, y_data, sen
    else:
      raise ValueError('Can not find dictionary file %s' % (fp) )

  def get_batch(self, x, y, sen = None):
    i = 0
    while i < len(x):
      start = i
      end = i + self.batch_size
      if end > len(x):
        end = len(x)
      x_batch = x[start:end]
      y_batch = y[start:end]
      if sen != None:
        sen_batch = sen[start:end]
        yield x_batch, y_batch, sen_batch
      yield x_batch, y_batch
      i = end

  def train(self):
    ckpt = tf.train.get_checkpoint_state(self.model_dir)
    if ckpt:
      print('load model from:', self.model_dir)
      self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    else:
      print("Creating model with fresh parameters.")
      self.sess.run(tf.global_variables_initializer())

    x_train, y_train, sen = self.build_dataset()
    x_test, y_test, sen = self.build_dataset(f='source_test')
    print ("start training...")
  
    step = 0

    for epoch in range(self.training_epochs):
      print (' epoch %3d :' %(epoch+1))
      total_acc = 0.0
      total_loss = 0.0
      for x_batch, y_batch in self.get_batch(x_train, y_train):
        step += 1
        loss, acc = self.step(x_batch, y_batch, False)
        total_acc += acc
        total_loss += loss

        if step % self.printing_step == 0:
          print ('\nStep {} =>  loss : {} ,  acc : {}'.format(step, total_loss/self.printing_step,total_acc/self.printing_step))
          total_acc = 0.0
          total_loss = 0.0

        if step % self.saving_step == 0:
          self.saver.save(self.sess, self.model_path, global_step=step)
          val_loss, val_acc = self.step(x_test, y_test, True)
          print ('\nTesting =>  val_loss: {} ,  acc : {}\n'.format(val_loss, val_acc))

        if step >= self.num_step:
          break
          
      if step >= self.num_step:
        break

  def val(self):
    print ('loading previous model...')
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
    
    x_d, y_d, sen_d = self.build_dataset(f='source_test')
    _acc = 0.0
    cnt = 0

    for x, y in self.get_batch(x_d, y_d):
      _, acc = self.step(x, y, True)
      _acc += acc*len(x)
      cnt += len(x)
    _acc /= cnt
    print ("source_test Acc : {}".format(_acc))

  def test(self):
    print ('loading previous model...')
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
    while True:
      input_sentence = input(">> Input your sentence: ")
      pat = re.compile('(\W+)')
      input_sentence = re.split(pat, input_sentence.lower())
      #print (' '.join(input_sentence))
      data = self.tokenizer(' '.join(input_sentence))
      #print (data)
      score = self.get_score(np.array([data]))
      print ('score: ' , score[0][0])

  def test_f(self):
    print ('loading previous model...')
    self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
    
    x_d, y_d, sen_d = self.build_dataset(f='source_train')
    f = open(os.path.join(self.data_dir, 'new_data'), 'w')
    cnt0, cnt1 = 0, 0

    for x, y, sen in self.get_batch(x_d, y_d, sen_d):
      score = self.get_score(x)

      for l, a, s in zip(sen, y, score):
        a, s = a[0], s[0]
        if s > 0.8:
          f.write('1 +++$+++ ' + l + '\n')
          print l, ' -> ', a, ',', s
          cnt1 += 1
        elif s < 0.2:
          f.write('0 +++$+++ ' + l + '\n')
          print l, ' -> ', a, ',', s
          cnt0 += 1
    f.close()
    print "Count -> 0 : [{}] , 1 : [{}]".format(cnt0, cnt1)

