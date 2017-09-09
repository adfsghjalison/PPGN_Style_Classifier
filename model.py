import collections
import os
import re
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
		self.data_file = os.path.join(self.data_dir, args.data_file)
		self.dict_file = args.dict_file
		self.sess = sess
		self.training_epochs = 2
		self.learning_rate = 0.001
		self.val_ratio = 0.98
		self.display_step = 50
		self.load_model = args.load_model
		self.embedding_dim = args.embedding_dim
		self.max_length = args.max_length
		self.unit_size = args.unit_size
		self.batch_size = args.batch_size
		self.dictionary, self.num_words = self.get_dictionary(self.dict_file)
		self.eos = self.dictionary['__EOS__']
		self.bos = self.dictionary['__BOS__']
		self.unk = self.dictionary['__UNK__']
		self.build_model()
		self.saver = tf.train.Saver(tf.all_variables())
	
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
		self.socre = tf.sigmoid(pred)
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
		output_feed = [self.socre]
		outputs = self.sess.run(output_feed, feed_dict=feed_dict)
		return outputs[0]

	def get_dictionary(self, dict_file):
		if os.path.exists(dict_file):
			print ('loading dictionary from : %s' %(dict_file))
			dictionary = dict()
			num_word = 0
			with open(dict_file, 'r', errors='ignore') as file:
				un_parse = file.readlines()
				for line in un_parse:
					line = line.strip('\n').split()
					dictionary[line[0]] = int(line[1])
					num_word += 1
			return dictionary, num_word
		else:
			raise ValueError('Can not find dictionary file %s' %(dict_file))

	def tokenizer(self, input_sentence):
		data = [self.eos]*self.max_length
		data[0] = self.bos
		data[self.max_length-1] = self.eos
		word_count = 1
		for word in input_sentence.split():
			if word_count>=self.max_length-1:
				break;
			if word in self.dictionary:
				data[word_count] = self.dictionary[word]
			else:
				data[word_count] = self.unk
			word_count += 1
		return data

	def build_dataset(self):
		if os.path.exists(self.data_file):
			print ('building dataset...')
			x_data = []
			y_data = []
			with open(self.data_file, 'r', errors='ignore') as file:
				un_parse = file.readlines()
				for line in un_parse:
					line = line.strip('\n').split(' +++$+++ ')
					y_data.append([int(line[0])])
					x_data.append(self.tokenizer(line[1]))
			x_train = np.array(x_data[:int(len(x_data)*self.val_ratio)])
			y_train = np.array(y_data[:int(len(x_data)*self.val_ratio)])
			x_test  = np.array(x_data[int(len(x_data)*self.val_ratio):])
			y_test  = np.array(y_data[int(len(y_data)*self.val_ratio):])
			return x_train, y_train, x_test, y_test
		else:
			raise ValueError('Can not find dictionary file %s' %(self.data_file))

	def get_batch(self, x, y):
		i = 0
		while i<len(x):
			start = i
			end = i + self.batch_size
			if end >len(x):
				end = len(x)
			x_batch = x[start:end]
			y_batch = y[start:end]
			yield x_batch, y_batch, i
			i = end

	def train(self):
		if self.load_model:
			print ('loading previous model...')
			self.saver.restore(self.sess, tf.train.latest_checkpoint(self.model_dir))
		else:
			print("Creating model with fresh parameters.")
			self.sess.run(tf.global_variables_initializer())
		x_train, y_train, x_test, y_test = self.build_dataset()
		print ("start training...")
		for epoch in range(self.training_epochs):
			print (' epoch %3d :' %(epoch+1))
			total_acc = 0.0
			total_loss = 0.0
			for x_batch, y_batch, i in self.get_batch(x_train, y_train):
				loss, acc = self.step(x_batch, y_batch, False)
				total_acc = (total_acc*i+ acc*self.batch_size)/(i+self.batch_size)
				total_loss = (total_loss*i+ loss*self.batch_size)/(i+self.batch_size)
				if (i / self.batch_size) % self.display_step == 0:
					print ( '\rIter %6d' %(i),'-- loss: %6f' %(total_loss), ' acc: %6f' %(total_acc),end='')
			val_loss, val_acc = self.step(x_test, y_test, True)
			print (' | testing -- val_loss: ', val_loss, ' val_acc: ', val_acc )
		self.saver.save(self.sess, self.model_path)

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