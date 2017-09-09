import collections
import os
import re
import tensorflow as tf 
import numpy as np
from lib.ops import *

class sentiment_classifier(object):
    def __init__(self, sess, args ):
    
    
        #VAE
        self.VAE_batch_size = 48
        self.word_embedding_dim = 300
        self.latent_dim = 500
        self.lstm_length = [args.max_length+1]*self.VAE_batch_size
        
        self.model_dir = args.model_dir
        try: 
            print ('creating the dicectory %s...' %(self.model_dir))
            os.mkdir(self.model_dir)
        except:
            print ('%s has created...' %(self.model_dir))
        self.model_path = os.path.join(self.model_dir,'model_{m_type}'.format(m_type='sent'))
        self.data_dir = args.data_dir
        self.data_file = os.path.join(self.data_dir,args.data_file)
        self.dict_file = args.dict_file
        self.sess = sess
        self.training_epochs = 2
        self.learning_rate = 0.001
        self.val_ratio = 0.98
        self.display_step = 50
        self.save_step = 5000
        self.load_model = args.load_model
        self.embedding_dim = args.embedding_dim
        self.max_length = args.max_length
        self.unit_size = args.unit_size
        self.batch_size = args.batch_size
        self.dictionary, self.num_words = self.get_dictionary(self.dict_file)
        self.eos = self.dictionary['__EOS__']
        self.bos = self.dictionary['__BOS__']
        self.unk = self.dictionary['__UNK__']
        self.build_VAE_graph()
        self.vae_saver = tf.train.Saver(var_list={v.op.name : v for v in self.dialogue_var_list}, max_to_keep=2)
        print(self.dialogue_var_list)
        self.vae_saver.restore(self.sess, tf.train.latest_checkpoint('/home_local/htsungy/vrnn/model_dir/')) 
        self.build_model()
        self.senti_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sentiment')
        self.saver = tf.train.Saver(var_list={v.op.name : v for v in self.senti_var_list}, max_to_keep = 3)
       
    
        
    
    def build_VAE_graph(self):
    
        print('starting building graph [VAE]')
            
        with tf.variable_scope("input") as scope:
            self.VAE_encoder_inputs = tf.placeholder(dtype=tf.int32, shape=(self.VAE_batch_size, self.max_length))
            self.train_decoder_sentence = tf.placeholder(dtype=tf.int32, shape=(self.VAE_batch_size, self.max_length))
   

            BOS_slice = tf.ones([self.VAE_batch_size, 1], dtype=tf.int32)*self.bos
            EOS_slice = tf.ones([self.VAE_batch_size, 1], dtype=tf.int32)*self.eos
            train_decoder_sentence = tf.concat([BOS_slice,self.train_decoder_sentence],axis=1)

        
        with tf.variable_scope("embedding") as scope:
            init = tf.contrib.layers.xavier_initializer()
          
            #word embedding
            word_embedding_matrix = tf.get_variable(
                name="word_embedding_matrix",
                shape=[self.num_words, self.word_embedding_dim],
                initializer=init,
                trainable = False)

            VAE_encoder_inputs_embedded = tf.nn.embedding_lookup(word_embedding_matrix, self.VAE_encoder_inputs)
            train_decoder_sentence_embedded = tf.nn.embedding_lookup(word_embedding_matrix, train_decoder_sentence)
             
        with tf.variable_scope("encoder") as scope:
            cell_fw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim, state_is_tuple=True)
            #bi-lstm encoder
            encoder_outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=cell_fw,
                cell_bw=cell_bw,
                dtype=tf.float32,
                sequence_length=self.lstm_length,
                inputs=VAE_encoder_inputs_embedded,
                time_major=False)
    
            output_fw, output_bw = encoder_outputs
            state_fw, state_bw = state
            encoder_outputs = tf.concat([output_fw,output_bw],2)      
            encoder_state_c = tf.concat((state_fw.c, state_bw.c), 1)
            encoder_state_h = tf.concat((state_fw.h, state_bw.h), 1)
            
        with tf.variable_scope("sample") as scope:
        
            w_mean = weight_variable([self.latent_dim*2,self.latent_dim*2],0.001)
            b_mean = bias_variable([self.latent_dim*2])
            scope.reuse_variables()
            b_mean_matrix = [b_mean] * self.batch_size
            
            w_logvar = weight_variable([self.latent_dim*2,self.latent_dim*2],0.001)
            b_logvar = bias_variable([self.latent_dim*2])
            scope.reuse_variables()
            b_logvar_matrix = [b_logvar] * self.batch_size
            
            mean = tf.matmul(encoder_state_h,w_mean) + b_mean
            logvar = tf.matmul(encoder_state_h,w_logvar) + b_logvar
            var = tf.exp( 0.5 * logvar)
            noise = tf.random_normal(tf.shape(var))
            sampled_encoder_state_h = mean + tf.multiply(var,noise)
            
        encoder_state = tf.contrib.rnn.LSTMStateTuple(c = encoder_state_c, h = sampled_encoder_state_h)         
        encoder_state_test = tf.contrib.rnn.LSTMStateTuple(c = encoder_state_c, h = encoder_state_h)
        decoder_inputs = batch_to_time_major(train_decoder_sentence_embedded ,self.max_length+1)  
        
        with tf.variable_scope("decoder") as scope:
        
            r_num = tf.reduce_sum(tf.random_uniform([1], seed=1))
            cell = tf.contrib.rnn.LSTMCell(num_units=self.latent_dim*2, state_is_tuple=True)                
    
            def test_decoder_loop(prev,i):
                prev_index = tf.stop_gradient(tf.argmax(prev, axis=-1))
                pred_prev = tf.nn.embedding_lookup(word_embedding_matrix, prev_index)
                next_input = pred_prev
                return next_input
            
            #the decoder of testing
            test_decoder_output,test_decoder_state = tf.contrib.legacy_seq2seq.attention_decoder(
                decoder_inputs = decoder_inputs,
                initial_state = encoder_state,
                attention_states = encoder_outputs,
                cell = cell,
                output_size = self.num_words,
                loop_function = test_decoder_loop,
                scope = scope
            )   #the test decoder input can be same as train
    
            
            test_decoder_logits = tf.stack(test_decoder_output, axis=1)
            test_pred = tf.argmax(test_decoder_logits,axis=-1)
            test_pred = tf.to_int32(test_pred,name='ToInt32')
            self.test_decoder_logits = test_decoder_logits 
        
        embedding_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding')
        encoder_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
        sample_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='sample')
        decoder_var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
        self.dialogue_var_list=embedding_var_list+encoder_var_list+decoder_var_list+sample_var_list
        
    
    def build_model(self):
        with tf.variable_scope("sentiment") as scope:
            #placeholder
            print ('placeholding...')
            self.encoder_inputs = tf.placeholder(tf.float32, shape=[self.batch_size, self.max_length, self.num_words])
            self.target = tf.placeholder(tf.float32, shape=[self.batch_size, 1])
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
                
            self.k = k = tf.get_variable(initializer = tf.random_uniform_initializer(3, 5, dtype=tf.float32), shape=[], name="softmax_k")
            #self.encoder_inputs_softmax = tf.nn.softmax(tf.scalar_mul(k,self.encoder_inputs))
            self.encoder_inputs = tf.argmax(self.encoder_inputs, axis=-1)
            self.encoder_inputs = tf.one_hot(self.encoder_inputs, depth=self.num_words, axis=-1)
            y_list=[]        
            for i in range(self.encoder_inputs.get_shape().as_list()[0]):        
                y = tf.matmul(self.encoder_inputs[i], weights['w2v'])
                y = tf.reshape(y, [1, self.max_length, self.embedding_dim])
                y_list.append(y)    
            
            embbed_layer = tf.concat(y_list,0)    
            layer_1 = BiRNN(embbed_layer)
            pred = tf.matmul(layer_1, weights['out_1']) + biases['out_1']
            self.cost = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=pred, labels=self.target) )
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
    
            ###testing part
            self.socre = tf.sigmoid(pred)
            pred_sig = tf.cast(tf.greater(tf.sigmoid(pred), 0.5), tf.float32)
            correct_ped = tf.equal(pred_sig,self.target)
            self.accuracy = tf.reduce_mean(tf.cast(correct_ped, tf.float32))

    def step(self, x_batch, y_batch, predict):
        x_batch_dis = self.get_distribution(x_batch)
        feed_dict = { self.encoder_inputs: x_batch_dis, self.target: y_batch }
        output_feed = [self.cost, self.accuracy, self.k]
        if predict:
            outputs = self.sess.run(output_feed, feed_dict=feed_dict)
        else:
            output_feed.append(self.optimizer)
            outputs = self.sess.run(output_feed, feed_dict=feed_dict)
        
        return outputs[0], outputs[1], outputs[2]

    def get_score(self, input_idlist):
        sent_vec = np.zeros((self.VAE_batch_size,self.max_length),dtype=np.int32)
        sent_vec[0] = input_idlist
        input_dis = self.get_distribution(sent_vec)
        #print(input_dis[0])
        feed_dict = { self.encoder_inputs: input_dis }
        output_feed = [self.socre]
        outputs = self.sess.run(output_feed, feed_dict=feed_dict)
        return outputs[0]
        
    def get_distribution(self,sent_vec):
        t = np.ones((self.VAE_batch_size, self.max_length),dtype=np.int32)
        feed_dict = {
                self.VAE_encoder_inputs:sent_vec,\
                self.train_decoder_sentence:t
        }
        distribution = self.sess.run(self.test_decoder_logits,feed_dict)
        return distribution[:, :self.max_length, :]
        

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
        word_count = 0
        for word in input_sentence.split():
            if word_count >= self.max_length-1:
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
                np.random.shuffle(un_parse)
                for line in un_parse:
                    line = line.strip('\n').split(' +++$+++ ')
                    y_data.append([int(line[0])])
                    x_data.append(self.tokenizer(line[1]))
            
            x_train = np.array(x_data[:int(len(x_data)*self.val_ratio)])
            y_train = np.array(y_data[:int(len(x_data)*self.val_ratio)])
            x_test  = np.array(x_data[int(len(x_data)*self.val_ratio):])
            y_test  = np.array(y_data[int(len(y_data)*self.val_ratio):])
            print("total data size: " + str(len(y_train) + len(y_test)))
            print("positive data size: " + str(np.sum(y_train) + np.sum(y_test)))
            return x_train, y_train, x_test, y_test
        else:
            raise ValueError('Can not find dictionary file %s' %(self.data_file))

    def get_batch(self, x, y):
        i = 0
        while i<len(x):
            start = i
            end = i + self.batch_size
            if end < len(x):   
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
                loss, acc ,k = self.step(x_batch, y_batch, False)
                total_acc = (total_acc*i+ acc*self.batch_size)/(i+self.batch_size)
                total_loss = (total_loss*i+ loss*self.batch_size)/(i+self.batch_size)
                if (i / self.batch_size) % self.display_step == 0:
                    print ( 'Iter %6d' %(i),'-- loss: %6f' %(total_loss), ' acc: %6f' %(total_acc) , ' k: ' , k )
                if (i / self.batch_size) % self.save_step == 0:
                    self.saver.save(self.sess, self.model_path)
            #val_loss, val_acc, val_k = self.step(x_test, y_test, True)
            #print (' | testing -- val_loss: ', val_loss, ' val_acc: ', val_acc, ' k: ', val_k )
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
