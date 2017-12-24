import tensorflow as tf
import math
from Models.attn_gru import AttnGRU

class RSTAN:
	def __init__(self, options):
		self.options = options
	
	def DropoutWrappedGRUCell(self, hidden_size, in_keep_prob, name=None):
		cell = tf.contrib.rnn.GRUCell(hidden_size)
		cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob = in_keep_prob)
		return cell

	def BiGRU(self, x, hidden_size, in_keep_prob, name = 'BiGRU'):
		with tf.name_scope(name):
			fw_lstm = self.DropoutWrappedGRUCell(hidden_size, in_keep_prob)
			bw_lstm = self.DropoutWrappedGRUCell(hidden_size, in_keep_prob)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(fw_lstm, bw_lstm, x, dtype=tf.float32)
			output_states = tf.concat([outputs[0][:, -1], outputs[1][:, 0]], 1)
			return tf.concat(outputs, 2, name = 'outputs'), tf.concat(output_states, 1, name = 'output_states')

	def AttnLayer(self, x, y, name=None):
		with tf.name_scope(name):
			initializer = tf.random_normal_initializer(stddev=0.1)
			Wx = tf.get_variable('Wx', [512, 512], initializer=initializer) # [Glove_dim, 300]
			Wy = tf.get_variable('Wy', [1024, 512], initializer=initializer) # [bGRU_dim, 300]
			Wt = tf.get_variable('Wt', [512, 1], initializer=initializer) # [300, 300]
			b = tf.get_variable('b', 512, initializer=tf.constant_initializer(0.0))
			Wxx = self.mat_weight_mul(tf.expand_dims(x, 1), Wx)
			Wyy = self.mat_weight_mul(y, Wy)
			s = self.mat_weight_mul(tf.tanh(Wxx + Wyy + b), Wt)
			return tf.nn.softmax(tf.squeeze(s), dim=1, name = 's')

	def mat_weight_mul(self, mat, weight):
		# [batch_size, n, m] * [m, p] = [batch_size, n, p]
		mat_shape = mat.get_shape().as_list()
		weight_shape = weight.get_shape().as_list()
		assert(mat_shape[-1] == weight_shape[0]), '{}, {}'.format(mat_shape[-1], weight_shape[0])
		mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
		mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
		return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])
		
	def sta(self, video, question, reuse=False):
		opts = self.options
		with tf.variable_scope('TemporalAttention') as scope:
			if reuse: scope.reuse_variables()
			h_s, _ = self.BiGRU(video, opts['bGRU_dim'], opts['in_keep_prob'], name = 'HiddenState') #bGRU_dim = 300?
			t_att_score = self.AttnLayer(question, h_s, name = 'TemporalAttentionScore')
			
			gru = AttnGRU(opts['aGRU_dim'])
			state = tf.zeros([opts['batch_size'], opts['aGRU_dim']])
			
			h_sp = [None for i in range(opts['v_length'])]	
			with tf.variable_scope('AttnGate') as scope2:
				for i in range(opts['v_length']):
					if i > 0:
						state = h_sp[i-1]
					h_sp[i] = gru(h_s[:, i], state, t_att_score[:, i], scope2)
					scope2.reuse_variables()  # share params
			return tf.identity(h_sp[-1], name = 'h_N')
			
	def question_encoding(self, question):
		output, q_enc = self.BiGRU(question, 256, 1.0, 'QuestionEncoding')
		return q_enc

	def build_model(self):
		opts = self.options

		# placeholders
		video = tf.placeholder(tf.float32, [opts['batch_size'], opts['v_length'], opts['v_dim']])
		question = tf.placeholder(tf.float32, [opts['batch_size'], opts['q_length'], opts['q_dim']])
		answer = tf.placeholder(tf.float32, [opts['batch_size'], opts['vocab']])

		print('Encoding')
		q_enc = self.question_encoding(question)
		y = q_enc
		for i in range(3):
			print('Step', i)
			with tf.name_scope('Step{}'.format(i)):
				reuse = True if i>0 else False
				sta = self.sta(video, q_enc, reuse=reuse)
				y = y + sta
		print(y)

		print('Decoding')
		with tf.variable_scope('decoder'):
			Wk = tf.get_variable('Wk', [opts['aGRU_dim'], opts['vocab']], initializer=tf.random_normal_initializer(stddev=0.1))
			Wky = self.mat_weight_mul(tf.expand_dims(y, 1), Wk)
			p = tf.nn.softmax(tf.squeeze(Wky), dim=1, name = 'p')
			print(p)	

		pred = tf.argmax(p, axis=1)
		ce = tf.nn.softmax_cross_entropy_with_logits(labels = answer, logits = Wky)
		loss = tf.reduce_sum(ce)
		correct = tf.equal(tf.cast(pred, tf.int64), tf.cast(tf.argmax(answer, axis=1), tf.int64)) 
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		input_tensors = {
			'v':video,
			'q':question,
			'a':answer,
		}
	
		print('Model built')
		for v in tf.global_variables():
			print(v.name, v.shape)
		
		return input_tensors, loss, accuracy, pred

