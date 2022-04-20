import tensorflow as tf
import sonnet as snt
from tensorflow.keras.models import Model
import tensorflow.keras.layers as kl
import numpy as np

"""
All the code is propietary exceppt for:
	*Relation Net, MHA and PrediNet, which come from https://github.com/deepmind/deepmind-research/tree/master/PrediNet, 
	distributed under license Apache 2.0. They were adapted to work within the bigger networks of this research

	*RIMs, adapted from https://github.com/dido1998/Recurrent-Independent-Mechanisms to tensorflow. No lincense available
"""

# ----- Self Attention Modules ---------------------------------------------

class CombineValues(tf.keras.layers.Layer):
	"""Custom module for computing tensor products in SelfAttentionNet."""

	def __init__(self, heads, w_init=None):
		"""Initialise the CombineValues module.

		Args:
			heads: scalar. Number of heads over which to combine the values.
			w_init: an initializer. A sonnet snt.initializers.
		"""
		super(CombineValues, self).__init__(name='CombineValues')
		self._heads = heads
		self.w_init = w_init

	def build(self, input_shape):
		num_features = input_shape[2]
		if self.w_init is None:
			self.w_init = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1)
		self.w=tf.Variable(
					self.w_init([self._heads, 1, num_features], tf.float32), name='w')	
			
	def call(self, inputs):
		return tf.einsum('bhrv,har->bhav', inputs, self.w)



# ----- Hybrids ---------------------------------------------
class AC_TPN_NMHA_MM(Model):
	"""
	Actor-Critic MultiModal PrediNet-MHA imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	"""
	def __init__(self, n_outputs, conv_out_size, trainable, board = 0, value_size = 20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False,
					extended = True, n_neurons = 128, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_TPN_NMHA_MM'):

		super(AC_TPN_NMHA_MM, self).__init__(name=name)
		self.trainable = trainable

		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._value_size = value_size
		self._visual = visual
		if visual: self._conv_out_size = int(conv_out_size[-2]/9 * conv_out_size[-1]/9)
		else: self._conv_out_size = conv_out_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		# Feature co-ordinate matrix
		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsT = tf.concat([cols,rows],1)

		self.flattenT = tf.keras.layers.Flatten()

		# Define all model components
		self.get_keysT = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysT')
		self.get_query1T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1T')
		self.get_query2T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2T')
		self.embed_entitiesT = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesT')

		# self.DenseT = {}
		# for i in range(0, n_hlayers):
		if Hlstm:
			self.lstmT = tf.keras.layers.LSTM(units=128, time_major = True)
		
		else:
			# for i in range(0, n_hlayers):
			self.DenseT = tf.keras.layers.Dense(units=relations_p, 
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												activation='relu', name='DenseT')
		self.Tout = tf.keras.layers.Dense(units=self.n_neurons,
											bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
											activation=None, name='Tout')

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(obs_dim)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self.obs_size,1])

		rows = tf.reshape(rows, [self.obs_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.get_keysN = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keys')
		self.get_queriesN = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_queries')
		self.get_valuesN = tf.keras.layers.Dense(
				units=self._heads * self._value_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_values')
		self.central_outputN = CombineValues(
				heads=self._heads, w_init=self._weight_initializer)

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			self.lstmN = tf.keras.layers.LSTM(units=128, time_major = True)

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_TPN_NMHA_MM, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None):
		s = tf.dtypes.cast(s, tf.float32)

		if self._visual:
			if len(s.shape)==2:
				s = tf.expand_dims(s, 0)
				s = tf.expand_dims(s, -1)
			elif len(s.shape)==3:
				s = tf.expand_dims(s, -1)
			# print("Input shape", s.shape)
			s = self.convA(s)
			s = tf.nn.relu(s)
			s = self.convB(s)
			s = tf.nn.relu(s)
			s = self.convC(s)
			s = self.flattenC(s)

		# tf.print(s, "Conv output", summarize=40)
		# a = tf.Print(a, [a[0, 0]], "Print part of a\n", summarize=100000)
		x = s
		batch_size = s.shape[0]
		if prev_act == None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
		prev_rew = tf.reshape(prev_rew, [batch_size, 1])
		prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		if len(s.shape)==2:
			obs = s[:,-self.obs_size:]
			x = tf.expand_dims(x, -1)
			obs = tf.expand_dims(obs, -1)
		else:
			x = tf.reshape(x, [batch_size, s.shape[2], s.shape[1]])
			obs = tf.reshape(x[:,-self.obs_size:], [batch_size,
				self.obs_size,1])	
		# print("Latent space", x.shape)
		# print("obs", obs.shape)		
		# jsakjsa+=1
			# jaja+=1
		with tf.device("/device:GPU:0"):
			# print("s",s.shape)
			# print("x",x.shape)
			# print("obs ",obs.shape)
			# Append location
			locs = tf.tile(tf.expand_dims(self._locsT, 0), [batch_size, 1, 1])
			# print("locs",locs.shape)
			# print("x",x.shape)
			features_locs = tf.concat([x, locs], 2)
			# (batch_size, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = self.flattenT(features_locs)
			# (batch_size, (conv_out_size+1)*conv_out_size*channels+2)
			

			# Keys
			keys = snt.BatchApply(self.get_keysT)(features_locs)
			# (batch_size,  (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 1), [1, self._heads, 1, 1])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, key_size)

			# Queries
			query1 = self.get_query1T(features_flat)
			# (batch_size, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, self._heads, self._key_size])
			# (batch_size, heads, key_size)
			query1 = tf.expand_dims(query1, 2)
			# (batch_size, heads, 1, key_size)

			query2 = self.get_query2T(features_flat)
			# (batch_size, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, self._heads, self._key_size])
			# (batch_size, heads, key_size)
			query2 = tf.expand_dims(query2, 2)
			# (batch_size, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 3, 2])
			# (batch_size, heads, key_size, conv_out_size*conv_out_size)
			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 1), [1, self._heads, 1, 1])
			# (batch_size, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))
			# (batch_size, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = snt.BatchApply(self.embed_entitiesT)(feature1)
			embedding2 = snt.BatchApply(self.embed_entitiesT)(feature2)
			# (batch_size, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2) #here

			# Positions
			pos1 = tf.slice(feature1, [0, 0, self._channels], [-1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, self._channels], [-1, -1, -1])
			# (batch_size, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 2)
			# (batch_size, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, self._heads * (self._relations + 4)])
			# (batch_size, heads*(relations+4))
			x = relations

			
			if self._Hlstm:
				# relations = tf.expand_dims(relations, 0)
				# x = tf.concat([relations, prev_act, prev_rew], -1)
				x = tf.concat([relations, prev_act], -1)
				x = tf.expand_dims(x, 0)
				x = self.lstmT(x)
			else: x =  self.DenseT(relations)

			IntTask = self.Tout(x)
			# x = tf.expand_dims(x, -1)
			# x = tf.concat([x,obs],1)

			#-------Navigator
			locs = tf.tile(tf.expand_dims(self._locsN, 0), [batch_size, 1, 1])
			features_locs = tf.concat([obs, locs], 2)
			# (batch_size, (conv_out_size+1)*conv_out_size,channels+2)


			# Keys
			keys = snt.BatchApply(self.get_keysN)(features_locs)
			# (batch_size,  (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.reshape(keys, [
				batch_size, self.obs_size, 
				self._heads, self._key_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, key_size)

			# Queries
			queries = snt.BatchApply(self.get_queriesN)(features_locs)
			#(batch_size, (conv_out_size+1)*conv_out_size, heads * key_size)
			queries = tf.reshape(queries, [
				batch_size, self.obs_size, 
				self._heads, self._key_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, key_size)

			# Values 
			values = snt.BatchApply(self.get_valuesN)(features_locs)
			#(batch_size, (conv_out_size+1)*conv_out_size, heads * key_size)
			values = tf.reshape(values, [
				batch_size, self.obs_size,
				self._heads, self._value_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, values_size)

			# Attention weights
			queries_t = tf.transpose(queries, perm=[0, 2, 1, 3])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, key_size)
			keys_t = tf.transpose(keys, perm=[0, 2, 3, 1])
			# (batch_size, heads, key_size, (conv_out_size+1)*conv_out_size)
			att = tf.nn.softmax(tf.matmul(queries_t, keys_t))
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, (conv_out_size+1)*conv_out_size)

			# Apply attention weights to values
			values_t = tf.transpose(values, perm=[0, 2, 1, 3])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)
			values_out = tf.matmul(att, values_t)
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)


			# # Compute self-attention head output
			central_out = snt.Flatten()(self.central_outputN(values_out))

			x = tf.concat([central_out, IntTask],1)

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				x = tf.expand_dims(x, 0)
				x = self.lstmN(x)
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
		return tf.squeeze(logits), tf.squeeze(values,-1)

class AC_TMHA_NPN_MM(Model):
	"""
	Actor-Critic MultiModal PrediNet, PrediNet imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	"""
	def __init__(self, n_outputs, conv_out_size, trainable, board = 0, value_size = 20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False,
					extended = True, n_neurons = 128, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_TMHA_NPN_MM'):

		super(AC_TMHA_NPN_MM, self).__init__(name=name)
		self.trainable = trainable

		self._value_size = value_size
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		if visual: self._conv_out_size = int(conv_out_size[-2]/9 * conv_out_size[-1]/9)
		else: self._conv_out_size = conv_out_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsT = tf.concat([cols,rows],1)

		# Define all model components
		self.get_keysT = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keys')
		self.get_queriesT = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_queries')
		self.get_valuesT = tf.keras.layers.Dense(
				units=self._heads * self._value_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_values')
		self.central_outputT = CombineValues(
				heads=self._heads, w_init=self._weight_initializer)

		# self.DenseT = {}
		# for i in range(0, n_hlayers):
		if Hlstm:
			self.lstmT = tf.keras.layers.LSTM(units=128, time_major = True)
		
		else:
			# for i in range(0, n_hlayers):
			self.DenseT = tf.keras.layers.Dense(units=relations_p, 
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												activation='relu', name='DenseT')
		self.Tout = tf.keras.layers.Dense(units=self.n_neurons,
											bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
											activation=None, name='Tout')

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(obs_dim)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self.obs_size,1])

		rows = tf.reshape(rows, [self.obs_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.flattenN = tf.keras.layers.Flatten()

		self.get_keysN = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysN')
		self.get_query1N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1N')
		self.get_query2N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2N')
		self.embed_entitiesN = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesN')

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			self.lstmN = tf.keras.layers.LSTM(units=128, time_major = True)

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_TMHA_NPN_MM, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None):
		s = tf.dtypes.cast(s, tf.float32)

		if self._visual:
			if len(s.shape)==2:
				s = tf.expand_dims(s, 0)
				s = tf.expand_dims(s, -1)
			elif len(s.shape)==3:
				s = tf.expand_dims(s, -1)
			# print("Input shape", s.shape)
			s = self.convA(s)
			s = tf.nn.relu(s)
			s = self.convB(s)
			s = tf.nn.relu(s)
			s = self.convC(s)
			s = self.flattenC(s)

		# tf.print(s, "Conv output", summarize=40)
		# a = tf.Print(a, [a[0, 0]], "Print part of a\n", summarize=100000)
		x = s
		batch_size = s.shape[0]
		if prev_act == None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
		prev_rew = tf.reshape(prev_rew, [batch_size, 1])
		prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		if len(s.shape)==2:
			obs = s[:,-self.obs_size:]
			x = tf.expand_dims(x, -1)
			obs = tf.expand_dims(obs, -1)
		else:
			x = tf.reshape(x, [batch_size, s.shape[2], s.shape[1]])
			obs = tf.reshape(x[:,-self.obs_size:], [batch_size,
				self.obs_size,1])	
		# print("Latent space", x.shape)
		# print("obs", obs.shape)		
		# jsakjsa+=1
			# jaja+=1
		with tf.device("/device:GPU:0"):
			# print("s",s.shape)
			# print("x",x.shape)
			# print("obs ",obs.shape)
			# Append location
			locs = tf.tile(tf.expand_dims(self._locsT, 0), [batch_size, 1, 1])
			features_locs = tf.concat([x, locs], 2)
			# (batch_size, (conv_out_size+1)*conv_out_size,channels+2)

			# Keys
			keys = snt.BatchApply(self.get_keysT)(features_locs)
			# (batch_size,  (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.reshape(keys, [
				batch_size, self._conv_out_size, 
				self._heads, self._key_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, key_size)

			# Queries
			queries = snt.BatchApply(self.get_queriesT)(features_locs)
			#(batch_size, (conv_out_size+1)*conv_out_size, heads * key_size)
			queries = tf.reshape(queries, [
				batch_size, self._conv_out_size, 
				self._heads, self._key_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, key_size)

			# Values 
			values = snt.BatchApply(self.get_valuesT)(features_locs)
			#(batch_size, (conv_out_size+1)*conv_out_size, heads * key_size)
			values = tf.reshape(values, [
				batch_size, self._conv_out_size,
				self._heads, self._value_size])
			#(batch_size, (conv_out_size+1)*conv_out_size, heads, values_size)

			# Attention weights
			queries_t = tf.transpose(queries, perm=[0, 2, 1, 3])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, key_size)
			keys_t = tf.transpose(keys, perm=[0, 2, 3, 1])
			# (batch_size, heads, key_size, (conv_out_size+1)*conv_out_size)
			att = tf.nn.softmax(tf.matmul(queries_t, keys_t))
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, (conv_out_size+1)*conv_out_size)

			# Apply attention weights to values
			values_t = tf.transpose(values, perm=[0, 2, 1, 3])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)
			values_out = tf.matmul(att, values_t)
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)


			# # Compute self-attention head output
			central_out = snt.Flatten()(self.central_outputT(values_out))

			x = central_out
			
			if self._Hlstm:
				# relations = tf.expand_dims(relations, 0)
				# x = tf.concat([relations, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				x = tf.expand_dims(x, 0)
				x = self.lstmT(x)
			else: x =  self.DenseT(x)

			IntTask = self.Tout(x)
			# x = tf.expand_dims(x, -1)
			# x = tf.concat([x,obs],1)

			#-------Navigator
			locs = tf.tile(tf.expand_dims(self._locsN, 0), [batch_size, 1, 1])
			features_locs = tf.concat([obs, locs], 2)
			# (batch_size, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = self.flattenN(features_locs)
			# (batch_size, (conv_out_size+1)*conv_out_size*channels+2)
			

			# Keys
			keys = snt.BatchApply(self.get_keysN)(features_locs)
			# (batch_size,  (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 1), [1, self._heads, 1, 1])
			# (batch_size, heads, (conv_out_size+1)*conv_out_size, key_size)

			# Queries
			query1 = self.get_query1N(features_flat)
			# (batch_size, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, self._heads, self._key_size])
			# (batch_size, heads, key_size)
			query1 = tf.expand_dims(query1, 2)
			# (batch_size, heads, 1, key_size)

			query2 = self.get_query2N(features_flat)
			# (batch_size, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, self._heads, self._key_size])
			# (batch_size, heads, key_size)
			query2 = tf.expand_dims(query2, 2)
			# (batch_size, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 3, 2])
			# (batch_size, heads, key_size, conv_out_size*conv_out_size)
			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 1), [1, self._heads, 1, 1])
			# (batch_size, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))
			# (batch_size, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = snt.BatchApply(self.embed_entitiesN)(feature1)
			embedding2 = snt.BatchApply(self.embed_entitiesN)(feature2)
			# (batch_size, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, self._channels], [-1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, self._channels], [-1, -1, -1])
			# (batch_size, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 2)
			# (batch_size, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, self._heads * (self._relations + 4)])

			x = tf.concat([relations, IntTask],1)

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				x = tf.expand_dims(x, 0)
				x = self.lstmN(x)
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
		return tf.squeeze(logits), tf.squeeze(values,-1)

class GroupLinearLayer(tf.keras.layers.Layer):
	def __init__(self, units, nRIM, initializer = 'random_normal'):
		super(GroupLinearLayer, self).__init__()
		self.units = units
		self.nRIM = nRIM
		self.initializer = initializer

	def build(self, input_shape):
		self.w = self.add_weight(name = 'group_linear_layer',
								shape = (self.nRIM, int(input_shape[-1]), self.units),
								initializer = self.initializer,
								trainable = True)

	def call(self, inputs):
		params = self.w
		output = tf.transpose(tf.matmul(tf.transpose(inputs, [1,0,2]), params),[1,0,2])
		return output

class GroupGRUCell(tf.keras.layers.Layer):
	def __init__(self, units, nRIM):
		super(GroupGRUCell, self).__init__()
		self.units = units
		self.nRIM = nRIM
		self.x2h = GroupLinearLayer(3 * units, nRIM, initializer = 'uniform')
		self.h2h = GroupLinearLayer(3 * units, nRIM, initializer = 'uniform')

	@property
	def state_size(self):
		return tf.TensorShape([self.nRIM, self.units])

	def build(self, input_shape):
		super(GroupGRUCell, self).build(input_shape)

	def call(self, inputs, h_state):
		# inputs in shape [batch, nRIM, din]
		# h, hidden_state in shape [batch, nRIM, units]

		preact_i = self.x2h(inputs)  #Group linear layer
		preact_h = self.h2h(h_state) 
		i_reset, i_input, i_new = tf.split(preact_i, 3, 2)
		h_reset, h_input, h_new = tf.split(preact_h, 3, 2)

		reset_gate = tf.sigmoid(i_reset + h_reset)
		input_gate = tf.sigmoid(i_input + h_input)
		new_gate = tf.tanh(i_new + tf.multiply(reset_gate, h_new))


		hy = new_gate + tf.multiply(input_gate, h_state - new_gate)
		# hy = tf.multiply(update_gate, new_gate) + tf.multiply(1-update_gate, h_state)
		# jajaja+=1
		return hy, hy


class IntegratedVisualTextEncoder(tf.keras.Model):
	"""
	Encoder designed assuming a 9x9 resolution per letter in the visual input
	"""
	def __init__(self, kernels=[3,3,1], filters=[8,16,1], strides=[3,3,1],
				name='IntegratedVisualTextEncoder'):
		super(IntegratedVisualTextEncoder, self).__init__(name=name)
		self.convA    = kl.Conv2D(filters[0], kernels[0], strides=strides[0])
		self.convB    = kl.Conv2D(filters[1], kernels[1], strides=strides[1])
		self.convC    = kl.Conv2D(filters[2], kernels[2], strides=strides[2])
		self.flattenC = kl.Flatten()

	def build(self, input_shape):
		super(IntegratedVisualTextEncoder, self).build(input_shape)

	def call(self, s):
		if len(s.shape)>4:
			s = kl.TimeDistributed(self.convA)(s)
			s = tf.nn.relu(s)
			s = kl.TimeDistributed(self.convB)(s)
			s = tf.nn.relu(s)
			s = kl.TimeDistributed(self.convC)(s)
			s = kl.TimeDistributed(self.flattenC)(s)
		else:
			s = self.convA(s)
			s = tf.nn.relu(s)
			s = self.convB(s)
			s = tf.nn.relu(s)
			s = self.convC(s)
			s = self.flattenC(s)
		return s	


class GeneralImgEncoder(tf.keras.Model):
	def __init__(self, kernels=[3,3,3], filters=[16,32,64], strides=[1,1,1]):
		super(GeneralImgEncoder, self).__init__()
		self.conv = {}
		self.drop = {}
		self.maxp = {}
		self.n_layers = len(filters)

		# For V5 all only this:
		# self.convA=kl.Conv2D(filters[0], kernels[0], strides=strides[0], activation='relu')
		# self.convB=kl.Conv2D(filters[1], kernels[1], strides=strides[1], activation='relu')

		# For V6 and 5_3 only this:
		# for i in range(self.n_layers):
		# 	self.conv[str(i)] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu')

		# For V6_2_2 all only this:
		for i in range(self.n_layers):
			if i < self.n_layers-1:
				self.conv[str(i)] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu')
			else:
				self.conv[str(i)] = kl.Conv2D(filters[i], kernels[i], strides=strides[i])

		#V3 
		# for i in range(self.n_layers):
		# 	self.conv[str(i)+'0'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu', padding='same')
		# 	self.maxp[str(i)] = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')
		# 	for l in range(2):
		# 		self.conv[str(i)+ str(l) +'1'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu', padding='same')
		# 		self.conv[str(i)+ str(l) +'2'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu', padding='same')


		# self.final_conv = kl.Conv2D(filters[self.n_layers-1], 1, strides=1, padding='same', activation='relu')

		#---- V5, not below

		# self.init_conv0 = kl.Conv2D(filters[0], kernels[0], strides=strides[0], activation='relu', padding='same')
		# self.init_conv1 = kl.Conv2D(filters[0], kernels[0], strides=strides[0], activation='relu', padding='same')
		# # For V5 all bellow should be removed
		# # self.drop['0'] = kl.Dropout(0.1)
		# for i in range(1,self.n_layers):
		# 	# self.maxp[str(i)] = kl.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid') #V1, 4
		# 	self.maxp[str(i)] = kl.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid') #V2, 3
		# 	self.conv[str(i)+'0'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu', padding='same')
		# 	self.conv[str(i)+'1'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], activation='relu', padding='same')
		# 	self.drop[str(i)] = kl.Dropout(0.1)

		# self.final_conv = kl.Conv2D(filters[self.n_layers-1], 1, strides=1, padding='same', activation='relu')
		# # self.dropout_enc = kl.Dropout(0.1)
		# # self.flattenC  = kl.Flatten()
	def build(self, input_shape):
		super(GeneralImgEncoder, self).build(input_shape)

	def call(self, s):
		# s_skip = s
		# s = kl.TimeDistributed(self.init_conv0)(s)
		# s = kl.TimeDistributed(self.init_conv1)(s)
		# s = tf.concat([s,s_skip], -1)
		# # s = kl.TimeDistributed(self.drop['0'])(s) 
		# for i in range(1,self.n_layers):
			# s = kl.TimeDistributed(self.maxp[str(i)])(s)
			# s_skip = s
			# s = kl.TimeDistributed(self.conv[str(i)+'0'])(s)
			# s = kl.TimeDistributed(self.conv[str(i)+'1'])(s)
			# s = tf.concat([s,s_skip], -1)
		# 	# s = kl.TimeDistributed(self.drop[str(i)])(s) 
		# # s = kl.TimeDistributed(self.flattenC)(s)
		# s = self.final_conv(s)
		#-- V3 -----
		# for i in range(self.n_layers):
		# 	s = kl.TimeDistributed(self.conv[str(i)+'0'])(s)
		# 	s = kl.TimeDistributed(self.maxp[str(i)])(s)
		# 	s_skip = s
		# 	for l in range(2):
		# 		s = kl.TimeDistributed(self.conv[str(i)+ str(l)+'1'])(s)
		# 		s = kl.TimeDistributed(self.conv[str(i)+ str(l)+'2'])(s)
		# 		s = tf.concat([s,s_skip], -1)
		# 		s_skip = s

		# s = self.final_conv(s)
		#-- V5 -----
		# s = kl.TimeDistributed(self.convA)(s)
		# s = kl.TimeDistributed(self.convB)(s)

		#-- V6 -----
		for i in range(self.n_layers):
			s = kl.TimeDistributed(self.conv[str(i)])(s)

		return s

class GeneralImgDecoder(tf.keras.Model):
	def __init__(self, kernels=[3,3,3], filters=[64,32,16,3], strides=[1,1,1], channels=3, ttl = 'TTL2'):
		super(GeneralImgDecoder, self).__init__()
		self.dconv = {}
		self.conv = {}
		self.n_layers = len(filters)
		output_padding	= None
		pad ='valid' if ttl == 'TTL1' else 'same'
		pad_last = 'same'
		# for i in range(self.n_layers-1):
		# 	l = i+1
		# 	if i == self.n_layers-2: pad=pad_last
		# 	output_padding = (1,1) #V3
		# 	if ttl == 'TTL2':
		# 		if i == 0 or i ==1: output_padding = (0,0) #V2,
		# 	# else: #V4
		# 	# 	if i == self.n_layers-2: output_padding = (0,0) #V4
		# 	# self.dconv[str(i)] = kl.Conv2DTranspose(filters[l], kernels[l], strides=(2,2), padding	= pad) #V1,
		# 	# self.dconv[str(i)] = kl.Conv2DTranspose(filters[l], kernels[l], strides=(2,2), output_padding = output_padding) #V2, V3, V4
		# 	self.conv[str(i)+'0'] = kl.Conv2D(filters[l], kernels[l], strides=1, activation='relu', padding='same')
		# 	self.conv[str(i)+'1'] = kl.Conv2D(filters[l], kernels[l], strides=1, activation='relu', padding='same')


		#----V5
		filters = filters[1:]
		filters.append(channels)
		for i in range(self.n_layers): #V5
			output_padding	= (0,0) #V6
			# output_padding	= (2,2) if i == self.n_layers-1 else (0,0) #V5
			self.dconv[str(i)] = kl.Conv2DTranspose(filters[i], kernels[i], strides=strides[i], output_padding = output_padding) #V5,V6
		# self.final_conv = kl.Conv2D(channels, 5, strides=1)

	def build(self, input_shape):
		super(GeneralImgDecoder, self).build(input_shape)

	def call(self, s):
		# for i in range(self.n_layers-1):
		# 	s = kl.TimeDistributed(self.dconv[str(i)])(s)
		# 	s_skip = s
		# 	s = kl.TimeDistributed(self.conv[str(i)+'0'])(s)
		# 	s = kl.TimeDistributed(self.conv[str(i)+'1'])(s)
		# 	s = tf.concat([s,s_skip], -1)

		# s = kl.TimeDistributed(self.final_conv)(s)
		

		# V5
		for i in range(self.n_layers):
			s = kl.TimeDistributed(self.dconv[str(i)])(s)

		# s = kl.TimeDistributed(self.final_conv)(s)
		# s = kl.TimeDistributed(self.flattenC)(s)
		return s

class CAE(tf.keras.Model):
	"""Convolutional autoencoder, inspired on the UNET from 
	"Deep Learning for SAR Image Despeckling" https://doi.org/10.3390/rs11131532
	"""
	def __init__(self, channels = 3, ttl = 'TTL2'):
		super(CAE, self).__init__()

		# kernels=[3,3,3,3] #V1 maxpool pool_size 2
		# strides=[1,1,1,1]
		# filters=[channels, channels*2, channels*4, channels*8]

		# kernels=[3,3,3] #V1_2 maxpool pool_size 2
		# strides=[1,1,1]
		# filters=[channels*2, channels*4, channels*8]

		# kernels=[3,3,3,3] #V2 maxpool pool_size 3
		# strides=[1,1,1,1]
		# filters=[channels,channels*5+1, channels*10+2, channels*10+2]

		# kernels=[3,3,3] #V3 maxpool pool_size 3
		# strides=[1,1,1]
		# filters=[16, 32, 32]

		# kernels=[3,3,3,3] #V3_2 maxpool pool_size 3
		# strides=[1,1,1,1]
		# filters=[channels, 16, 32, 32]

		# kernels=[2,2,2] #V4 maxpool pool_size 2
		# strides=[1,1,1]
		# filters=[16, 32, 64]

		# kernels=[2,2,2,2] #V4_2 maxpool pool_size 2
		# strides=[1,1,1,1]
		# filters=[channels, 16, 32, 64]

		# kernels=[8,4] #V5 No Maxpool
		# strides=[4,2]
		# filters=[16, 32]


		# kernels=[3,3] #V5_2 No Maxpool
		# strides=[3,3]
		# filters=[16, 32]

		# kernels=[8,4,1] #V5_3 No Maxpool
		# strides=[4,2,1]
		# filters=[16, 32,1]

		# # V6 no Maxpool
		# kernels=[3,3,1]
		# filters=[8,16,1]
		# strides=[3,3,1]

		
		kernels=[3,3,1] # V6_2 and 6_2_2
		filters=[16,32,1]
		strides=[3,3,1]


		dec_filters = filters[::-1]
		dec_kernels = kernels[::-1]
		dec_strides = strides[::-1]

		self.Encoder    = GeneralImgEncoder(kernels, filters, strides)
		self.Decoder 	= GeneralImgDecoder(dec_kernels, dec_filters, dec_strides, channels, ttl)
		# self.flatten  	= kl.Flatten()

	def build(self, input_shape):
		super(CAE, self).build(input_shape)

	def call(self, s):
		# print('s', s.shape)
		if s.shape[2] != s.shape[3]: # We need to make it a square image in case we have a rectangular image
			if s.shape[2] < s.shape[3]:
				aux = tf.zeros([s.shape[0], s.shape[1], s.shape[3]-s.shape[2], s.shape[3], s.shape[4]])
				s = tf.concat([s,aux], 2)
			elif s.shape[2] > s.shape[3]:
				aux = tf.zeros([s.shape[0], s.shape[1], s.shape[2], s.shape[2]-s.shape[3], s.shape[4]])
				s = tf.concat([s,aux], 3)
		# print('s edited', s.shape)
		enc = self.Encoder(s)
		# print('enc', enc.shape)
		dec = self.Decoder(enc)
		# dec = s
		# print('dec', dec.shape)
		# jdjdjd+=1
		assert s.shape == dec.shape
		return enc, dec, s

class AuxCNN(tf.keras.layers.Layer):
	def __init__(self, channels = 3, mapT = 'minigrid',
					ResNet = False):
		super(AuxCNN, self).__init__()
		self.conv = {}
		self.ResNet = ResNet
		# kernels=[8,4] #V5 No Maxpool
		# strides=[4,2]
		# filters=[16, 32]

		# kernels=[9,3] #V5_2 No Maxpool
		# strides=[3,3]
		# filters=[16, 32]

		# kernels=[8,4,1] #V5_3_2 No Maxpool
		# strides=[4,2,1]
		# filters=[16, 32,1]

		# kernels=[8,4,1] #V5_3_3 No Maxpool no relu	
		# strides=[4,2,1]
		# filters=[16, 32,channels]

		# V6 ICML NETWORK
		# kernels=[3,3,1]
		# filters=[8,16,1]
		# strides=[3,3,1]

		# kernels=[3,3,1] # V6_2_2, no relu
		# filters=[16,32,1]
		# strides=[3,3,1]
		
		# kernels=[3,3,1] # V6_2_3, no relu
		# filters=[16,32,channels]
		# strides=[3,3,1]

		# kernels=[6,3,1] # V8, no relu, additional padding req
		# filters=[16,32,channels]
		# strides=[2,2,1]

		# kernels=[6,2,2] # #V8_2, no relu
		# filters=[16,32,channels]
		# strides=[2,2,2]


		# kernels=[3,3,3] # ResNet
		# filters=[16,32,32]
		# strides=[1,1,1]

		# kernels=[3,3,3] # ResNet_3
		# filters=[16,32,32]
		# strides=[3,3,1]


		# -- minigrid

		# ---------CNN
		if not self.ResNet:
			# THIS ONE WORKED BEST
			if mapT == 'minigrid':
				kernels=[2,2,2] #V7_3_3 intended for minigrid
				strides=[2,2,2]
				filters=[16, 32, 32]
			else:
				kernels=[3,3,1] # V6_2_2, no relu
				filters=[16,32,1]
				strides=[3,3,1]

			self.n_layers = len(filters)

			for i in range(self.n_layers):
				self.conv[str(i)] = kl.Conv2D(filters[i], kernels[i], strides=strides[i])

		
		#-----------------ResNet
		else:
			kernels=[3,3,3] # ResNet HCAM
			filters=[16,32,32]
			strides=[2,2,2]
			print('Encoding with ResNet')

			self.n_layers = len(filters)

			for i in range(self.n_layers):
				self.conv[str(i)+'0'] = kl.Conv2D(filters[i], kernels[i], strides=2, padding='valid') # Minigrid only
				# self.conv[str(i)+'0'] = kl.Conv2D(filters[i], kernels[i], strides=strides[i], padding='valid') # Minecraft only
				self.conv[str(i)+'1'] = kl.Conv2D(filters[i], kernels[i], strides=1, activation='relu', padding='same')
				self.conv[str(i)+'2'] = kl.Conv2D(filters[i], kernels[i], strides=1, padding='same')


	def build(self, input_shape):
		super(AuxCNN, self).build(input_shape)

	@tf.function(experimental_relax_shapes=True)
	def call (self,s):
		#-------Shallow convs
		if not self.ResNet:
			for i in range(self.n_layers):
				# if i == 1: s = kl.TimeDistributed(self.maxp)(s) # For V9 variants only
				s = kl.TimeDistributed(self.conv[str(i)])(s)
				# if i < self.n_layers-1: s =  tf.nn.relu(s) # For V6, 7 only tf.nn.relu(s)
				s =  tf.nn.relu(s)
				# print ('s', s.shape)

		# print ('s', s.shape)
		# jajaja+=1
		else:
			# -------ResNet
			for i in range(self.n_layers):
				s = kl.TimeDistributed(self.conv[str(i)+'0'])(s)
				s_skip = s
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.conv[str(i)+'1'])(s)
				s = kl.TimeDistributed(self.conv[str(i)+'2'])(s)
				s = tf.math.add(s,s_skip)
			s = tf.nn.relu(s)
		
		# print ('s', s.shape)
		# jajaja+=1
		return s

class ImgDecoder(tf.keras.layers.Layer):
	# Reconstructs the image from the 640 output from 
	def __init__(self, ResNet = False, channels = 3):
		super(ImgDecoder, self).__init__()
		self.dconv = {}
		self.conv = {}
		self.ups = {}
		self.ResNet = ResNet
		output_padding	= None
		pad_last = 'same'


		if ResNet:
			print('Decoding with ResNet!')
			embedding_size = 512
			dil_rate=(1,1)
			kernels=[3,3,3] # ResNet
			filters=[32,32,16,16]
			strides=[2,2,2]

			dil_rate=(1,1)
			output_padding	= (0,0)
			for i in range(3):
				output_padding	= (0,0) if i != 2 else (1,1)	
				self.dconv[str(i)+'1'] = kl.Conv2DTranspose(filters[i], kernels[i], strides=1, activation='relu', padding='same')
				self.dconv[str(i)+'2'] = kl.Conv2DTranspose(filters[i], kernels[i], strides=1, padding='same')
				self.dconv[str(i)+'0'] = kl.Conv2DTranspose(filters[i+1], kernels[i], strides=2, output_padding = output_padding, 
											dilation_rate=dil_rate, padding='valid') # Minigrid only
		else:
			embedding_size = 800
			kernels=[2,2,2] #V7_3_2, 7_3_3 is the same but with no realu in the last layer
			strides=[2,2,2]
			filters= [32,32,16]

			
			dil_rate=(1,1)
			for i in range(3): #V5
				output_padding	= (0,0) #V7, v9
				self.dconv[str(i)] = kl.Conv2DTranspose(filters[i], kernels[i], strides=strides[i], output_padding = output_padding, 
											dilation_rate=dil_rate, activation='relu') #V5,V6,v7

		self.embedding_layer = tf.keras.layers.Dense(units=embedding_size,
													activation='relu',
													name='Decoder_embedding')

		self.final_conv = kl.Conv2D(channels, 3, padding='same', strides=1, activation='sigmoid')

	def build(self, input_shape):
		super(ImgDecoder, self).build(input_shape)

	@tf.function(experimental_relax_shapes=True)	
	def call(self, embed):
		s = kl.TimeDistributed(self.embedding_layer)(embed)
		shape = tf.shape(embed)
		
		if self.ResNet:
			s = tf.reshape(s, [shape[0], shape[1], 4, 4,32])
			for i in range(3):
				s_skip = s
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.dconv[str(i)+'1'])(s)
				s = kl.TimeDistributed(self.dconv[str(i)+'2'])(s)
				s = tf.math.add(s,s_skip)
				s = kl.TimeDistributed(self.dconv[str(i)+'0'])(s)
			s = tf.nn.relu(s)
		else:
			s = tf.reshape(s, [shape[0], shape[1], 5, 5,32])
			for i in range(3):
				s = kl.TimeDistributed(self.dconv[str(i)])(s)
		# ALL 
		s = kl.TimeDistributed(self.final_conv)(s)
		return s


class TextDecoder(tf.keras.Model):
	def __init__(self, rnn_type='lstm', num_layers=1, hidden_dim=32, textLength=7,
					dic_size = 32):
		super(TextDecoder, self).__init__()
		self.num_layers = num_layers
		# self.word_embedding = kl.Embedding(dic_size, word_embedding_size)
		self.textLength = textLength
		self.rnnL={}
		self.embedding_layer = tf.keras.layers.Dense(units=hidden_dim*2,
													activation='relu',
													name='Decoder_embedding')
		for i in range(self.num_layers):
			if rnn_type == 'gru' or rnn_type == 'GRU':
				self.rnnL[str(i)] = kl.Bidirectional(kl.GRU(hidden_dim, return_sequences=True), merge_mode='sum')
			else:
				self.rnnL[str(i)] = kl.Bidirectional(kl.LSTM(hidden_dim, return_sequences=True, activation='tanh'), merge_mode='concat')
		self.output_layer = kl.Dense(dic_size, activation='softmax')

	def build(self, input_shape):
		super(TextDecoder, self).build(input_shape)

	@tf.function(experimental_relax_shapes=True)
	def call(self, embed):
		text = kl.TimeDistributed(self.embedding_layer)(embed)
		# [B,T,64]
		text = tf.broadcast_to(tf.expand_dims(text,2),
					[tf.shape(text)[0], text.shape[1], self.textLength,  text.shape[-1]])
		# [B,T, textLength, 64]

		# text = tf.expand_dims(text,2)
		# print('broad text', text.shape)
		for i in range(self.num_layers):
			text= kl.TimeDistributed(self.rnnL[str(i)])(text)

		decoded_text = kl.TimeDistributed(self.output_layer)(text)
		return decoded_text


class TextEmbed(tf.keras.Model):
	def __init__(self, rnn_type='lstm', num_layers=1, hidden_dim=32, output_dim = 32, dic_size= 32,
		word_embedding_size=32, textLength=7):
		super(TextEmbed, self).__init__()
		# output_dim =64
		self.textLength = textLength
		self.num_layers = num_layers

		# self.word_embedding = kl.Embedding(dic_size, word_embedding_size, input_length=textLength)
		self.rnnL={}
		for i in range(self.num_layers):
			seq=True if i < self.num_layers-1 else False
			if rnn_type == 'gru' or rnn_type == 'GRU':
				self.rnnL[str(i)] =  kl.Bidirectional(kl.GRU(hidden_dim, return_sequences=seq), merge_mode='concat')
			else:
				self.rnnL[str(i)] = kl.Bidirectional(kl.LSTM(hidden_dim, return_sequences=seq, activation='tanh' ), merge_mode='concat')


	def build(self, input_shape):
		super(TextEmbed, self).build(input_shape)

	@tf.function(experimental_relax_shapes=True)
	def call (self, text):
		assert text.shape[-2] == self.textLength
		# em_text = kl.TimeDistributed(self.word_embedding)(s)

		for i in range(self.num_layers):
			encoded_text = kl.TimeDistributed(self.rnnL[str(i)])(text)
		# encoded_text = kl.TimeDistributed(self.output_layer)(encoded_text)

		# s = tf.reshape(em_text, [em_text.shape[0], em_text.shape[1], -1])
		# print('text reshaped', inp.shape)
		# s = kl.TimeDistributed(self.d1)(s)
		# s = kl.TimeDistributed(self.d2)(s)

		# jsjsjsjs+=1
		return encoded_text

class TextEncoder(tf.keras.Model):
	def __init__(self, rnn_type='gru', num_layers=1, hidden_dim=16, output_dim = 32, dic_size = 64,
					word_embedding_size=20, textLength = 7):
		super(TextEncoder, self).__init__()
		self.num_layers = num_layers
		# self.word_embedding = kl.Embedding(dic_size, word_embedding_size, input_length=textLength)

		self.rnnL={}
		for i in range(self.num_layers):
			seq=True if i < self.num_layers-1 else False
			if rnn_type == 'gru' or rnn_type == 'GRU':
				self.rnnL[str(i)] =  kl.Bidirectional(kl.GRU(hidden_dim, return_sequences=seq), merge_mode='concat')
			else:
				self.rnnL[str(i)] = kl.Bidirectional(kl.LSTM(hidden_dim, return_sequences=seq, activation='tanh' ), merge_mode='concat')
		# self.dropout_enc = kl.Dropout(0.1)
		# self.output_layer = kl.Dense(output_dim)

	def build(self, input_shape):
		super(TextEncoder, self).build(input_shape)

	def call(self, text):
		# em_text = kl.TimeDistributed(self.word_embedding)(text)
		# print('emb_text', em_text.shape)
		for i in range(self.num_layers):
			encoded_text = kl.TimeDistributed(self.rnnL[str(i)])(text)
		# text = kl.TimeDistributed(self.dropout_enc)(text)
		# encoded_text = kl.TimeDistributed(self.output_layer)(text)
		return encoded_text

class GTextDecoder(tf.keras.Model):
	def __init__(self, rnn_type='gru', num_layers=1, hidden_dim=16, textLength=7,
					dic_size = 32):
		super(TextDecoder, self).__init__()
		self.num_layers = num_layers
		# self.word_embedding = kl.Embedding(dic_size, word_embedding_size)
		self.textLength = textLength
		self.rnnL={}
		for i in range(self.num_layers):
			if rnn_type == 'gru' or rnn_type == 'GRU':
				self.rnnL[str(i)] = kl.Bidirectional(kl.GRU(hidden_dim, return_sequences=True), merge_mode='concat')
			else:
				self.rnnL[str(i)] = kl.Bidirectional(kl.LSTM(hidden_dim, return_sequences=True, activation='tanh'), merge_mode='concat')
		self.output_layer = kl.Dense(dic_size, activation='softmax')

	def build(self, input_shape):
		super(TextDecoder, self).build(input_shape)

	def call(self, text):
		# print('inp text', text.shape)
		text = tf.broadcast_to(tf.expand_dims(text,2), [tf.shape(text)[0], text.shape[1], self.textLength,  text.shape[-1]])
		# text = tf.expand_dims(text,2)
		# print('broad text', text.shape)
		for i in range(self.num_layers):
			text= kl.TimeDistributed(self.rnnL[str(i)])(text)
		# print('lstm_dec_text', text.shape)
		decoded_text = kl.TimeDistributed(self.output_layer)(text)
		# print('decoded_text', decoded_text.shape)
		return decoded_text


class TAE(tf.keras.Model):
	"""
	Text autoencoder
	"""
	def __init__(self, rnn_type='lstm', num_layers=1, hidden_dim=32, output_dim = 32, dic_size= 64,
		word_embedding_size=32, input_dim=32):
		super(TAE, self).__init__()
		self.rnn_type = rnn_type
		self.num_layers = num_layers
		self.input_dim = input_dim
		self.dic_size = dic_size 
		# self.word_embedding = kl.Embedding(dic_size, word_embedding_size)
		self.Encoder = TextEncoder(rnn_type, num_layers, hidden_dim, output_dim)
		self.Decoder = TextDecoder(rnn_type, num_layers, hidden_dim, dic_size, input_dim)
	def build(self, input_shape):
		super(TAE, self).build(input_shape)

	def call(self, text):
		assert text.shape[-2] == self.input_dim
		# em_text = kl.TimeDistributed(self.word_embedding)(text)
		enc = self.Encoder(text)
		dec = self.Decoder(enc)
		# print('enc', enc.shape)
		# print('dec', dec.shape)
		# print('text', text.shape)
		return enc, dec, text

class PrediNet(tf.keras.layers.Layer):
	def __init__(self, w_initializer, b_initializer, 
					heads_p = 32, key_size=16, relations_p = 16,
					view_size =5, name='PrediNet'):
		super(PrediNet, self).__init__(name=name)
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self._view_size = view_size

		self._locs = None
		# self.central_ouput_size = heads_p * (relations_p+4)

		self._weight_initializer = w_initializer
		self._bias_initializer = b_initializer

		self.flatten = tf.keras.layers.Flatten()
		self.get_keys = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keys')
		self.get_query1 = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1')
		self.get_query2 = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2')
		self.embed_entities = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entities')

	def build(self, input_shape):
		assert len(input_shape) == 4
		conv_out_size = input_shape[2]
		channels = input_shape[3]
		Wsize = self._view_size
		Hsize = conv_out_size//Wsize
		rest = conv_out_size%Wsize

		# assert conv_out_size%Wsize == 0
		
		cols = tf.constant([[[x / float(Wsize)]
						 for x in range(Hsize)]
						for _ in range(Wsize)])

		if rest > 0:
			instr = tf.constant([[1+ x / float(Wsize)]
							 for x in range(rest)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [Wsize*Hsize,1])
		rows = tf.reshape(rows, [Wsize*Hsize,1])

		if rest > 0:
			cols = tf.concat([cols, instr], 0)
			rows = tf.concat([rows, instr], 0)

		self._locs = tf.concat([cols, rows], -1)

		self._channels = channels

		# print('self._locs pre slice', self._locs.shape)
		# print('self._locs ', self._locs)
		# jajaja+=1
		# print('observation', observation.shape)

		# if Wsize*Hsize != conv_out_size:
		# 	self._locs = tf.slice(self._locs,[0,0], [conv_out_size, 2])
		super(PrediNet, self).build(input_shape)

	@tf.function(experimental_relax_shapes=True)
	def call(self, observation):
		batch_size = tf.shape(observation)[0]
		tboostrap = observation.shape[1]
		conv_out_size = observation.shape[2]
		channels = observation.shape[3]
		# tf.print('obs', observation)
		# print('conv_out_size', conv_out_size)

		# t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)

		locs = tf.tile(tf.expand_dims(tf.expand_dims(self._locs, 0),0), [batch_size, tboostrap, 1, 1])
		# tf.print('locs', self._locs)
		# tf.print(s.shape, "Conv output", summarize=40)
		# tf.print(locs.shape, "locs", summarize=40)

		features_locs = tf.concat([observation, locs], 3)
		# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)
		# tf.print('features_locs', features_locs)

		features_flat = kl.TimeDistributed(self.flatten)(features_locs)
		# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
		# Keys
		keys = kl.TimeDistributed(self.get_keys)(features_locs)
		# tf.print('keys', keys)
		# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
		keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
		# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
		# Queries
		query1 = kl.TimeDistributed(self.get_query1)(features_flat)
		# (batch_size, time, heads*key_size)
		query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
		# (batch_size, time, heads, key_size)
		query1 = tf.expand_dims(query1, 3)
		# (batch_size, time, heads, 1, key_size)
		# tf.print('query1', query1)

		query2 = kl.TimeDistributed(self.get_query2)(features_flat)
		# (batch_size, time, heads*key_size)
		query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
		# (batch_size, time, heads, key_size)
		query2 = tf.expand_dims(query2, 3)
		# (batch_size, time, heads, 1, key_size)

		# Attention weights
		keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
		# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

		att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
		att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
		# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

		# Reshape features
		features_tiled = tf.tile(
			tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
		# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

		# Compute a pair of features using attention weights
		feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
		feature2 = tf.squeeze(tf.matmul(att2, features_tiled))
		# (batch_size, time, heads, (channels+2))

		feature1 = tf.reshape(feature1, [batch_size, tboostrap, self._heads, channels+2])
		feature2 = tf.reshape(feature2, [batch_size, tboostrap, self._heads, channels+2])
		
		# Spatial embedding
		embedding1 = kl.TimeDistributed(self.embed_entities)(feature1)
		embedding2 = kl.TimeDistributed(self.embed_entities)(feature2)
		# (batch_size, time, heads, relations)

		# Comparator
		dx = tf.subtract(embedding1, embedding2)

		# Positions
		pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
		pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
		# (batch_size, time, heads, 2)
		# Collect relations and concatenate positions
		relations = tf.concat([dx, pos1, pos2], 3)
		# (batch_size, time, heads, relations+4)
		relations = tf.reshape(relations,
						[batch_size, tboostrap, self._heads * (self._relations + 4)])
		# print('contrast ouput dimension with prediNet, should be the same, do we need a flat?')
		# print('relations', relations.shape)
		# jajajaja+=1
		return relations	


class RelationNet(tf.keras.layers.Layer):
	def __init__(self, w_initializer, b_initializer, 
					central_hidden_size = 256,
					view_size =5, name='RelationNet'):
		super(RelationNet, self).__init__(name=name)

		self._view_size = view_size

		self._locs = None
		# self.central_ouput_size = heads_p * (relations_p+4)

		self._weight_initializer = w_initializer
		self._bias_initializer = b_initializer

		self.central_hidden = tf.keras.layers.Dense(
				units=central_hidden_size, use_bias=False,
				kernel_initializer=self._weight_initializer, activation='relu',
				name='central_hidden')
		self.central_output = tf.keras.layers.Dense(units=640, activation='relu',
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												name='central_out')

	def build(self, input_shape):
		assert len(input_shape) == 4
		conv_out_size = input_shape[2]
		channels = input_shape[3]
		Wsize = self._view_size
		Hsize = conv_out_size//Wsize
		rest = conv_out_size%Wsize

		# assert conv_out_size%Wsize == 0
		
		cols = tf.constant([[[x / float(Wsize)]
						 for x in range(Hsize)]
						for _ in range(Wsize)])

		if rest > 0:
			instr = tf.constant([[1+ x / float(Wsize)]
							 for x in range(rest)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [Wsize*Hsize,1])
		rows = tf.reshape(rows, [Wsize*Hsize,1])

		if rest > 0:
			cols = tf.concat([cols, instr], 0)
			rows = tf.concat([rows, instr], 0)

		self._locs = tf.concat([cols, rows], -1)

		self._conv_out_size = self._locs.shape[0]

		self._channels = channels

		# print('self._locs pre slice', self._locs.shape)
		# print('observation', observation.shape)

		# if Wsize*Hsize != conv_out_size:
		# 	self._locs = tf.slice(self._locs,[0,0], [conv_out_size, 2])
		super(RelationNet, self).build(input_shape)

	def call(self, observation):
		batch_size = tf.shape(observation)[0]
		tboostrap = observation.shape[1]
		conv_out_size = observation.shape[2]
		channels = observation.shape[3]
		# tf.print('obs', observation)
		# print('conv_out_size', conv_out_size)

		# t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)

		locs = tf.tile(tf.expand_dims(tf.expand_dims(self._locs, 0),0), [batch_size, tboostrap, 1, 1])
		# tf.print('locs', self._locs)
		# tf.print(s.shape, "Conv output", summarize=40)
		# tf.print(locs.shape, "locs", summarize=40)

		features_locs = tf.concat([observation, locs], 3)
		# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)
		# tf.print('features_locs', features_locs)

		features_flat = tf.reshape(features_locs,[batch_size, tboostrap,
					self._conv_out_size, self._channels + 2])

		# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)

		num_features = self._conv_out_size

		# Compute all possible pairs of features
		indexes = tf.range(num_features)
		receiver_indexes = tf.tile(indexes, [num_features])
		sender_indexes = tf.reshape(
			tf.transpose(
				tf.reshape(receiver_indexes, [num_features, num_features])),
												[-1])
		receiver_objects = tf.gather(features_flat, receiver_indexes, axis=2)
		sender_objects = tf.gather(features_flat, sender_indexes, axis=2)
		object_pairs = tf.concat([sender_objects, receiver_objects], -1)
		# (batch_size, ((conv_out_size+1)*conv_out_size)^2, (channels+2)*2)

		# tf.print(object_pairs.shape, "Should be batch, 1600, 6", summarize=40)
		# Compute "relations"
		central_activations = kl.TimeDistributed(self.central_hidden)(object_pairs)
		# (batch_size, ((conv_out_size+1)*conv_out_size)^2, central_hidden_size)

		central_out_activations = kl.TimeDistributed(self.central_output)(central_activations)
		# (batch_size, time, ((conv_out_size+1)*conv_out_size)^2, central_output_size)

		# Aggregate relations
		central_out_mean = tf.reduce_mean(central_out_activations, 2)
		# (batch_size, time, central_output_size)

		return central_out_mean	



class MHA(tf.keras.layers.Layer):
	def __init__(self, w_initializer, b_initializer, value_size = 20, 
					heads_p = 32, key_size=16, relations_p = 16,
					view_size =5, name='MHA'):
		super(MHA, self).__init__(name=name)
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self._view_size = view_size
		self._value_size = value_size
		self._key_size = key_size

		self._locs = None
		# self.central_ouput_size = heads_p * (relations_p+4)

		self._weight_initializer = w_initializer
		self._bias_initializer = b_initializer

		self.flatten = tf.keras.layers.Flatten()
		self.get_keys = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keys')
		self.get_queries = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_queries')
		self.get_values = tf.keras.layers.Dense(
				units=self._heads * self._value_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_values')
		self.central_output = CombineValues(
				heads=self._heads, w_init=self._weight_initializer)

	def build(self, input_shape):
		assert len(input_shape) == 4
		conv_out_size = input_shape[2]
		channels = input_shape[3]
		Wsize = self._view_size
		Hsize = conv_out_size//Wsize
		rest = conv_out_size%Wsize

		# assert conv_out_size%Wsize == 0
		
		cols = tf.constant([[[x / float(Wsize)]
						 for x in range(Hsize)]
						for _ in range(Wsize)])

		if rest > 0:
			instr = tf.constant([[1+ x / float(Wsize)]
							 for x in range(rest)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [Wsize*Hsize,1])
		rows = tf.reshape(rows, [Wsize*Hsize,1])

		if rest > 0:
			cols = tf.concat([cols, instr], 0)
			rows = tf.concat([rows, instr], 0)

		self._locs = tf.concat([cols, rows], -1)

		self._conv_out_size = self._locs.shape[0]
		self._channels = channels

		# print('self._locs pre slice', self._locs.shape)
		# print('observation', observation.shape)

		# if Wsize*Hsize != conv_out_size:
		# 	self._locs = tf.slice(self._locs,[0,0], [conv_out_size, 2])
		super(MHA, self).build(input_shape)

	def call(self, observation):
		batch_size = tf.shape(observation)[0]
		tboostrap = observation.shape[1]
		conv_out_size = observation.shape[2]
		channels = observation.shape[3]
		# tf.print('obs', observation)
		# print('conv_out_size', conv_out_size)

		# t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)

		locs = tf.tile(tf.expand_dims(tf.expand_dims(self._locs, 0),0), [batch_size, tboostrap, 1, 1])
		# tf.print('locs', self._locs)
		# tf.print(s.shape, "Conv output", summarize=40)
		# tf.print(locs.shape, "locs", summarize=40)

		features_locs = tf.concat([observation, locs], 3)
		# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)
		# tf.print('features_locs', features_locs)

		# features_flat = kl.TimeDistributed(self.flatten)(features_locs) #ssems that MHA does not need this
		# Keys
		keys = kl.TimeDistributed(self.get_keys)(features_locs)

		keys = tf.reshape(keys, [
				batch_size, tboostrap, self._conv_out_size, 
				self._heads, self._key_size])
		# tf.print('keys', keys)
		# (batch_size, time, (conv_out_size+1)*conv_out_size, heads, key_size)

		# Queries
		queries = kl.TimeDistributed(self.get_queries)(features_locs)
		# (batch_size, time, heads*key_size)
		queries = tf.reshape(queries, [
				batch_size, tboostrap, self._conv_out_size, 
				self._heads, self._key_size])

		values = kl.TimeDistributed(self.get_values)(features_locs)
		# (batch_size, time, heads*key_size)
		values = tf.reshape(values, [
				batch_size, tboostrap, self._conv_out_size, 
				self._heads, self._value_size])

		# Attention weights
		queries_t = tf.transpose(queries, perm=[0, 1, 3, 2, 4])
		# (batch_size, times, heads, (conv_out_size+1)*conv_out_size, key_size)
		keys_t = tf.transpose(keys, perm=[0, 1, 3, 4, 2])
		# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

		att = tf.nn.softmax(tf.matmul(queries_t, keys_t))
		# (batch_size, heads, (conv_out_size+1)*conv_out_size, (conv_out_size+1)*conv_out_size)

		# Reshape features
		values_t = tf.transpose(values, perm=[0, 1, 3, 2, 4])
		# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)
		values_out = tf.matmul(att, values_t)
		# (batch_size, heads, (conv_out_size+1)*conv_out_size, value_size)
		values_out = tf.reshape(values_out,[batch_size*tboostrap,values_out.shape[2],
										 values_out.shape[3], values_out.shape[4]])
		central_out = self.central_output(values_out)
		
		central_out = tf.reshape(central_out,[batch_size, tboostrap,central_out.shape[1], 
														central_out.shape[2], central_out.shape[3]])
		central_out = kl.TimeDistributed(self.flatten)(central_out)
		# (batch_size, time, heads*value_size)


		return central_out	




class GroupLSTMCell(tf.keras.layers.Layer):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, units, nRIM):
		super(GroupLSTMCell, self).__init__()
		self.units = units
		self.nRIM = nRIM
		self.x2h = GroupLinearLayer(4 * units, nRIM, initializer = 'uniform')
		self.h2h = GroupLinearLayer(4 * units, nRIM, initializer = 'uniform')
	@property
	def state_size(self):
		return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))

	def build(self, input_shape):
		super(GroupLSTMCell, self).build(input_shape)
	def call(self, inputs, states):
		# inputs in shape [batch, nRIM, din]
		h,c = states

		preact_i = self.x2h(inputs)  #Group linear layer
		preact_h = self.h2h(h) 

		preact = preact_i + preact_h

		gates = tf.sigmoid(preact[:,:,:3*self.units])
		g_t = tf.tanh(preact[:,:,3*self.units:])
		input_gate = gates[:,:,:self.units]
		forget_gate = gates[:,:,self.units:(self.units*2)]
		output_gate = gates[:,:,-self.units:] #pytorch uses -self.hidden_size:

		c_t = tf.multiply(c, forget_gate) + tf.multiply(input_gate, g_t)
		h_t = tf.multiply(output_gate, tf.tanh(c_t))

		return h_t, (h_t, c_t)


class RIMCell(tf.keras.layers.Layer):
	"""Implementation of RimCell, adapted from David F. Li aka fuyuan-li
	https://github.com/fuyuan-li"""
	def __init__(self, units, nRIM, k, rnn_type, name='RIMCell',
				num_input_heads = 1, input_key_size = 64, input_query_size= 64, input_keep_prob = 0.1,
				num_comm_heads = 4, comm_key_size = 32, comm_value_size = 100, comm_query_size = 32, comm_keep_prob = 0.1):
		super(RIMCell, self).__init__(name=name)
		self.units = units
		self.nRIM = nRIM
		self.k = k

		if comm_value_size != units:
			#print('INFO: Changing communication value size to match hidden_size')
			comm_value_size = units
		
		self.num_input_heads = num_input_heads
		self.input_key_size = input_key_size
		self.input_value_size = units*4
		self.input_query_size = input_query_size
		self.input_keep_prob = input_keep_prob
		
		self.num_comm_heads = num_comm_heads
		self.comm_key_size = comm_key_size
		self.comm_value_size = comm_value_size
		self.comm_query_size = comm_query_size
		self.comm_keep_prob = comm_keep_prob

		self.rnn_type = rnn_type

		assert input_key_size == input_query_size, 'input_key_size == input_query_size required'
		assert comm_key_size == comm_query_size, 'comm_key_size == comm_query_size required'

		# -----------------

		self.key   = tf.keras.layers.Dense(units=self.num_input_heads*self.input_key_size,   activation=None, use_bias=True)
		self.value = tf.keras.layers.Dense(units=self.num_input_heads*self.input_value_size, activation=None, use_bias=True)
		self.query = GroupLinearLayer(units=self.num_input_heads*self.input_query_size, nRIM=self.nRIM)
		self.input_attention_dropout = tf.keras.layers.Dropout(rate = self.input_keep_prob)
		if self.rnn_type == 'gru': self.rnn_cell = GroupGRUCell(units=self.units, nRIM=self.nRIM)
		else: self.rnn_cell = GroupLSTMCell(units=self.units, nRIM=self.nRIM)
		self.key_   = GroupLinearLayer(units=self.num_comm_heads*self.comm_key_size,   nRIM=self.nRIM)
		self.value_ = GroupLinearLayer(units=self.num_comm_heads*self.comm_value_size, nRIM=self.nRIM)
		self.query_ = GroupLinearLayer(units=self.num_comm_heads*self.comm_query_size, nRIM=self.nRIM)
		self.comm_attention_output  = GroupLinearLayer(units=self.comm_value_size, nRIM=self.nRIM)#!! tf said units here
		self.comm_attention_dropout = tf.keras.layers.Dropout(rate = self.comm_keep_prob)

	@property
	def state_size(self):
		return (tf.TensorShape([self.nRIM, self.units]), tf.TensorShape([self.nRIM, self.units]))

	def build(self, input_shape):
		super(RIMCell, self).build(input_shape)		

	def call(self, inputs, states=None):
		# inputs of shape (batch_size, input_feature_size)
		# if states is not None: 
		# 	tf.print('RIMCell input', inputs.shape)
		# 	tf.print('RIMCell h', states[0].shape)
		if self.rnn_type != 'gru': 
			hs, cs = states
		else:
			hs,_ = states
		# print('RIMC hs', hs.shape)
		# print('RIMC cs', cs.shape)
		# tf.print('RIMCell hs', hs.shape)
		# tf.print('RIMCell cs', cs.shape)
		rnn_inputs, mask = self.input_attention_mask(inputs, hs)#, training=training)
		h_old = hs*1.0

		if self.rnn_type != 'gru': 
			c_old = cs*1.0
			_, (h_rnnout, c_rnnout) = self.rnn_cell(rnn_inputs, (hs, cs))
		else: 
			_, h_rnnout = self.rnn_cell(rnn_inputs, hs)
			# print(h_rnnout)
			# ksksk+=1
		
		h_new = tf.stop_gradient(h_rnnout*(1-mask)) + h_rnnout*mask
		
		h_comm = self.comm_attention(h_new, mask)#, training=training)
		
		h_update = h_comm*mask + h_old*(1-mask)
		if self.rnn_type != 'gru': 

			c_update = c_rnnout*mask + c_old*(1-mask)

			return tf.reshape(h_update, [tf.shape(inputs)[0], self.units*self.nRIM]), (h_update, c_update)
		else:
			return tf.reshape(h_update, [tf.shape(inputs)[0], self.units*self.nRIM]), (h_update, h_update)
			# tf.stack([h_update, c_update], axis=0)

	def input_attention_mask(self, x, hs):#, training=False):
		# x of shape (batch_size, input_feature_size)
		# hs of shape (batch_size, nRIM, hidden_size = units)
		xx = tf.stack([x, tf.zeros_like(x)], axis=1)

		key_layer   = self.key(xx)
		value_layer = self.value(xx)
		query_layer = self.query(hs)

		key_layer1   = tf.stack(tf.split(key_layer,   num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
		value_layer1 = tf.stack(tf.split(value_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
		query_layer1 = tf.stack(tf.split(query_layer, num_or_size_splits=self.num_input_heads, axis=-1), axis=1)
		value_layer2 = tf.reduce_mean(value_layer1, axis=1)

		attention_scores1 = tf.matmul(query_layer1, key_layer1, transpose_b=True)/np.sqrt(self.input_key_size)
		attention_scores2 = tf.reduce_mean(attention_scores1, axis=1)

		signal_attention = attention_scores2[:,:,0]
		topk = tf.math.top_k(signal_attention, self.k)
		indices = topk.indices
		mesh = tf.meshgrid( tf.range(indices.shape[1]), tf.range(tf.shape(indices)[0]) )[1]
		full_indices = tf.reshape(tf.stack([mesh, indices], axis=-1), [-1,2])

		sparse_tensor = tf.sparse.SparseTensor(indices=tf.cast(full_indices, tf.int64),
										  values=tf.ones(tf.shape(full_indices)[0]),
										  dense_shape=[tf.shape(x)[0],self.nRIM])
		sparse_tensor = tf.sparse.reorder(sparse_tensor)
		mask_ = tf.sparse.to_dense(sparse_tensor)
		mask  = tf.expand_dims(mask_, axis=-1)

		attention_prob = self.input_attention_dropout(tf.nn.softmax(attention_scores2, axis=-1))#, training=training)
		inputs = tf.matmul(attention_prob, value_layer2)
		inputs1 = inputs*mask
		return inputs1, mask

	def comm_attention(self, h_new, mask):
		# h_new of shape (batch_size, nRIM, hidden_size = units)
		# mask of shape (batch_size, nRIM, 1)
		comm_key_layer   = self.key_(h_new)
		comm_value_layer = self.value_(h_new)
		comm_query_layer = self.query_(h_new)

		comm_key_layer1 = tf.stack(tf.split(comm_key_layer,   num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
		comm_value_layer1 = tf.stack(tf.split(comm_value_layer, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)
		comm_query_layer1 = tf.stack(tf.split(comm_query_layer, num_or_size_splits=self.num_comm_heads, axis=-1), axis=1)

		comm_attention_scores = tf.matmul(comm_query_layer1, comm_key_layer1, transpose_b=True)/np.sqrt(self.comm_key_size)
		comm_attention_probs  = tf.nn.softmax(comm_attention_scores, axis=-1)

		comm_mask_ = tf.tile(tf.expand_dims(mask, axis=1), [1,self.num_comm_heads, 1, 1])

		comm_attention_probs1 = self.comm_attention_dropout(comm_attention_probs*comm_mask_)#, training=training)
		context_layer = tf.matmul(comm_attention_probs1, comm_value_layer1)
		context_layer1= tf.reshape(tf.transpose(context_layer, [0,2,1,3]), [tf.shape(h_new)[0], 
										self.nRIM, self.num_comm_heads*self.comm_value_size])
		comm_out = self.comm_attention_output(context_layer1) + h_new
		return comm_out


class BRIM(tf.keras.layers.Layer):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, units, nRIM, k, rnn_type='lstm', n_layers=2, bidirectional = True,
					return_sequences=False):
		super(BRIM, self).__init__()
		self.units = units
		self.nRIM = nRIM
		self.k = k
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		self.n_layers = n_layers
		self.rimcell = {}
		for i in range(n_layers):
			self.rimcell[str(i)] = RIMCell(units=units, nRIM=nRIM, k=k, rnn_type = self.rnn_type)
		self.return_states = return_sequences

	def build(self, input_shape):
		super(BRIM, self).build(input_shape)

	def call(self, inputs, states = None):
		# inputs in shape [batch, t_seq, din]
		inputs = tf.transpose(inputs, perm=[1, 0, 2])
		# states = None
		# print('BLayer_Input_shape', inputs.shape)
		# tf.print('BLayer_Input_shape', inputs.shape)
		output = []
		nstates = []
		for kt in range(inputs.shape[0]): #cannot do for t in inputs because of build call
			t = inputs[kt]
			if self.bidirectional:
				if states == None:
					states = tf.zeros([self.n_layers,2, t.shape[0],
								self.nRIM, self.units], dtype=tf.dtypes.float32)
				else:
					# tf.print('Size into BRIM', states.shape)
					states = tf.reshape(states, [t.shape[0], self.n_layers,2,
								self.nRIM, self.units])
					states = tf.transpose(states, perm=[1, 2, 0,3,4])
			# for the upper layer only the lower is given as input (while retaining the hiddden state, for the rest of them the input is a concat)
			for i in range(self.n_layers):
				if i < self.n_layers-1 and self.bidirectional: 
					Ht = tf.reshape(states[i+1, 0], [t.shape[0], self.units*self.nRIM])

					t = tf.concat([t, Ht], axis=1)
				# tf.print('h shape', states[0][i].shape)
				# h = tf.reshape(states[:,i, 0], [t.shape[0], self.units*self.nRIM])
				# c = tf.reshape(states[:,i, 1], [t.shape[0], self.units*self.nRIM])
				t, newStates = self.rimcell[str(i)](t, states= (states[i, 0], states[i, 1]))
				nstates.append(newStates)
			if self.return_states and kt>0:
				output = tf.concat([output, tf.expand_dims(t, 1)], axis = 1)
			else:
				if self.return_states: output = tf.expand_dims(t, 1)
				else: output = t
				
		# tf.print('BLayer_Out_shape', output.shape)
		# tf.print('N of h', len(states[0]))
		# tf.print('h shape', states[0][0].shape)
		# output in shape [batch, dout]
		nstates= tf.transpose(tf.convert_to_tensor(nstates, dtype=tf.float32), perm=[2, 0, 1,3,4])
		# print('out shape',nstates.shape)
		# tf.print('out shape',nstates.shape)
		return output, nstates



class MMBRIM_V1(tf.keras.layers.Layer):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, units, nRIM, k, rnn_type='lstm', n_layers=2, bidirectional = True,
					return_sequences=False, LtaskSize = 16):
		super(MMBRIM_V1, self).__init__(name='MMBRIM_V1')
		self.units = units
		self.nRIM = nRIM
		self.k = k
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		assert bidirectional == True
		self.n_layers = n_layers
		self.rimcell = {}
		for i in range(n_layers):
			self.rimcell[str(i)] = RIMCell(units=units, nRIM=nRIM, k=k, rnn_type = self.rnn_type)
		self.return_states = return_sequences

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)
		# self.Ltask = tf.keras.layers.Dense(units=LtaskSize, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 										activation='relu', name='Ltask')

	def build(self, input_shape):
		super(MMBRIM_V1, self).build(input_shape)

	def call(self, Tinputs, Ninputs,  states = None):
		# inputs in shape [batch, t_seq, din]
		# states= None
		Tinputs = tf.transpose(Tinputs, perm=[1, 0, 2])
		Ninputs = tf.transpose(Ninputs, perm=[1, 0, 2])
		# tf.print('BLayer_Input_shape', inputs.shape)
		output = []
		nstates = []
		aux_States = []
		for kt in range(Tinputs.shape[0]): #cannot do for t in inputs because of build call
			nstates = []
			t = Tinputs[kt]
			batch_size = tf.shape(t)[0]
			if self.bidirectional:
				if states == None:
					states = tf.zeros([self.n_layers,2, batch_size,
								self.nRIM, self.units], dtype=tf.dtypes.float32)
				else:
					states = tf.reshape(states, [batch_size, self.n_layers,2,
								self.nRIM, self.units])
					states = tf.transpose(states, perm=[1, 2, 0,3,4])
			# for the upper layer only the lower is given as input (while retaining the hiddden state, for the rest of them the input is a concat) 
			for i in range(self.n_layers):
				if i < self.n_layers-1 and self.bidirectional: 
					Ht = tf.reshape(states[i+1, 0], [batch_size, self.units*self.nRIM])
					t = tf.concat([t, Ht], axis=1)
				else:
					# t = self.Ltask(t)
					t = tf.concat([Ninputs[kt],t], axis=1)
				# tf.print('i', i)
				t, newStates = self.rimcell[str(i)](t, states = (states[i, 0], states[i, 1]))
				# if kt < (Tinputs.shape[0]-1):
				nstates.append(newStates)	

			states = tf.transpose(tf.convert_to_tensor(nstates, dtype=tf.float32), perm=[2, 0, 1,3,4])
			# if kt < (Tinputs.shape[0]-1):
			if self.return_states and kt>0:
				output = tf.concat([output, tf.expand_dims(t, 1)], axis = 1)
			else:
				if self.return_states: output = tf.expand_dims(t, 1)
				else: output = t
		
		# nstates= tf.transpose(tf.convert_to_tensor(nstates, dtype=tf.float32), perm=[2, 0, 1,3,4])
		# tf.print('BLayer_Out_shape', output.shape)
		# print('BLayer_Out_shape', output.shape)
		# tf.print('BLayer_H_shape', nstates.shape)
		# print('BLayer_H_shape', nstates.shape)
		# tf.print('N of h', len(states[0]))
		# tf.print('h shape', states[0][0].shape)
		# output in shape [batch, dout]
		return output, states

class MMBRIM_V2(tf.keras.layers.Layer):
	"""
	GroupLSTMCell can compute the operation of N LSTM Cells at once.
	"""
	def __init__(self, units, nRIM, k, rnn_type='lstm', n_layers=3, bidirectional = True,
					return_sequences=False, LtaskSize = 16):
		super(MMBRIM_V2, self).__init__(name='MMBRIM_V2')
		self.units = units
		self.nRIM = nRIM
		self.k = k
		self.rnn_type = rnn_type
		self.bidirectional = bidirectional
		self.n_layers= n_layers
		assert bidirectional == True
		self.rimcellN = RIMCell(units=units, nRIM=nRIM, k=k, rnn_type = self.rnn_type, name='RIMCellN')
		# Try this one with a higher dropout, lower k?
		self.rimcellT = RIMCell(units=units, nRIM=nRIM, k=k, rnn_type = self.rnn_type, name='RIMCellT')
		self.rimcellH = RIMCell(units=units, nRIM=nRIM, k=k, rnn_type = self.rnn_type, name='RIMCellH')
		self.return_states = return_sequences

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)
		self.Ltask = tf.keras.layers.Dense(units=LtaskSize, 
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												activation='relu', name='Ltask')

	def build(self, input_shape):
		super(MMBRIM_V2, self).build(input_shape)

	def call(self, Tinputs, Ninputs,  states = None):
		# inputs in shape [batch, t_seq, din]
		Tinputs = tf.transpose(Tinputs, perm=[1, 0, 2])
		Ninputs = tf.transpose(Ninputs, perm=[1, 0, 2])
		# tf.print('BLayer_Input_shape', inputs.shape)
		# states = None
		output = []
		nstates = []
		for kt in range(Tinputs.shape[0]): #cannot do for t in inputs because of build call
			t = Tinputs[kt]
			n = Ninputs[kt]
			if self.bidirectional:
				if states == None:
					states = tf.zeros([self.n_layers,2, t.shape[0],
								self.nRIM, self.units], dtype=tf.dtypes.float32)
				else:
					# tf.print('Size into BRIM', states.shape)
					states = tf.reshape(states, [t.shape[0], self.n_layers,2,
								self.nRIM, self.units])
					states = tf.transpose(states, perm=[1, 2, 0,3,4])
			# for the upper layer only the lower is given as input (while retaining the hiddden state, for the rest of them the input is a concat) 
			Ht = tf.reshape(states[2, 0], [t.shape[0], self.units*self.nRIM])
			# Translator
			t = tf.concat([t, Ht], axis=1)
			t, newStates = self.rimcellT(t, states= (states[0, 0], states[0, 1]))
			nstates.append(newStates)
			t = self.Ltask(t)
			# Env
			n = tf.concat([n, Ht], axis=1)
			n, newStates = self.rimcellN(n,  states= (states[1, 0], states[1, 1]))
			nstates.append(newStates)

			# Higher
			nt = tf.concat([n,t], axis=1)
			nt, newStates = self.rimcellH(nt, states= (states[2, 0], states[2, 1]))
			nstates.append(newStates)

			if self.return_states and kt>0:
				output = tf.concat([output, tf.expand_dims(nt, 1)], axis = 1)
			else:
				if self.return_states: output = tf.expand_dims(nt, 1)
				else: output = nt
		
		nstates= tf.transpose(tf.convert_to_tensor(nstates, dtype=tf.float32), perm=[2, 0, 1,3,4])
		# tf.print('BLayer_H_shape', nstates.shape)
		# print('BLayer_H_shape', nstates.shape)
		# tf.print('BLayer_Out_shape', output.shape)
		# tf.print('N of h', len(states[0]))
		# tf.print('h shape', states[0][0].shape)
		# output in shape [batch, dout]
		return output, nstates



class AC_FCRIM(Model):
	"""
	Actor-Critic  + RIM 
	"""
	def __init__(self, n_outputs, trainable, board = 0, n_hlayers = 1,
		input_size = 5, extended = True, n_lstm = 128, visual=False,
		n_neurons = 640, n_BrimsN=20,
		name='AC_FCRIM'):

		super(AC_FCBRIM, self).__init__(name=name)
		self.trainable = trainable
		Bias_init= tf.keras.initializers.Constant(value=0.1)
		rand_trun = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed = 12345)
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self.n_hlayers = n_hlayers
		self._visual = visual
		# Encoder, works well only if the input is = resolution

		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		# Hidden layers
		self.Dense = {}
		for i in range(0, n_hlayers):
			self.Dense[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
												bias_initializer=Bias_init,
												kernel_initializer=rand_trun,
												activation='relu')
		# self.lstm = tf.keras.layers.LSTM(units=n_lstm, time_major=True)
		self.rimcell = RIMCell(units=n_BrimsN, nRIM=6, k=4, rnn_type='gru')
		self.recnet = tf.keras.layers.RNN(cell=self.rimcell)

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											activation='softmax',
											kernel_initializer=rand_trun,
											bias_initializer=Bias_init)
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=Bias_init,
												kernel_initializer=rand_trun,
												 activation=None)
	def build(self, input_shape):
		super(AC_FCRIM, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None):
		s = tf.dtypes.cast(s, tf.float32)
		s = tf.reshape(s, [s.shape[0], s.shape[2], s.shape[3]])
		# print('input',s)
		# jajaj+=1


		if self._visual:
			if len(s.shape)==2:
				s = tf.expand_dims(s, 0)
				s = tf.expand_dims(s, -1)
			elif len(s.shape)==3:
				s = tf.expand_dims(s, -1)
			s = self.convA(s)
			s = tf.nn.relu(s)
			s = self.convB(s)
			s = tf.nn.relu(s)
			s = self.convC(s)
			s = self.flattenC(s)

		batch_size = s.shape[0]
		s = tf.squeeze(s)	
		if len(s.shape)==1:
			s = tf.expand_dims(s, 0)
		if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
		prev_rew = tf.reshape(prev_rew, [batch_size, 1])
		prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		with tf.device("/device:GPU:0"):
			for i in range(0, self.n_hlayers):
				s = self.Dense[str(i)](s)
			# s = tf.concat([s, prev_act, prev_rew], -1)
			s = tf.concat([s, prev_act], -1)
			s = tf.expand_dims(s, 0)
			# print("rec input shape",s.shape)
			tf.print('recnet_in:', s, summarize=40)
			s = self.recnet(s)
			# s = self.lstm(s)
			# tf.print('recnet_out:', s, summarize=40)
			values = self.value_l(s)
			logits = self.act_l(s)

		return tf.squeeze(logits), tf.squeeze(values,-1)

class AC_FCBRIM_test(Model):
	"""
	Actor-Critic  Bidirectional RIM 
	"""
	def __init__(self, n_outputs, trainable, board = 0, n_hlayers = 1,
		input_size = 5, extended = True, n_lstm = 128, visual=False,
		n_neurons = 128, n_BrimsN=20,
		name='AC_FCBRIM_test'):

		super(AC_FCBRIM_test, self).__init__(name=name)
		self.trainable = trainable
		Bias_init= tf.keras.initializers.Constant(value=0.1)
		rand_trun = tf.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed = 12345)
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self.n_hlayers = n_hlayers
		self._visual = visual
		# Encoder, works well only if the input is = resolution

		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		# Hidden layers
		self.Dense = {}
		for i in range(0, n_hlayers):
			self.Dense[str(i)] = tf.keras.layers.Dense(units=n_BrimsN, 
												bias_initializer=Bias_init,
												kernel_initializer=rand_trun,
												activation='relu')

		# self.rimcell = RIMCell(units=n_BrimsN, nRIM=6, k=4, rnn_type='lstm')
		self.lstm = tf.keras.layers.LSTM(units=n_lstm, time_major=True)
		# self.recnet = tf.keras.layers.RNN(cell=self.rimcell)

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											activation='softmax',
											kernel_initializer=rand_trun,
											bias_initializer=Bias_init)
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=Bias_init,
												kernel_initializer=rand_trun,
												 activation=None)
	def build(self, input_shape):
		super(AC_FCBRIM_test, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None):
		s = tf.dtypes.cast(s, tf.float32)
		s = tf.reshape(s, [s.shape[0], s.shape[2], s.shape[3]])
		# print('input',s)
		# jajaj+=1


		if self._visual:
			if len(s.shape)==2:
				s = tf.expand_dims(s, 0)
				s = tf.expand_dims(s, -1)
			elif len(s.shape)==3:
				s = tf.expand_dims(s, -1)
			s = self.convA(s)
			s = tf.nn.relu(s)
			s = self.convB(s)
			s = tf.nn.relu(s)
			s = self.convC(s)
			s = self.flattenC(s)

		batch_size = s.shape[0]
		s = tf.squeeze(s)	
		if len(s.shape)==1:
			s = tf.expand_dims(s, 0)
		if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
		prev_rew = tf.reshape(prev_rew, [batch_size, 1])
		prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		with tf.device("/device:GPU:0"):
			for i in range(0, self.n_hlayers):
				s = self.Dense[str(i)](s)

			# s = tf.concat([s, prev_act, prev_rew], -1)
			s = tf.concat([s, prev_act], -1)
			s = tf.expand_dims(s, 0)
			# print("rec input shape",s.shape)
			s = self.lstm(s)
			values = self.value_l(s)
			logits = self.act_l(s)

		return tf.squeeze(logits), tf.squeeze(values,-1)


class AC_MMRIM_PrediNet(Model):
	"""
	Actor-Critic MultiModal PrediNet, PrediNet imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	This is the standard M-PrediNet
	"""
	def __init__(self, n_outputs, conv_out_size, trainable, board = 0, n_BrimsN=20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False, rnn_type = 'lstm',
					extended = True, n_neurons = 16, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_MMRIM_PrediNet'):

		super(AC_MMRIM_PrediNet, self).__init__(name=name)
		self.trainable = trainable

		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		if visual: self._conv_out_size = int(conv_out_size[-2]/9 * conv_out_size[-1]/9)
		else: self._conv_out_size = conv_out_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		self.rnn_type= rnn_type
		self.mem = Hlstm
		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		# Feature co-ordinate matrix
		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsT = tf.concat([cols,rows],1)

		self.flattenT = tf.keras.layers.Flatten()

		# Define all model components
		self.get_keysT = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysT')
		self.get_query1T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1T')
		self.get_query2T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2T')
		self.embed_entitiesT = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesT')

		# self.DenseT = {}
		# for i in range(0, n_hlayers):
		if Hlstm:
			# self.lstmT = tf.keras.layers.LSTM(units=128, time_major = False, return_sequences=True)
			self.rimcell = RIMCell(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)
			self.recnetT = tf.keras.layers.RNN(cell=self.rimcell, return_sequences=True)
			
		else:
			# for i in range(0, n_hlayers):
			self.DenseT = tf.keras.layers.Dense(units=relations_p, 
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												activation='relu', name='DenseT')
		self.Tout = tf.keras.layers.Dense(units=self.n_neurons,
											bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
											activation=None, name='Tout')

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(obs_dim)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self.obs_size,1])

		rows = tf.reshape(rows, [self.obs_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.flattenN = tf.keras.layers.Flatten()

		self.get_keysN = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysN')
		self.get_query1N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1N')
		self.get_query2N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2N')
		self.embed_entitiesN = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesN')

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			# self.lstmN = tf.keras.layers.LSTM(units=128, time_major = False)
			self.rimcell2 = RIMCell(units=n_BrimsN, nRIM=6, k=4, rnn_type= self.rnn_type)
			self.recnetN = tf.keras.layers.RNN(cell=self.rimcell2)

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_MMRIM_PrediNet, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None, states=None):
		s = tf.dtypes.cast(s, tf.float32)
		if self.mem: tboostrap=s.shape[1]
		batch_size = s.shape[0]
		if self._visual:
			if not self.mem:
				if len(s.shape)==2:
					s = tf.expand_dims(s, 0)
					s = tf.expand_dims(s, -1)
				elif len(s.shape)==3:
					s = tf.expand_dims(s, -1)
				s = self.convA(s)
				s = tf.nn.relu(s)
				s = self.convB(s)
				s = tf.nn.relu(s)
				s = self.convC(s)
				s = self.flattenC(s)
			else:
				if len(s.shape)==4: s = tf.expand_dims(s, -1)
				s = kl.TimeDistributed(self.convA)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convB)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convC)(s)
				s = kl.TimeDistributed(self.flattenC)(s)
		if not self.mem:
			s = tf.squeeze(s)
			if len(s.shape)==1:
				s = tf.expand_dims(s, 0) #until here remove if it's not mem
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		else:
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((tboostrap,self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		s = tf.expand_dims(s, -1)
		obs = tf.reshape(s[:, :, -self.obs_size:], [batch_size, tboostrap,
				self.obs_size,1])	
		# print("Latent space", s.shape)
		# print("obs", obs.shape)		
		# jsakjsa+=1
		with tf.device("/device:GPU:0"):
			# print("s",s.shape)
			# print("x",x.shape)
			# print("obs ",obs.shape)
			# Append location
			t_locs=tf.expand_dims(tf.expand_dims(self._locsT, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([s, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1T)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2T)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			if len(feature1.shape) < 4:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesT)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesT)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			if self._Hlstm:
				# relations = tf.expand_dims(relations, 0)
				# x = tf.concat([relations, prev_act, prev_rew], -1)
				x = tf.concat([relations, prev_act], -1)
				# x = self.lstmT(x)
				x = self.recnetT(x)
			else: x =  self.DenseT(relations)
			# (batch size, Num_LSTM)

			# tf.print(x.shape, "RNN output with time seq", summarize=20)
			# print("LSTM output", x)

			IntTask = kl.TimeDistributed(self.Tout)(x)
			# hshshs+=1

			# IntTask = self.Tout(x)
			# # (batch size, num_neurons)
			# IntTask = tf.tile(tf.expand_dims(IntTask, 1), [1, tboostrap,1])
			# (batch size, time,  num_neurons)

			# x = tf.expand_dims(x, -1)
			# x = tf.concat([x,obs],1)

			#-------Navigator
			t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([obs, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1N)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2N)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)
			if len(feature1.shape) < 4:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesN)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesN)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			x = tf.concat([relations, IntTask],-1)

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				# x = self.lstmN(x)
				x = self.recnetN(x)
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
		return tf.squeeze(logits), tf.squeeze(values,-1), [0]


class AC_MMBRIM_PrediNet(Model):
	"""
	Actor-Critic MultiModal PrediNet, PrediNet imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	This is the standard M-PrediNet
	"""
	def __init__(self, n_outputs, conv_out_size, trainable, bidirectional =True,
					board = 0, n_BrimsN=20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False, rnn_type = 'lstm',
					extended = True, n_neurons = 16, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_MMBRIM_PrediNet'):

		super(AC_MMBRIM_PrediNet, self).__init__(name=name)
		self.trainable = trainable
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		if visual: self._conv_out_size = int(conv_out_size[-2]/9 * conv_out_size[-1]/9)
		else: self._conv_out_size = conv_out_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		self.rnn_type= rnn_type
		self.mem = Hlstm
		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		# Feature co-ordinate matrix
		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsT = tf.concat([cols,rows],1)

		self.flattenT = tf.keras.layers.Flatten()

		# Define all model components
		self.get_keysT = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysT')
		self.get_query1T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1T')
		self.get_query2T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2T')
		self.embed_entitiesT = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesT')

		# self.DenseT = {}
		# for i in range(0, n_hlayers):
		if Hlstm:
			# self.lstmT = tf.keras.layers.LSTM(units=128, time_major = False, return_sequences=True)
			self.recnetT = BRIM(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type, return_sequences=True)			
		else:
			# for i in range(0, n_hlayers):
			self.DenseT = tf.keras.layers.Dense(units=relations_p, 
												bias_initializer=self._bias_initializer,
												kernel_initializer=self._weight_initializer,
												activation='relu', name='DenseT')
		self.Tout = tf.keras.layers.Dense(units=self.n_neurons,
											bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
											activation=None, name='Tout')

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(obs_dim)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self.obs_size,1])

		rows = tf.reshape(rows, [self.obs_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.flattenN = tf.keras.layers.Flatten()

		self.get_keysN = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysN')
		self.get_query1N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1N')
		self.get_query2N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2N')
		self.embed_entitiesN = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesN')

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			# self.lstmN = tf.keras.layers.LSTM(units=128, time_major = False)
			self.recnetN = BRIM(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)	

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_MMBRIM_PrediNet, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None, hstates= None):
		s = tf.dtypes.cast(s, tf.float32)
		if hstates is None: pass
		else:
			if len(hstates.shape)<6: hstates= tf.expand_dims(hstates, axis=0)
			# print('hstates in call', hstates.shape)
			# tf.print('tf hstates',hstates.shape)
			# jjdjd+=1

		if self.mem: tboostrap=s.shape[1]
		batch_size = s.shape[0]
		if self._visual:
			if not self.mem:
				if len(s.shape)==2:
					s = tf.expand_dims(s, 0)
					s = tf.expand_dims(s, -1)
				elif len(s.shape)==3:
					s = tf.expand_dims(s, -1)
				s = self.convA(s)
				s = tf.nn.relu(s)
				s = self.convB(s)
				s = tf.nn.relu(s)
				s = self.convC(s)
				s = self.flattenC(s)
			else:
				if len(s.shape)==4: s = tf.expand_dims(s, -1)
				s = kl.TimeDistributed(self.convA)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convB)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convC)(s)
				s = kl.TimeDistributed(self.flattenC)(s)
		if not self.mem:
			s = tf.squeeze(s)
			if len(s.shape)==1:
				s = tf.expand_dims(s, 0) #until here remove if it's not mem
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		else:
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((tboostrap,self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		s = tf.expand_dims(s, -1)
		obs = tf.reshape(s[:, :, -self.obs_size:], [batch_size, tboostrap,
				self.obs_size,1])	
		# print("Latent space", s.shape)
		# print("obs", obs.shape)		
		with tf.device("/device:GPU:0"):
			# Append location
			t_locs=tf.expand_dims(tf.expand_dims(self._locsT, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([s, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1T)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2T)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			if len(feature1.shape) < 4:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesT)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesT)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			if self._Hlstm:
				# relations = tf.expand_dims(relations, 0)
				# x = tf.concat([relations, prev_act, prev_rew], -1)
				x = tf.concat([relations, prev_act], -1)
				# x = self.lstmT(x)
				if hstates is None:
					x, hstatesT = self.recnetT(x)
				else:
					x, hstatesT = self.recnetT(x, hstates[:,0,:,:,:,:])
			else: x =  self.DenseT(relations)
			# (batch size, Num_LSTM)

			# tf.print(x, "LSTM output", summarize=40)
			# print("LSTM output", x)
			IntTask = kl.TimeDistributed(self.Tout)(x)

			# IntTask = self.Tout(x)
			# # (batch size, num_neurons)
			# IntTask = tf.tile(tf.expand_dims(IntTask, 1), [1, tboostrap,1])
			# (batch size, time,  num_neurons)

			# x = tf.expand_dims(x, -1)
			# x = tf.concat([x,obs],1)

			#-------Navigator
			t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([obs, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1N)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2N)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)
			if len(feature1.shape) < 4:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesN)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesN)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)
			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			x = tf.concat([relations, IntTask],-1)

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				# x = self.lstmN(x)
				if hstates is None:
					x, hstatesN = self.recnetN(x)
				else:
					x, hstatesN = self.recnetN(x, hstates[:,1,:,:,:,:])
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
			
		return tf.squeeze(logits), tf.squeeze(values,-1), tf.squeeze(tf.stack([hstatesT,hstatesN], axis=1))

class AC_MMBRIM_V_PrediNet(Model):
	"""
	Actor-Critic MultiModal PrediNet, PrediNet imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	This is the standard M-PrediNet
	"""
	def __init__(self, n_outputs, input_size, trainable, bidirectional =True,
					board = 0, n_BrimsN=20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False, rnn_type = 'lstm',
					extended = True, n_neurons = 16, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_MMBRIM_V_PrediNet',
					text_params = [64,32]):

		super(AC_MMBRIM_V_PrediNet, self).__init__(name=name)
		self.trainable = trainable
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		self.AE = False

		if visual: self._conv_out_size = int(input_size[-3]/9 * input_size[-2]/9)
		else: self._conv_out_size = input_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		self.rnn_type= rnn_type
		self.mem = Hlstm
		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		# print("obs_dim", obs_dim)
		# print('self.Nrows', self.Nrows)

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		# Feature co-ordinate matrix
		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		# print("cols ", cols.shape)

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		# print("cols ", cols.shape)
		# jfdk+=1

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsT = tf.concat([cols,rows],1)

		self.flattenT = tf.keras.layers.Flatten()

		# Define all model components
		self.get_keysT = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysT')
		self.get_query1T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1T')
		self.get_query2T = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2T')
		self.embed_entitiesT = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesT')

		# self.DenseT = {}
		# for i in range(0, n_hlayers):

		# Navigator
		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(obs_dim)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self.obs_size,1])

		rows = tf.reshape(rows, [self.obs_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.flattenN = tf.keras.layers.Flatten()

		self.get_keysN = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysN')
		self.get_query1N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1N')
		self.get_query2N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2N')
		self.embed_entitiesN = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesN')

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			# self.lstmN = tf.keras.layers.LSTM(units=128, time_major = False)
			self.Hrecnet = MMBRIM_V1(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)	
			# self.Hrecnet = MMBRIM_V2(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)	

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_MMBRIM_V_PrediNet, self).build(input_shape)  # Be sure to call this at the end

	@tf.function(experimental_relax_shapes=True)
	def call(self, s, prev_act = None, prev_rew = None, hstates= None, mission = None, enc_text=None):
		s = tf.dtypes.cast(s, tf.float32)
		hstates = None
		if self.mem: tboostrap=s.shape[1]
		batch_size = tf.shape(s)[0]
		if self._visual:
			if not self.mem:
				if len(s.shape)==2:
					s = tf.expand_dims(s, 0)
					s = tf.expand_dims(s, -1)
				elif len(s.shape)==3:
					s = tf.expand_dims(s, -1)
				s = self.convA(s)
				s = tf.nn.relu(s)
				s = self.convB(s)
				s = tf.nn.relu(s)
				s = self.convC(s)
				s = self.flattenC(s)
			else:
				# if len(s.shape)==4: s = tf.expand_dims(s, -1)
				s = kl.TimeDistributed(self.convA)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convB)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convC)(s)
				s = kl.TimeDistributed(self.flattenC)(s)
		if not self.mem:
			s = tf.squeeze(s)
			if len(s.shape)==1:
				s = tf.expand_dims(s, 0) #until here remove if it's not mem
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		else:
			if prev_act is None:
					aux = np.zeros((tboostrap,1)) 
					prev_rew = tf.constant(aux, dtype=tf.float32)
					aux = np.zeros((tboostrap,self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, tboostrap, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		s = tf.expand_dims(s, -1)
		obs = tf.reshape(s[:, :, -self.obs_size:], [batch_size, tboostrap,
				self.obs_size,1])	
		channels = 1
		# print("Latent space", s.shape)
		# print("obs", obs.shape)		
		
		with tf.device("/device:GPU:0"):
			# print("s",s.shape)
			# print("x",x.shape)
			# print("obs ",obs.shape)
			# Append location
			t_locs=tf.expand_dims(tf.expand_dims(self._locsT, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# print("self._locsT", self._locsT.shape)
			# print('self._locsn', self._locsN.shape)
			# print("t_locs", t_locs.shape)
			# print("locs", locs.shape)


			features_locs = tf.concat([s, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)
			# print("features_locs", features_locs.shape)
			# jsakjsa+=1


			features_flat = kl.TimeDistributed(self.flattenT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysT)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1T)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2T)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			feature1 = tf.reshape(feature1, [batch_size, tboostrap, self._heads, channels+2])
			feature2 = tf.reshape(feature2, [batch_size, tboostrap, self._heads, channels+2])

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesT)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesT)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations


			Tinputs = tf.concat([relations, prev_act], -1)

			# IntTask = self.Tout(x)
			# # (batch size, num_neurons)
			# IntTask = tf.tile(tf.expand_dims(IntTask, 1), [1, tboostrap,1])
			# (batch size, time,  num_neurons)

			# x = tf.expand_dims(x, -1)
			# x = tf.concat([x,obs],1)

			#-------Navigator
			t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([obs, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1N)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2N)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			feature1 = tf.reshape(feature1, [batch_size, tboostrap, self._heads, channels+2])
			feature2 = tf.reshape(feature2, [batch_size, tboostrap, self._heads, channels+2])

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesN)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesN)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)

			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			Ninputs = tf.concat([relations, prev_act], -1)

			# Ninputs = Tinputs

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				# x = self.lstmN(x)
				if hstates is None:
					x, Nhstates = self.Hrecnet(Tinputs = Tinputs, Ninputs= Ninputs)
				else:
					if len(hstates.shape)<5: hstates = tf.expand_dims(hstates, 0)
					x, Nhstates = self.Hrecnet(Tinputs = Tinputs, Ninputs= Ninputs, states= hstates[:,:,:,:,:])
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
		return tf.squeeze(logits), tf.squeeze(values,-1), tf.squeeze(Nhstates), None


class AC_BRIM_PrediNet(Model):
	"""
	Actor-Critic MultiModal PrediNet, PrediNet imported from 
	https://github.com/deepmind/deepmind-research/blob/master/PrediNet/PrediNet.ipynb
	This is the standard M-PrediNet
	"""
	def __init__(self, n_outputs, conv_out_size, trainable, bidirectional =True,
					board = 0, n_BrimsN=20,
					n_hlayers = 2, Hlstm = False, obs_dim=5, visual=False, rnn_type = 'lstm',
					extended = True, n_neurons = 16, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, name='AC_BRIM_PrediNet'):

		super(AC_BRIM_PrediNet, self).__init__(name=name)
		self.trainable = trainable
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		if visual: self._conv_out_size = int(conv_out_size[-3]/9 * conv_out_size[-2]/9)
		else: self._conv_out_size = conv_out_size[-1]
		self.obs_size = obs_dim**2
		assert(self._conv_out_size% obs_dim == 0)
		self.Nrows = int(self._conv_out_size / obs_dim)
		self.n_hlayers = n_hlayers
		self._key_size = key_size
		self._heads = heads_p
		self._relations = relations_p
		self.central_ouput_size = heads_p * (relations_p+4)
		self._channels = channels
		self._Hlstm = Hlstm
		self._visual = visual
		self.rnn_type= rnn_type
		self.mem = Hlstm
		self.AE = False

		# Encoder, works well only if the input is = resolution

		# self.flatten = tf.keras.layers.Flatten()
		# Hidden layers

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		# Feature co-ordinate matrix
		if self._visual:
			kernels=[3,3,1]
			filters=[8,16,1]
			strides=[3,3,1]
			self.convA=tf.keras.layers.Conv2D(filters[0], kernels[0], strides=strides[0])
			self.convB=tf.keras.layers.Conv2D(filters[1], kernels[1], strides=strides[1])
			self.convC=tf.keras.layers.Conv2D(filters[2], kernels[2], strides=strides[2])
			self.flattenC = tf.keras.layers.Flatten()

		# Feature co-ordinate matrix
		cols = tf.constant([[[x / float(obs_dim)]
						 for x in range(self.Nrows)]
						for _ in range(obs_dim)])

		rows = tf.transpose(cols, [1, 0, 2])

		cols = tf.reshape(cols, [self._conv_out_size,1])

		rows = tf.reshape(rows, [self._conv_out_size,1])

		self._locsN = tf.concat([cols,rows],1)

		self.flattenN = tf.keras.layers.Flatten()

		self.get_keysN = tf.keras.layers.Dense(
				units=self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_keysN')
		self.get_query1N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query1N')
		self.get_query2N = tf.keras.layers.Dense(
				units=self._heads * self._key_size, use_bias=False,
				kernel_initializer=self._weight_initializer, name='get_query2N')
		self.embed_entitiesN = tf.keras.layers.Dense(
				units=self._relations, use_bias=False,
				kernel_initializer=self._weight_initializer, name='embed_entitiesN')

		# self.DenseN = {}
		# for i in range(0, n_hlayers):
		# 	self.DenseN[str(i)] = tf.keras.layers.Dense(units=n_neurons, 
		# 										bias_initializer=self._bias_initializer,
		# 										kernel_initializer=self._weight_initializer,
		# 									 	activation='relu', name='DenseN'+str(i))
		if Hlstm:
			# self.lstmN = tf.keras.layers.LSTM(units=128, time_major = False)
			self.recnet = BRIM(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)	

		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(AC_BRIM_PrediNet, self).build(input_shape)  # Be sure to call this at the end

	def call(self, s, prev_act = None, prev_rew = None, hstates= None, mission = None, enc_text=None):
		s = tf.dtypes.cast(s, tf.float32)
		if hstates is None: pass
		else:
			if len(hstates.shape)<6: hstates= tf.expand_dims(hstates, axis=0)
			# print('hstates in call', hstates.shape)
			# tf.print('tf hstates',hstates.shape)
			# jjdjd+=1
		hstates = None
		if self.mem: tboostrap=s.shape[1]
		batch_size = s.shape[0]
		if self._visual:
			if not self.mem:
				if len(s.shape)==2:
					s = tf.expand_dims(s, 0)
					s = tf.expand_dims(s, -1)
				elif len(s.shape)==3:
					s = tf.expand_dims(s, -1)
				s = self.convA(s)
				s = tf.nn.relu(s)
				s = self.convB(s)
				s = tf.nn.relu(s)
				s = self.convC(s)
				s = self.flattenC(s)
			else:
				if len(s.shape)==4: s = tf.expand_dims(s, -1)
				s = kl.TimeDistributed(self.convA)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convB)(s)
				s = tf.nn.relu(s)
				s = kl.TimeDistributed(self.convC)(s)
				s = kl.TimeDistributed(self.flattenC)(s)
		if not self.mem:
			s = tf.squeeze(s)
			if len(s.shape)==1:
				s = tf.expand_dims(s, 0) #until here remove if it's not mem
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, self.n_outputs])
		else:
			if prev_act is None:
					prev_rew = tf.constant([0], dtype=tf.float32)
					aux = np.zeros((tboostrap,self.n_outputs)) 
					prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		s = tf.expand_dims(s, -1)	
		# print("Latent space", s.shape)
		# print("obs", obs.shape)		
		with tf.device("/device:GPU:0"):

			t_locs=tf.expand_dims(tf.expand_dims(self._locsN, 0),0)
			locs = tf.tile(t_locs, [batch_size, tboostrap, 1, 1])

			# tf.print(s.shape, "Conv output", summarize=40)
			# tf.print(locs.shape, "locs", summarize=40)

			features_locs = tf.concat([s, locs], 3)
			# (batch_size, time, (conv_out_size+1)*conv_out_size,channels+2)

			features_flat = kl.TimeDistributed(self.flattenN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size*channels+2)
			# Keys
			keys = kl.TimeDistributed(self.get_keysN)(features_locs)
			# (batch_size, time, (conv_out_size+1)*conv_out_size, key_size)
			keys = tf.tile(tf.expand_dims(keys, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, (conv_out_size+1)*conv_out_size, key_size)
			# Queries
			query1 = kl.TimeDistributed(self.get_query1N)(features_flat)
			# (batch_size, time, heads*key_size)
			query1 = tf.reshape(query1, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query1 = tf.expand_dims(query1, 3)
			# (batch_size, time, heads, 1, key_size)

			query2 = kl.TimeDistributed(self.get_query2N)(features_flat)
			# (batch_size, time, heads*key_size)
			query2 = tf.reshape(query2, [batch_size, tboostrap, self._heads, self._key_size])
			# (batch_size, time, heads, key_size)
			query2 = tf.expand_dims(query2, 3)
			# (batch_size, time, heads, 1, key_size)

			# Attention weights
			keys_t = tf.transpose(keys, perm=[0, 1, 2, 4, 3])
			# (batch_size, time, heads, key_size, conv_out_size*conv_out_size)

			att1 = tf.nn.softmax(tf.matmul(query1, keys_t))
			att2 = tf.nn.softmax(tf.matmul(query2, keys_t))
			# (batch_size, time, heads, 1, (conv_out_size+1)*conv_out_size)

			# Reshape features
			features_tiled = tf.tile(
				tf.expand_dims(features_locs, 2), [1, 1, self._heads, 1, 1])
			# (batch_size, time, heads, conv_out_size*conv_out_size, channels+2)

			# Compute a pair of features using attention weights
			feature1 = tf.squeeze(tf.matmul(att1, features_tiled))
			feature2 = tf.squeeze(tf.matmul(att2, features_tiled))

			# (batch_size, time, heads, (channels+2))
			if len(feature1.shape) < 3:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)
			if len(feature1.shape) < 4:
				feature1 = tf.expand_dims(feature1, 0)
				feature2 = tf.expand_dims(feature2, 0)

			# Spatial embedding
			embedding1 = kl.TimeDistributed(self.embed_entitiesN)(feature1)
			embedding2 = kl.TimeDistributed(self.embed_entitiesN)(feature2)
			# (batch_size, time, heads, relations)

			# Comparator
			dx = tf.subtract(embedding1, embedding2)

			# Positions
			pos1 = tf.slice(feature1, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			pos2 = tf.slice(feature2, [0, 0, 0, self._channels], [-1, -1, -1, -1])
			# (batch_size, time, heads, 2)
			# Collect relations and concatenate positions
			relations = tf.concat([dx, pos1, pos2], 3)
			# (batch_size, time, heads, relations+4)
			relations = tf.reshape(relations,
							[batch_size, tboostrap, self._heads * (self._relations + 4)])
			# (batch_size, time, heads*(relations+4))
			x = relations

			if self._Hlstm:
				# x = tf.concat([x, prev_act, prev_rew], -1)
				x = tf.concat([x, prev_act], -1)
				# x = self.lstmN(x)
				if hstates is None:
					x, hstates = self.recnet(x)
				else:
					x, hstates = self.recnet(x, hstates[:,:,:,:,:])
			else:
				raise Exception('Not implemented')

			values = self.value_l(x)
			logits = self.act_l(x)
			
		return tf.squeeze(logits), tf.squeeze(values,-1), tf.squeeze(hstates), None


class Core_MM_Recurrent(Model):
	def __init__(self, n_outputs, bidirectional =True, mem_net='rim', view_size =5, core_net = "PrediNet",
					board = 0, n_BrimsN=50, n_hlayers = 1, Hlstm = False, text_hidden_dim = "32",
					ResNet = False,
					visual=True, rnn_type = 'lstm', n_neurons = 16, heads_p = 32, key_size=16,
					channels=1, relations_p = 16, mem_type= 'sequential', name='Core_MM_Recurrent',
					mode = 0):
		super(Core_MM_Recurrent, self).__init__(name=name)
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		self.rnn_type = rnn_type
		self._channels = channels
		self.mem_net=mem_net
		self.mode = mode
		self.text_hidden_dim = text_hidden_dim


		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)


		if core_net == 'default':
			self.central_moduleT = tf.keras.layers.Dense(units=640,
												kernel_initializer=self._weight_initializer,
												activation='relu',
												bias_initializer=self._bias_initializer,
												name='CentralT_Dense_layer')
			self.central_moduleN = tf.keras.layers.Dense(units=640,
												kernel_initializer=self._weight_initializer,
												activation='relu',
												bias_initializer=self._bias_initializer,
												name='CentralN_Dense_layer')
		elif core_net == 'PrediNet':	
			#Relational Modules
			self.central_moduleT = PrediNet(self._weight_initializer, self._bias_initializer, 
									 	 		view_size = view_size, name= 'PrediNetT')

			self.central_moduleN = PrediNet(self._weight_initializer, self._bias_initializer,
									 	 		 view_size = view_size, name= 'PrediNetN')

		elif core_net == 'MHA':
			self.central_moduleT = MHA(self._weight_initializer, self._bias_initializer, 
										  		name='CentralT_MHA')
			self.central_moduleN = MHA(self._weight_initializer, self._bias_initializer, 
										  		name='CentralN_MHA')
		elif core_net == 'RelNet':
			self.central_moduleT = RelationNet(self._weight_initializer, self._bias_initializer, 
										  		name='CentralT_MHA')
			self.central_moduleN = RelationNet(self._weight_initializer, self._bias_initializer, 
										  		name='CentralN_MHA')

		else:
			print('The net is missing yet')
			raise Exception('This kind of core layer is not implemented')

		#rec_Net
		#  add non_rec mem option and output size = mem_embed_size in the dense layer
		if mem_net == 'rim':
			self.Hrecnet = MMBRIM_V1(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)	
			# self.Hrecnet = MMBRIM_V2(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)
		else:
			self.lstmT = tf.keras.layers.LSTM(units=128, time_major = False, return_sequences=True, return_state=True)

			self.task_bottleneck = tf.keras.layers.Dense(units=4,
											bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
											activation='relu', name='task_embedding')

			# self.ablation_bottleneck = tf.keras.layers.Dense(units=n_neurons,
			# 								bias_initializer=self._bias_initializer,
			# 								kernel_initializer=self._weight_initializer,
			# 								activation='relu', name='task_embedding')

			self.lstmN = tf.keras.layers.LSTM(units=128, time_major = False, return_sequences=False, return_state=True)

		#actor-critic
		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation='softmax',
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
		self.used_obs = None
		self.mode=mode

	def build(self, input_shape):
		super(Core_MM_Recurrent, self).build(input_shape)  # Be sure to call this at the end

	@tf.function(experimental_relax_shapes=True)
	def call(self, t_input, used_obs = None, prev_rew = None, prev_act = None,
						hstates = None, instr = None):
		batch_size = tf.shape(t_input)[0]
		tboostrap = t_input.shape[1]

		if used_obs is None:
			used_obs = tf.zeros_like(self.used_obs)
			aux = np.zeros((tboostrap,1)) 
			prev_rew = tf.constant(aux, dtype=tf.float32)
			aux = np.zeros((tboostrap,self.n_outputs)) 
			prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, tboostrap, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
			if self.mode>0:
				instr_aux = np.zeros((tboostrap, self.text_hidden_dim*2))
				instr = tf.reshape(tf.constant(instr_aux, dtype=tf.float32),
									 [batch_size, tboostrap, self.text_hidden_dim*2])

		central_embed_T = self.central_moduleT(t_input)

		# tf.print('relationsT', relationsT)
		
		
		if self.mode==1: Tinputs = tf.concat([Tinputs, instr],-1)

		central_embed_N = self.central_moduleN(used_obs)
		# central_embed_N = kl.TimeDistributed(self.ablation_bottleneck)(central_embed_N)

				# x = tf.concat([x, prev_act, prev_rew], -1)
				# x = self.lstmN(x)


		mem_queryT = None

		Tinputs = tf.concat([central_embed_T, prev_act], -1)

		if self.mem_net == 'rim':
			Ninputs = tf.concat([central_embed_N, prev_act], -1)
			if hstates is None:
				x, hstates = self.Hrecnet(Tinputs = Tinputs, Ninputs= Ninputs)
			else:
				if len(hstates.shape)<5: hstates = tf.expand_dims(hstates, 0)
				x, hstates = self.Hrecnet(Tinputs = Tinputs, Ninputs= Ninputs, states = hstates)
		else:
			if hstates is None: 
				x, h1, c1 = self.lstmT(Tinputs)
			else:
				# tf.print('hstates used')
				if len(hstates.shape)<4 : hstates = tf.expand_dims(hstates,0)
				hstates = tf.transpose(hstates, perm=[1, 2, 0, 3])
				x, h1, c1 = self.lstmT(Tinputs, initial_state=[hstates[0][0], hstates[0][1]])
			Thstates = [h1, c1]

			embedding =  kl.TimeDistributed(self.task_bottleneck)(x)

			Ninputs = tf.concat([central_embed_N, embedding, prev_act], -1)
			if hstates is None: 
				x, h2, c2 = self.lstmN(Ninputs)
			else:
				x, h2, c2 = self.lstmN(Ninputs, initial_state=[hstates[1][0], hstates[1][1]])
			Nhstates = [h2, c2]

			hstates = [Thstates,Nhstates]

		values = self.value_l(x)
		logits = self.act_l(x)

		# check the AE maybe it's better as output from the  lstm

		return logits, values, hstates, central_embed_T

class AC_AE_MM_Recurrent(Model):
	"""
	Actor-Critic MultiModal PrediNet with BRIMS
	"""
	def __init__(self, n_outputs, input_size, trainable, AE = False, bidirectional =True,
					board = 0, n_BrimsN=50, n_hlayers = 1, Hlstm = False, obs_dim=5, 
					visual=True, rnn_type = 'lstm', text_rnn_type = 'lstm', view_size = 5,
					extended = True, n_neurons = 16, heads_p = 32, key_size=16, mode = 0,
					channels=1, relations_p = 16, name='AC_AE_MM_Recurrent', text_hidden_dim = 32,
					core_net= 'default', mem_net='rim', text_params = [64,32], ttl='TTL2',
					mem_type='sequential', ResNet = False, mapT = 'minigrid'):
		super(AC_AE_MM_Recurrent, self).__init__(name=name)
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		self.text_params = text_params
		self.rnn_type = rnn_type
		self._channels = channels
		self.text_hidden_dim = text_hidden_dim
		self.AE = AE
		self.ttl = ttl
		self._view_size = view_size
		self.mode = mode
		self.core_net = core_net
		self.Rec_instr = False


		#Encoders of the input
		# self.cae = CAE(channels= channels, ttl = ttl)

		# if self.text_params is not None:
		# 	self.cae = CAE(channels= channels, ttl = ttl)
		# else:
		# 	self.cae = AuxCNN(channels= channels, ttl = ttl)
		self.cae = AuxCNN(channels= channels, ResNet = ResNet, mapT = mapT)

		if self.text_params is not None:
			self.textLength = text_params[1]
			self.dic_size = text_params[0]

			if mode > 0 or core_net == 'default': #from experimental results 0 is best performer
				self.Rec_instr = True
				# if self.AE:
				# 	self.tae = TAE(dic_size=text_params[0],  textLength=self.textLength,
				# 		rnn_type = text_rnn_type, hidden_dim = text_hidden_dim)
				# else:
				self.tae = TextEmbed(dic_size=text_params[0],  textLength=self.textLength,
					rnn_type = text_rnn_type, hidden_dim = text_hidden_dim)

		# self.pre_flat = tf.keras.layers.Flatten()


		self.core = Core_MM_Recurrent(n_outputs = n_outputs, bidirectional = bidirectional, 
										n_BrimsN = n_BrimsN, n_hlayers = n_hlayers, Hlstm= Hlstm, 
										visual = visual, rnn_type = rnn_type, channels= channels, 
										view_size = view_size, mode = mode, text_hidden_dim = text_hidden_dim,
										n_neurons = n_neurons, heads_p = heads_p, key_size = key_size, 
										relations_p = relations_p, core_net = core_net, mem_net = mem_net,
										ResNet= ResNet)
		if self.AE: 
			self.img_decoder = ImgDecoder(ResNet = ResNet)
			if self.text_params is not None:
				self.text_decoder = TextDecoder(textLength=self.textLength, dic_size=self.dic_size)

	def build(self, input_shape):
		super(AC_AE_MM_Recurrent, self).build(input_shape)  # Be sure to call this at the end

	@tf.function(experimental_relax_shapes=True)
	def call(self, s, prev_act = None, prev_rew = None, hstates= None, 
				text = None):
		# with tf.device("/device:GPU:0"):
		building = False
		s = tf.dtypes.cast(s, tf.float32)
		if len(s.shape) == 4:
			 s = tf.expand_dims(s,0)
		tboostrap=s.shape[1]
		batch_size = tf.shape(s)[0]

		if prev_act is None:
			building = True
			aux = np.zeros((tboostrap,1)) 
			prev_rew = tf.constant(aux, dtype=tf.float32)
			aux = np.zeros((tboostrap,self.n_outputs)) 
			prev_act = tf.constant(aux, dtype=tf.float32)
			if self.text_params is not None:
				if self.Rec_instr:
					text_aux = np.zeros((tboostrap, self.textLength, self.dic_size))
					text = tf.constant(text_aux, dtype=tf.float32)
				else:
					text_aux = np.zeros((tboostrap, self.textLength))
					text = tf.constant(text_aux, dtype=tf.float32)


		prev_rew = tf.reshape(prev_rew, [batch_size, tboostrap, 1])
		prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		

		enc_obs = self.cae(s)

		# hstates = None

		# print('enc_obs', enc_obs.shape)

		if self.core_net == 'default':
			used_obs = tf.reshape(enc_obs, [batch_size, tboostrap, enc_obs.shape[2]*enc_obs.shape[3]*enc_obs.shape[4]])
		else:
			used_obs = tf.reshape(enc_obs, [batch_size, tboostrap, enc_obs.shape[2]*enc_obs.shape[3], enc_obs.shape[4]])

		# print('t_input', t_input.shape)

		em_instr = None
		if self.text_params is not None:
			if self.Rec_instr:
				text = tf.reshape(text, [batch_size, tboostrap, self.textLength, self.dic_size])
				enc_text = self.tae(text)

				if self.mode == 0:
					t_input = used_obs
					t_input = tf.concat([used_obs, enc_text],-1)
				elif self.mode == 1:
					em_instr = enc_text
					t_input = used_obs
				elif self.mode == 2:
					n_rows = math.ceil(enc_text.shape[-1]/enc_obs.shape[3])
					pad_size = n_rows*enc_obs.shape[3]-enc_text.shape[-1]
					paddings = tf.constant([[0,0], [0,0], [0, pad_size]])
					enc_text = tf.pad(enc_text, paddings, "CONSTANT")
					enc_text = tf.reshape(enc_text, [batch_size, tboostrap, n_rows, enc_obs.shape[3],1])
					enc_text = tf.cast(tf.broadcast_to(enc_text, [batch_size, tboostrap, n_rows, 
													enc_obs.shape[3], enc_obs.shape[4]]),dtype=tf.float32)
					t_input = tf.concat([enc_obs, enc_text],2)
					t_input = tf.reshape(t_input, [batch_size, tboostrap, t_input.shape[2]*t_input.shape[3],t_input.shape[4]])
				else: #Mode 3, the good one
					if self.core_net != 'default':
						assert enc_text.shape[-1] %  used_obs.shape[-1] == 0
						enc_text = tf.reshape(enc_text, [batch_size, tboostrap, enc_text.shape[-1] //  used_obs.shape[-1],
														 used_obs.shape[-1]])
						t_input = tf.concat([used_obs, enc_text],2)
					else:
						t_input = tf.concat([used_obs, enc_text],-1)
			else:
				text = tf.reshape(text, [batch_size, tboostrap, self.textLength])
				if self.mode == 0:
					n_rows = math.ceil(self.textLength/enc_obs.shape[3])
					pad_size = n_rows*enc_obs.shape[3]-self.textLength
					paddings = tf.constant([[0,0], [0,0], [0, pad_size]])
					text = tf.pad(text, paddings, "CONSTANT")
					text = tf.reshape(text, [batch_size, tboostrap, n_rows, enc_obs.shape[3],1])
					text = tf.cast(tf.broadcast_to(text, [batch_size, tboostrap, n_rows, 
													enc_obs.shape[3], enc_obs.shape[4]]),dtype=tf.float32)

					t_input = tf.concat([enc_obs, text],2)
					t_input = tf.reshape(t_input, [batch_size, tboostrap, t_input.shape[2]*t_input.shape[3],t_input.shape[4]])


		else:
			inst_rows = 3 if self.ttl == 'TTL2' else 1
			t_input = used_obs
			used_obs = tf.reshape(enc_obs[:,:,inst_rows:, :], [batch_size, tboostrap, self._view_size**2, enc_obs.shape[4]])
			if self.core_net == 'default':
				used_obs = tf.reshape(used_obs, [batch_size, tboostrap, self._view_size**2 * enc_obs.shape[4]])

			# For task-agnostic ablation
			# used_obs = t_input
		
		if building: self.core.used_obs = used_obs

		logits, values, Nhstates, latent_rep = self.core(t_input, 
								used_obs, prev_rew, prev_act, hstates, em_instr)

		if self.AE:
			dec_obs = self.img_decoder(latent_rep)
			# print('s.shape', s.shape)
			# tf.print('s.shape', s.shape)
			# print('dec_obs.shape', dec_obs.shape)
			# tf.print('dec_obs.shape', dec_obs.shape)
			assert s.shape[2:] == dec_obs.shape[2:]
			if self.text_params is not None:
				dec_text = self.text_decoder(latent_rep)
				assert text.shape[2:] == dec_text.shape[2:]
		else: 
			dec_obs = None
			dec_text = None


		return tf.squeeze(logits), tf.squeeze(values,-1), tf.squeeze(Nhstates),\
			dec_obs, dec_text

class Core_Sequential_Recurrent(Model):
	def __init__(self, n_outputs, core_net= "default", mem_net='rim', mode=0,
					board = 0, n_lstmN=128, n_hlayers = 1, Hlstm = True, 
					visual=True, n_BrimsN=50, rnn_type = 'lstm', n_neurons = 128,
					text_hidden_dim=32, ResNet = False,
					name='Core_Sequential_Recurrent'):
		super(Core_Sequential_Recurrent, self).__init__(name=name)
		self.n_neurons = n_neurons
		self.n_lstmN = n_lstmN
		self.n_outputs = n_outputs
		self._visual = visual
		self.rnn_type = rnn_type
		self.Hlstm = Hlstm
		self.mem_net=mem_net
		self.mode = mode
		self.text_hidden_dim = text_hidden_dim
		self.ResNet = ResNet
		# self.Hlstm = False

		self._weight_initializer = tf.initializers.TruncatedNormal(
														mean=0.0, stddev=0.1)
		self._bias_initializer = tf.keras.initializers.Constant(0.1)

		#dense
		if core_net == 'default':
			self.embedding_layer = tf.keras.layers.Dense(units=640,
												kernel_initializer=self._weight_initializer,
												activation='relu',
												bias_initializer=self._bias_initializer,
												name='Central_Dense_layer')
		elif core_net == 'PrediNet':
			self.embedding_layer =  PrediNet(self._weight_initializer, self._bias_initializer, 
										  		name='Central_PrediNet')

		elif core_net == 'MHA':
			self.embedding_layer =  MHA(self._weight_initializer, self._bias_initializer, 
										  		name='Central_MHA')
		elif core_net == 'RelNet':
			self.embedding_layer =  RelationNet(self._weight_initializer, self._bias_initializer, 
										  		name='Central_RelNet')
		else:
			raise Exception('This kind of core layer is not implemented')

		#rec_Net
		if self.Hlstm:
			if mem_net == 'rim':
				self.Hrecnet = BRIM(units=n_BrimsN, nRIM=6, k=4, rnn_type = self.rnn_type)
			else:
				self.Hrecnet = tf.keras.layers.LSTM(units=self.n_lstmN, return_state=True)

		else: self.dense = tf.keras.layers.Dense(units=mem_embed_size,
											kernel_initializer=self._weight_initializer,
											activation='relu',
											bias_initializer=self._bias_initializer,
											name='output_layer')

		#actor-critic
		self.act_l = tf.keras.layers.Dense(units=n_outputs,
											kernel_initializer=self._weight_initializer,
											activation=None,
											bias_initializer=self._bias_initializer,
											name='act_l')
		self.value_l = tf.keras.layers.Dense(1, bias_initializer=self._bias_initializer,
											kernel_initializer=self._weight_initializer,
												 activation=None, name='value_l')
	def build(self, input_shape):
		super(Core_Sequential_Recurrent, self).build(input_shape)  # Be sure to call this at the end

	@tf.function(experimental_relax_shapes=True)
	def call(self, inp, prev_rew = None, prev_act = None, hstates = None,
						instr = None):
		batch_size = tf.shape(inp)[0]
		tboostrap = inp.shape[1]
		if prev_act is None:
			aux = np.zeros((tboostrap,1)) 
			prev_rew = tf.constant(aux, dtype=tf.float32)
			aux = np.zeros((tboostrap,self.n_outputs)) 
			prev_act = tf.constant(aux, dtype=tf.float32)
			prev_rew = tf.reshape(prev_rew, [batch_size, tboostrap, 1])
			prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
			if self.mode>0:
				instr_aux = np.zeros((tboostrap, self.text_hidden_dim*2))
				instr = tf.reshape(tf.constant(instr_aux, dtype=tf.float32),
									 [batch_size, tboostrap, self.text_hidden_dim*2])

		embedding = self.embedding_layer(inp)

		if self.mode==1: embedding = tf.concat([embedding, instr],-1)
		

		latent_rep = embedding

		# hstates = None

		embedding = tf.concat([embedding, prev_act], -1)
			
		if self.Hlstm:		
			if self.mem_net == 'rim':
				if hstates is None: x, Nhstates = self.Hrecnet(embedding)
				else:			
					if len(hstates.shape)<5: hstates = tf.expand_dims(hstates, 0)
					x, Nhstates = self.Hrecnet(embedding, states = hstates)

			else:
				if hstates is None: x, h, c = self.Hrecnet(embedding)
				else:
					hstates = tf.reshape(hstates, [batch_size, 2, self.n_lstmN])
					hstates = tf.transpose(hstates, perm=[1, 0, 2])
					x, h, c = self.Hrecnet(embedding, initial_state=[hstates[0], hstates[1]])
				Nhstates= [h,c]
		else: 
			embedding = tf.reshape(embedding, [batch_size, -1])
			x = self.dense(embedding)
			Nhstates = 0.0


		# latent_rep = tf.broadcast_to(tf.expand_dims(x, 1), [batch_size, tboostrap, x.shape[-1]])

		# a= tf.zeros([2,3])
		# b = tf.math.log(a)
		values = self.value_l(x)
		logits_pre = self.act_l(x)

		logits = tf.nn.softmax(logits_pre)

		return logits, values, Nhstates, latent_rep

class AC_AE_Sequential_Recurrent(Model):
	"""
	Actor-Critic MultiModal PrediNet with BRIMS
	"""
	def __init__(self, n_outputs, input_size, trainable, bidirectional =True,
					board = 0,n_lstmN=128, n_BrimsN=50, n_hlayers = 1, Hlstm = True, obs_dim=5, 
					visual=True, rnn_type = 'lstm', text_rnn_type = 'lstm', mem_embed_size= 80,
					core_net= 'default', mem_net='rim', mode = 0, text_hidden_dim = 32,
					extended = True, n_neurons = 128, heads_p = 32, key_size=16,
					channels=1, name='AC_AE_Sequential_Recurrent', AE = False,
					text_params = [64,32], ttl='TTL2', ResNet = False, mapT = 'minigrid',
					mem_type='sequential'):
		super(AC_AE_Sequential_Recurrent, self).__init__(name=name)
		self.bidirectional = bidirectional
		self.n_neurons = n_neurons
		self.n_outputs = n_outputs
		self._visual = visual
		self.text_params = text_params
		self.rnn_type = rnn_type
		self._channels = channels
		self.AE = AE
		self.core_net = core_net
		self.Rec_instr = False
		self.mode = mode

		#Encoders of the input
		# self.cae = CAE(channels= channels, ttl = ttl) if AE else AuxCNN(channels= channels, 
																				# ttl = ttl)
		# self.cae = CAE(channels= channels, ttl = ttl)
		self.cae = AuxCNN(channels = channels, mapT = mapT, ResNet = ResNet)

		if self.text_params is not None:
			self.textLength = text_params[1]
			self.dic_size = text_params[0]
			if mode>0 or core_net == 'default':
				self.Rec_instr = True
				# if self.AE:
				# 	self.tae = TAE(dic_size=text_params[0],  textLength=self.textLength,
				# 		rnn_type = text_rnn_type, hidden_dim = text_hidden_dim)
				# else:
				self.tae = TextEmbed(dic_size=text_params[0],  textLength=self.textLength,
					rnn_type = text_rnn_type, hidden_dim = text_hidden_dim)
				# self.tae = TAE(dic_size=text_params[0],  textLength=self.textLength,
				# 	rnn_type = text_rnn_type)

		# self.pre_flat = tf.keras.layers.Flatten()

		self.core = Core_Sequential_Recurrent(n_outputs = n_outputs, core_net = core_net, mem_net = mem_net,
										mode=mode, text_hidden_dim = text_hidden_dim, n_BrimsN = n_BrimsN,
										n_lstmN = n_lstmN, n_hlayers = n_hlayers, Hlstm= Hlstm, 
										visual = visual, n_neurons = n_neurons, ResNet= ResNet)

		if self.AE: 
			self.img_decoder = ImgDecoder(ResNet = ResNet)
			if self.text_params is not None:
				self.text_decoder = TextDecoder(textLength=self.textLength, dic_size=self.dic_size)

	def build(self, input_shape):
		super(AC_AE_Sequential_Recurrent, self).build(input_shape)  # Be sure to call this at the end

	@tf.function(experimental_relax_shapes=True)
	def call(self, s, prev_act = None, prev_rew = None, hstates= None, 
						text = None):
		building = False
		s = tf.dtypes.cast(s, tf.float32)
		if len(s.shape) == 4:
			 s = tf.expand_dims(s,0)
		tboostrap = s.shape[1]
		batch_size = tf.shape(s)[0]

		if prev_act is None:
			building = True
			aux = np.zeros((tboostrap,1)) 
			prev_rew = tf.constant(aux, dtype=tf.float32)
			aux = np.zeros((tboostrap,self.n_outputs)) 
			prev_act = tf.constant(aux, dtype=tf.float32)
			if self.text_params is not None:
				if self.Rec_instr:
					text_aux = np.zeros((tboostrap, self.textLength, self.dic_size))
					text = tf.constant(text_aux, dtype=tf.float32)
				else:
					text_aux = np.zeros((tboostrap, self.textLength))
					text = tf.constant(text_aux, dtype=tf.float32)

		prev_rew = tf.reshape(prev_rew, [batch_size, tboostrap, 1])
		prev_act = tf.reshape(prev_act, [batch_size, tboostrap, self.n_outputs])
		
		# hstates= None
		# print("Latent space", s.shape)
		# print("obs", obs.shape)		
		# jsakjsa+=1
		enc_obs = self.cae(s)


		if self.core_net == 'default':
			t_input = tf.reshape(enc_obs, [batch_size, tboostrap, enc_obs.shape[2]*enc_obs.shape[3]*enc_obs.shape[4]])
		else:
			t_input = tf.reshape(enc_obs, [batch_size, tboostrap, enc_obs.shape[2]*enc_obs.shape[3],enc_obs.shape[4]])
		em_instr = None
		if self.text_params is not None:
			if self.Rec_instr:
				text = tf.reshape(text, [batch_size, tboostrap, self.textLength, self.dic_size])
				enc_text = self.tae(text)
				if self.mode == 0:
					t_input = tf.concat([t_input, enc_text],-1)
				elif self.mode == 1:
					em_instr = enc_text
				elif self.mode == 2:
					n_rows = math.ceil(enc_text.shape[-1]/enc_obs.shape[3])
					pad_size = n_rows*enc_obs.shape[3]-enc_text.shape[-1]
					paddings = tf.constant([[0,0], [0,0], [0, pad_size]])
					enc_text = tf.pad(enc_text, paddings, "CONSTANT")
					enc_text = tf.reshape(enc_text, [batch_size, tboostrap, n_rows, enc_obs.shape[3],1])
					enc_text = tf.cast(tf.broadcast_to(enc_text, [batch_size, tboostrap, n_rows, 
													enc_obs.shape[3], enc_obs.shape[4]]),dtype=tf.float32)
					t_input = tf.concat([enc_obs, enc_text],2)
					t_input = tf.reshape(t_input, [batch_size, tboostrap, t_input.shape[2]*t_input.shape[3],t_input.shape[4]])
				else:
					if self.core_net != 'default':
						assert enc_text.shape[-1] %  t_input.shape[-1] == 0
						enc_text = tf.reshape(enc_text, [batch_size, tboostrap, enc_text.shape[-1] //  t_input.shape[-1],
														 t_input.shape[-1]])
						t_input = tf.concat([t_input, enc_text],2)
					else:
						t_input = tf.concat([t_input, enc_text],-1)


			else:
				text = tf.reshape(text, [batch_size, tboostrap, self.textLength])
				if self.mode == 0:
					n_rows = math.ceil(self.textLength/enc_obs.shape[3])
					pad_size = n_rows*enc_obs.shape[3]-self.textLength
					paddings = tf.constant([[0,0], [0,0], [0, pad_size]])
					text = tf.pad(text, paddings, "CONSTANT")
					text = tf.reshape(text, [batch_size, tboostrap, n_rows, enc_obs.shape[3],1])
					text = tf.cast(tf.broadcast_to(text, [batch_size, tboostrap, n_rows, 
													enc_obs.shape[3], enc_obs.shape[4]]),dtype=tf.float32)

					t_input = tf.concat([enc_obs, text],2)
					t_input = tf.reshape(t_input, [batch_size, tboostrap, t_input.shape[2]*t_input.shape[3],t_input.shape[4]])

		logits, values, Nhstates, latent_rep = self.core(t_input, prev_rew, prev_act, hstates, 
												em_instr)

		if self.AE:
			dec_obs = self.img_decoder(latent_rep)
			assert s.shape[2:] == dec_obs.shape[2:]
			if self.text_params is not None:
				dec_text = self.text_decoder(latent_rep)
				assert text.shape[2:] == dec_text.shape[2:]
			else:dec_text = None

		else: 
			dec_obs = None
			dec_text = None

		return tf.squeeze(logits), tf.squeeze(values,-1), tf.squeeze(Nhstates),\
			dec_obs, dec_text 



