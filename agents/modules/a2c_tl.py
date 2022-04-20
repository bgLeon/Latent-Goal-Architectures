from __future__ import absolute_import, division, print_function,\
					   unicode_literals
import tensorflow as tf
import numpy as np
import random, time, os.path, shutil, copy, datetime, sys
# import tensorflow_probability as tfp
from utils.networks import *

np.set_printoptions(threshold=sys.maxsize)

class A2C_tl:
	"""
	A2C with TL specifications
	"""
	def __init__(self, num_actions, num_features, obs_size, pretrained_encoder,
		training_params,  env_params, board, visual, mem = True,
		policy_name='Main'):
		# initialize attributes
		self.num_actions = num_actions
		self.num_features = num_features
		self.training_params = training_params
		self.visual = visual
		self.env_params = env_params
		self.mem = mem
		self.policy_name = policy_name # could be of interest later
		self.n_neurons = training_params.num_neurons 
		self.n_hlayers = training_params.num_layers
		self.multiModal = training_params.multiModal
		self.network = training_params.network
		self.timeSize = training_params.timeSize
		self.rnn_type = training_params.rnn_type
		self.mem_net_type = training_params.mem_net_type
		self.loss_func = training_params.loss
		self.text_mode = training_params.text_mode
		resnet = True if training_params.cnn_type == 'resnet' else False
		if self.network == 'default': self.text_mode=1


		self.IsStart = True
		self.UpdateAE = True
		AE =  training_params.AE

		if training_params.mapType == 'minecraft-insp':
			res = env_params.resolution
			self.offset = 1 if self.env_params.specs_type=='TTL1' else 3
				# input_size = (1,(obs_size+2)*obs_size)
			if visual:
				if mem:
					input_size = (1, self.timeSize, res*(obs_size+self.offset),res*obs_size, 1)
				else:
					input_size = (1, res*(obs_size+self.offset),res*obs_size, 1)
			else:
				if mem:
					input_size = (1, self.timeSize, (obs_size+self.offset)*obs_size)
				else:
					input_size = (1,(obs_size+self.offset)*obs_size)
			self.input_size = input_size
			self.text_params = None
			self.channels = 1
			self.textAE = False
		else:
			self.textAE = True
			image_s = num_features['image'].shape 
			self.text_params = [env_params.vocabSize, env_params.textLength]
			if True:
				self.input_size = (1, self.timeSize, image_s[0], image_s[1], image_s[2])
			else:
				self.input_size = (1, image_s[0], image_s[1], image_s[2])

			self.channels = image_s[2]

		print("input size", self.input_size)
			# self.NModel = RecNet(num_features, num_actions, trainable = True,
			# 					n_neurons = self.n_neurons, n_lstm = 4)
		if self.multiModal:
			# if training_params.mapType=='minigrid':
			self.NModel = AC_AE_MM_Recurrent(num_actions, 
									trainable = True, core_net = self.network,
									Hlstm = mem, visual = visual, 
									n_hlayers = self.n_hlayers, 
									input_size = self.input_size,
									rnn_type= self.rnn_type,
									n_neurons= self.n_neurons,
									text_params = self.text_params,
									mode = training_params.text_mode,
									mem_net = self.mem_net_type,
									channels = self.channels,
									view_size = self.env_params.agent_view_size,
									ttl = self.env_params.specs_type,
									AE = AE, ResNet= resnet,
									mapT = training_params.mapType)

		else:
			self.NModel = AC_AE_Sequential_Recurrent(num_actions, 
									trainable = True,
									Hlstm = mem, visual = visual, 
									n_hlayers = self.n_hlayers, 
									input_size = self.input_size,
									rnn_type = self.rnn_type,
									core_net = self.network,
									n_neurons = self.n_neurons,
									mode = training_params.text_mode,
									mem_net = self.mem_net_type,
									text_params = self.text_params,
									channels = self.channels,
									ttl = self.env_params.specs_type,
									AE = AE, ResNet= resnet,
									mapT = training_params.mapType)
		self.AE = self.NModel.AE

		# self.NModel.cae.trainable = False

		if pretrained_encoder:
			#path to pretrained encoder
			if training_params.mapType == 'minecraft-insp':
				if self.env_params.specs_type=='TTL1':
					path = '' # give here the path to your pretrained CNN

					if self.multiModal:
						# if training_params.mapType=='minigrid':
						self.AuxModel= AC_AE_MM_Recurrent(num_actions, 
												trainable = True, core_net = self.network,
												Hlstm = mem, visual = visual, 
												n_hlayers = self.n_hlayers, 
												input_size = self.input_size,
												rnn_type= self.rnn_type,
												n_neurons= self.n_neurons,
												text_params = self.text_params,
												mode = training_params.text_mode,
												mem_net = self.mem_net_type,
												channels = self.channels,
												view_size = self.env_params.agent_view_size,
												ttl = self.env_params.specs_type,
												AE = AE, ResNet= resnet,
												mapT = training_params.mapType)

					else:
						self.AuxModel = AC_AE_Sequential_Recurrent(num_actions, 
												trainable = True,
												Hlstm = mem, visual = visual, 
												n_hlayers = self.n_hlayers, 
												input_size = self.input_size,
												rnn_type = self.rnn_type,
												core_net = self.network,
												n_neurons = self.n_neurons,
												mode = training_params.text_mode,
												mem_net = self.mem_net_type,
												text_params = self.text_params,
												channels = self.channels,
												ttl = self.env_params.specs_type,
												AE = AE, ResNet= resnet,
												mapT = training_params.mapType)

					if os.path.exists(path):
						print("Loading a ConvLay and freezing: ", seedN)
						self.AuxModel.load_weights(path)
						self.NModel.cae = self.AuxModel.cae
						self.NModel.cae.trainable = False 
						print("loaded")
					else:
						self.NModel.cae.trainable = False  
						print("No model found, using random encoder")	

				else:
					path = './Saved_Models/3L_Encoder/TTL2_Aff/checkpoint'
					aux_input_size = (1, res*(obs_size+3),res*obs_size)
					if self.multiModal:
						# if training_params.mapType=='minigrid':
						self.AuxModel= AC_AE_MM_Recurrent(num_actions, 
												trainable = True, core_net = self.network,
												Hlstm = mem, visual = visual, 
												n_hlayers = self.n_hlayers, 
												input_size = self.input_size,
												rnn_type= self.rnn_type,
												n_neurons= self.n_neurons,
												text_params = self.text_params,
												mode = training_params.text_mode,
												channels = self.channels,
												view_size = self.env_params.agent_view_size,
												ttl = self.env_params.specs_type,
												AE = AE, ResNet= resnet)

					else:
						self.AuxModel = AC_AE_Sequential_Recurrent(num_actions, 
												trainable = True,
												Hlstm = mem, visual = visual, 
												n_hlayers = self.n_hlayers, 
												input_size = self.input_size,
												rnn_type = self.rnn_type,
												core_net = self.network,
												n_neurons = self.n_neurons,
												mode = training_params.text_mode,
												text_params = self.text_params,
												channels = self.channels,
												ttl = self.env_params.specs_type,
												AE = AE, ResNet= resnet)
					if os.path.exists(path):
						print("Loading a ConvLay and freezing: ", seedN)
						self.AuxModel.load_weights(path)
						self.NModel.cae = self.AuxModel.cae
						self.NModel.cae.trainable = False 
						print("loaded")
					else:
						self.NModel.cae.trainable = False  
						print("No model found, using random encoder")
			else:
				raise Exception('no pretrained encoder for this map')

			# self.AuxModel.load_weights('./Saved_Models/ConvLayer_Only/'\
			# 			+'RandomLayer/Glorot_uniform/checkpoint')
			# print("using NOT pre-trained layer")
			print("frozen") 
		self.NModel.run_eagerly = True
		self.NModel.build(self.input_size)
		self.NModel.summary()

		
		print("Alg: A2C-tl")
		# Creating the experience replay buffer
		self.batch_size = training_params.batch_size
		self.replay_buffer = ReplayBuffer(visual)
		self.lr = self.training_params.lr
		self.core_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
		if self.AE:
			self.imgAE_optimizer = tf.keras.optimizers.Adam(1e-3,epsilon=1e-8) #1e-4, 1e-5
			if self.textAE: self.textAE_optimizer = tf.keras.optimizers.Adam(1e-3, epsilon=1e-8)
		self.gamma = self.training_params.gamma
		# count of the number of environmental steps
		self.step = 0
		self.counter = 0
		self.text_recB = np.empty((100)); self.text_recB.fill(0.8)
		self.img_recB = np.empty((100)); self.img_recB.fill(0.8)
		self.rl_loss = np.empty((100)); self.rl_loss.fill(0.8)
		self.entropy_loss = np.empty((100)); self.entropy_loss.fill(0.8)
		
	def save_transition(self, s, act, reward, value, logit, done, prev_act, 
							prev_rew, hstates, mission):
		self.replay_buffer.add(s, act, reward, value, logit, float(done), 
								prev_act, prev_rew, hstates, mission)

	def shuffle_in_unison(self, a, b, c, d, e):
		n_elem = a.shape[0]
		indeces = np.random.permutation(n_elem)
		return a[indeces], b[indeces], c[indeces], d[indeces], e[indeces]


	@tf.function(experimental_relax_shapes=True)
	def learn(self, observations, Qvals, a_idxs, prev_acts, prev_rews, hstates, text_input):
		if text_input is not None and self.text_mode>0:
			check = tf.zeros_like(text_input)
			hidden_instr = False
			text_input = tf.one_hot(text_input, depth=self.text_params[0])
		observations = tf.math.divide(observations,255)
		loss_func = tf.keras.losses.SparseCategoricalCrossentropy() if self.loss_func == 'cce' else\
			tf.keras.losses.MeanSquaredError()
		# self.NModel.cae.Decoder.trainable = False
		if self.AE and self.textAE:
			with tf.device("/device:GPU:0"):
				with tf.GradientTape() as tape:
					logits, values, _, dec_obs, dec_text = self.NModel.call(observations, prev_acts, prev_rews, 
															hstates, text_input)
					# Img reconstruction
					observations = tf.clip_by_value(observations, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)
					# tf.cast(x, dtype)

					dec_obs = tf.clip_by_value(dec_obs, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)


					Lleft = tf.multiply(-observations,tf.math.log(dec_obs))
					Lright = tf.multiply(tf.subtract(1.0,observations), tf.math.log(1-dec_obs))


					self.img_rec_loss = tf.reduce_mean(tf.math.subtract(Lleft,Lright),[1, 2, 3 ,4]) 
					self.img_rec_loss = tf.reduce_mean(self.img_rec_loss)

					

					text_input = tf.clip_by_value(text_input, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)

					dec_text = tf.clip_by_value(dec_text, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)

					Lleft = tf.multiply(-text_input,tf.math.log(dec_text))
					Lright = tf.multiply(tf.subtract(1.0,text_input), tf.math.log(1-dec_text))

					self.text_rec_loss = tf.reduce_sum(tf.math.subtract(Lleft,Lright),[1, 2, 3])
					self.text_rec_loss = tf.reduce_mean(self.text_rec_loss)


					# Reinforcement learning (core)
					values = tf.reshape(values,[tf.shape(values)[0],1])

					advantage = tf.subtract(Qvals, values)
					self.value_loss = self.training_params.value_loss_w*tf.reduce_mean(\
											tf.math.pow(advantage, 2))
					action_mask = tf.one_hot(indices=a_idxs, depth=self.num_actions,
						dtype=tf.float32)
					clipped_logits = tf.clip_by_value(logits, 
													clip_value_min=10**-21,
													clip_value_max=1)

					log_logits = tf.math.log(clipped_logits)

					check =  tf.reduce_mean(log_logits)


					log_probs=tf.multiply(log_logits, action_mask)

					self.actor_loss = tf.reduce_mean(-log_probs * advantage)

					entropy = -tf.reduce_sum(tf.reduce_mean(logits) * log_logits)

					self.entropy_loss = self.training_params.entropy*entropy

					self.ac_loss = self.value_loss + self.actor_loss +\
											self.entropy_loss

					self.total_loss = self.ac_loss + self.img_rec_loss\
											+ 0.3*self.text_rec_loss
				
				gradients = tape.gradient(self.total_loss, self.NModel.trainable_variables)

				gradients, _ = tf.clip_by_global_norm(gradients, 1) #1.5 /1

				self.core_optimizer.apply_gradients(\
									zip(gradients, self.NModel.trainable_variables))

				return self.img_rec_loss, self.text_rec_loss, self.ac_loss, self.entropy_loss, dec_obs, dec_text

				# tf.print('self', self)

		elif self.AE:
			with tf.device("/device:GPU:0"):
				with tf.GradientTape() as tape:
					logits, values, _, dec_obs, _ = self.NModel.call(observations, prev_acts, prev_rews, 
															hstates, text_input)
					# Img reconstruction
					observations = tf.clip_by_value(observations, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)
					# tf.cast(x, dtype)

					dec_obs = tf.clip_by_value(dec_obs, 
											clip_value_min=1e-7,
											clip_value_max=1-1e-7)


					Lleft = tf.multiply(-observations,tf.math.log(dec_obs))
					Lright = tf.multiply(tf.subtract(1.0,observations), tf.math.log(1-dec_obs))


					self.img_rec_loss = tf.reduce_mean(tf.math.subtract(Lleft,Lright),[1, 2, 3 ,4]) 
					self.img_rec_loss = tf.reduce_mean(self.img_rec_loss)

					# Reinforcement learning (core)
					values = tf.reshape(values,[tf.shape(values)[0],1])

					advantage = tf.subtract(Qvals, values)
					self.value_loss = self.training_params.value_loss_w*tf.reduce_mean(\
											tf.math.pow(advantage, 2))
					action_mask = tf.one_hot(indices=a_idxs, depth=self.num_actions,
						dtype=tf.float32)
					clipped_logits = tf.clip_by_value(logits, 
													clip_value_min=10**-21,
													clip_value_max=1)

					log_logits = tf.math.log(clipped_logits)

					check =  tf.reduce_mean(log_logits)


					log_probs=tf.multiply(log_logits, action_mask)

					self.actor_loss = tf.reduce_mean(-log_probs * advantage)

					entropy = -tf.reduce_sum(tf.reduce_mean(logits) * log_logits)

					self.entropy_loss = self.training_params.entropy*entropy

					self.ac_loss = self.value_loss + self.actor_loss +\
											self.entropy_loss

					self.total_loss = self.ac_loss + self.img_rec_loss
				
				gradients = tape.gradient(self.total_loss, self.NModel.trainable_variables)

				gradients, _ = tf.clip_by_global_norm(gradients, 1) #1.5 /1

				self.core_optimizer.apply_gradients(\
									zip(gradients, self.NModel.trainable_variables))

				return self.img_rec_loss, self.text_rec_loss, self.ac_loss, self.entropy_loss, dec_obs, dec_text
		else:
			with tf.device("/device:GPU:0"):
				with tf.GradientTape() as tape:
					logits, values,*_ = self.NModel.call(observations, prev_acts, prev_rews, 
															hstates, text_input)
					# To keep in mind for future recurrent approaches
					# 	logits, values = self.NModel.call(observations, action_mask, 
															# rewards)
					# print("values pre", values)
					values = tf.reshape(values,[tf.shape(values)[0],1])
					# print("values post", values)
					advantage = tf.subtract(Qvals, values)
					self.value_loss = self.training_params.value_loss_w*tf.reduce_mean(\
											tf.math.pow(advantage, 2))
					action_mask = tf.one_hot(indices=a_idxs, depth=self.num_actions,
						dtype=tf.float32)
					clipped_logits = tf.clip_by_value(logits, 
													clip_value_min=10**-21,
													clip_value_max=1)

					log_logits = tf.math.log(clipped_logits)

					log_probs=tf.multiply(log_logits, action_mask)

					self.actor_loss = tf.reduce_mean(-log_probs * advantage)

					entropy = -tf.reduce_sum(tf.reduce_mean(logits) * log_logits)

					self.entropy_loss = self.training_params.entropy*entropy

					# self.entropy_loss = -self.training_params.entropy*\
					# 				tf.reduce_mean(dists.entropy())

					# tf.print('self.entropy_loss', self.entropy_loss)
					# tf.print('self.actor_loss', self.actor_loss)

					self.total_loss =  self.value_loss + self.actor_loss +\
											self.entropy_loss

				gradients = tape.gradient(self.total_loss,
											self.NModel.trainable_variables)
				self.core_optimizer.apply_gradients(\
									zip(gradients, self.NModel.trainable_variables))

			return None, None, None, None, None, None


	def learn_step(self, next_value):
		"""
		Minimize the error in Bellman's equation on a batch sampled from
		replay buffer.
		"""
		dones =  np.array(self.replay_buffer.dones)
		f_Qvals = np.zeros_like(np.array(self.replay_buffer.values))
		rewards = self.replay_buffer.rewards
		gamma = self.training_params.gamma
		num_minibatches = self.training_params.num_minibatches
		values = self.replay_buffer.values
		values.append(next_value)
		last_return = next_value
		ft_hstates = self.replay_buffer.hstates
		

		# for i in reversed(range(len(f_Qvals))):
		# 	if dones[i]:discounted_reward = 0
		# 	discounted_reward = rewards[i] + gamma*discounted_reward 
		# 	f_Qvals[i] = discounted_reward

		# Generalized Advantage Estimation
		trace_decay = 0.97
		# last_return = values[len(f_Qvals)] 
		for i in reversed(range(len(f_Qvals))):
			bootstrap = (
				(1 - trace_decay) * values[i] + trace_decay * last_return)
			bootstrap *= (1 - dones[i])
			f_Qvals[i] = last_return = rewards[i] + gamma * bootstrap
		# 	bootstrap += dones[i] * next_values[t]

		ft_observations = np.array(self.replay_buffer.observations)
		ft_prev_act = np.array(self.replay_buffer.prev_act)
		ft_prev_rew = np.array(self.replay_buffer.rewards)
		if self.textAE: ft_missions = np.array(self.replay_buffer.missions)
		# if self.mem:
		if True:
			batch_size=len(self.replay_buffer.observations)
			 

			# origk = k #control to not get beyond indeces
			# if k<self.timeSize:k=self.timeSize
			# total = k
			# print("It should enter now")
			# nt = False
			if self.timeSize == 1:
				f_observations = np.expand_dims(ft_observations,1)
				f_prev_act = np.expand_dims(ft_prev_act,1)
				f_prev_rew = np.expand_dims(ft_prev_rew,1)
				if self.textAE: f_missions = np.expand_dims(ft_missions,1)
			else:
				if self.visual:
					f_observations = np.zeros((batch_size, self.timeSize, 
											self.input_size[2], self.input_size[3], self.channels)) 
				else:
					f_observations = np.zeros((batch_size, self.timeSize, 
											self.input_size[2])) 
				f_prev_act = np.zeros((batch_size, self.timeSize, 
											self.num_actions)) 
				f_prev_rew = np.zeros((batch_size, self.timeSize, 
											1)) 
				if self.textAE: f_missions = np.zeros((batch_size, self.timeSize, 
											self.text_params[1]))
				for i in reversed(range(batch_size)):
					k=i
					for t in reversed(range(self.timeSize)):
						if k < 0 or (dones[k] == 1 and t != self.timeSize-1): 
						# if k < 0 or (dones[k] == 1 and k != batch_size-1):
							# if not k <0: nt= True
							break
						# print("Ok")
						f_observations[i][t] = ft_observations[k]
						f_prev_act[i][t] = ft_prev_act[k]
						f_prev_rew[i][t] = ft_prev_rew[k]
						if self.textAE: f_missions[i][t] = ft_missions[k]
						k-=1

		else: 
			f_observations = ft_observations
		f_a_idxs = np.array(self.replay_buffer.a_idx)

		mbatch_s = int(batch_size/num_minibatches)
		# print("mbatch_s", mbatch_s)
		for i in range(num_minibatches):
			observations = f_observations[mbatch_s*i:mbatch_s*(i+1)]
			Qvals = f_Qvals[mbatch_s*i:mbatch_s*(i+1)]
			a_idxs = f_a_idxs[mbatch_s*i:mbatch_s*(i+1)]
			prev_rew = f_prev_rew[mbatch_s*i:mbatch_s*(i+1)]
			prev_act = f_prev_act[mbatch_s*i:mbatch_s*(i+1)]
			hstates = ft_hstates[mbatch_s*i:mbatch_s*(i+1)]


			if self.training_params.mapType == 'minigrid':
				missions = f_missions[mbatch_s*i:mbatch_s*(i+1)]



				missions = tf.convert_to_tensor(missions, dtype=tf.int32)
				
			else: missions = None

			img_rec_loss, text_rec_loss, rl_loss, entropy_loss, dec_obs, dec_text = self.learn(
						tf.convert_to_tensor(observations, dtype=tf.float32),
						tf.convert_to_tensor(Qvals, dtype=tf.float32),
						tf.convert_to_tensor(a_idxs),
						tf.convert_to_tensor(prev_act, dtype=tf.float32),
						tf.convert_to_tensor(prev_rew, dtype=tf.float32),
						tf.convert_to_tensor(hstates, dtype=tf.float32),
						missions)

		if self.NModel.AE and False:
			self.counter+=1
			self.text_recB[self.counter%100] = text_rec_loss
			self.img_recB[self.counter%100] =  img_rec_loss
			self.rl_loss[self.counter%100] =  rl_loss
			if self.counter % 20 == 0:
				print('self.img_rec_loss', np.average(self.img_recB))
				print('self.text_rec_loss', np.average(self.text_recB))
				print('RL loss ', np.average(self.rl_loss))
				if self.counter % 40 == 0:
					print(observations.shape)
					print(dec_obs.shape)
					if len(dec_obs.shape) == 5 and len(observations.shape) == 5:
						try:
							print('mission', missions[0])
							print('dec_text', tf.math.argmax(dec_text, axis =-1)[0])
							print('img', observations[0][0][15][10])
							print('dec_img', dec_obs[0][0][15][10]*255)
						except:
							print("Error with reward batch!")
							print('observations', observations.shape)
							print('dec_obs', dec_obs.shape)

	@tf.function
	def get_action_value(self, s, action_mask, reward, ohstates, mission):
		with tf.device("/device:GPU:0"):
			if mission is not None and self.text_mode>0:
				mission = tf.one_hot(mission, depth=self.text_params[0])

			s = tf.math.divide(s,255)
			logits, value, hstates, *_, dec_obs, dec_text = self.NModel.call(
						s, action_mask, reward, ohstates, mission)

		return logits, value, hstates

	def get_action_value_Np(self, s, prev_action_mask, reward, hstates,
							mission = None):
		if True:
			tmissions = mission
			tback = self.timeSize -1

			if self.visual:
				ts = np.zeros((1, self.timeSize, 
								self.input_size[2], self.input_size[3], self.channels))
				if len(s.shape) == 2 : s = np.expand_dims(s,-1)
			else:
				ts = np.zeros((1, self.timeSize, 
										self.input_size[2])) 
			tam = np.zeros((1, self.timeSize, self.num_actions))
			trew = np.zeros((1, self.timeSize, 1))
			if mission is not None:
				tmissions = np.zeros((1, self.timeSize, self.text_params[1]))
				t_missions = self.replay_buffer.missions
				tmissions[0][tback] = mission
			t_obs = self.replay_buffer.observations
			t_am =  self.replay_buffer.prev_act
			t_rew =  self.replay_buffer.prev_rew
			dones = self.replay_buffer.dones
	

			ts[0][tback] = s
			tam[0][tback] = prev_action_mask
			trew[0][tback] = reward

			k = len(t_obs)-1
			if t_obs:
				for t in reversed(range(tback)):
					if k < 0 or dones[k] == 1: break
					ts[0][t] = t_obs[k]
					tam[0][t] = t_am[k]
					trew[0][t] = t_rew[k]
					if mission is not None: tmissions[0][t] = t_missions[k]
		else:
			ts = s
			tam =  prev_action_mask
			trew = reward
			tmissions = mission

		if hstates == None: pass
		else: 
			hstates=tf.convert_to_tensor(hstates, dtype=tf.float32)
			# jaja+=1

		if mission is not None: tmissions=tf.convert_to_tensor(tmissions, dtype=tf.int32)

		logits, value, hstates = self.get_action_value(
			tf.convert_to_tensor(ts, dtype=tf.float32),
			tf.convert_to_tensor(tam, dtype=tf.float32),
			tf.convert_to_tensor(trew, dtype=tf.float32),
			hstates, tmissions)
		return np.array(logits), np.array(value), logits, hstates

	def get_steps(self):
		return self.step

	def add_step(self):
		self.step += 1

class ReplayBuffer(object):
	def __init__(self, visual = False):
		self.observations = []
		self.a_idx = []
		self.rewards = []
		self.values = []
		self.logits = []
		self.dones = []
		self.prev_act = []
		self.prev_rew = []
		self.hstates = []
		self.missions = []
		self.size = 0
		self.prev_h = 0
		self.visual = visual

	def add(self, s, a, r, v, logit, done, prev_act, prev_rew, hstate, mission):
		if not self.visual:
			num_features = s.shape[0]*s.shape[1]
			self.observations.append(s.reshape((1, num_features)))
		else:
			self.observations.append(s)
		self.a_idx.append(a)
		self.rewards.append(r)
		self.values.append(v)
		self.logits.append(logit)
		self.dones.append(done)
		self.prev_act.append(prev_act)
		self.prev_rew.append(prev_rew)
		self.missions.append(mission)

		if len(self.hstates) == 0: self.prev_h = tf.zeros_like(hstate)
		self.hstates.append(self.prev_h)
		if done: self.prev_h = tf.zeros_like(hstate)
		else: self.prev_h = hstate
		self.size+=1

	def clear(self):
		self.observations = []
		self.a_idx = []
		self.rewards = []
		self.values = []
		self.logits = []
		self.dones = []
		self.prev_act = []
		self.prev_rew = []
		self.mission = []
		self.size = 0
		self.hstates = []
		self.prev_h = 0