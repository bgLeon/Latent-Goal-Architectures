from __future__ import absolute_import, division, print_function,\
					   unicode_literals
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import random, time, os.path, shutil, copy, datetime, sys
# import tensorflow_probability as tfp
from utils.networks import *

class PPO_tl:
	"""
	PPO with TL specifications
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
		self.mem_type = training_params.mem_type
		self.loss_func = training_params.loss


		self.IsStart = True
		self.UpdateAE = True

		if training_params.mapType == 'minecraft-insp':
			res = env_params.resolution
			self.offset = 1 if self.env_params.specs_type=='TTL1' else 3
				# input_size = (1,(obs_size+2)*obs_size)
			if visual:
				if mem:
					input_size = (None, self.timeSize, res*(obs_size+self.offset),res*obs_size, 1)
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
			if mem:
				self.input_size = (1, self.timeSize, image_s[0], image_s[1], image_s[2])
			else:
				self.input_size = (1, image_s[0], image_s[1], image_s[2])

			self.channels = image_s[2]

		print("input size", self.input_size)
			# self.NModel = RecNet(num_features, num_actions, trainable = True,
			# 					n_neurons = self.n_neurons, n_lstm = 4)
		if self.multiModal:
			if self.network == 'default':
				self.NModel = AC_FCLSTM_MM(num_actions, trainable = True,  
										n_hlayers = self.n_hlayers, 
										input_size = input_size,
										visual = visual,
										n_neurons = self.n_neurons)
			elif self.network == 'PrediNet':

				# self.NModel = AC_MM_PrediNet(num_actions, 
				# 						trainable = True,
				# 						Hlstm = mem, visual = visual, 
				# 						n_hlayers = self.n_hlayers, 
				# 						conv_out_size = input_size,
				# 						n_neurons = self.n_neurons)
				# self.NModel = AC_MMRIM_PrediNet(num_actions, 
				# 						trainable = True,
				# 						Hlstm = mem, visual = visual, 
				# 						n_hlayers = self.n_hlayers, 
				# 						conv_out_size = input_size,
				# 						rnn_type= self.rnn_type,
				# 						n_BrimsN = self.n_neurons)
				# self.NModel = AC_MMBRIM_PrediNet(num_actions, 
				# 						trainable = True,
				# 						Hlstm = mem, visual = visual, 
				# 						n_hlayers = self.n_hlayers, 
				# 						conv_out_size = input_size,
				# 						rnn_type= self.rnn_type,
				# 						n_BrimsN = self.n_neurons)
				# self.NModel = AC_MMBRIM_V_PrediNet(num_actions, 
				# 						trainable = True,
				# 						Hlstm = mem, visual = visual, 
				# 						n_hlayers = self.n_hlayers, 
				# 						input_size = self.input_size,
				# 						rnn_type= self.rnn_type,
				# 						n_BrimsN = self.n_neurons,
				# 						text_params = self.text_params,
				# 						channels = self.channels)

				self.NModel = AC_AE_MM_PrediNet(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										input_size = self.input_size,
										rnn_type= self.rnn_type,
										n_BrimsN = self.n_neurons,
										text_params = self.text_params,
										channels = self.channels,
										mem_net = self.mem_type,
										view_size = self.env_params.agent_view_size,
										ttl = self.env_params.specs_type)
			elif self.network == 'SelfAttentionNet':
				self.NModel = AC_MM_MHA(num_actions, 
											trainable = True,  
											n_hlayers = self.n_hlayers, 
											conv_out_size = self.input_size,
											Hlstm = mem, visual = visual,
											n_neurons = self.n_neurons)
			elif self.network == 'PN_MHA':
				self.NModel = AC_TPN_NMHA_MM(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										conv_out_size = input_size,
										n_neurons = self.n_neurons)
			elif self.network == 'MHA_PN':
				self.NModel = AC_TMHA_NPN_MM(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										conv_out_size = input_size,
										n_neurons = self.n_neurons)
			else:
				self.NModel = AC_RelationNet_MM(num_actions, 
											trainable = True,  
											n_hlayers = self.n_hlayers, 
											conv_out_size = input_size,
											Hlstm = mem, visual = visual,
											n_neurons = self.n_neurons)

		else:
			if self.network == 'default':
				# self.NModel = AC_FCLSTM(num_actions, trainable = True,  
				# 							n_hlayers = self.n_hlayers, 
				# 							n_neurons = self.n_neurons,
				# 							visual = visual)

				# self.NModel = AC_FCBRIM(num_actions, trainable = True,  
				# 							n_hlayers = self.n_hlayers, 
				# 							n_BrimsN = self.n_neurons,
				# 							visual = visual)
				self.NModel = AC_AE_DenseNet_LSTM(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										input_size = self.input_size,
										rnn_type= self.rnn_type,
										n_neurons = self.n_neurons,
										text_params = self.text_params,
										channels = self.channels,
										ttl = self.env_params.specs_type)
				# self.NModel = AC_FCBRIM_test(num_actions, trainable = True,  
				# 							n_hlayers = self.n_hlayers, 
				# 							n_BrimsN = self.n_neurons,
				# 							visual = visual)
				# self.NModel = AC_FCLSTM_2(num_actions, trainable = True,  
				# 							n_hlayers = self.n_hlayers, 
				# 							n_neurons = self.n_neurons,
				# 							visual = visual)
			elif self.network == 'PrediNet':
				# self.NModel = AC_PrediNet(num_actions, 
				# 						trainable = True,  
				# 						n_hlayers = self.n_hlayers, 
				# 						conv_out_size = input_size,
				# 						Hlstm = mem, visual = visual,
				# 						n_neurons = self.n_neurons)
				self.NModel = AC_BRIM_PrediNet(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										conv_out_size = input_size,
										rnn_type= self.rnn_type,
										n_BrimsN = self.n_neurons)
				

			elif self.network == 'SelfAttentionNet':

				self.NModel = AC_MHA(num_actions, 
										trainable = True,  
										n_hlayers = self.n_hlayers, 
										conv_out_size = input_size,
										Hlstm = mem, visual = visual,
										n_neurons = self.n_neurons)

			elif self.network == 'RelationNet':
				self.NModel = AC_RelationNet(num_actions, 
										trainable = True,  
										n_hlayers = self.n_hlayers, 
										conv_out_size = input_size,
										Hlstm = mem, visual = visual,
										n_neurons = self.n_neurons)

		self.AE = self.NModel.AE

		# self.NModel.cae.trainable = False
		# print ('Model', self.NModel.count_params())
		# print ('Cnn', self.NModel.cae.count_params())
		# print ('PrediNetT', self.NModel.core.PrediNetT.count_params())
		# print ('PrediNetN', self.NModel.core.PrediNetN.count_params())


		if pretrained_encoder:
			#path to pretrained encoder
			if training_params.mapType == 'minecraft-insp':
				if self.env_params.specs_type=='TTL1':
					seedB = self.env_params.seedB
					seedN = str(seedB) if seedB>=5 and seedB<=7 else '5'
					# path = './Checkpoints/Neurips21/minecraft-insp_/Spec_TTL1/PretrainA2C/Used/'\
					# +'V5_pretrain/seedB'+seedN+'/checkpoint'

					path = './Checkpoints/Neurips21/minecraft-insp_/Spec_TTL1/PretrainA2C/Used/'\
					+'V5_3_2_pretrain/seedB'+seedN+'/checkpoint'

					# path = './Checkpoints/Neurips21/minecraft-insp_/Spec_TTL1/PretrainA2C/Used/'\
					# +'V6_2_2_pretrain/seedB'+seedN+'/checkpoint'

					self.AuxModel = AC_AE_DenseNet_LSTM(num_actions, 
										trainable = True,
										Hlstm = mem, visual = visual, 
										n_hlayers = self.n_hlayers, 
										input_size = self.input_size,
										rnn_type= self.rnn_type,
										n_neurons = self.n_neurons,
										text_params = self.text_params,
										channels = self.channels,
										ttl = self.env_params.specs_type)
					if os.path.exists(path):
						print("Loading a ConvLay and freezing: ", seedN)
						self.AuxModel.load_weights(path)
						self.NModel.cae = self.AuxModel.cae
						self.NModel.cae.trainable = False 
						print("loaded")
					else: print("No model found, using random encoder")	

				else:
					path = './Saved_Models/3L_Encoder/TTL2_Aff/checkpoint'
					aux_input_size = (1, res*(obs_size+3),res*obs_size)
					self.AuxModel = AC_PrediNet_MM3(4, 
													trainable = True,
													Hlstm = True, visual = True, 
													n_hlayers = 1, 
													conv_out_size = aux_input_size,
													n_neurons = 16)
					if os.path.exists(path):
						print("Loading a ConvLay and freezing")
						self.AuxModel.load_weights(path)
						# self.NModel.convA = self.AuxModel.convA
						# self.NModel.convB = self.AuxModel.convB
						# self.NModel.convC = self.AuxModel.convC

						# self.NModel.convA.trainable = False
						# self.NModel.convB.trainable = False
						# self.NModel.convC.trainable = False 

						self.NModel.cae.conv['0'] = self.AuxModel.convA
						self.NModel.cae.conv['1'] = self.AuxModel.convB
						self.NModel.cae.conv['2'] = self.AuxModel.convC

						self.NModel.cae.conv['0'].trainable = False
						self.NModel.cae.conv['1'].trainable = False
						self.NModel.cae.conv['2'].trainable = False 
						print("loaded")

				
					else: print("No model found, using random encoder")
			else:
				raise Exception('no pretrained encoder for this map')

			# self.AuxModel.load_weights('./Saved_Models/ConvLayer_Only/'\
			# 			+'RandomLayer/Glorot_uniform/checkpoint')
			# print("using NOT pre-trained layer")
			print("frozen") 
		self.NModel.run_eagerly = True
		self.NModel.build(self.input_size)
		self.NModel.summary()
		# hrue+=1
		# print('cae', self.NModel.core.PrediNetN.get_keys.variables)
		


		# print('oldcae',  self.NModel.get_keysN.variables)

		# ajajajaja+=1
		print("Alg: PPO")
		# Creating the experience replay buffer
		self.batch_size = training_params.batch_size
		self.replay_buffer = ReplayBuffer(visual)
		self.lr = self.training_params.lr
		self.core_optimizer = tf.optimizers.Adam(learning_rate=self.lr)
		if self.AE:
			self.imgAE_optimizer = tf.keras.optimizers.Adam(self.lr) #1e-4, 1e-5
			if self.textAE: self.textAE_optimizer = tf.keras.optimizers.Adam(self.lr)
		self.gamma = self.training_params.gamma
		# count of the number of environmental steps
		self.step = 0
		
	def save_transition(self, s, act, reward, value, logit, done, prev_act,
						prev_rew, hstates, mission):
		self.replay_buffer.add(s, act, reward, value, logit, float(done),
								prev_act, prev_rew, hstates, mission)

	def shuffle_in_unison(self, a, b, c, d, e, f, g):
		n_elem = a.shape[0]
		indeces = np.random.permutation(n_elem)
		return a[indeces], b[indeces], c[indeces], d[indeces], e[indeces], f[indeces], g[indeces]

	@tf.function(experimental_relax_shapes=True)
	def learn(self, observations, Qvals, a_idxs, prev_acts, prev_rews, old_logprobs, old_vpred, hstates, missions):
		"""
		Minimize the error in Bellman's equation on a batch sampled from
		replay buffer.
		"""
		# dones =  np.array(self.replay_buffer.dones)
		# f_Qvals = np.zeros_like(np.array(self.replay_buffer.values))
		# f_logits = np.array(self.replay_buffer.logits)
		# f_values = np.array(self.replay_buffer.values)
		# rewards = self.replay_buffer.rewards
		# discounted_reward = 0
		# gamma = self.training_params.gamma
		# eps_clip = self.training_params.ratio_clip
		# num_minibatches = self.training_params.num_minibatches
		# # print("Rewards", Qvals.shape)
		# # print("len", len(Qvals))
		# for i in reversed(range(len(f_Qvals))):
		# 	if dones[i]:
		# 		discounted_reward = 0
		# 	discounted_reward = rewards[i] + gamma*discounted_reward 
		# 	f_Qvals[i] = discounted_reward
		

		# f_observations = np.array(self.replay_buffer.observations)
		# f_a_idxs = np.array(self.replay_buffer.a_idx)
		# # print("a_idxs", a_idxs)
		# f_observations, f_Qvals, f_a_idxs, f_logits, f_values = self.shuffle_in_unison(f_observations, f_Qvals,
		# 												f_a_idxs, f_logits, f_values)

		# mbatch_s = int(len(f_observations)/num_minibatches)
		# # print("mbatch_s", mbatch_s)
		# for i in range(num_minibatches):
		# 	observations = f_observations[mbatch_s*i:mbatch_s*(i+1)]
		# 	Qvals = f_Qvals[mbatch_s*i:mbatch_s*(i+1)]
		# 	a_idxs = f_a_idxs[mbatch_s*i:mbatch_s*(i+1)]
		# 	old_logprobs = f_logits[mbatch_s*i:mbatch_s*(i+1)]
		# 	old_vpred = f_values[mbatch_s*i:mbatch_s*(i+1)]

			# print("New Qvals", Qvals)
			# print("New observations", observations)
			# print("New a_idxs", a_idxs)
		eps_clip = self.training_params.ratio_clip
		with tf.GradientTape() as tape:
			# print('observations', observations.shape)
			# print('prev_acts', prev_acts.shape)
			# print('prev_rews', prev_rews.shape)
			# print('hstates', hstates.shape)
			# print('missions', missions.shape)
			logits, values,*_ = self.NModel.call(observations, prev_acts, prev_rews, hstates, missions)
			# print('logits',logits)
			dists = tfp.distributions.Categorical(logits)
			action_logprobs = dists.log_prob(a_idxs)

			# print('action_logprobs',action_logprobs)
			# print('dist_entropy',dist_entropy)
			# Finding the ratio (pi_theta / pi_theta__old):
			ratios = tf.exp(tf.subtract(action_logprobs, old_logprobs))

			# Finding Surrogate Loss:
			# print('Values pre', values.shape)
			# advantages = tf.subtract(Qvals, values)
			# print('advantages', advantages)
			values = tf.reshape(values,[tf.shape(values)[0],1])
			advantages = tf.subtract(Qvals, values)
			# print('Values post', values.shape)
			# print('advantages', advantages)
			# jajaja+=1

			# jajaja+=1

			vpredclipped = old_vpred  + tf.clip_by_value(\
				tf.subtract(values, old_vpred), -eps_clip, eps_clip)

			vf_losses1 = tf.square(advantages)
			vf_losses2 = tf.square(vpredclipped)
			# print('vf_losses1', vf_losses1.shape)
			# print('vf_losses2', vf_losses2.shape)
			# jajaja+=1

			# print('vf_losses1', vf_losses1)
			# print('vf_losses2', vf_losses2)
			self.value_loss = self.training_params.value_loss_w*tf.reduce_mean(\
									tf.maximum(vf_losses1, vf_losses2))
			# print('self.value_loss', self.value_loss)

			# self.old_value_loss = self.training_params.value_loss_w*tf.reduce_mean(\
			# 						vf_losses1)

			surr1 = tf.multiply(ratios,advantages)

			surr2 = tf.multiply(tf.clip_by_value(ratios, 
									1-eps_clip, 1+eps_clip), advantages)

			clipped_logits = tf.clip_by_value(logits, 
											clip_value_min=10**-21,
											clip_value_max=1)

			log_logits = tf.math.log(clipped_logits)

			# entropy = tf.reduce_sum(tf.reduce_mean(logits) * log_logits)

			# self.entropy_loss = self.training_params.entropy*entropy
			# print('self.entropy_loss', self.entropy_loss)

			self.entropy_loss = self.training_params.entropy*\
							tf.reduce_mean(dists.entropy())

			self.actor_loss = tf.reduce_mean(tf.minimum(surr1,surr2))

			# tf.print('self.entropy_loss', self.entropy_loss)
			# tf.print('self.actor_loss', self.actor_loss)
			
			self.total_loss = -(self.actor_loss - self.value_loss + self.entropy_loss)
			# self.old_total_loss = -(self.actor_loss - self.old_value_loss + self.entropy_loss)									
			# kaka+=1

		gradients = tape.gradient(self.total_loss,
									self.NModel.trainable_variables)
		self.core_optimizer.apply_gradients(\
							zip(gradients, self.NModel.trainable_variables))

		kl = tf.reduce_mean(old_logprobs - action_logprobs)
		if kl > self.training_params.kl_threshold: return True
		# print('LOSSES')
		# print('total_loss',self.total_loss)
		# print('old_total_loss',self.old_total_loss)
		# # print('actor_loss', self.actor_loss)
		# print('value_loss', self.value_loss)
		# print('old_value_loss', self.old_value_loss)
		# print('entropy_loss', self.entropy_loss)
		# print('New_entropy_loss', self.New_entropy_loss)
		# hdshs+=1
		return False

	def learn_step(self, next_value):
		"""
		Minimize the error in Bellman's equation on a batch sampled from
		replay buffer.
		"""

		dones =  np.array(self.replay_buffer.dones)
		f_Qvals = np.zeros_like(np.array(self.replay_buffer.values))
		f_logits = np.array(self.replay_buffer.logits)
		rewards = self.replay_buffer.rewards
		gamma = self.training_params.gamma
		num_minibatches = self.training_params.num_minibatches
		values =  copy.deepcopy(self.replay_buffer.values)
		f_values = np.array(self.replay_buffer.values)
		values.append(next_value)
		last_return = next_value
		ft_hstates = self.replay_buffer.hstates

		# for i in reversed(range(len(f_Qvals))):
		# 	if dones[i]:discounted_reward = 0
		# 	discounted_reward = rewards[i] + gamma*discounted_reward 
		# 	f_Qvals[i] = discounted_reward

		# Generalized Advantage Estimation
		trace_decay = 0.97

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

		if self.mem:
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
						# hwjvghkervua+=1
						f_observations[i][t] = ft_observations[k]
						f_prev_act[i][t] = ft_prev_act[k]
						f_prev_rew[i][t] = ft_prev_rew[k]
						if self.textAE: f_missions[i][t] = ft_missions[k]
						k-=1
						# if nt:
						# 	print("i", i)
						# 	print("k", k)
						# 	print("Old act batch", ft_prev_act[i])
						# 	print("new act batch", f_prev_act[i])
						# 	nt = False
							# jdjdjs+=1
			# print("Old batch", ft_observations)
			# print("nuevo", f_observations)
			# print("Old act batch", ft_prev_act)
			# print("new act batch", f_prev_act)
			# jdsjsjs+=1
		else: 
			f_observations = ft_observations
		# print("old observations", ft_observations[-1])
		# print("new observations", f_observations[-1])
		# print("old bacth actions", ft_prev_act)
		# print("new batch actions", f_prev_act)
		# jdsjsjs+=1
		f_a_idxs = np.array(self.replay_buffer.a_idx)
		# f_prev_rew = np.array(self.replay_buffer.prev_rew)
		# print("a_idxs", a_idxs)

		# f_observations, f_Qvals, f_a_idxs, f_prev_rew, f_prev_act, f_logits, f_values = self.shuffle_in_unison(\
		# 	f_observations, f_Qvals, f_a_idxs, f_prev_rew, f_prev_act, f_logits, f_values)

		ep_idxs = []
		for i in reversed(range(len(dones))):
			if dones[i]: 
				ep_idxs.append(i+1)
		ep_idxs.append(0)		


		mbatch_s = int(batch_size/num_minibatches)
		# print("mbatch_s", mbatch_s)
		# print('ep_idxs', ep_idxs)
		for i in reversed(range(len(ep_idxs))):
			# print('i:', ep_idxs[i], '; i+1:', ep_idxs[i-1])
			if i == 0: break
			observations = f_observations[ep_idxs[i]:ep_idxs[i-1]]
			Qvals = f_Qvals[ep_idxs[i]:ep_idxs[i-1]]
			a_idxs = f_a_idxs[ep_idxs[i]:ep_idxs[i-1]]
			prev_rew = f_prev_rew[ep_idxs[i]:ep_idxs[i-1]]
			prev_act = f_prev_act[ep_idxs[i]:ep_idxs[i-1]]
			old_logprobs = f_logits[ep_idxs[i]:ep_idxs[i-1]]
			old_vpred = f_values[ep_idxs[i]:ep_idxs[i-1]]
			hstates = ft_hstates[ep_idxs[i]:ep_idxs[i-1]]

			# print('i:', ep_idxs[i], '; i+1:', ep_idxs[i+1])
			# print('ft_hstates',len(ft_hstates))
			# print('hstates',len(hstates))
			# print('a_idxs', a_idxs)



			if self.training_params.mapType == 'minigrid':
				missions = f_missions[ep_idxs[i]:ep_idxs[i-1]]
				missions = tf.convert_to_tensor(missions, dtype=tf.float32)
			else: missions = None

			limit = self.learn(tf.convert_to_tensor(observations, dtype=tf.float32),
					tf.convert_to_tensor(Qvals, dtype=tf.float32),
					tf.convert_to_tensor(a_idxs),
					tf.convert_to_tensor(prev_act, dtype=tf.float32),
					tf.convert_to_tensor(prev_rew, dtype=tf.float32),
					tf.convert_to_tensor(old_logprobs, dtype=tf.float32),
					tf.convert_to_tensor(old_vpred, dtype=tf.float32),
					tf.convert_to_tensor(hstates, dtype=tf.float32),
					missions)
			if limit: 
				# print('Kl reached! after', 20-i, 'updates')
				break
		return	limit


	@tf.function()
	def get_action_value(self, s, action_mask, reward, ohstates, mission, enc_text):
		with tf.device("/device:CPU:0"):
			logits, value, hstates, *_, enc_text = self.NModel.call(s, action_mask, reward, ohstates, mission, enc_text)
		return logits, value, hstates, enc_text

	def get_action_value_Np(self,  s, prev_action_mask, reward, hstates, mission = None, enc_text=None):
		if self.mem:
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
			# print("ts", ts.shape)
			# print("s", s.shape)

			ts[0][tback] = s
			tam[0][tback] = prev_action_mask
			trew[0][tback] = reward

			k = len(t_obs)-1
			if t_obs:
				for t in reversed(range(tback)):
					if k < 0 or dones[k] == 1: break
					# print("should enter once")
					ts[0][t] = t_obs[k]
					tam[0][t] = t_am[k]
					trew[0][t] = t_rew[k]
					if mission is not None: tmissions[0][t] = t_missions[k]
				# 	k-=1
				# print("old observations", len(t_obs))
				# print("old actions shape", len(t_am))
				# print("old actions", t_am)
				# print("last action", prev_action_mask)
				# print("new actions", tam)
		else: 
			ts = s
			tam =  prev_action_mask
			trew = reward
			tmissions = mission
		# print("Old obs", s)
		# print("new obs", ts)
		# print("Old act", prev_action_mask)
		# print("new act", tam)
		# print('hstates in', hstates)
		if hstates == None: pass
		else: 
			hstates=tf.convert_to_tensor(hstates, dtype=tf.float32)
			# jaja+=1

		if mission is not None: tmissions=tf.convert_to_tensor(tmissions, dtype=tf.float32)
		logits, value, hstates, enc_text = self.get_action_value(
			tf.convert_to_tensor(ts, dtype=tf.float32),
			tf.convert_to_tensor(tam, dtype=tf.float32),
			tf.convert_to_tensor(trew, dtype=tf.float32),
			hstates, tmissions, enc_text)
		dist= tfp.distributions.Categorical(logits)		
		# logits, value = self.NModel.call(s)
		return np.array(logits), np.array(value), dist, hstates, enc_text

	def get_steps(self):
		return self.step

	def add_step(self):
		self.step += 1

class ReplayBuffer(object):
	def __init__(self, visual = False):
		self.observations = []
		self.a_idx= []
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