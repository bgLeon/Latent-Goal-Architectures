from agents.modules.a2c_tl import *
from agents.modules.ppo_tl import *
from agents.modules.symbolic_module import SymbMod
from utils.safety_random_game import *
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import random, time, os, datetime, sys
import gym
from gym_minigrid.register import env_list
from gym_minigrid.minigrid import Grid, OBJECT_TO_IDX

# Test specifically importing a specific environment
from gym_minigrid.envs import DoorKeyEnv

# Test importing wrappers
from gym_minigrid.wrappers import *

os.environ['TF_DETERMINISTIC_OPS'] = '1'

# to limit memory
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
	try:
		# Currently, memory growth needs to be the same across GPUs
		for gpu in gpus:
		  tf.config.experimental.set_memory_growth(gpu, True)
		logical_gpus = tf.config.experimental.list_logical_devices('GPU')
		print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
	except RuntimeError as e:
		# Memory growth must be set before GPUs have been initialized
		print(e)


class A2CTrainingParameters:
	def __init__(self, rnn_type, learning_rate=8e-5, #lr=8e-5, lr=2e-5
				print_freq=2000, multiModal = False, cnn_type = 'cnn',
				entropy = 0.001, value_loss_w = 0.5, num_layers = 2,
				batch_size = 256, gamma=0.99, num_neurons= 128, 
				num_epochs = 1, num_minibatches=1, timeSize = 4,
				mem_net_type = 'lstm',
				loss = 'cce', text_mode = 0, AE = False,
				network = "default", mapType='minecraft-insp'):
		"""Parameters
		-------
		lr: float
			 learning rate
		final_exploration: float
			final value of exploration
		model_update_freq: int
			the model is updated every `update_freq` steps.
		batch_size: int
			max size of the batch to train
		print_freq: int
			how often to print out progress
			set to None to disable printing
		gamma: float
			discount factor for future rewards
		indepText: boolean
			if False we assume the instruction to be embedded in the image
		"""
		self.lr = learning_rate
		self.loss = loss
		self.rnn_type = rnn_type
		self.mem_net_type = mem_net_type
		self.alg_name = "A2C"
		self.network = network
		self.value_loss_w = value_loss_w
		self.entropy = entropy
		self.batch_size = batch_size
		self.print_freq = print_freq
		self.gamma = gamma
		self.num_neurons = num_neurons
		self.num_layers = num_layers
		self.multiModal = multiModal
		self.num_epochs = num_epochs
		self.num_minibatches = num_minibatches
		self.timeSize = timeSize
		self.mapType = mapType
		self.text_mode = text_mode
		self.AE = AE
		self.cnn_type = cnn_type

		if mapType=='minecraft-insp':self.indepText=False 
		elif mapType=='minigrid': self.indepText=True
		else: raise Exception('This benchmark has not been implemented')

class PPOTrainingParameters:
	def __init__(self, rnn_type, learning_rate=8e-5, #lr=8e-5,
				print_freq=2000, multiModal = False, value_loss_w = 0.5,
				entropy = 0.001, num_layers = 2, ratio_clip = 0.2,
				batch_size = 2048, gamma = 0.99, num_neurons = 128, 
				num_epochs = 4, num_minibatches=16, kl_threshold=0.015,
				mem_type= 'sequential', mem_net_type = 'lstm', loss = 'cce',
				timeSize = 4, mapType='minecraft-insp', network = "default"):
		"""Parameters
		-------
		lr: float
			 learning rate
		final_exploration: float
			final value of exploration
		model_update_freq: int
			the model is updated every `update_freq` steps.
		batch_size: int
			max size of the batch to train
		print_freq: int
			how often to print out progress
			set to None to disable printing
		gamma: float
			discount factor for future rewards
		
		"""
		self.lr = learning_rate
		self.alg_name = "PPO"
		self.network = network
		self.value_loss_w = value_loss_w
		self.loss = loss
		self.rnn_type = rnn_type
		self.mem_type = mem_type
		self.mem_net_type = mem_net_type
		self.entropy = entropy
		self.batch_size = batch_size
		self.print_freq = print_freq
		self.gamma = gamma
		self.num_neurons = num_neurons
		self.num_layers = num_layers
		self.multiModal = multiModal
		self.num_epochs = num_epochs
		self.ratio_clip = ratio_clip
		self.num_minibatches = num_minibatches
		self.kl_threshold = kl_threshold
		self.timeSize = timeSize
		self.mapType = mapType

class NSAgent:
	def __init__ (self, alg_name, network, batch_size, learning_rate, num_neurons = 128, 
					num_layers = 2, multiModal = False, timeSize = 4, rnn_type = 'lstm', 
					mem_net_type= 'lstm', mapType='minecraft-insp',
					loss = 'cce', text_mode = 0, AE = False, cnn_type= 'cnn'):

		if alg_name == "a2c":
			self.training_params = A2CTrainingParameters(network = network,
												num_neurons = num_neurons,
												num_layers = num_layers,
												batch_size = batch_size,
												multiModal= multiModal,
												timeSize = timeSize,
												rnn_type = rnn_type,
												mem_net_type =  mem_net_type,
												mapType = mapType,
												learning_rate = learning_rate,
												loss = loss, text_mode = text_mode,
												AE = AE, cnn_type= cnn_type)
		elif alg_name == "ppo":
			self.training_params = PPOTrainingParameters(network = network,
												num_neurons = num_neurons,
												num_layers = num_layers,
												multiModal= multiModal,
												timeSize = timeSize,
												rnn_type = rnn_type,
												mem_net_type = mem_net_type,
												mapType = mapType,
												learning_rate = learning_rate,
												loss = loss)
		else:
			raise Exception('Undefined algorithm for neural module')


	def	call(self, env_params, num_times, show_print, pretrained_encoder, 
				render, test, mem, seedB, max_steps, random_walker, loaded_path, loaded_point,
				loaded_test_avg, loaded_train_avg, loaded_best_test):

		run_experiments(self.training_params, env_params, num_times,
								show_print, pretrained_encoder, render, test,
								mem, seedB, max_steps, loaded_path, loaded_point, 
								loaded_test_avg, loaded_train_avg, loaded_best_test, random_walker)


def _initialize_modules(training_params, env_params, mem, pretrained_encoder):
		"""
		This method initializes the policy of the neural module
		
		"""
		policy_dict = {}
		symbM_dict = {}
		board = 0 # in case we use tensorboard

		visual = env_params.visual

		if training_params.mapType == 'minecraft-insp':
			aux_env = Game(env_params)

		# aux_symbM = SymbMod(env_params, env)
			agent = aux_env.agents[0]

			aux_s = aux_env.get_observation(agent)
			num_features = aux_s.shape[0]*aux_s.shape[1]
			num_actions  = len(aux_env.get_actions(aux_env.agents[0]))

			if env_params.fully_obs: obs_size = env_params.size
			else: obs_size = env_params.obs_range*2+1 
			n_agents = aux_env.n_agents

		else:
			aux_env = gym.make(env_params.env_name, populationSz = env_params.trainSz, 
								mapsize=env_params.mapsize)
			aux_env = RGBImgPartialObsWrapper(aux_env)
			aux_obs = aux_env.reset()
			num_actions = aux_env.n_actions
			num_features = aux_obs 
			n_agents = env_params.n_agents
			obs_size = aux_env.agent_view_size

		# We create a different A2C per agent
		for agent in range(n_agents):
			symbM_dict[agent] = SymbMod(env_params, aux_env)

			if training_params.alg_name == "A2C":
				policy_dict[agent] = A2C_tl(num_actions, num_features, obs_size,
												 pretrained_encoder, training_params,
												 env_params, board, 
												 visual = visual, 
												 mem = mem)
			else:
				policy_dict[agent] = PPO_tl(num_actions, num_features, obs_size,
												 pretrained_encoder, training_params,
												 env_params, board, 
												 visual = visual, 
												 mem = mem)
		return policy_dict, symbM_dict


def run_NSA(policy_dict, symbM_dict, training_params, env_params, pretrained_encoder,
				show_print, render, test, seedB, max_steps, random_walker, loaded_path, loaded_point,
				loaded_test_avg, loaded_train_avg, loaded_best_test):
	# Initializing parameters
	NeurM = policy_dict[0] # Only single-agent framework implemented so far
	symbM = symbM_dict[0]
	best_rew = -5000
	best_test = -5000
	mapType = env_params.mapType
	current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	r1 = np.random.default_rng(100) #different seed for choosing options
	best10 = -500

	taskT = env_params.specs_type
	# if taskT ==  3: taskT ='Full_Syntax'
	# elif taskT ==  2: taskT = 'Pos_Neg'
	# else: taskT = 'Pos'

	if not test:
		mm_t = 'MM_' if NeurM.multiModal else ''
		ptr = 'pretrained' if pretrained_encoder else ''
		ae = 'AE_WLoss1'+ training_params.cnn_type if NeurM.AE else training_params.cnn_type #'pretrained_' Pretrained_V5_3_2_ #'Petrain_ES_V6_2_2'+str(env_params.trainSz)
		rec = '_Recurrent_T' if NeurM.mem else '_Dense' 
		lossN = NeurM.loss_func
		mapName = '' if  training_params.mapType=='minecraft-insp' else env_params.env_name
		f_path = '/LatentG/'\
				+ training_params.mapType +'_'+ mapName+'/'\
				+ training_params.alg_name +'_LR_'+str(training_params.lr)+'_B'+str(training_params.batch_size)+'/'\
				+ ae+'_'+mm_t+training_params.network +'_'+ str(training_params.num_neurons) +'/Loss_'\
				+ lossN+'/'+ rec+str(training_params.timeSize)+'/seedB'\
				+ str(seedB)+ '__'+current_time
		trainStats = {}
		testStats = {}
		print('Checkpoint path is:', f_path)
		stats_file = './logs' + f_path

		test_perfs = []

				
		if not os.path.isdir(stats_file) and not test:
				os.makedirs(stats_file)				
		train_stats_file = stats_file+'/TrainStats.npy'
		test_stats_file = stats_file+'/TestStats.npy'

		checkpoint_path = 'Checkpoints' +f_path


		trainCheckpoint_path =  checkpoint_path +'/train'
		if not os.path.isdir(checkpoint_path) and not test:
				os.makedirs(checkpoint_path)			
		checkpoint_path = checkpoint_path +'/checkpoint'

		if not os.path.isdir(trainCheckpoint_path) and not test:
				os.makedirs(trainCheckpoint_path)
		trainCheckpoint_path = trainCheckpoint_path +'/checkpoint'

			# for tensorboard
	# actor_loss = tf.keras.metrics.Mean('actor_loss', dtype=tf.float32)
	# critic_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
	# entropy_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
	# total_loss = tf.keras.metrics.Mean('critic_loss', dtype=tf.float32)
	
	# train_log_dir = 'logs/gradient_tape/' + current_time + '/train'


	# tensorboard
	# train_summary_writer = tf.summary.create_file_writer(train_log_dir)
	
	load_point = 0
	if mapType == 'minecraft-insp':
		rew_batch = np.empty((10000)); rew_batch.fill(-50) #Negative case
		steps_batch =  np.zeros((10000))
	else:
		if env_params.env_name == 'MiniGrid-ttl2Reachability-R-3A-v0':
			rew_batch = np.empty((5000)); rew_batch.fill(-100) #Negative case
			steps_batch =  np.zeros((5000))
		else:
			rew_batch = np.empty((500)); rew_batch.fill(-2) #Negative case
			steps_batch =  np.zeros((500))

	if test:
		# # Binary Choice Maps, no need of so many episodes
		# num = 120 if env_params.mapsize < 20 else 80
		# rew_batch = np.empty((num)); rew_batch.fill(-31) #Afirmative case
		# steps_batch =  np.zeros((num))

		if mapType == 'minecraft-insp' and False:
		
				NeurM.NModel.load_weights('') #write here the path to your modelr 

		else:
			pass

		print('loaded')
	
	#initializing with the first environment ans specification 
	if mapType == 'minecraft-insp':
		env = Game(env_params)
		agent = env.agents[0]
		symbM.new_Specification(env)
		action_set = env.get_actions(agent)
		num_actions = len(action_set)
		obs = env.get_observation(agent)
		num_features = obs.shape[0]*obs.shape[1]
		mission = None
		proc_mission = None
	else:
		BCM = False
		Objs_type = 2 if test else 0 #0 is train objects, 2 is test, 3 is train objects depective, 4 test deceptive
		steps = 0 if loaded_path == 'None' else loaded_point+1
		env = gym.make(env_params.env_name, Obj_set = Objs_type, BCM = BCM, 
							populationSz = env_params.trainSz, mapsize=env_params.mapsize,
							total_steps =  steps)
		env = RGBImgPartialObsWrapper(env) # Get pixel observations
		env.seed(seedB)

		eval_env = gym.make(env_params.env_name, Obj_set = 1, populationSz = env_params.trainSz,
							 mapsize=env_params.mapsize) #
		eval_env = RGBImgPartialObsWrapper(eval_env) # Get pixel observations
		eval_env.seed(seedB)
		# eval_env.Obj_type = 1

		c_obs = env.reset()
		num_actions = env.n_actions
		obs = c_obs['image']
		mission = c_obs['mission']
		pre_mission = mission
		proc_mission = symbM.preProcessText(mission)

	training_reward = 0
	episode_count = 0  # episode counter
	prev_action = 0
	prev_rew = 0

	if show_print: sys.stdout.write("Executing " + str(max_steps-load_point) +" steps...\n")
	sys.stdout.flush()
	#We start iterating with the environment
	if render:
			# time.sleep(0.8)
			# clear_screen()
			env.render()
	hstate = None
	FixTaskH = 10 #To fix the type of task for a number of episodes
	Tasktimer = 0

	Nupdates = 0


	if loaded_path != 'None':
		load_point = loaded_point+1
		NeurM.step = loaded_point+1
		best_test =  loaded_best_test
		rew_batch.fill(loaded_train_avg)
		test_perfs = 10 * [loaded_test_avg]
		NeurM.NModel.load_weights(loaded_path)
		print('loaded ', loaded_path)
		

		if training_params.lr == 8e-5:
			# if t == 0: print("Will be lr updates!")

			# # std
			if load_point > 30e6:
				NeurM.core_optimizer.learning_rate=6e-5
				print("Lr update!")

			if load_point > 50e6:
				NeurM.core_optimizer.learning_rate=4e-5
				print("Lr update 2!")

		# if taskT == 'TTL2' and training_params.network == "PrediNet" and seedB==4:
		# 		NeurM.NModel.load_weights('./Checkpoints/Neurips21/minecraft-insp_/Spec_TTL2/'\
		# 						+'RecNetsA2C/pretrained_encoderMMPrediNet_rim_Netw/Loss_cce/'\
		# 						+'Res_BRIMV1_NoD_NoHstates_1x50/TSize1/seedB4__20210509-172337/checkpoint')
		# 		load_point = 30000000
		# 		NeurM.step = 30000000
		# 		rew_batch = np.empty((20000)); rew_batch.fill(-5.31)
		# 		NeurM.core_optimizer.learning_rate=6e-5
		# 		print("loaded Res MBrim PrediNet seed 4")
		# 		print ("Weights succesfully loaded")	

	# timeBatch_size = 300
	# time_acting = np.zeros((timeBatch_size))
	# time_learning = np.zeros((timeBatch_size))
	# time_learn_call = np.zeros((timeBatch_size))
	# time_tf_learning = np.zeros((timeBatch_size))
	# time_missions = np.zeros((timeBatch_size))

	for t in range (load_point,max_steps):
		# Astart_time = time.time()
		# if t == 155e5 or t == 405e5:
		# if episode_count%400== 0:
		# 	render = True
		# else: render = False
		# if t == 305e5 or t == 505e5 and env_params.env_name == 'MiniGrid-ttl2Reachability-R-3A-v0':
		# 	print('Map Sizes were recently redistributed, updating train best_rew')
		# 	best_rew=np.average(rew_batch)

		if training_params.lr == 8e-5:
			# if t == 0: print("Will be lr updates!")

			# # std
			if t == 30e6:
				NeurM.core_optimizer.learning_rate=6e-5
				print("Lr update!")

			elif t == 50e6:
				NeurM.core_optimizer.learning_rate=4e-5
				print("Lr update 2!")

			# LLR
			# if t == 8e6:
			# 	NeurM.core_optimizer.learning_rate=6e-5
			# 	print("Lr update!")
			# if t == 20e6:
			# 	NeurM.core_optimizer.learning_rate=4e-5
			# 	print("Lr update!")

			# elif t == 35e6:
			# 	NeurM.core_optimizer.learning_rate=2e-5
			# 	print("Lr update 2!")

			# LLR_2
			# if t == 8e6:
			# 	NeurM.core_optimizer.learning_rate=4e-5
			# 	print("Lr update!")
			# if t == 20e6:
			# 	NeurM.core_optimizer.learning_rate=2e-5
			# 	print("Lr update 2!")

		# if not random_walker or not test:
		prev_action_mask = tf.one_hot(indices=prev_action, depth=num_actions,
			dtype=tf.float32)

			# print('Time for mem', start_M - time.time()) 

		# start_A = time.time()
		# action_mask = 0 # this is currently not in use
		logit, value, dist, hstate = NeurM.get_action_value_Np(obs, 
												prev_action_mask, prev_rew, hstate, 
												proc_mission)
		# logit = [0,1,2]
		# value = [2]
		# dis = [0,1,2,3]
		# hstate =np.zeros((256))
		# last_mem = None
		# print('Time to act', start_A - time.time())

		# print('logit', logit)
		# print('value', value)
		# ajajaj+=1

		if training_params.alg_name == "PPO":
			a_idx = dist.sample(seed=seedB)
		else:
			a_idx = r1.choice(num_actions, p=np.squeeze(logit))	
			# if test:
			# 	a_idx= np.argmax(np.squeeze(logits))
				# print("a_idx max", a_idx)
		# except:
			# print("NAN in logits")
			# print("logits", logit)
			# raise Exception('Imposible action')
				# a_idx = np.random.choice((0,1,2,3))
			# print("ACT", act)
			# print("ACT_old", act_old)
			
		# For random walker's variance
		if test:
			# ty = tf.random.uniform([1], minval=0, maxval=12, dtype=tf.dtypes.int32)
			# a = ty
			# a = ty%4
			# if env.spec[0] == "N":
			# 	a+=4
			# elif env.spec[0] == "F":
			# 	a+=8
			if random_walker:
				# a_idx = a.numpy()[0]
				a_idx = np.random.choice(num_actions)
				
		NeurM.add_step()
		# Executing the action
		if mapType == 'minecraft-insp':
			reward = symbM.execute_actions(a_idx, hand_pol = False)
			done = symbM.ltl_game_over or symbM.env_game_over
		else:
			c_obs, reward, done, info = env.step(a_idx)

		if render:
			if mapType == 'minecraft-insp': time.sleep(0.2)
			else: 
				# time.sleep(0.2)
				if reward>0:
					print('reward rend', training_reward+reward)
			# clear_screen()
			env.render()
		training_reward += reward


		# Saving this transition
		if training_params.alg_name == "PPO":
			NeurM.save_transition(obs, a_idx, reward, value,
									dist.log_prob(a_idx), done, prev_action_mask, prev_rew , hstate, proc_mission)
		else:
			NeurM.save_transition(obs, a_idx, reward, value,
									logit, done, prev_action_mask, prev_rew, hstate, proc_mission)

		if not test:

			# Learning
			if (done or (NeurM.replay_buffer.size >= training_params.batch_size)):
			# if NeurM.replay_buffer.size >= training_params.batch_size:
				# print('NeurM.replay_buffer.size', NeurM.replay_buffer.size )
				# time_acting[episode_count%timeBatch_size] = time.time()- Astart_time
				# Lstart_time = time.time()
				Tasktimer +=1
				if training_params.alg_name == "PPO" and Tasktimer < FixTaskH and NeurM.replay_buffer.size < training_params.batch_size:
					pass
					# print('continue')
					# NeurM.replay_buffer.clear()
					print('shouldn\'t be here')
				else:
					Tasktimer = 0
					
					# print('learn')
					if done:
						next_value = 0
					else:
						if mapType == 'minecraft-insp':
							s_end = env.get_observation(agent) 
							if not NeurM.visual:
								obs = s_end.reshape((1,num_features)) 
							else: obs = s_end
						else:
							mission = c_obs['mission']
							obs = c_obs['image']
							if mission != pre_mission:
								pre_mission = mission
								proc_mission = symbM.preProcessText(mission)

						prev_action_mask = tf.one_hot(indices=a_idx, depth=num_actions,
													dtype=tf.float32)
						_, next_value, *_ = NeurM.get_action_value_Np(obs, prev_action_mask, reward, hstate, proc_mission)


					for i in range(training_params.num_epochs):
						if training_params.alg_name == "PPO":
							# print('Loop', i)
							limit_reached = NeurM.learn_step(next_value)
							if limit_reached: 
								break
						else:
							# Cstart_time = time.time()
							NeurM.learn_step(next_value)
							# time_tf_learning[episode_count%timeBatch_size] = t_learning
							# time_missions[episode_count%timeBatch_size] = t_missions
							# time_learn_call[episode_count%timeBatch_size] = time.time()- Cstart_time
					Nupdates +=1

				
					NeurM.IsStart = True if Nupdates < 1000 else False
					NeurM.UpdateAE = True if Nupdates% 100 == 0 else False

					NeurM.replay_buffer.clear()
			


					# time_learning[episode_count%timeBatch_size] = time.time() - Lstart_time
		prev_action = a_idx
		prev_rew = reward
		# Printing
		if show_print and (NeurM.get_steps()+1) % training_params.print_freq\
							== 0:
			sys.stdout.write("Step:"+ str(NeurM.get_steps()+1)+\
							"\tAvg 2000 ep reward:" + str(np.average(rew_batch))\
							+ "\tNumber of episodes:" + str(episode_count) +"\n")	
			sys.stdout.flush()

		# Saving training progress for painting
		if not test and (NeurM.get_steps()+1) % 10000 == 0:
			trainStats[str(NeurM.get_steps()+1)] = str(np.average(rew_batch))

		if done:
			# Game over occurs for one of two reasons: 
			# 1) Episode's Horizon Reached, 
			# 2) TTL instruction has been satisfied
			# print('done!')
			# Restarting
			rew_batch[episode_count%len(rew_batch)] = training_reward
			steps_batch[episode_count%len(rew_batch)] = env.step_count

			hstate = None
			# if render:
			# 	data = np.clip(env.heat_map*10, 0, 255)
			# 	data= data.astype(np.uint8) 
			# 	image = Image.fromarray(data, "L")
			# 	newsize = (420, 420) 
			# 	image = image.resize(newsize) 
			# 	image.show()
			# 	print("shape", env.heat_map.shape)
			# 	a11=env.heat_map
			# 	plt.imshow(a11, cmap='hot')
			# 	plt.colorbar()
			# 	plt.show()
			# 	time.sleep(2)

			if env_params.complexT:
				env_params.complexT+=1
				symbM.complexT+=1


			if mapType == 'minecraft-insp':

				env = Game(env_params)


				if env.AllSolved:
					break
				symbM.new_Specification(env)
				agent = env.agents[0]
				
			else:
				c_obs = env.reset()

			# if env_params.mapType == "default": start_track = -30
			# else:  start_track = -10
			prev_action = 0
			prev_rew = 0
			if  mapType == 'minigrid': 
				if env_params.env_name == 'MiniGrid-ttl2Reachability-R-3A-v0':
					start_track = -25 
				else:
					start_track = 0
			else: start_track = -10 if taskT == 'TTL1' else -30
			
			if best_rew<np.average(rew_batch) and not test:
				try:
					best_rew=np.average(rew_batch)
					if best_rew > start_track:
						print("New BEST!", best_rew)
				except:
					print("Error with reward batch!") 
			if test:
				if episode_count == len(rew_batch):
					print("200 episodes reward is:", np.average(rew_batch))
					print("1Avg n_steps is:", np.average(steps_batch))
					break
			episode_count+=1
			training_reward = 0
			# if np.average(rew_batch)>0.99:
			# 	break

		if NeurM.get_steps()%200000 == 0 and not test: #500000
				if mapType == 'minecraft-insp':
					testStats, best_test = Mcraft_evaluation(NeurM, symbM, env_params, checkpoint_path, 
														test_stats_file, testStats, best_test, r1,
														training_params = training_params)
					try:
						np.save(train_stats_file, trainStats)
						print("train stats saved!")
						NeurM.NModel.save_weights(trainCheckpoint_path)
						print("train weigths saved!")
					except Exception as e:
						print(e)
						print("train stats couldn't be saved!")
				else:
					if env_params.env_name == 'MiniGrid-ttl2Reachability-R-3A-v0':
						testStats, best_test, last_test = MGrid_evaluation(NeurM, symbM, eval_env, checkpoint_path, 
															test_stats_file, testStats, best_test, r1, 
															render = render, training_params = training_params)
						test_perfs.append(last_test)
						if len(test_perfs)>10: 
							test_perfs.pop(0)
						tr = np.average(test_perfs)
						print('Test 10 average', tr)
						if tr > best10:
							best10 = tr
							try:
								print('NEW BEST 10 AVERAGE!--------------------------------------------------------------')
								NeurM.NModel.save_weights(checkpoint_path)
								print("Weights saved!")
							except Exception as e:
								print(e)
								print("It couldn't be saved!")

					# try:
					np.save(train_stats_file, trainStats)
					print("train stats saved!")
					NeurM.NModel.save_weights(trainCheckpoint_path)
					print("train weigths saved!")
					# except Exception as e:
					# 	print(e)
					# 	print("It couldn't be saved!")
		# if NeurM.get_steps()%20000 == 0:
		# 		print('Checking time performance')
		# 		print('Avg time acting', np.mean(time_acting))
		# 		print('Avg time learning', np.mean(time_learning))
		# 		print('Avg time learn call', np.mean(time_learn_call))
		# 		print('Avg time_tf_learning', np.mean(time_tf_learning))
		# 		print('Avg time_missions', np.mean(time_missions))

		# Getting the current state and ltl goal
		if mapType == 'minecraft-insp':
			obs = env.get_observation(agent) 
			if not NeurM.visual:
				obs = obs.reshape((1,num_features)) 

		elif mapType == 'minigrid':
			mission = c_obs['mission']
			obs = c_obs['image']

			# time.sleep(0.5)
			# jdjd+=1
			if mission != pre_mission:
				pre_mission = mission
				proc_mission = symbM.preProcessText(mission)
		else: raise Exception('This kind of map has not been implemented')
		# print('mission', mission)
		# print('proc_mission', proc_mission)
	if show_print: 
		print("Done! Total reward:", training_reward)


def Mcraft_evaluation(NeurM, symbM, env_params, checkpoint_path, test_stats_file, testStats, best_test, r1, Nsamples = 100):
	#generate test environment
	aux_env_params = copy.deepcopy(env_params)
	aux_env_params.test = True
	aux_env_params.validation = True
	aux_env_params.SpecSets = aux_env_params.testSpecSets
	aux_env_params.candidates = aux_env_params.test_candidates
	env = Game(aux_env_params)
	agent = env.agents[0]
	
	# Initialize parameters
	symbM.new_Specification(env)
	action_set = env.get_actions(agent)
	num_actions = len(action_set)
	obs = env.get_observation(agent)
	num_features = obs.shape[0]*obs.shape[1]
	mission = None
	proc_mission = None
	hstate= None
	prev_action = 0
	prev_rew = 0
	testing_reward = 0

	rew_batch =  np.zeros((Nsamples))
	t=0
	# if True:
	# 		time.sleep(0.2)
	# 		# clear_screen()
	# 		env.render()
	# run evaluation loop
	rew_batchH =  np.zeros((Nsamples))
	testing_rewardH = 0
	while t < Nsamples:
		prev_action_mask = tf.one_hot(indices=prev_action, depth=num_actions,
			dtype=tf.float32)
		logit, value, dist, hstate,  = NeurM.get_action_value_Np(obs, 
												prev_action_mask, prev_rew, hstate, proc_mission)
		a_idx = r1.choice(num_actions, p=np.squeeze(logit))	


		reward = symbM.execute_actions(a_idx, hand_pol = False)
		done = symbM.ltl_game_over or symbM.env_game_over

		testing_reward += reward

		prev_action = a_idx
		prev_rew = reward

		obs = env.get_observation(agent)

		if reward == -1: reward = -20
		testing_rewardH += reward
		# if True:
		# 	time.sleep(0.2)
		# 	# clear_screen()
		# 	env.render()

		if done: 
			rew_batch[t] = testing_reward
			rew_batchH[t] = testing_rewardH
			t+=1
			prev_action = 0
			prev_rew = 0
			env = Game(aux_env_params)
			agent = env.agents[0]
			symbM.new_Specification(env)
			obs = env.get_observation(agent)
			hstate= None
			testing_reward = 0
			testing_rewardH = 0
	perf = np.average(rew_batch)
	perfH = np.average(rew_batchH)
	# print('reward batch', rew_batch)

	testStats[str(NeurM.get_steps()+1)] = str(perf)
	np.save(test_stats_file, testStats)
	print("test stat saved!")
	print('perfH', perfH)
	print('perf', perf)
	print('best_test', best_test)
	if perf > best_test:
		best_test = perf
		print("----------------------------------------------new best!")
		try:
			NeurM.NModel.save_weights(checkpoint_path)
			print("Weights saved!")
		except Exception as e:
			print(e)
			print("It couldn't be saved!")

	return testStats, best_test

def MGrid_evaluation(NeurM, symbM, eval_env, checkpoint_path, test_stats_file, testStats, best_test, r1, training_params,
					Nsamples = 20, render =False):
	#generate test environment
	c_obs = eval_env.reset()
	num_actions = eval_env.n_actions
	obs = c_obs['image']
	mission = c_obs['mission']
	pre_mission = mission
	proc_mission = symbM.preProcessText(mission)
	# Initialize parameters


	hstate = None
	prev_action = 0
	prev_rew = 0
	testing_reward = 0

	rew_batch = np.zeros((Nsamples))
	t=0
	# if True:
	# 		time.sleep(0.2)
	# 		# clear_screen()
	# 		env.render()
	if render:
			eval_env.render()
	# run evaluation loop
	rew_batchH =  np.zeros((Nsamples))
	testing_rewardH = 0
	while t < Nsamples:

		prev_action_mask = tf.one_hot(indices=prev_action, depth=num_actions,
			dtype=tf.float32)
		logit, value, dist, hstate = NeurM.get_action_value_Np(obs, 
												prev_action_mask, prev_rew, hstate, proc_mission)
		a_idx = r1.choice(num_actions, p=np.squeeze(logit))	

		c_obs, reward, done, info = eval_env.step(a_idx)
		if render:
			eval_env.render()
		testing_reward += reward

		prev_action = a_idx
		prev_rew = reward

		mission = c_obs['mission']
		obs = c_obs['image']
		if mission != pre_mission:
				pre_mission = mission
				proc_mission = symbM.preProcessText(mission)

		if reward == -1: reward = -20
		testing_rewardH += reward
		# if True:
		# 	time.sleep(0.2)
		# 	# clear_screen()
		# 	env.render()

		if done: 
			rew_batch[t] = testing_reward
			rew_batchH[t] = testing_rewardH
			t+=1
			prev_action = 0
			prev_rew = 0
			c_obs = eval_env.reset()
			mission = c_obs['mission']
			obs = c_obs['image']
			pre_mission = mission
			proc_mission = symbM.preProcessText(mission)
			hstate= None
			testing_reward = 0
			testing_rewardH = 0

	perf = np.average(rew_batch)
	perfH = np.average(rew_batchH)
	# print('reward batch', rew_batch)

	testStats[str(NeurM.get_steps()+1)] = str(perf)
	np.save(test_stats_file, testStats)
	print("test stat saved!")
	print('perfH', perfH)
	print('perf', perf)
	print('best_test', best_test)
	if perf > best_test:
		best_test = perf
		print("----------------------------------------------new best!")

	return testStats, best_test, perf

def run_experiments(training_params, env_params, num_times, show_print, 
							pretrained_encoder, render, test, mem, seedB,
							max_steps, loaded_path, loaded_point, 
							loaded_test_avg, loaded_train_avg, loaded_best_test, 
							random_walker = False):
	for t in range(num_times):
		# Setting the random seed to 't' + an offset
		random.seed(t+seedB)
		tf.random.set_seed(t+seedB)
		np.random.seed(t+seedB)
		os.environ['PYTHONHASHSEED']=str(t+seedB)
		# Initializing NeurMs, currently one per agent
		policy_dict, symbM_dict = _initialize_modules(training_params,
														env_params,
														mem, pretrained_encoder)
		if show_print: print("Current n_experiments:", t,
							"from", num_times)
		run_NSA(policy_dict = policy_dict, symbM_dict = symbM_dict,
					training_params = training_params, 
					env_params = env_params, pretrained_encoder = pretrained_encoder,
					show_print = show_print, render = render, test = test, 
					seedB = t+seedB, max_steps = max_steps, 
					random_walker = random_walker, loaded_path = loaded_path, 
					loaded_point = loaded_point, loaded_train_avg = loaded_train_avg,
					loaded_test_avg = loaded_test_avg, loaded_best_test = loaded_best_test)