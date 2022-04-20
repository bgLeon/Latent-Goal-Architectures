from utils.specifications import *
from utils.safety_game_objects import Actions
import numpy as np
import re
import tensorflow as tf

class Vocabulary:
    """A mapping from tokens to ids with a capacity of `max_size` words.
    It can be saved in a `vocab.json` file."""

    def __init__(self, max_size):
        self.max_size = max_size
        self.vocab = {}

    def load_vocab(self, vocab):
        self.vocab = vocab

    def __getitem__(self, token):
        if not token in self.vocab.keys():
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum vocabulary capacity reached")
            self.vocab[token] = len(self.vocab) + 1
        return self.vocab[token]

class SymbMod:
	"""
	Internal labelling function
	Internal Reward function
	Extractor
	Progression
	"""
	def __init__(self, env_params, aux_env):
		"""
		Inputs:
			-env_params: parameters of the environment
			-aux_env: the environment  
		"""
		self.mapType = env_params.mapType
		if env_params.mapType == 'minecraft-insp':
			self.visual = env_params.visual
			self.class_ids = aux_env.class_ids
			self.resolution = env_params.resolution
			self.fully_obs = env_params.fully_obs
			self.spec = aux_env.spec
			self.moving_spec = env_params.moving_spec
			self.obs_range = env_params.obs_range
			self.complexT = env_params.complexT
			# self.t_limit = env_params.t_limit
			self.specs_type = env_params.specs_type
			self.nSteps = 0
			self.env = aux_env
			self.t_limit = aux_env.t_limit
			self.agent = aux_env.agents[0]
			if self.moving_spec:
				# Sets where in the extension are we placing the TTL instruction
				self.beg = np.random.randint(5-len(self.spec))
			self.spec = aux_env.spec
			if self.complexT:
				self.spec = aux_env.spec
				self.SubTidx = 0 #auxiliary counter of the current complex subTask
				print("specification", self.spec)
				self.SubTasksList = self._Extractor(self.spec)
				self.SubTasksList.append('T') # Means specification is done
				print('List', self.SubTasksList)
				self._progression()
			
			if env_params.specs_type == 'TTL1':
				reward, self.ltl_game_over, self.env_game_over = self._get_TTL1_rewards()
			else:
				reward, self.ltl_game_over, self.env_game_over = self._get_TTL2_rewards(init=True)
			reward = 0 # for initialization we don't penalize the agents
			self.extend_observation()
		elif env_params.mapType == 'minigrid':
			self.vocab = Vocabulary(env_params.vocabSize)
			self.maxTextLen = env_params.textLength
			self._generate_Vocab(aux_env)
		else:
			raise Exception('This Symbolic Module is not ready to preprocess the given environment')

	def new_Specification(self, env):
		"""
		Restarting the goals of the Symbolic Module
		"""
		self.spec = env.spec
		self.env = env
		self.t_limit = env.t_limit
		self.agent = env.agents[0]
		if self.complexT:
			self.SubTidx = 0 #auxiliary counter of the current complex subTask
			self.SubTasksList = self._Extractor(self.spec)
			self.SubTasksList.append('T')
			self._progression()
		self.extend_observation() 
		self.nSteps = 0

		"""
		We should automatically detect if it's a complex specification
		"""
	def _Extractor(self, complexS):
		"""
		Extracts the list of subtasks, should be called only when starting with 
		a new specification
		"""
		length = len(complexS)
		SubTasksList = []
		subtasks = []

		if len(complexS) <= 2:
			# Simple subtasks
			SubTasksList.append(complexS)
			return SubTasksList
		else:
			i = 0
			while i < length:
				if complexS[i] == "(":
					# We jump to the end of the bracket
					e_idx= _get_bracket_idxs(complexS, i)
					i = e_idx+1
					if i >= length: break

				if complexS[i] == ";":
					PriorSubTasks = []
					PostSubTasks= []

					# We extract the subtasks prior to this operator
					if complexS[i-1] == ")":
						i_idx= _get_bracket_idxs(complexS, i-1, True)
						PriorSubTasks = self._Extractor(complexS[i_idx+1:i-1])
					elif complexS[i-1] == "!":
						PriorSubTasks = [complexS[i-2:i]]
					else:
						PriorSubTasks = [complexS[i-1]]
					for subtask in PriorSubTasks:
						SubTasksList.append(subtask)

					"""
					We extract the subtasks posterior to this operator,
					They will be added only if there aren't future operators
					"""
					if complexS[i+1] == "(":
						e_idx= _get_bracket_idxs(complexS, i+1, False)
						if e_idx+1 == length:
							PostSubTasks = self._Extractor(complexS[i+2:e_idx])
					elif i+2 == length:
						PostSubTasks = [complexS[i+1]]
					elif complexS[i+2] == "!" and i+3 == length:
						PostSubTasks = [complexS[i+1:i+3]]
					if PostSubTasks:
						for subtask in PostSubTasks:
							SubTasksList.append(subtask)

				# Non-deterministic choices
				elif complexS[i] == "V" or complexS[i] == "U" :
					origIdx = i
					PriorSubTasks = []
					PostSubTasks= []
					# We extract the subtasks prior to this operator
					if complexS[i-1] == ")":
						i_idx= _get_bracket_idxs(complexS, i-1, True)
						PriorSubTasks = self._Extractor(complexS[i_idx+1:i-1])
					elif complexS[i-1] == "!":
						PriorSubTasks = [complexS[i-2:i]]
					else:
						PriorSubTasks = [complexS[i-1]]

					# We extract the subtasks posterior to this operator
					if complexS[i+1] == "(":
						e_idx= _get_bracket_idxs(complexS, i+1, False)
						PostSubTasks = self._Extractor(complexS[i+2:e_idx])
						i = e_idx+1
					elif i+2<length:
						if complexS[i+2] == "!":
							PostSubTasks = [complexS[i+1:i+3]]
					else:
						PostSubTasks = [complexS[i+1]]

					"""
					If there are negated clauses or complex disjunctions
					the symbolic module will use a sequential approach
					"""
					sequential = False
					for subtask in PriorSubTasks:
						for symbol in subtask:
							if symbol == "!" or symbol == "V":
								sequential = True
					for subtask in PostSubTasks:
						for symbol in subtask:
							if symbol == "!" or symbol == "V":
								sequential = True
					path1 = len(PriorSubTasks)
					path2 = len(PostSubTasks)
					if sequential:
						if complexS[origIdx] == "U":
							for subtask in PriorSubTasks:
									SubTasksList.append(subtask)
							for subtask in PostSubTasks:
									SubTasksList.append(subtask)

						elif path1<= path2:
							for subtask in PriorSubTasks:
									SubTasksList.append(subtask)
						else:
							for subtask in PostSubTasks:
									SubTasksList.append(subtask)
					else:
						"""
						The SM will present a path of non-deterministic choices
						to the NM
						"""
						totalP = max(path1, path2)

						for j in range (totalP):
							if j < path1 and j < path2:
								# If in operators both of them, go to the next one
								subtask = PriorSubTasks[j] + 'V'\
											+ PostSubTasks[j]
								SubTasksList.append(subtask)

							elif j >= path1:
								subtask = '_V'\
											+ PostSubTasks[j]
								SubTasksList.append(subtask)
							else:
								subtask = PriorSubTasks[j] + 'V_'
								SubTasksList.append(subtask)

							if complexS[origIdx] == "U":
								SubTasksList.append("CAP")
							elif complexS[origIdx] == "V" and (j+1 < totalP):
								SubTasksList.append("CUP")			
				i+=1
		return SubTasksList

	def _progression(self, true_props=''):

		"""
		In case we have complex specifications, the SM calls this function
		to extract the next subtask for the NM
		"""
		n = self.SubTidx

		# Inclusive disjunction
		if self.SubTasksList[n] == 'CAP':
			spec = self.SubTasksList[n-1]
			if true_props == spec[0]: self.spec = spec[2]
			else: self.spec = spec[0]
			if self.spec == "_":
				self.SubTidx += 1
				self.spec = self.SubTasksList[self.SubTidx]

		# Exclusive Non-deterministic Path 
		elif self.SubTasksList[n] == 'CUP':
			spec1 = self.SubTasksList[n-1]
			spec2 = self.SubTasksList[n+1]
			if true_props == spec1[0]: self.spec = spec2[0]
			else: self.spec = spec2[2]
			self.SubTidx+=1
			if self.spec == "_":
				self.SubTidx += 1
				self.spec = self.SubTasksList[self.SubTidx]
		else:
			self.spec = self.SubTasksList[n]

		if len(self.spec) == 3:
			if self.spec[0] == "_":
				self.spec = self.spec[2]
			elif self.spec[2] == "_":
				self.spec = self.spec[0]

	def labelling_function(self, agent):
		"""
		Returns the string with the propositions that are True in the current 
		state
		"""
		back_map = self.env.back_map
		ret = str(back_map[agent.i][agent.j]).strip()
		return ret

	def _get_TTL1_rewards(self):
		"""
		Internal Reward Function
		"""
		agents = self.env.agents
		reward = -0.1
		spec_copy = self.spec
		ltl_game_over = False
		env_game_over = False if self.nSteps < self.t_limit else True
		for agent in agents.values():
			true_props = self.labelling_function(agent)
			if len(self.spec) < 2 or \
				(len(self.spec) == 2 and self.spec[0] == 'P'):
				if true_props in 'abcdefghijklmnopqrstuvwxyz' and \
														true_props.strip() != '':
					if true_props ==  self.spec [0]:
					# if true_props ==  'k':						
						reward = 1
						if self.complexT:
							self.SubTidx += 1
							self._progression(true_props)
						else:
							self.spec = 'T'
						# Finish if the full instruction is solved
						if self.spec== 'T': ltl_game_over = True
					else: reward = -1
			elif len(self.spec) == 2:
				if true_props in 'abcdefghijklmnopqrstuvwxyz' and true_props.strip() != '':
					s = 0
					if self.spec[0] == "!": s=1
					if true_props ==  self.spec[s]: reward = -1
					else: 
						reward = 1
						if self.complexT:
							self.SubTidx += 1
							self._progression(true_props)
						else:
							self.spec = 'T'
						if self.spec== 'T': ltl_game_over = True
			elif len(self.spec)>4:
				if true_props in 'abcdefghijklmnopqrstuvwxyz' and true_props.strip() != '':
					if self.spec[2] == '&':
						if true_props == self.spec[1]: reward = -1
						elif true_props == self.spec[4]: 
							# reward = -1
							reward = 1
							self.SubTidx += 1
							if self.complexT:
								self._progression(true_props)
							else:
								self.spec = 'T'

							if self.spec== 'T': ltl_game_over = True
						else:
							reward = -1
					else:
						reward = 1
						if self.complexT:
							self.SubTidx += 1
							self._progression(true_props)
						else:
							self.spec = 'T'
						if self.spec== 'T': ltl_game_over = True
			else:
				if true_props in 'abcdefghijklmnopqrstuvwxyz' and true_props.strip() != '':
					if true_props ==  self.spec[0] or true_props ==  self.spec[2]:
					# if true_props !=  self.spec[0] and true_props !=  self.spec[2]:
					# if true_props ==  'j':						
						reward = 1
						if self.complexT:
							self.SubTidx += 1
							self._progression(true_props)
						else:
							self.spec = 'T'
						if self.spec== 'T': ltl_game_over = True
					else: reward = -1

		# ---- Deceptive
		# if reward == -1: 
		# 	reward=1
		# 	ltl_game_over = True
		# 	self.spec = spec_copy
		# elif reward == 1:
		# 	reward=-1
		# 	ltl_game_over = False
		# 	self.spec = spec_copy

		self.agent.update_reward(reward) # used in rendering 
		return reward, ltl_game_over, env_game_over

	def _get_TTL2_rewards(self, action = 0, init = False):
		"""
		Internal Reward Function
		"""
		agents = self.env.agents
		reward = -0.05
		# reward = -0.1
		ltl_game_over = False
		env_game_over = False if self.nSteps < self.t_limit else True
		task = self.env.spec[:6]
		if init:
			return reward, ltl_game_over, env_game_over
		for agent in agents.values():
			# Checking the action type
			# if self.spec[0] == 'S' and action>3:
			# 	reward-=1
			# 	feedb="B"
			# elif self.spec[0] == 'N' and (action<4 or action>7):
			# 	reward-=1
			# 	feedb="B"
			# elif self.spec[0] == 'F' and action<8:
			# 	reward-=1
			# 	feedb="B"
			# else:
			# 	feedb="G"
			# feedb=" "
			# self.env.spec = task + feedb
			# self.spec = self.env.spec
			# self.agent.observation[1][1]=feedb
			true_props = self.labelling_function(agent)
			prog_idx=len(self.spec)-1
			if true_props in ' abcdefghijklmnopqrstuvwxyz0123456789BCDFGHIJKLMNOPQRSWYZ':
				if (self.spec[prog_idx-2] == 'V' and\
				 	true_props != self.spec[prog_idx-3] and\
				 	true_props != self.spec[prog_idx]) or\
				 	(self.spec[prog_idx-2] != 'V' and\
				 	 self.spec[prog_idx-1] == '+' and\
				 	true_props !=  self.spec[prog_idx]) or\
				 	(self.spec[prog_idx-1] == '-' and\
				 	true_props == self.spec[prog_idx]):

					if 	(self.spec[2] == 'V' and self.spec[4] != true_props and\
						self.spec[1] != true_props) or\
						(self.spec[2] != 'V' and self.spec[0] == '+' and\
							self.spec[1] != true_props) or\
						(self.spec[0] == '-' and self.spec[1] == true_props):
						reward-=1
					# else:
						# if true_props.strip() != '': reward-=0.5
				else:
					reward += 1
					self.spec = 'T'			
					ltl_game_over = True
		self.agent.update_reward(reward) # used in rendering
		return reward, ltl_game_over, env_game_over

	def extend_observation(self):
		raw_obs = self.agent.observation
		if self.fully_obs:
			obs = []
			row = []
			for i in range(self.obs_range*2+1):
				if self.moving_spec:
					 if i < self.beg or i >=(self.beg+len(self.spec)):
					 	row.append('X')
					 else:
					 	row.append(self.spec[i-self.beg])
				else:
					if i < len(self.spec):
						row.append(self.spec[i])
					else:
						row.append('X')
			obs.append(row)

			# Adding the raw obs
			for row in raw_obs:
				obs.append(row)
		else:
			obs = []
			row = []
			i=0
			# stri="abcdefghijklmnopqrstuvwxyz0123456789-+AXVUETFNSGB"
			# stri="BCDFGHIJKLMNOPQRSWYZ"
			offset = 5 if self.specs_type == 'TTL1' else 15
			for j in range(offset):
			# for j in range(5):
				i+=1
				# if self.moving_spec:
				# 	 if i < self.beg or i >=(self.beg+len(self.spec)):
				# 	 	row.append(' ')
				# 	 else: 
				# 	 	row.append(self.spec[i-self.beg])
				# else:
				if j < len(self.spec): v = self.spec[j]
				# if j < len(stri):
				# 	row.append(stri[j])
				else: v = ' '
				# Control experiment 1
				# if j== 0: v = '+'
				# Control experiment 2
				# if j== 0: v = 'E'
				# if j== 1: v = 'E'
				# Control experiment 3
				# if j== 0: v = '-'
				row.append(v)
				if i == 5:
					i=0
					# row[0]='+'
					obs.append(row)
					row = []
			# Adding the raw obs
			for row in raw_obs:
				obs.append(row)
		# obs[0][0]="S"
		self.agent.observation = obs

	def execute_actions(self, a_idx, hand_pol = False):
		"""
		We execute 'action' in the game
		Returns the reward that the agent gets after executing the action 
		"""
		if hand_pol:
			a_idx = self._handcrafted_policy()

		actions = [Actions(a_idx)]
		"""
		If the multi-agent extension has a decentralized SM this action 
		execution should be changed
		"""
		suceeded = self.env.execute_actions(actions)
		# If the environment produces a new obsrevation it's extended for the NM
		if suceeded: self.extend_observation()  
		# Progressing the LTL reward and dealing with the consequences...

		if self.specs_type == 'TTL1':
			reward, self.ltl_game_over, self.env_game_over = self._get_TTL1_rewards()
		else:
			reward, self.ltl_game_over, self.env_game_over = self._get_TTL2_rewards(action=a_idx)
		self.nSteps+=1
		return reward
	def _handcrafted_policy(self):
		reachable_pos = []
		agent = self.agent
		# To select the center of the observation
		ni = agent.i
		nj = agent.j

		reachable_pos.append(str(self.env.back_map[ni-1][nj]).strip()) #up
		reachable_pos.append(str(self.env.back_map[ni][nj+1]).strip()) #right
		reachable_pos.append(str(self.env.back_map[ni+1][nj]).strip()) #down
		reachable_pos.append(str(self.env.back_map[ni][nj-1]).strip()) #left
		a=0 #Up is 0
		Bad_move = []

		# Analizing neigbours and moving if satisfies
		for p in reachable_pos: 
			# print ("position", p)
			if p in 'abcdefghijklmnopqrstuvwxyz' and p.strip() != '':
				if len(self.spec)<2:
					if p == self.spec[0]:
						return a
					else: Bad_move.append(a)

				elif len(self.spec)==2:
					s = 0
					if self.spec[0] == "!": s=1
					if p != self.spec[s]: return a
					else: Bad_move.append(a)
				else:
					if p == self.spec[0] or p == self.spec[2]: return a
					else:Bad_move.append(a)
			a+=1
		# Nothing good
		counter = 10
		while counter>0:
			a = np.random.randint(4)
			if a not in Bad_move: return a
			else: counter-=1
		# if sorrounded by bad objects
		return np.random.randint(4)

	def _generate_Vocab(self, aux_env):
		words = aux_env.vocab 
		for word in words:
			self.vocab[word]

	def preProcessText(self, text):
		# print('text', text)
		tokens = re.findall("([a-zA-Z\+\-\&]+)", text)
		var_indexed_text = [[self.vocab[token] for token in tokens]]
		preproc_tex = tf.keras.preprocessing.sequence.pad_sequences(var_indexed_text, 
						maxlen=self.maxTextLen, dtype='int32', padding='post')
		# print('var_indexed_text ',var_indexed_text)
		# print('preproc_tex ', preproc_tex )
		

		# cat_preproc_tex = tf.one_hot(preproc_tex[0], self.vocab.max_size) #very uneffcicient use rather numpy
		# not used:
		# cat_preproc_tex = np.zeros((self.maxTextLen,self.vocab.max_size))
		# cat_preproc_tex[np.arange(self.maxTextLen),preproc_tex[0]] = 1

		# print('preproc_tex', preproc_tex[0])
		# print('cat_preproc_tex', cat_preproc_tex.shape)

		# jajaj+=1
		# return cat_preproc_tex
		return preproc_tex[0]

	# @tf.function
	# def _cat_text(self,text, max_size):
	# 	return tf.one_hot(text, max_size)

def _get_bracket_idxs(instruction, Idx, closing = False):
	"""
	Returns the idx of the corresponding closing/opening bracket
	 	inputs:
	 	- instruction (str): the formula that contains the bracket
	 	- start (int): the idx of the anchor bracket
	 	- reversed (boolean): looks for the opening bracket instead of the 
	 		closing one
	"""
	if not closing:
		e_idx = Idx+1
		bracket = 1
		for j in range(Idx+1, len(instruction)):
			if instruction[j] == "(" : bracket +=1
			elif instruction[j]== ")":
				bracket -=1
				if bracket == 0:
					e_idx = j
					break
	else:
		e_idx = Idx-1
		bracket = 1
		for j in range(Idx-1, -1, -1):
			if instruction[j] == ")" : bracket +=1
			elif instruction[j]== "(":
				bracket -=1
				if bracket == 0:
					e_idx = j
					break

	return e_idx


