from utils.safety_game_objects import *

import random, math, os, sys, copy, time
import numpy as np

class Game:
	def __init__(self, params, taskType=None):
		self.params = params
		self.fully_obs = params.fully_obs
		self.obs_range = params.obs_range
		self.offset = params.offset
		self.visual = params.visual
		self.t_limit = params.t_limit
		self.complexT = params.complexT
		self.specType = params.specs_type
		sizes = [5,6,7,8]
		# specT = np.random.choice(4, p=[0.25, 0.4, 0.2, 0.15])
		if self.specType == 'TTL2':
			specT = np.random.choice(5, p=[0.2, 0.2, 0.15, 0.25, 0.2]) #---this one
		else:
			specT = np.random.randint(len(params.SpecSets))

		if taskType is not None:
			specT = taskType
		self.specT = specT
		# print('task in the game', specT)
		# specT = np.random.choice([1,2,4])
		# specT = np.random.choice([1,4]) #--This other
		# specT = 0
		# specT = 1
		# specT = 2
		self.specifications = params.SpecSets[specT]
		prob = [0.5, 0.1, 0.1, 0.3]
		if specT == 2: prob = [0.1, 0.1, 0.3, 0.5]

		if self.params.test or params.size == 12345:
			# self.size = np.random.choice(sizes, p=[0.85, 0.05, 0.05, 0.05])
			# self.size = np.random.choice(sizes, p=[0.76, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
			# self.size = np.random.choice(sizes, p=[0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])
			self.size = np.random.choice(sizes, p=[0.5, 0.1, 0.1, 0.3])
			self.size = 5
			if self.params.test : self.size = params.size
			if self.size > 5:
				self.t_limit = 1000
			else:
				self.t_limit = 300
		else:
			# self.size = np.random.choice(sizes, p=[0.85, 0.05, 0.05, 0.05])
			if self.specType == 'TTL2':
				self.size = np.random.choice(sizes, p=prob)
				if self.size > 5:
					self.t_limit = 1000
				else:
					self.t_limit = 300
			else:
				self.size = 5
				self.t_limit = 300
			# self.size = np.random.choice(sizes, p=[0.76, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
			# self.size = np.random.choice(sizes, p=[0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])

		# print("Size", self.size)
		self.complexSubT = 0 #auxiliary counter of the current complex subTask
		# complexSubT is for symbolic Module
		self.AllSolved = False
		if self.visual:
			self.class_ids = params.class_ids
			self.resolution = params.resolution
		# getting specifications and objects available
		# specT = np.random.choice(4, p=[0.5, 0.35, 0.15])
		# self.specifications = params.SpecSets[0]
		self.candidates = params.candidates

		if self.complexT:
			self.specifications = params.SpecSets #HEREE
			self.spec = np.random.choice( self.specifications)
			# print('complexT', self.spec)
			# if self.complexT>len(self.specifications):

		else:
			n_spec=np.random.randint(len(self.specifications))
			self.spec = self.specifications[n_spec]
		if params.file_map is not None:
				print("It shouldn't be used anymore")
				self._load_map(params.file_map)
		else:
			# print("Spec", self.spec)
			self._generate_map()
		self.action = 0
		self.step_count = 0
		
		# reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
		# reward = 0 # for initialization we don't penalize the agents
		for agent in self.agents.values():
			# agent.update_reward(reward)
			self._update_agent_observation(agent)


	def execute_actions(self, actions):
		"""
		We execute 'action' in the game
		"""
		agents = self.agents

		# So that the agents do not always make an action in the same order
		r = list(range(self.n_agents))
		random.shuffle(r)
		
		# Getting new position after executing action
		for i in r:
			agent = agents[i]
			self.action = actions[i]

			# Getting new position after executing action
			i,j = agent.i,agent.j
			ni,nj = self._get_next_position(self.action, i, j)

			# Interacting with the objects that is in the next position
			action_succeeded = self.map_array[ni][nj].interact(agent)

			# Action can only fail if the new position is a wall or another agent
			if action_succeeded:
				# changing agent position in the map
				self._update_agent_position(agent, ni, nj)
				self._update_agent_observation(agent)
			self.heat_map[ni][nj] += 1
		self.step_count+=1
		return action_succeeded


	def _get_next_position(self, action, ni, nj):
		"""
		Returns the position where the agent would be if we execute action
		"""
		if action == Actions.S_up and ni>0  : ni-=1
		if action == Actions.S_down and ni<(self.map_height-1): ni+=1
		if action == Actions.S_left and nj>0: nj-=1
		if action == Actions.S_right and nj<(self.map_width-1): nj+=1
		
		return ni,nj

	def _update_agent_position(self, agent, ni, nj):
		"""
		Updates the map with the new position of the agent
		"""
		i, j = agent.i, agent.j
		# We recover what was there before the agent came
		self.map_array[i][j] = self.back_map[i][j] 
		agent.change_position(ni,nj)
		# we place the agent in the new position
		self.map_array[ni][nj] = agent 

	def get_actions(self, agent):
		"""
		Returns the list with the actions that the given agent can perform
		"""
		return agent.get_actions()


	def _update_agent_observation(self, agent):
		"""
		Update the observation for the given agent
		"""
		if self.fully_obs:
			obs =  self.map_array

		# Partial Observable Environment
		else:
			obs = []
			ci, cj = agent.i, agent.j
			for i in range(ci-self.obs_range, ci+self.obs_range+1):
				row = []
				for j in range(cj-self.obs_range, cj+self.obs_range+1):
					if i<0 or i>=self.map_height or j<0 or j>=self.map_width:
						row.append(' ')
					else:
						row.append(self.map_array[i][j])
				obs.append(row)
		agent.observation = obs

	def get_observation(self, agent):
		obs = agent.observation
		N = len(obs[:])
		M = len(obs[0][:])
		vis = self.visual
		if not vis:
			image = np.zeros((N,M), dtype=int)
		else:
			res = self.resolution
			image = np.zeros((N*res,M*res), dtype=int)
		if not vis:
			for i in range(N):
				for j in range(M):
					try:

							image[i][j] = self.class_ids[str(obs[i][j])]
					except:
						e = str(obs[i][j])
						self.class_ids[e] = len(self.class_ids) + self.offset
						print("An exception occurred")
						print("i", i)
						print("j", j)
						print("what is image", image[i][j])
						print("what is obs", str(obs[i][j]))
						print("what is e", e)
						raise Exception('Undefined object in the environment')

		else:
			aux_list = []
			for i in range(N):
				row = self.class_ids[str(obs[i][0])]
				for j in range(1,M):
					try:
						row = np.hstack((row,self.class_ids[str(obs[i][j])]))
					except:
						e = str(obs[i][j])
						self.class_ids[e] = len(self.class_ids) + self.offset
						print("An exception occurred")
						print("i", i)
						print("j", j)
						print("what is image", image[i][j])
						print("what is obs", str(obs[i][j]))
						print("what is e", e)
						# print("DFA obs:", self.spec)
						raise Exception('Undefined object in the environment')
				aux_list.append(row)

			image = aux_list[0]
			for idx in range(1,N):
				image = np.vstack((image, aux_list[idx]))
		if len(image.shape)<3:
			image = np.expand_dims(image, -1)
		return image


	def render(self):
		"""
		Prints a representation of the current observation, for rendering
		"""

		
		obs = self.agents[0].observation
		obs_str = self._get_obs_str(obs)
		Rmap = self._get_obs_str(self.map_array)
		image = self.get_observation(self.agents[0])
		print('The map')
		print(Rmap)
		print("obs:")
		# print("The Map becomes:", image)

		print("Action", self.action)

		print(obs_str, 
			"Reward:", self.agents[0].reward,
			"Agent has", self.agents[0].num_keys, "keys.", 
			"Goal", self.spec)
		return self._get_map_str()

	def _get_obs_str(self, obs):
		"""
		Transforms the observation into a string for rendering
		"""
		r = ""
		for i in range(len(obs[:])):
			s = ""
			for j in range(len(obs[0][:])):
				s += str(obs[i][j])
			if(i > 0):
				r += "\n"
			r += s
		return r

	def _get_map_str(self):
		"""
		Transforms the env into a srting representation for render debugging
		"""
		r = ""
		agent = self.agents[0]
		for i in range(self.map_height):
			s = ""
			for j in range(self.map_width):
				# if agent.idem_position(i,j):
				# 	s += str(agent)
				# else:
					s += str(self.map_array[i][j])
			if(i > 0):
				r += "\n"
			r += s
		return r

	def _allocate_population(population, goal, np_map):
		for item in population:
			count = 0
			while True:
					x = random.randint(0,size-1)
					y = random.randint(0,size-1)
					if np_map[x][y] != goal or np_map[x][y]=="A": continue
					else:
						np_map[x][y] = item
						count+=1
					if count > 0 and item == "A": break
					if item in " abcdefghijklmnopqrstuvwxyzBCDFGHIJKLMNOPQRSWYZ" and count > 0: break
		return np_map

	def _populateTTL2_map(self, np_map, size):
		"""
		Returns the map populated with objects and the agent
		"""
		candidates = self.candidates
		# population.append(self.spec[0])
		# population.append(self.spec[2])
		# Selects the number of objects to be present

		prog_idx = len(self.spec)-1
		

		Ax = random.randint(1,size-2)
		Ay = random.randint(1,size-2)
		np_map[Ax][Ay] = 'A'
		# goal = self.spec[prog_idx]

		goal_placed = False

		if self.spec[prog_idx-1] == "-": #"get out with a specific move type"
			goal = self.spec[prog_idx]
			population=[]
			c = np.random.randint(1,3)
			for i in range(c):
				obj = goal
				while obj == goal:
					t = np.random.randint(len(candidates))
					obj = candidates[t]
				population.append(obj)
			for item in population:
				count = 0
				while True:
						x = random.randint(0,size-1)
						y = random.randint(0,size-1)
						if np_map[x][y] != goal or np_map[x][y]=="A": continue
						else:
							np_map[x][y] = item
							break
		else:
			goal = [self.spec[prog_idx]]
			if self.spec[prog_idx-2] == 'V':
				goal.append(self.spec[prog_idx-3])
			if self.spec[0] == "+": #follow a path to the goal

				# initialize the variables to create the path
				path_length = np.random.randint(2,size*2+2)
				ni = copy.copy(Ax)
				nj = copy.copy(Ay)
				goal_placed = False
				obj = []
				obj.append(self.spec[1])
				if self.spec[2] == 'V':
					obj.append(self.spec[4])
				gl = np.random.choice(goal)
				for i in range(path_length):
					next_obj = np.random.choice(obj)
					if i == (path_length-1):
						next_obj = gl
					counter = 10
					while counter>0:
						d = np.random.randint(4) #direction to extend the path
						pi =  copy.copy(ni)
						pj = copy.copy(nj)
						# Check we are not going outside the boundaries
						if d == 0:
							if ni<=0:
								counter-=1
								continue
							pi -= 1
						elif d == 1:
							if nj>=size-1:
								counter-=1
								continue
							pj += 1
						elif d == 2 :
							if ni>=size-1:
								counter-=1
								continue
							pi += 1
						elif d == 3:
							if nj<=0:
								counter-=1
								continue
							pj -= 1
						# place the object if there is nothing there already
						if np_map[pi][pj] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
							and np_map[pi][pj].strip() != '':
							counter-=1
							continue
						np_map[pi][pj] = next_obj
						if next_obj == gl: goal_placed = True
						ni, nj = pi, pj
						break
				if not goal_placed: 
					if np_map[ni][nj] != "A": 
						np_map[ni][nj] = np.random.choice(goal)
					else:
						# goal = self.spec[5]
						while True:
							x = random.randint(0,size-1)
							y = random.randint(0,size-1)
							if np_map[x][y] == "A": continue
							else:
								np_map[x][y] = np.random.choice(goal)

				fill_population = []
				c = np.random.randint(3,8)
				for i in range(c):
						t = np.random.randint(len(candidates))
						fill_population.append(candidates[t])
						# adding the posibility of a second solution
						# if self.spec[prog_idx-2] == 'V': 
						# 	extra_goal = self.spec[prog_idx]
						# 	if extra_goal == gl: 
						# 		extra_goal = self.spec[prog_idx-3]
						# 	fill_population.append(extra_goal)

				attemps = np.random.randint(3,20)
				for _ in range(attemps):
					item = np.random.choice(fill_population)
					x = random.randint(1,size-2)
					y = random.randint(1,size-2)
					if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
						and np_map[x][y].strip() != '':
						continue
					else:
						np_map[x][y] = item
			else:
				goal = [self.spec[prog_idx]]
				if self.spec[prog_idx-2] == 'V':
					goal.append(self.spec[prog_idx-3])
				while True:
					x = random.randint(0,size-1)
					y = random.randint(0,size-1)
					if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
						and np_map[x][y].strip() != '':
						continue
					else:
						np_map[x][y] = np.random.choice(goal)
						break
				if self.spec[0] == "-": # avoid a type of object until the goal
					attemps = np.random.randint(5,15) #This one
					# attemps = np.random.randint(2,6)

					item = self.spec[1] #object to avoid
					for _ in range(attemps):
						x = random.randint(1,size-2)
						y = random.randint(1,size-2)
						if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
							and np_map[x][y].strip() != '':
							continue
						else:
							np_map[x][y] = item
					fill_population = []
					c = min(23-attemps, np.random.randint(3,8)) #This one
					# c = min(23-attemps, np.random.randint(6,12))
					for i in range(c):
							t = np.random.randint(len(candidates))
							fill_population.append(candidates[t])
							# if self.spec[prog_idx-2] == 'V': 
							# 	extra_goal = self.spec[prog_idx]
							# 	if extra_goal == np_map[x][y]: 
							# 		extra_goal = self.spec[prog_idx-3]
							# 	fill_population.append(extra_goal)
					attemps = np.random.randint(3,15)
					for _ in range(attemps):
						item = np.random.choice(fill_population)
						x = random.randint(1,size-2)
						y = random.randint(1,size-2)
						if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
							and np_map[x][y].strip() != '':
							continue
						else:
							np_map[x][y] = item
				else:
					# We only care about the goal
					population = []
					for i in range(np.random.randint(3,8)):
						t = np.random.randint(1,len(candidates))
						population.append(candidates[t])
						if self.spec[prog_idx-2] == 'V': 
								extra_goal = self.spec[prog_idx]
								if extra_goal == np_map[x][y]: 
									extra_goal = self.spec[prog_idx-3]
								population.append(extra_goal)
					for item in population:
						while True:
							x = random.randint(0,size-1)
							y = random.randint(0,size-1)
							if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyz0123456789XABCDFGHIJKLMNOPQRSWYZ'\
								and np_map[x][y].strip() != '':
								continue
							else:
								np_map[x][y] = item
								break
		return np_map


	def _populate_map(self, np_map, size):
		"""
		Returns the map populated with objects and the agent
		"""
		candidates = self.candidates
		population = list('A')

		# Selects the number of objects to be present
		if len(self.spec)<2:
			# Positive task
			c = np.random.randint(3,8)
			for i in range(c):
				t = np.random.randint(len(candidates))
				population.append(candidates[t])

		elif len(self.spec)==2:
			# Negative task
			c = np.random.randint(2,6)
			for i in range(c):
				t = np.random.randint(1,len(candidates))
				population.append(candidates[t])

		else:
			# Non-deterministic-choice task
			choice = np.random.randint(4) 
			# To have more envs with the 2nd obj only
			if choice  < 2: 
				# Second object only
				if self.spec[0] in candidates:
					candidates = candidates.replace(self.spec[0], '')
			elif choice  == 3:
				# First object only
				if self.spec[2] in candidates:
					candidates = candidates.replace(self.spec[2], '')
			# else: both are present
			c = np.random.randint(3,8)
			for i in range(c):
				t = np.random.randint(1,len(candidates))
				population.append(candidates[t])
		# ------ In case we need specific populations
		# c = 2
		# population =	list('Acxkm')
		# population =	list('Afl')
		# population =	list('Afq')
		# population =	list('Acj')
		# population =	list('Aan')
		# population =	list('Aandq')
		# population =	list('Amx')



		# ----- likelihood
		# population = list('Avb')
		# population = list('Awj')

		# Inserting the objects of the spec in the population
		if len(self.spec) == 2:
			# Negative Task
			for e in self.spec:
				if e != '!' :
					if e not in population:
						t = np.random.randint(1,c+1)
						population[t] = e
					else:
						#to check that not all of them are the negated object
						diversity = False
						for ob in population[1:]:
							if e != ob: diversity = True
						if not diversity:
							t = np.random.randint(1,c+1)
							obj = e
							loop_break = 0 
							while obj == e:
								obj = candidates[np.random.randint(\
															len(candidates))]
								loop_break+=1
								if loop_break > 20:
									break
							population[t] = obj
		elif len(self.spec) == 3:
			# Non-deterministic choice
			if self.spec[0] in candidates:
				if self.spec[0] not in population:
					t = np.random.randint(1,c+1)
					population[t] = self.spec[0]
			if self.spec[2] in candidates:
				if self.spec[2] not in population:
					counting = 10
					while True:
						p = np.random.randint(1,c+1)
						if population[p] == self.spec[0] and counting > 0:
							counting-=1
							continue
						else:
							population[p]=self.spec[2]
							break
		else:
			# Positive task
			if self.spec[0] not in population:
				t = np.random.randint(1,c+1)
				population[t] = self.spec[0]

		# If Complex Tasks we populate with the test objects
		if self.complexT: population = list('Acjkmwx')

		# Interting the list of objects and the agent in the environment
		for item in population:
			count = 0
			while True:
				x = random.randint(0,size-1)
				y = random.randint(0,size-1)
				if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyzXABCDFGHIJKLMNOPQRSWYZ'\
					and np_map[x][y].strip() != '':
					continue
				else:
					np_map[x][y] = item
					count+=1
				if count > 0 and item == "A": break
				if item in " abcdefghijklmnopqrstuvwxyzBCDFGHIJKLMNOPQRSWYZ" and count > 0: break
		return np_map


	def _generate_map(self):
		"""
		Generates a random map of the given size
		"""
		size = self.size
		if not self.fully_obs: 
			# If it is not p.o. we need walls to show the limits
			size +=2
		np_map = np.empty((size, size), dtype='str')
		self.heat_map = np.zeros((size, size), dtype=np.uint16)
		default_ground = ' '
		prog_idx=len(self.spec)-1
		if self.spec[prog_idx-1] == "-": default_ground = self.spec[prog_idx]
		for i in range(size):
			for j in range (size):
				# The limits of the map will have X as an obstacle
				if (i == 0 or i == size-1 or j == 0 or j == size-1) \
					and (not self.fully_obs): 
						np_map[i,j] ='X'
				else: np_map[i,j] = default_ground

		if self.specType == 'TTL2':
			np_map = self._populateTTL2_map(np_map, size=size-2)
		else:
			np_map = self._populate_map(np_map, size=size-2)

		# contains all the actions that the agent can perform
		actions = self._load_actions()

		# loading the map
		self.map_array = []
		if not self.visual:
			"""
			Simpler game version with a different interger for each simbol
			We need to create a dictionary for each symbol, in the visual case
			this has been already done by vdic
			"""
			self.class_ids = {} 
			# Assign numbers to the objects
			for e in " AESX!-+VGNTFUBabcdefghijklmnopqrstuvwxyz0123456789":
				self.class_ids[e] = len(self.class_ids)*2

			# for e in "abcdefghijklmnopqrstuvwxyz":
			# 	self.class_ids[e] = len(self.class_ids)*24

			# if not self.small:
			# 	for e in "abcdefghijklmnopqrstuvwxyz":
			# 		if e != self.spec [0]:
			# 			noise = np.random.randint(-2,3)
			# 			self.class_ids[e] = self.class_ids[e] + noise
		self.agents = {}
		i,j = 0,0
		ag = 0
		for l in np_map:
			row = []
			b_row = []
			j = 0
			for e in l:
				if e in "abcdefghijklmnopqrstuvwxyz0123456789HBCDFGHIJKLMNOPQRSWYZ":
					entity = Empty(i,j,label=e)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)+self.offset

				# we need to declare the initial positions of agents 
				# to be empty espaces (after they moved somewhere else)
				elif e in " ": 
					entity = Empty(i,j)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids) + self.offset
				elif e == "X": 
					entity = Obstacle(i,j)	
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids) +self.offset		
				elif e == "A":
					self.agents[ag] = Agent(i,j,actions)
					entity = self.agents[ag]
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids) +self.offset
					ag+=1
				else:
					raise ValueError('Unkown entity ', e)
				row.append(entity)
				j += 1
			self.map_array.append(row)
			i += 1
		"""
		We use this back map to check what was there when an agent leaves a 
		position
		"""
		self.back_map = copy.deepcopy(self.map_array) 
		for agent in self.agents.values():
			i,j = agent.i, agent.j
			self.map_array[i][j] = agent
			self.back_map[i][j] =  Empty(i,j,label=default_ground)
			self.heat_map[i][j] += 1 
		# height width
		self.map_height, self.map_width = len(self.map_array),\
			len(self.map_array[0])
		self.n_agents = len(self.agents)
		for agent in self.agents.values():
			# We initialize the agent's observation
			self._update_agent_observation(agent)

	def _load_map(self, file_map):
		"""
		Similar to the method above, this method generates a map, 
		however it does it from a pre-created map that is used as input
		"""
		# contains all the actions that the agent can perform
		actions = self._load_actions()
		# loading the map
		self.map_array = []
		if not self.visual:
			self.class_ids = {} # Lower case letters define objects
			for e in " AX!VTUabcdefghijklmnopqrstuvwxyz":
				self.class_ids[e] = len(self.class_ids)+self.offset
		self.agents = {}
		f = open(file_map)
		i,j = 0,0
		ag = 0
		for l in f:
			# I don't consider empty lines!
			if(len(l.rstrip()) == 0): continue
			# this is not an empty line!
			row = []
			b_row = []
			j = 0
			for e in l.rstrip()[:-1]:
			# for e in l.rstrip()[:]:
				# print(l.rstrip(), "check")
				# print(l.rstrip()[:-1], "check")
				if e in "abcdefghijklmnopqrstuvwxyzH":
					entity = Empty(i,j,label=e)
					if e not in self.class_ids:
						raise ValueError( "Missed Object") 
						self.class_ids[e] = len(self.class_ids) + self.offset
				elif e in " ": 
					entity = Empty(i,j)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids) + self.offset
				elif e == "X": 
					entity = Obstacle(i,j)	
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)	+ self.offset	
				elif e == "A":
					self.agents[ag] = Agent(i,j,actions)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids) + self.offset
					ag+=1
				else:
					raise ValueError('Unkown entity ', e)
				row.append(entity)
				j += 1
			self.map_array.append(row)
			i += 1
		# We use this back map to check what was there when an agent leaves a 
		# position
		self.back_map = copy.deepcopy(self.map_array) 
		for agent in self.agents.values():
			i,j = agent.i, agent.j
			self.map_array[i][j] = agent
			self.back_map[i][j] =  Empty(i,j,label=" ")
		f.close()
		# height width
		self.map_height, self.map_width = len(self.map_array),\
			len(self.map_array[0])
		self.n_agents = len(self.agents)
		for agent in self.agents.values():
			# we initialize agent observation
			self._update_agent_observation(agent)

	def _load_actions(self):
		# return [Actions.S_up, Actions.S_right, Actions.S_down, Actions.S_left,
		# 		Actions.N_up, Actions.N_right, Actions.N_down, Actions.N_left,
		# 		Actions.F_up, Actions.F_right, Actions.F_down, Actions.F_left]
		return [Actions.S_up, Actions.S_right, Actions.S_down, Actions.S_left]