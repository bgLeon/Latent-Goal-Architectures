from utils.game_objects import *
from utils.specifications import *


import random, math, os, sys, copy, time
import numpy as np
"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class GameParams:
	def __init__(self, specs_type, test, fully_obs= False,
		file_map= None, visual = False, vdic = False, obs_range = 2, 
		extended_obs = True, moving_spec =  False, trainSz = 10, 
		complexT = 0, offset = 0, size = 5, resolution = 9, mapType="default"):
		"""
		Parameters
		-------
		specs_type: int
			TTL syntax used for training
		test: boolean
			If True this is a testing environment
		fully_obs: boolean
			optional accesd to the full map as input 
			(stops the visual invariance)
		visual: boolean 
			If True renders the map in a manner that each object is represented 
			as a matrix of a resolution given by vdic. Each object is 
			represented by a single value otherwise
		vdic: dictionary
			Maps objects to their matrix representation
		obs_range: int
			Tells how far the aget can perceive objects
		extended_obs: boolean
			Tells if the observation is etended with the TTL task
		moving_spec: boolean
		 	If True the TTL task can be placed in any position of the extension
		trainSz: int
			The variety of objects that belong to the training set
		complexT: boolean
			Tells if we are testing with complex TTL instructions
		offset: int
			For non-visual envs, adds an offset to the value assigned to an 
			object
		size: int
			Map dimensions are size x size
		resolution: int
			Each object in this game is represented by a resolution x resolution
			matrix
		t_limit: int
			Total number of steps to finish the episode 
		"""
		self.file_map = file_map
		self.test = test
		self.specs_type = specs_type
		self.fully_obs = fully_obs
		self.obs_range = obs_range
		self.extended_obs = extended_obs 
		self.moving_spec =  moving_spec
		self.trainSz= trainSz
		self.visual = visual
		self.mapType = mapType
		if visual:
			self.class_ids = vdic.class_ids
			self.resolution = vdic.resolution
		self.offset = offset
		self.complexT = complexT
		self.size = size
		self.resolution = resolution
		if size > 5:
			self.t_limit = 1000
		else:
			self.t_limit = 300

class Game:
	def __init__(self, params):
		self.params = params
		self.fully_obs = params.fully_obs
		self.obs_range = params.obs_range
		self.offset = params.offset
		self.visual = params.visual
		self.t_limit = params.t_limit
		self.complexT = params.complexT
		self.mapType = params.mapType
		sizes = [5,6,7,8]
		# sizes = [5,4,6,7,8,9,10]
		if self.params.test or params.size == 12345:
			self.size = np.random.choice(sizes, p=[0.85, 0.05, 0.05, 0.05])
			# self.size = np.random.choice(sizes, p=[0.76, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
			# self.size = np.random.choice(sizes, p=[0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])

			# self.size = 5
			if self.params.test : self.size = params.size
		else:
			 self.size = np.random.choice(sizes, p=[0.85, 0.05, 0.05, 0.05])
			# self.size = np.random.choice(sizes, p=[0.76, 0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
			# self.size = np.random.choice(sizes, p=[0.64, 0.06, 0.06, 0.06, 0.06, 0.06, 0.06])

		# print("Size", self.size)
		self.complexSubT = 0 #auxiliary counter of the current complex subTask
		# complexSubT is for symbolic Module
		self.AllSolved = False
		if self.visual:
			self.class_ids = params.class_ids
			self.resolution = params.resolution
		trainSz = params.trainSz

		# getting specifications and objects available
		if params.mapType == "default":
			AffSpecs, NegSpecs, DisjSpecs, \
			 	self.candidates = get_learning_specs(trainSz, params.test)

			specT = np.random.choice(params.specs_type, p=[0.3, 0.3, 0.4])
			# specT = 1
			if specT == 0:
				self.specifications = AffSpecs
				# Positive
			elif specT == 1:	
				self.specifications = NegSpecs
				# Negative
			else:  
				self.specifications = DisjSpecs
				# Non-deterministic choice

			if self.complexT:
				self.specifications, self.candidates = get_Complex_specs()

				if self.complexT>len(self.specifications):
					self.spec = self.specifications[self.complexT-1]
					self.AllSolved = True
				else:
					self.spec = self.specifications[self.complexT]

			else:
				n_spec=np.random.randint(len(self.specifications))
				self.spec = self.specifications[n_spec]

		else:
			self.specifications, self.candidates = get_Until_specs(\
				trainSz, params.test)
			n_spec = np.random.randint(len(self.specifications))
			self.spec = self.specifications[n_spec]
		if params.file_map is not None:
				self._load_map(params.file_map)
		else:
			self._generate_map()
		self.nSteps = 0
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
			action = actions[i]

			# Getting new position after executing action
			i,j = agent.i,agent.j
			ni,nj = self._get_next_position(action, i, j)

			# Interacting with the objects that is in the next position
			action_succeeded = self.map_array[ni][nj].interact(agent)

			# Action can only fail if the new position is a wall or another agent
			if action_succeeded:
				# changing agent position in the map
				self._update_agent_position(agent, ni, nj)
				self._update_agent_observation(agent)
			self.heat_map[ni][nj] += 1
		return action_succeeded

	def _get_next_position(self, action, ni, nj):
		"""
		Returns the position where the agent would be if we execute action
		"""
		if action == Actions.up and ni>0  : ni-=1
		if action == Actions.down and ni<(self.map_height-1): ni+=1
		if action == Actions.left and nj>0: nj-=1
		if action == Actions.right and nj<(self.map_width-1): nj+=1
		
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
		return image


	def show_map(self):
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
		print("The Map becomes:", image)

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

	def _populatePath_map(self, np_map, size):
		"""
		Returns the map populated with objects and the agent
		"""
		candidates = self.candidates
		population = list('A')
		population.append(self.spec[0])
		population.append(self.spec[2])

		# Selects the number of objects to be present
		path_length = np.random.randint(3,size*2+2)

		Ax = random.randint(1,size-2)
		Ay = random.randint(1,size-2)
		np_map[Ax][Ay] = 'A'
		ni = copy.copy(Ax)
		nj = copy.copy(Ay)

		goal_placed = False

		for i in range(path_length):
			obj = population[1] 
			if i == (path_length-1):
				obj = population[2]
			counter = 10
			while counter>0:

				d = np.random.randint(4)
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
				if np_map[pi][pj] in 'abcdefghijklmnopqrstuvwxyzXA'\
					and np_map[pi][pj].strip() != '':
					counter-=1
					continue
				np_map[pi][pj] = obj
				if obj == population[2]: goal_placed = True
				ni, nj = pi, pj
				break
		if not goal_placed: np_map[ni][nj] = population[2]

		fill_population = []
		c = np.random.randint(3,8)
		for i in range(c):
				t = np.random.randint(len(candidates))
				fill_population.append(candidates[t])

		attemps = np.random.randint(3,30)
		for _ in range(attemps):
			item = np.random.choice(fill_population)
			x = random.randint(1,size-2)
			y = random.randint(1,size-2)
			if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyzXA'\
				and np_map[x][y].strip() != '':
				continue
			else:
				np_map[x][y] = item
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
				if np_map[x][y] in 'abcdefghijklmnopqrstuvwxyzXA'\
					and np_map[x][y].strip() != '':
					continue
				else:
					np_map[x][y] = item
					count+=1
				if count > 0 and item == "A": break
				if item in "abcdefghijklmnopqrstuvwxyz" and count > 0: break
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
		for i in range(size):
			for j in range (size):
				# The limits of the map will have X as an obstacle
				if (i == 0 or i == size-1 or j == 0 or j == size-1) \
					and (not self.fully_obs): 
						np_map[i,j] ='X'
				else: np_map[i,j] =' '

		if self.mapType == "default":
			if not self.fully_obs: 
				np_map = self._populate_map(np_map, size=size-2)
			else:
				np_map = self._populate_map(np_map, size=size)
		else:
			np_map = self._populatePath_map(np_map, size=size-2)

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
			for e in " AX!VTUabcdefghijklmnopqrstuvwxyz":
				self.class_ids[e] = len(self.class_ids)*8

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
				if e in "abcdefghijklmnopqrstuvwxyzH":
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
			self.back_map[i][j] =  Empty(i,j,label=" ")
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
		return [Actions.up, Actions.right, Actions.down, Actions.left]