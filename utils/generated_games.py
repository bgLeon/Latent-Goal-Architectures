from utils.game_objects import *
from utils.dfa import *
import random, math, os, sys, copy, time
import numpy as np

"""
Auxiliary class with the configuration parameters that the Game class needs
"""
class GameParams:
	def __init__(self, file_map, ltl_spec, consider_night, fully_obs,
		obs_range = 2, extended_obs = True):
		self.file_map = file_map
		self.ltl_spec = ltl_spec
		self.consider_night = consider_night
		# optional partially observable environment
		self.fully_obs = fully_obs

		# how far the agent "sees"
		self.obs_range = obs_range
		# The optional extension of the observation with the specification"
		self.extended_obs = extended_obs 

class Game:
	def __init__(self, params, gen_file_map = False, test = False):
		self.params = params
		# Adding options if needed
		self.consider_night = params.consider_night
		self.fully_obs = params.fully_obs
		self.extended_obs = params.extended_obs
		self.obs_range = params.obs_range
		# Loading the DFA
		self.dfa = DFA(params.ltl_spec)

		if test:
			self._load_map(params.file_map)
		else:
			self._load_map(gen_file_map)

		self.nSteps = 0
		self.hour = 12
		if self.consider_night:
			self.sunrise = 5
			self.sunset  = 21
		reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
		reward = 0 # for initialization we don't penalize the agents
		for agent in self.agents.values():
			agent.update_reward(reward)
			self._update_agent_observation(agent)

	def execute_actions(self, actions):
		"""
		We execute 'action' in the game
		Returns the reward that the agent gets after executing the action 
		"""
		agents = self.agents
		self.hour = (self.hour + 1)%24

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

		# Progressing the LTL reward and dealing with the consequences...
		# As it is right now all the agents share reward and consequences
		reward, self.ltl_game_over, self.env_game_over = self._get_rewards()
		for agent in agents.values():
			agent.update_reward(reward)
		self.nSteps+=1
		# we continue playing
		return reward

	def _get_next_position(self, action, ni, nj):
		"""
		Returns the position where the agent would be if we execute action
		"""
			
		# OBS: Invalid actions behave as wait
		if action == Actions.up and ni>0  : ni-=1
		if action == Actions.down and ni<(self.map_height-1): ni+=1
		if action == Actions.left and nj>0: nj-=1
		if action == Actions.right and nj<(self.map_width-1): nj+=1
		
		return ni,nj

	# Updates the map with the new position of the agent
	def _update_agent_position(self, agent, ni, nj):

		i, j = agent.i, agent.j
		self.map_array[i][j] = self.back_map[i][j] # we recover what was previously there
		agent.change_position(ni,nj)
		self.map_array[ni][nj] = agent # we update the map with the agent

	def get_actions(self, agent):
		"""
		Returns the list with the actions that the given agent can perform
		"""
		return agent.get_actions()

	def _get_rewards(self):
		"""
		This method progress the dfa and returns the 'reward' and if its game over
		"""
		if not self.consider_night:
			reward = -0.1
			ltl_game_over = False
			env_game_over = False if self.nSteps < 300 else True
			# env_game_over = False
			for agent in self.agents.values():
				true_props = self.get_true_propositions(agent)
				progressed = self.dfa.progress(true_props)
				if true_props in 'abcdefgh' and true_props.strip() != '' and not progressed:
					# print("truepropos: ", true_props)
					if true_props != self.dfa.state_representation[0]:
						reward = -1
						# print("Penalization!!!!")
					# else:
						# print("WELL DONE!")


				ltl_game_over = self.dfa.is_game_over() or ltl_game_over
				if progressed: reward = 0
				if self.dfa.in_terminal_state(): 
					reward = 1
					"""
					For now we allow only one dfa progression at the same time, 
					This is in order to easily keep track of the optimal policies
					"""
					# break 
			return reward, ltl_game_over, env_game_over
		else:
			reward = -1
			ltl_game_over = False
			env_game_over = False if self.nSteps < 500 else True
			for agent in self.agents.values():
				true_props = self.get_true_propositions(agent)
				progressed = self.dfa.progress(true_props)
				if progressed: reward = 0
				ltl_game_over = self.dfa.is_game_over() or ltl_game_over
				if ltl_game_over: reward = -1*(500-self.nSteps)
				if self.dfa.in_terminal_state(): 
					reward = 1
					game.show_map()
					"""
					For now we allow only one dfa progression at the same time, 
					This is in order to easily keep track of the optimal policies
					"""
					# break
			return reward, ltl_game_over, env_game_over

	def get_LTL_goal(self):
		"""
		Returns the next LTL goal
		"""
		return self.dfa.get_LTL()

	def _is_night(self):
		return not(self.sunrise <= self.hour <= self.sunset)

	def _steps_before_dark(self):
		if self.sunrise - 1 <= self.hour <= self.sunset:
			return 1 + self.sunset - self.hour
		return 0 # it is night

	"""
	Returns the string with the propositions that are True in this state
	"""
	def get_true_propositions(self, agent):
		ret = str(self.back_map[agent.i][agent.j]).strip()
		# adding the is_night proposition
		if self.consider_night and self._is_night():
			ret += "n"
		return ret
	"""
	The following methods return a feature representations of the map ------------
	for a given agent
	"""
	def _update_agent_observation(self, agent):
		# Fully Observable Environment
		if self.fully_obs:
			# Extending with the DFA
			if self.extended_obs:
				obs = []
				row = []
				needs_space = False # to give spaces between specifications
				for i in range(self.map_width):
					if i < len(self.dfa.state_representation) and not needs_space:
						row.append(self.dfa.state_representation[i])
						needs_space = True
					else:
						needs_space = False
						row.append(' ')
				obs.append(row)
				# Adding the map
				for row in self.map_array:
					obs.append(row)
				# full map without the DFA
			else: obs =  self.map_array

		# Partial Observable Environment
		else:
			obs = []
			# Extending with the DFA
			if self.extended_obs:
				row = []
				needs_space = False # to give spaces between specifications
				for i in range(self.obs_range*2+1):
					if i < len(self.dfa.state_representation) and not needs_space:
						row.append(self.dfa.state_representation[i])
						needs_space = True
					else:
						needs_space = False
						row.append(' ')
				obs.append(row)
			# The partial Obs
			ci, cj = agent.i, agent.j
			for i in range(ci-self.obs_range, ci+self.obs_range+1):
				row = []
				for j in range(cj-self.obs_range, cj+self.obs_range+1):
					if i<0 or i>=self.map_height or j<0 or j>=self.map_width:
						row.append(' ')
					else:
						# print('test', type(str(self.map_array[i][j])))
						row.append(self.map_array[i][j])
						# print('ok')
					# print('obs', obs)
				obs.append(row)
		agent.observation = obs

	def get_observation(self, agent):
		obs = agent.observation
		N = len(obs[:])
		M = len(obs[0][:])
		image = np.zeros((N,M), dtype=int)
		for i in range(N):
			for j in range(M):
				try:
					image[i][j] = self.class_ids[str(obs[i][j])]
				except:
					print("An exception occurred")
					print("i", i)
					print("j", j)
					print("what is image", image[i][j])
					print("what is obs", str(obs[i][j]))
					print("DFA obs:", self.dfa.state_representation)
					ghAG+=1
		return image
		
		# Adding the number of steps before night (if need it)
		if self.consider_night:
			ret.append(self._steps_before_dark())

		return np.array(ret, dtype=np.float64)


	def _manhattan_distance(self, obj, agent):
		"""
		Returns the Manhattan distance between 'obj' and the agent
		"""
		return abs(obj.i - agent.i) + abs(obj.j - agent.j)


	# The following methods create a string representation of the current state ---------
	def show_map(self):
		"""
		Prints the current map
		"""
		if self.consider_night:
			print(self.__str__(),"\n",
				"Steps before night:", self._steps_before_dark(), "Current time:", self.hour,
				"\n" ,"Reward:", self.agents[0].reward,
				"Agent has", self.agents[0].num_keys, "keys.", "Goal", self.get_LTL_goal())
		else:
			print(self.__str__(), "\n obs:")
			obs = self.agents[0].observation
			obs_str = self._get_obs_str(obs)
			print(obs_str, 
				"Reward:", self.agents[0].reward,
				"Agent has", self.agents[0].num_keys, "keys.", 
				"Goal", self.get_LTL_goal())
			# obs_str = game._get_obs_str(agent.observation)
			# print(obs_str)
			# print("\nend of show \n")
# print(game.get_observation(agent))
	def __str__(self):
		return self._get_map_str()

	def _get_obs_str(self, obs):
		r = ""
		# print("debugging")
		# print("obs", obs)
		# print("i range:", len(obs))
		# print("j range:", len(obs[:][0]))
		for i in range(len(obs[:])):
			s = ""
			for j in range(len(obs[0][:])):
				# print("i:", i, "j:", j)
				s += str(obs[i][j])
			if(i > 0):
				r += "\n"
			r += s
		return r

	def _get_map_str(self):
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


	def _generate_map (self, size = 5):
		# random.seed(seed)
		np_map = np.empty((size, size), dtype='str')
		for i in range(size):
			for j in range (size):
				# The limits of the map will have X as an obstacle
				if i == 0 or i == size-1 or j == 0 or j == size-1: 
					# np_map[i,j] ='X'
					np_map[i,j] =' '
				else: np_map[i,j] =' '
				# np_map[i,j] =' '
		# population = list('Aabcdefgh')
		population = list('Aabcd')
		for item in population:
			count = 0
			while True:
				x = random.randint(1,size-2)
				y = random.randint(1,size-2)
				# x = random.randint(0,size-1)
				# y = random.randint(0,size-1)
				if np_map[x][y] == ' ': 
					np_map[x][y] = item
					count+=1
				if count > 0 and item == "A": break
				if item in "abcdegh" and count > 0: break
				if item in "adf" and count > 0: break	
		# contains all the actions that the agent can perform
		actions = self._load_actions()
		# loading the map
		self.map_array = []
		self.class_ids = {} # I use the lower case letters to define the features
		self.agents = {}
		i,j = 0,0
		ag = 0
		for l in np_map:
			# I don't consider empty lines!
			# if(len(l.rstrip()) == 0): continue
			
			# this is not an empty line!
			row = []
			b_row = []
			j = 0
			for e in l:
				if e in "abcdefghijklmnopqrstuvwxyzH":
					entity = Empty(i,j,label=e)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)
				# we need to declare the initial positions of agents 
				# to be potentially empty espaces (after they moved)
				elif e in " ": 
					entity = Empty(i,j)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)
				elif e == "X": 
					entity = Obstacle(i,j)	
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)		
				elif e == "A":
					self.agents[ag] = Agent(i,j,actions)
					entity = self.agents[ag]
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)
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
		# height width
		self.map_height, self.map_width = len(self.map_array),\
			len(self.map_array[0])
		self.n_agents = len(self.agents)
		for agent in self.agents.values():
			self._update_agent_observation(agent)
		# self.show_map()
		# return np_map


	# The following methods create the map ----------------------------------------------
	def _load_map(self,file_map):
		"""
		This method adds the following attributes to the game:
			- self.map_array: array containing all the static objects in the map
				- e.g. self.map_array[i][j]: contains the object located on row 'i' and column 'j'
			- self.agents: are the agents
			- self.map_height: number of rows in every room 
			- self.map_width: number of columns in every room
		The inputs:
			- file_map: path to the map file
		"""
		# contains all the actions that the agent can perform
		actions = self._load_actions()
		# loading the map
		self.map_array = []
		self.class_ids = {} # I use the lower case letters to define the features
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
						self.class_ids[e] = len(self.class_ids)
				# we need to declare the initial positions of agents 
				# to be potentially empty espaces (after they moved)
				elif e in " ": 
					entity = Empty(i,j)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)
				elif e == "X": 
					entity = Obstacle(i,j)	
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)		
				elif e == "A":
					entity = Empty(i,j)
					self.agents[ag] = Agent(i,j,actions)
					if e not in self.class_ids:
						self.class_ids[e] = len(self.class_ids)
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
			self._update_agent_observation(agent)
		# self.show_map()
		# we initialize agent observation

	def _load_actions(self):
		return [Actions.up, Actions.right, Actions.down, Actions.left]
		# ,
			# Actions.wait]


def play(params, max_time):
	# commands
	str_to_action = {"w":Actions.up,"d":Actions.right,"s":Actions.down,
		"a":Actions.left}#, "e":Actions.wait}
	# play the game!
	game = Game(params)
	for t in range(max_time):
		# Showing game
		agent = game.agents[0]
		game._update_agent_observation(agent)
		game.show_map()
		# obs_str = game._get_obs_str(agent.observation)
		# print(obs_str)
		# print(game.get_observation(agent))
		acts = game.get_actions(game.agents[0])
		# Getting action
		print("\nSteps ", t)
		print("Action? ", end="")
		a = input()
		print()
		# Executing action
		if a in str_to_action and str_to_action[a] in acts:
			aactions = [str_to_action[a], Actions.up, Actions.up]
			reward = game.execute_actions(aactions)
			if game.ltl_game_over or game.env_game_over: # Game Over
				break 
		else:
			print("Forbidden action")
	game.show_map()

	return reward


# This code allow to play a game (for debugging purposes)
if __name__ == '__main__':
	import specifications
	map = "./debug/map_1.txt"
	#specs = get_sequence_of_subspecs
	#specs = specs.get_interleaving_subspecs()
	specs = specifications.get_sequence_of_subspecs()
	max_time = 100
	consider_night = False
	fully_obs = False
	for t in specs:
		# t = specs[-1]
		while True:
			params = GameParams(map, t, consider_night, fully_obs)
			if play(params, max_time) > 0:
				break