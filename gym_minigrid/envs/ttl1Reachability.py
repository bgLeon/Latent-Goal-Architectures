from gym_minigrid.minigrid import *
from gym_minigrid.register import register
class ttl1ReachabilityEnv(MiniGridEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""

	def __init__(
		self,
		size=7,
		agent_start_pos=(1,1),
		agent_start_dir=0,
		rand_sizes = None,
		Obj_set = 0, #0 is training, 1 is validation, 2 is test
		# n_actions = 3
		BCM = False,
		n_actions = 6,
		populationSz = 'large',
		mapsize = 5
	):
		self.agent_start_pos = agent_start_pos
		self.agent_start_dir = agent_start_dir
		self.Obj_set = Obj_set
		self.vocab = self._getVocab()
		self.n_actions = n_actions

		# To modify the set without modifying neuro-symbilc file
		# self.Obj_set = 4
		# BCM = True
		# --------

		self.BCM = BCM
		self.populationSz = populationSz
		print('loaded mode with', n_actions, 'actions' )
		super().__init__(
			grid_size=size,
			# max_steps=30,
			max_steps=8*size*size,
			rand_sizes=rand_sizes,
			# Set this to True for maximum speed
			agent_view_size=5,
			see_through_walls=True
		)
	def _gen_positive_spec(self, colors, population, actions):
		i1 = [np.random.choice(colors), np.random.choice(population)]
		a1 =  np.random.choice(actions)
		self.mission = a1+' '+ i1[0] + ' ' + i1[1]

		self.goals = [[a1,i1]]

	def _getVocab(self):
		tlOperators=['-','V']
		vocab = []
		for word in TL_OBJECTS:
			vocab.append(word)
		for word in COLOR_NAMES:
			vocab.append(word)
		for word in tlOperators:
			vocab.append(word)
		return vocab

	def _gen_disj_spec(self, colors, population, actions):
		i1 = [np.random.choice(colors), np.random.choice(population)]
		a1 =  np.random.choice(actions)
		i2 = [np.random.choice(colors), np.random.choice(population)]
		a2 =  np.random.choice(actions)

		if i1 == i2:
			counter = 20
			while i1 == i2:
				i2 = [np.random.choice(colors), np.random.choice(population)]
				counter-=1
				if counter==0: break
		self.mission= a1+' ' +i1[0]+' '+i1[1]+' V '+a2+' '+ i2[0] + ' ' + i2[1]

		self.goals = [[a1,i1],[a2,i2]]


	def _gen_negative_spec(self, colors, population, actions):
		i1 = [np.random.choice(colors), np.random.choice(population)]
		a1 =  np.random.choice(actions)
		self.mission = a1+' ' +i1[0] + ' ' + i1[1]+ ' -'

		self.goals = [[a1,i1]]

	def _get_object_Call(self, obj_type):
		if obj_type == 'wall':
			v = Wall
		elif obj_type == 'floor':
			v = Floor
		elif obj_type == 'ball':
			v = Ball
		elif obj_type == 'tile':
			v = Tile
		elif obj_type == 'rhombus':
			v = Rhombus
		elif obj_type == 'star':
			v = Star
		elif obj_type == 'key':
			v = Key
		elif obj_type == 'ring':
			v = Ring
		elif obj_type == 'box':
			v = Box
		elif obj_type == 'door':
			v = Door
		elif obj_type == 'goal':
			v = Goal
		elif obj_type == 'lava':
			v = Lava
		else:
			assert False, "unknown object type in decode '%s'" % obj_type
		return v

	def _generate_fillerPopulation(self, n_fillers, colors, population):
		fill_pop=[]
		# print('colors', colors)
		# print('population', population)
		# print('self.goals', self.goals)
		for i in range(n_fillers):
			old = True
			counter = 20
			while old:
				i1 = [np.random.choice(colors), np.random.choice(population)]
				old = False
				for goal in self.goals: 
					if goal[1] == i1: 
						old = True       
				counter-=1
				if counter==0: break
			if not old: fill_pop.append(i1)
		return fill_pop

	def _gen_grid(self, width, height):
		# Create an empty grid
		self.grid = Grid(width, height)

		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# if self.BCM:
		# 	self.specT = 1
		MaxActions = ['reach', 'get', 'moveUp', 'consume']

		if self.populationSz == 'small':
			shapes_min = 2 
			colors_min = 2 if (self.Obj_set == 2 or self.Obj_set == 4) else 3 #To have at least 3 test objects (combining colors and shapes)

			shapes_max = 4
			colors_max = 4

		elif self.populationSz == 'Lsmall': #2 and 5 in the old style
			shapes_min = 3 
			colors_min = 3 if (self.Obj_set == 2 or self.Obj_set == 4) else 4 #To have at least 3 test objects

			shapes_max = 5
			colors_max = 5

		elif self.populationSz == 'medium':
			shapes_min = 4
			colors_min = 4 if (self.Obj_set == 2 or self.Obj_set == 4) else 5

			shapes_max = 6
			colors_max = 6

		else:
			shapes_min = len(TL_OBJECTS)-3
			colors_min = len(COLOR_NAMES)-4

			shapes_max = len(TL_OBJECTS)
			colors_max = len(COLOR_NAMES)


		if self.Obj_set == 0 or self.Obj_set == 3:
			self.specT = np.random.choice(3, p=[0.35,0.4, 0.25]) # 
			# self.specT = np.random.choice([1,2]) # When testing only!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

			population = TL_OBJECTS[:shapes_min]  if self.specT != 0 else TL_OBJECTS[:shapes_max]
			colors = COLOR_NAMES[:colors_min] if self.specT != 0 else COLOR_NAMES[:colors_max]
			actions = MaxActions[:len(MaxActions)-2] if self.specT != 0 else MaxActions[:]
		elif self.Obj_set == 1:
			self.specT = np.random.choice([1,2]) # 
			population = [TL_OBJECTS[shapes_min]]
			colors = COLOR_NAMES[:colors_min]
			actions = MaxActions[len(MaxActions)-2]
		else:
			self.specT = np.random.choice([1,2])
			population = TL_OBJECTS[shapes_min:shapes_max]
			colors = COLOR_NAMES[colors_min:colors_max]
			actions = MaxActions[:]

		# print('-----New-----')
		# print('specT', self.specT)
		# print('Obj_set', self.Obj_set)
		# print('population', self.populationSz)
		# print('shape', population)
		# print('colors', colors)

		# self.specT = 2

		if self.n_actions == 3 : actions= ['reach'] #Change N_actions too!!

		if self.specT == 0:   self._gen_positive_spec(colors, population, actions)
		elif self.specT == 1: self._gen_negative_spec(colors, population, actions)
		elif self.specT == 2: self._gen_disj_spec(colors, population, actions)

		# To fill with new objects to use as distractors
		maxP= 8
		n_fillers = np.random.randint(4,maxP) if self.specT != 1 else np.random.randint(1,4)


		if self.Obj_set == 1 and self.populationSz == 'small':
		# 	for i in range(4):
				population.append(TL_OBJECTS[0])
		# 		colors.append(COLOR_NAMES[i])

		# print('n_fillers', n_fillers)
		fill_pop = self._generate_fillerPopulation(n_fillers, colors, population)

		# print('fill_pop', fill_pop)

		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
			self.agent_dir = self.agent_start_dir
		else:
			self.place_agent()

		# Place the agent
		if self.agent_start_pos is not None:
			self.agent_pos = self.agent_start_pos
			self.agent_dir = self.agent_start_dir
		else:
			self.place_agent()
		
		us_width = width-2 #removing the walls

		# Place the goals
		pickable = True
		n_goals = np.random.randint(1,3)

		if self.BCM: n_goals = 1 
		elif self.goals[0][0] == 'moveUp': n_goals = 3
		elif self.specT == 1: 
			n_goals = np.random.randint(2,5)

		#-----------comment if BCM with disjuntions
		for _ in range(n_goals): 
			goal = self.goals[np.random.randint(len(self.goals))]
			Object= self._get_object_Call(goal[1][1])
			self.place_obj(Object(goal[1][0], overlap=True, pickup=pickable))

		#Hyperparameters for safety conditions 

		#Place filler population
		max_fillers = us_width*2 if self.specT != 1 else 4
		min_fillers = 1 if self.specT == 1 else 3
		n_fillers = np.random.randint(min_fillers,max_fillers)
		if self.BCM:
			n_fillers = 1

		# -----------------till here

		# for goal in self.goals: #  2 objects are offered
		# 	Object= self._get_object_Call(goal[1][1])
		# 	self.place_obj(Object(goal[1][0], overlap=True, pickup=pickable))
		# n_fillers = 2

		# 1 choice 

		# goal = self.goals[0] #  1st object is offered
		# # goal = self.goals[1] #  2nd object is offered
		# Object= self._get_object_Call(goal[1][1])
		# self.place_obj(Object(goal[1][0], overlap=True, pickup=pickable))
		# n_fillers = 1

		# ----------------------------

		for _ in range(n_fillers):
			filler = fill_pop[np.random.randint(len(fill_pop))]
			Filler = self._get_object_Call(filler[1])
			self.place_obj(Filler(filler[0], overlap=True, pickup=pickable))
		
		# print(self.mission)

	def step(self, action):
		self.preCarrying = self.carrying
		obs, reward, done, info = MiniGridEnv.step(self, action)

		reward, done = self._get_rewards(action)
		if self.step_count >= self.max_steps:
			done = True
		# print('action', action, '; reward',reward)

		# if self.carrying: #'consume' action, not in use
		#     if action == self.actions.toggle: 
		#         self.carrying = None

		# if self.carrying:
		#     if self.carrying.color == self.targetColor and \
		#        self.carrying.type == self.targetType:
		#         reward = self._reward()
		#         done = True
		#     else:
		#         reward = 0
		#         done = True

		return obs, reward, done, info

	def _get_rewards(self, action):
		cell= self.grid.get(self.agent_pos[0], self.agent_pos[1])
		reward = -0.05 
		# if cell is not None or self.carrying
		done = False
		#Only negative goal condition
		for goal in self.goals:
			if goal[0] == 'reach':
				if cell is not None:
					if (goal[1][0] == cell.color and goal[1][1] == cell.type):	
						reward = 1 if self.specT != 1 else -1
					else:
						reward = 1 if self.specT == 1 else -1
			elif goal[0] == 'get':
				if self.carrying: 
					if (goal[1][0] == self.carrying.color and \
						goal[1][1] == self.carrying.type): 
						reward = 1 if self.specT != 1 else -1
					else:
						reward = 1 if self.specT == 1 else -1
			elif goal[0] == 'moveUp':
				if action == self.actions.drop and self.preCarrying:
					u, v = self.dir_vec
					ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
					if not self.carrying and oy==1:
						if self.grid.get(ox, oy) is self.preCarrying:
							if (goal[1][0] == self.preCarrying.color \
								and goal[1][1] == self.preCarrying.type):
								reward = 1 if self.specT != 1 else -1
							else:
								reward = 1 if self.specT == 1 else -1
			elif goal[0] == 'consume':
				if action == self.actions.toggle and self.preCarrying:
					if (goal[1][0] == self.preCarrying.color \
						and goal[1][1] == self.preCarrying.type):
						reward = 1 if self.specT != 1 else -1
					else:
						reward = 1 if self.specT == 1 else -1
			else:
				raise Exception("This action has no reward associated")

			if reward == 1: break
		if self.Obj_set == 4 or self.Obj_set == 3: #Deceptive
			if reward == 1: reward = -1
			elif reward == -1: reward = 1
		done = True if reward == 1 else False
		return reward, done




class ttl1ReachabilityEnv5x5(ttl1ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=5, **kwargs)

class ttl1ReachabilityEnvRandom5x5(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(size=5, agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnv7x7(ttl1ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=7, agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnv7x7_3A(ttl1ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=7, agent_start_pos=None, n_actions=3, **kwargs)

class ttl1ReachabilityEnvRandom7x7(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(size=7, agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnvRandom58(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(rand_sizes=[7,10], agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnvRandom22x22(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(size=22, agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnvRandom14x14(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(size=14, agent_start_pos=None, **kwargs)

class ttl1ReachabilityEnv6x6(ttl1ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=6, **kwargs)
		random.seed(0)

class ttl1ReachabilityEnvRandom6x6(ttl1ReachabilityEnv):
	def __init__(self):
		super().__init__(size=6, agent_start_pos=None)

class ttl1ReachabilityEnv16x16(ttl1ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=16, **kwargs)

register(
	id='MiniGrid-ttl1Reachability-5x5-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnv5x5'
)

register(
	id='MiniGrid-ttl1Reachability-7x7-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnv7x7'
)

register(
	id='MiniGrid-ttl1Reachability-7x7-3A-v1',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnv7x7_3A'
)


register(
	id='MiniGrid-ttl1Reachability-R-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnvRandom58'
)


register(
	id='MiniGrid-ttl1Reachability-22x22-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnvRandom22x22'
)


register(
	id='MiniGrid-ttl1Reachability-14x14-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnvRandom14x14'
)

register(
	id='MiniGrid-ttl1Reachability-R5x5-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnvRandom5x5'
)

register(
	id='MiniGrid-ttl1Reachability-R-7x7-v0',
	entry_point='gym_minigrid.envs:ttl1ReachabilityEnvRandom7x7'
)


# register(
#     id='MiniGrid-Empty-6x6-v0',
#     entry_point='gym_minigrid.envs:ttl1ReachabilityEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-Random-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyRandom6x6'
# )

# register(
#     id='MiniGrid-Empty-8x8-v0',
#     entry_point='gym_minigrid.envs:ttl1ReachabilityEnv'
# )

# register(
#     id='MiniGrid-Empty-16x16-v0',
#     entry_point='gym_minigrid.envs:ttl1ReachabilityEnv16x16'
# )
