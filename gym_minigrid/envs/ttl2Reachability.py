from gym_minigrid.minigrid import *
from gym_minigrid.register import register
class ttl2ReachabilityEnv(MiniGridEnv):
	"""
	Empty grid environment, no obstacles, sparse reward
	"""

	def __init__(
		self,
		agent_start_pos=(1,1),
		agent_start_dir=0,
		rand_sizes = None,
		train = True,
		n_actions = 6,
		Obj_set = 0, #0 is training, 1 is validation, 2 is test
		BCM = False,
		populationSz = 'large',
		mapsize = 5,
		total_steps = 0
	):
		self.agent_start_pos = agent_start_pos
		self.agent_start_dir = agent_start_dir
		self.train = train
		self.vocab = self._getVocab()
		self.n_actions = n_actions
		self.rand_sizes = rand_sizes
		self.Obj_set = Obj_set
		self.__mapsize=mapsize
		self.orig_size= mapsize
		print('loaded mode with', n_actions, 'actions' )
		super().__init__(
			grid_size=mapsize,
			# max_steps=30,
			max_steps=4*mapsize*mapsize,
			rand_sizes=rand_sizes,
			Obj_set = Obj_set,
			# Set this to True for maximum speed
			agent_view_size=5,
			see_through_walls=True,
			total_steps = total_steps
		)
	def _gen_reachability_spec(self, colors, population, actions):
		i1 = [np.random.choice(colors), np.random.choice(population)]
		a1 =  np.random.choice(actions)
		self.mission ='True '+ ' U + '+a1+' '+ i1[0] + ' ' + i1[1]

		self.conditions = []
		self.goals = [[a1,i1]]

	def _getVocab(self):
		tlOperators=['+','-','V','U','&']
		vocab = []
		for word in TL_OBJECTS:
			vocab.append(word)
		for word in COLOR_NAMES:
			vocab.append(word)
		for word in tlOperators:
			vocab.append(word)
		return vocab

	def _gen_positive_spec(self, colors, population, actions):
		specT = np.random.choice(4, p=[0.3, 0.25, 0.25, 0.2]) #to decide the number of disj
		if specT == 0:
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
			self.mission='+ '+a1+' ' +i1[0]+' '+i1[1]+' U + '+a2+' '+ i2[0] + ' ' + i2[1]

			self.conditions = [[a1,i1]]
			self.goals = [[a2,i2]]

		elif specT ==3: # 2 disjunctions
			i1 = [np.random.choice(colors), np.random.choice(population)]
			a1 =  np.random.choice(actions)
			i2 = [np.random.choice(colors), np.random.choice(population)]
			a2 =  np.random.choice(actions)
			i3 = [np.random.choice(colors), np.random.choice(population)]
			a3 =  np.random.choice(actions)
			i4 = [np.random.choice(colors), np.random.choice(population)]
			a4 =  np.random.choice(actions)

			if i1 == i2:
				counter = 20
				while i1 == i2:
					i2 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			if i3 == i2 or i3 == i1:
				counter = 20
				while 3 == i2 or i3 == i1:
					i3 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break
			if i4 == i2 or i4 == i1 or i4 == i3:
				counter = 20
				while i4 == i2 or i4 == i1 or i4 == i3:
					i4 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			self.mission='+ '+a1+' '  +i1[0]+' '+i1[1]+ ' V + '+a2+' '\
			  			+ i2[0]+' '+i2[1]+' U + '+a3+' ' + i3[0] + ' '\
			  			+ i3[1] + ' V + '+a4+' '  +i4[0]+' '+i4[1]

			self.conditions = [[a1,i1],[a2,i2]]
			self.goals = [[a3,i3],[a4,i4]]

		else: #1 disjunction
			i1 = [np.random.choice(colors), np.random.choice(population)]
			a1 =  np.random.choice(actions)
			i2 = [np.random.choice(colors), np.random.choice(population)]
			a2 =  np.random.choice(actions)
			i3 = [np.random.choice(colors), np.random.choice(population)]
			a3 =  np.random.choice(actions)

			if i1 == i2:
				counter = 20
				while i1 == i2:
					i2 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			if i3 == i2 or i3 == i1:
				counter = 20
				while 3 == i2 or i3 == i1:
					i3 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			if specT == 2:
				self.mission='+ '+a1+' '  +i1[0]+' '+i1[1]+ ' V + '+a2+' '  + i2[0]+' '+i2[1]+' U + '+a3+' ' + i3[0] + ' ' + i3[1]
				self.conditions = [[a1,i1],[a2,i2]]
				self.goals = [[a3,i3]]

			else:
				self.mission='+ '+a1+' ' +i1[0]+' '+i1[1]+' U + '+a2+' '+ i2[0] + ' ' + i2[1] + ' V + '+a3+' ' +i3[0]+' '+i3[1]

				self.conditions = [[a1,i1]]
				self.goals = [[a2,i2],[a3,i3]]


	def _gen_negative_spec(self, colors, population, actions):
		specT = np.random.randint(2) #to decide the number of disj
		if specT == 0:
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
			self.mission='- '+a1+' ' +i1[0]+' '+i1[1]+' U + '+a2+' '+ i2[0] + ' ' + i2[1]
			self.conditions = [[a1,i1]]
			self.goals = [[a2,i2]]
		else:
			i1 = [np.random.choice(colors), np.random.choice(population)]
			a1 =  np.random.choice(actions)
			i2 = [np.random.choice(colors), np.random.choice(population)]
			a2 =  np.random.choice(actions)
			i3 = [np.random.choice(colors), np.random.choice(population)]
			a3 =  np.random.choice(actions)

			if i1 == i2:
				counter = 20
				while i1 == i2:
					i2 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			if i3 == i2 or i3 == i1:
				counter = 20
				while 3 == i2 or i3 == i1:
					i3 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break

			self.mission='- '+a1+' ' +i1[0]+' '+i1[1]+ ' U + '+a2+' '+ i2[0] + ' ' + i2[1] + ' V + ' +a3+' ' +i3[0]+' '+i3[1]

			self.conditions = [[a1,i1]]
			self.goals = [[a2,i2],[a3,i3]]

	def _gen_scape_spec(self, colors, population, actions):
		specT = np.random.randint(2) #to decide the number of disj
		a =  'reach'
		if specT == 0:
			i1 = [np.random.choice(colors), np.random.choice(population)]
			self.mission='True '+ ' U - '+a+' '+ i1[0] + ' ' + i1[1]
			self.conditions = []
			self.goals = [[a,i1]]
		else:
			i1 = [np.random.choice(colors), np.random.choice(population)]
			i2 = [np.random.choice(colors), np.random.choice(population)]

			if i1 == i2:
				counter = 20
				while i1 == i2:
					i2 = [np.random.choice(colors), np.random.choice(population)]
					counter-=1
					if counter==0: break
			self.mission='True '+ ' U - '+a+' '+ i1[0] + ' ' + i1[1]+ ' V - '+a+' ' +i2[0]+' '+i2[1]
			self.conditions = []
			self.goals = [[a,i1],[a,i2]]

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
		for i in range(n_fillers):
			old = True
			counter = 20
			while old:
				i1 = [np.random.choice(colors), np.random.choice(population)]
				old = False
				for condition in self.conditions:
					if condition[1] == i1: 
						old = True
						break
				if not old:
					for goal in self.goals: 
						if goal[1] == i1: 
							old = True       
				counter-=1
				if counter==0: break
			if not old: fill_pop.append(i1)
		return fill_pop

	def _gen_grid(self, width, height):
		# Create an empty grid
		# if self.orig_size < 7:
		# 	if self.Obj_set == 0:
		# 		self.mapsize = width
		# 	elif self.Obj_set == 1:
		# 		self.mapsize = 14
		# 	elif self.Obj_set == 2:
		# 		self.mapsize = 22
		self.grid = Grid(width, height)
		
		# Generate the surrounding walls
		self.grid.wall_rect(0, 0, width, height)

		# Place a goal square in the bottom-right corner
		# self.put_obj(Goal(), width - 2, height - 2)
		# self.place_obj(Goal())
		self.specT = np.random.choice(4, p=[0.20,0.35, 0.35, 0.1]) # 

		MaxActions = ['get', 'reach', 'moveUp', 'consume']
		# self.specT = 3
		if  self.Obj_set == 0 or self.Obj_set == 3:
			population = TL_OBJECTS[:len(TL_OBJECTS)-3]  if self.specT == 0 else TL_OBJECTS[2:]
			colors = COLOR_NAMES[:len(COLOR_NAMES)-4] if self.specT != 0 else COLOR_NAMES[3:]
			actions = MaxActions[:len(MaxActions)-2] if self.specT != 0 else MaxActions[1:]
		elif self.Obj_set == 1:
			if self.specT != 0:
				population = [TL_OBJECTS[len(TL_OBJECTS)-3]]
				colors = COLOR_NAMES[:len(COLOR_NAMES)-4]
				actions = MaxActions[:]
			else:
				population = [TL_OBJECTS[0]]
				colors = COLOR_NAMES[3:]
				actions = MaxActions[:]

		else:
			if self.specT != 0:
				population = TL_OBJECTS[len(TL_OBJECTS)-3:]
				colors = COLOR_NAMES[len(COLOR_NAMES)-4:]
				actions = MaxActions[:]
			else:
				population = TL_OBJECTS[:2]
				colors = COLOR_NAMES[3:]
				actions = MaxActions[:]

		if self.n_actions == 3 : actions= ['reach'] #Change N_actions too!!

		if self.specT == 0:   self._gen_reachability_spec(colors, population, actions)
		elif self.specT == 1: self._gen_positive_spec(colors, population, actions)
		elif self.specT == 2: self._gen_negative_spec(colors, population, actions)
		else:                 self._gen_scape_spec(colors, population, actions)

		# To fill with new objects to use as distractors
		if self.Obj_set == 0:
			max_filler_variety = 15
		else:
			max_filler_variety = 5

		# maxP= max(5,min(width//2,max_filler_variety))
		n_fillers = np.random.randint(3,max_filler_variety) #if self.specT != 2 else np.random.randint(1,4)
		fill_pop = self._generate_fillerPopulation(n_fillers , colors, population)

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
		if self.specT != 3:
			n_goals = np.random.randint(1,3)
			if self.goals[0][0] == 'moveUp': n_goals = us_width
		else:
			pickable = False
			n_scapes = max(us_width//5,2)
			n_goals = us_width*us_width-n_scapes
			for i in range(n_scapes):
				if i == 0:
					Object= self._get_object_Call(self.goals[0][1][1])
					self.put_obj(Object(self.goals[0][1][0], True, True), self.agent_pos[0], self.agent_pos[1])
					continue
				scape = fill_pop[np.random.randint(len(fill_pop))]
				Object= self._get_object_Call(scape[1])
				self.place_obj(Object(scape[0], overlap=True, pickup=pickable))
		for _ in range(n_goals):
			goal = self.goals[np.random.randint(len(self.goals))]
			Object= self._get_object_Call(goal[1][1])
			self.place_obj(Object(goal[1][0], overlap=True, pickup=pickable))

		#Hyperparameters for safety conditions 
		N_ReachPositive = us_width*us_width//2
		N_OtherPositive = max(us_width//2,1)
		N_ReachNegative = us_width*us_width//3
		N_OtherNegative =  us_width*us_width//4
		
		# Place the conditions
		for condition in self.conditions:
			if self.specT == 1: #Conditions that the agent should mantain as true
				if condition[0] == 'reach': n_condition = N_ReachPositive // len(self.conditions)
				else: n_condition = max(N_OtherPositive // len(self.conditions),1)
			else: #conditions the agent should avoid
				if condition[0] == 'reach': n_condition = N_ReachNegative // len(self.conditions)
				else: n_condition = max(N_OtherNegative // len(self.conditions),1)
			for _ in range(n_condition):
				Object = self._get_object_Call(condition[1][1])
				self.place_obj(Object(condition[1][0], overlap=True, pickup=pickable))

		#Place filler population
		if self.specT != 3:
			for _ in range(us_width*2):
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
		done = False
		#Only negative goal condition
		if self.specT == 3: 
			done = True
			reward = 1
			if cell is not None:
				for goal in self.goals:
					if (goal[1][0] == cell.color and goal[1][1] == cell.type): 
						reward = -0.05
						done = False
		else:
			#general final goals
			for goal in self.goals:
				if goal[0] == 'reach':
					if cell is not None:
						if (goal[1][0] == cell.color and goal[1][1] == cell.type): 
							reward = 1
							done = True
				elif goal[0] == 'get':
					if self.carrying: 
						if (goal[1][0] == self.carrying.color and \
							goal[1][1] == self.carrying.type): 
							reward = 1
							done = True
				elif goal[0] == 'moveUp':
					if action == self.actions.drop and self.preCarrying:
						u, v = self.dir_vec
						ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
						if not self.carrying and self.grid.get(ox, oy) is self.preCarrying and oy==1:
							if (goal[1][0] == self.preCarrying.color \
								and goal[1][1] == self.preCarrying.type):
									reward = 1
									done = True
				elif goal[0] == 'consume':
					if action == self.actions.toggle and self.preCarrying:
						if (goal[1][0] == self.preCarrying.color \
							and goal[1][1] == self.preCarrying.type):
								reward = 1
								done = True
				else:
					raise Exception("This action has no reward associated")

			#"safety" conditions
			if not done:
				if self.specT == 1 or self.specT == 2: # Reachability goals (spec 0) do not have conditions
					reward = -1 if self.specT == 1 else -0.05
					for condition in self.conditions:
						if condition[0] == 'reach':
							if cell is not None:
								if (condition[1][0] == cell.color and condition[1][1] == cell.type): 
									reward = -0.05 if self.specT == 1 else -1
						elif condition[0] == 'get':
							if self.carrying: 
								if (condition[1][0] == self.carrying.color and \
									condition[1][1] == self.carrying.type): 
									reward = -0.05 if self.specT == 1 else -1
						elif condition[0] == 'moveUp':
							if action == self.actions.drop and self.preCarrying:
								u, v = self.dir_vec
								ox, oy = (self.agent_pos[0] + u, self.agent_pos[1] + v)
								if not self.carrying and self.grid.get(ox, oy) is self.preCarrying and oy==1:
										reward = -0.05 if self.specT == 1 else -1
						elif condition[0] == 'consume':
							if action == self.actions.toggle and self.preCarrying:
								if (condition[1][0] == self.preCarrying.color \
									and condition[1][1] == self.preCarrying.type):
										reward = -0.05 if self.specT == 1 else -1
		return reward, done




class ttl2ReachabilityEnv5x5(ttl2ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=5, **kwargs)

class ttl2ReachabilityEnvRandom5x5(ttl2ReachabilityEnv):
	def __init__(self):
		super().__init__(size=5, agent_start_pos=None)

class ttl2ReachabilityEnvRandom7x7(ttl2ReachabilityEnv):
	def __init__(self):
		super().__init__(size=7, agent_start_pos=None)

class ttl2ReachabilityEnvR58_3A(ttl2ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(rand_sizes=[7, 8, 9, 10], agent_start_pos=None,
							n_actions=3, **kwargs)

class ttl2ReachabilityEnvR58_6A(ttl2ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(rand_sizes=[7, 8, 9, 10], agent_start_pos=None,
							 n_actions=6, **kwargs)

class ttl2ReachabilityEnvRandom22x22(ttl2ReachabilityEnv):
	def __init__(self):
		super().__init__(size=22, agent_start_pos=None)

class ttl2ReachabilityEnvRandom14x14(ttl2ReachabilityEnv):
	def __init__(self):
		super().__init__(size=14, agent_start_pos=None)

class ttl2ReachabilityEnv6x6(ttl2ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=6, **kwargs)
		random.seed(0)

class ttl2ReachabilityEnvRandom6x6(ttl2ReachabilityEnv):
	def __init__(self):
		super().__init__(size=6, agent_start_pos=None)

class ttl2ReachabilityEnv16x16(ttl2ReachabilityEnv):
	def __init__(self, **kwargs):
		super().__init__(size=16, **kwargs)

register(
	id='MiniGrid-ttl2Reachability-R-3A-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvR58_3A'
)

register(
	id='MiniGrid-ttl2Reachability-R-6A-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvR58_6A'
)


register(
	id='MiniGrid-ttl2Reachability-22x22-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvRandom22x22'
)


register(
	id='MiniGrid-ttl2Reachability-14x14-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvRandom14x14'
)

register(
	id='MiniGrid-ttl2Reachability-5x5-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvRandom5x5'
)

register(
	id='MiniGrid-ttl2Reachability-7x7-v0',
	entry_point='gym_minigrid.envs:ttl2ReachabilityEnvRandom7x7'
)


# register(
#     id='MiniGrid-Empty-6x6-v0',
#     entry_point='gym_minigrid.envs:ttl2ReachabilityEnv6x6'
# )

# register(
#     id='MiniGrid-Empty-Random-6x6-v0',
#     entry_point='gym_minigrid.envs:EmptyRandom6x6'
# )

# register(
#     id='MiniGrid-Empty-8x8-v0',
#     entry_point='gym_minigrid.envs:ttl2ReachabilityEnv'
# )

# register(
#     id='MiniGrid-Empty-16x16-v0',
#     entry_point='gym_minigrid.envs:ttl2ReachabilityEnv16x16'
# )
