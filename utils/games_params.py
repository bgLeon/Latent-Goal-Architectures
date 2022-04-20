"""
Auxiliary classes with the configuration parameters that the Game class needs
"""
class MinecSafetyGameParams:
	def __init__(self, specs_type, SpecSets, candidates, test , fully_obs= False,
		file_map= None, visual = False, vdic = False, obs_range = 2, testSpecSets = None,
		extended_obs = True, moving_spec =  False, trainSz = 10, seedB=0, test_candidates = None,
		validation = False,
		complexT = 0, offset = 0, size = 5, resolution = 9, mapType="minecraft-insp"):
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
		self.SpecSets = SpecSets
		self.candidates = candidates
		self.test = test
		self.specs_type = specs_type
		self.fully_obs = fully_obs
		self.obs_range = obs_range
		self.extended_obs = extended_obs 
		self.moving_spec =  moving_spec
		self.env_name = None
		self.trainSz= trainSz
		self.visual = visual
		self.validation = validation
		self.mapType = mapType
		self.testSpecSets = testSpecSets
		self.test_candidates = test_candidates
		self.agent_view_size = obs_range*2+1
		self.seedB=seedB
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

class MinigridGameParams:
	def __init__(self, mapType='minigrid', visual = True, vocabSize = 32,
				textMaxLength = 7, env_name= 'MiniGrid-tlReachability-R-v0',
				agent_view_size = 5, test=False, trainSz = 'large', mapsize = 5,
				n_agents=1, seedB =0, complexT = False, specs_type = 'TTL2'):

		self.mapType = mapType
		self.visual = visual
		self.vocabSize = vocabSize
		self.textLength = textMaxLength
		self.env_name = env_name
		self.n_agents = n_agents
		self.complexT = complexT
		self.trainSz = trainSz
		self.seedB = seedB
		self.specs_type = specs_type
		self.agent_view_size = agent_view_size
		self.test = test
		self.mapsize = mapsize
		if textMaxLength == 7 and specs_type == 'TTL2':
			textMaxLength == 19