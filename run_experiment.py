import argparse, time
from agents.neuro_symbolic_agent  import NSAgent
from utils.games_params import *
from utils.utils import VisualDic
from utils.specifications import *

def run_experiment(alg_name, specs_type, max_steps, fully_obs, network, learning_rate, 
					batch_size, pretrained_encoder, show_print, render, test, text_mode,
					visual, mem, random_walker,trainSz, seedB, complexT, loss,
					num_neurons, mapType, num_layers, size, multiModal, mem_type,
					TBPTT, rnn_type, minigridType, loaded_path, loaded_point, loaded_best_test,
					loaded_test_avg, loaded_train_avg, num_times = 1, resolution = 9):
	"""
	This method adds the parameters: 
		-num_times: (int)
			tells how many times we run the experiment
		-resolution: (int)
			Refers to the resolution of the objects in the environment
	
	"""
	time_init = time.time()
	file_map=None #place here route to a map if needed for testing
	if complexT: complexT = 1
	else: complexT = 0

	# configuration of learning params
	if mapType=='minecraft-insp':
		if trainSz == 'small':
			trainSz=6
			print("Working with small set")
		elif trainSz == 'medium':
			trainSz=10
			print("Working with medium set")
		else:
			trainSz=20
			print("Working with large set")
		if visual: vdic = VisualDic(resolution = resolution, test = test)
		else:  vdic = None

		# Getting the specifications
		# Modes:
		#    0: standard training specficiations
		#    1: standard test specifications
		#    2: Slow movement specifications only
		#    3: Normal movement specifications only
		#    4: Fast movement specifications only
		if specs_type == 'TTL2':
			TrueTilG, FalseTilG, MoveTilEsc, PDisjSpecs, NDisjSpecs,\
				candidates = get_TTL2_specs(\
				trainSz, test, mode = 0)
			SpecSets = [TrueTilG, FalseTilG, MoveTilEsc, PDisjSpecs, NDisjSpecs]

			TrueTilG, FalseTilG, MoveTilEsc, PDisjSpecs, NDisjSpecs,\
				test_candidates = get_TTL2_specs(trainSz, True, mode = 0)
			testSpecSets = [TrueTilG, FalseTilG, MoveTilEsc, PDisjSpecs, NDisjSpecs]
		else:
			if not complexT:
				AffSpecs, NegSpecs, DisjSpecs, candidates = get_TTL_specs(trainSz, test)
				SpecSets = [AffSpecs, NegSpecs, DisjSpecs]

				AffSpecs, NegSpecs, DisjSpecs, test_candidates = get_TTL_specs(trainSz, True)
				testSpecSets =[AffSpecs, NegSpecs, DisjSpecs]
			else:
				SpecSets, candidates = get_Complex_specs()
				testSpecSets, test_candidates = SpecSets, candidates
	# Setting the environment params
		env_params = MinecSafetyGameParams(specs_type, SpecSets, candidates, test, fully_obs,  
								file_map, visual, vdic,
								trainSz = trainSz, complexT = complexT,
								resolution = resolution, size = size, test_candidates = test_candidates,
								mapType = mapType, seedB= seedB, testSpecSets = testSpecSets)
	else:
		if minigridType == 'MiniGrid-ttl1Reachability-7x7-3A-v1' and specs_type == 'TTL2':
			minigridType = 'MiniGrid-ttl2Reachability-R-3A-v0'
		env_params = MinigridGameParams(env_name = minigridType, complexT = complexT, seedB= seedB, 
										trainSz= trainSz, mapsize=size, specs_type=specs_type)

	learner = NSAgent(alg_name, network, batch_size, learning_rate, num_neurons, num_layers, 
						multiModal, TBPTT, rnn_type, mem_type, mapType, loss, text_mode)

	learner.call(env_params, num_times, show_print, pretrained_encoder, render, test,
							 mem, seedB, max_steps, random_walker, loaded_path, loaded_point,
							 loaded_test_avg, loaded_train_avg, loaded_best_test)

	print("Total time:", "%0.2f"%((time.time() - time_init)/60), "mins")

if __name__ == "__main__":

	algorithms = ["a2c", "ppo"]
	operators = ["TTL1", "TTL2"]
	trainSzs = ["small", "Lsmall", "medium", "large"]
	networks = ["default", "PrediNet", "RelNet", "MHA", "PN_MHA", "MHA_PN"]
	maps = ["minecraft-insp", "minigrid"]
	rnn_types = ["lstm", "gru"]
	minigridTypes = ['MiniGrid-ttl2Reachability-R-3A-v0', 
					'MiniGrid-ttl1Reachability-7x7-v0', 
					'MiniGrid-ttl1Reachability-7x7-3A-v1',
					'MiniGrid-ttl2Reachability-R-6A-v0']
	mem_types = ['rim', 'lstm']
	losses = ['cce', 'mse']

	parser = argparse.ArgumentParser(prog="run_experiments",
									description='Runs a neuro-symbolic\
									agent over a gridworld\
									domain that is inspired on Minecraft.')

	parser.add_argument('--num_steps', default = 30000000, type=int,
						help='numner of training steps')
	parser.add_argument('--num_neurons', default = 128, type=int,
						help='numner of neurons per hidden layer')
	parser.add_argument('--text_mode', default = 3, type=int,
						help='Determines how instructions are procesed within the NNs (rel nets only). 0 text is concatenated with the visual output, 1 text is given after the relnet and preprocessed by a bidirectional LSTM')
	parser.add_argument('--num_layers', default = 1, type=int,
						help='numner of dense hidden layers')
	parser.add_argument('--batch_size', default = 1000, type=int,
						help='default batch size')
	parser.add_argument('--show_Print', default = True, 
						action ='store_false',
						help='this paremeter tells if print progress')
	parser.add_argument('--fobs', default = False, 
						action ='store_true', help=\
						'this parameter tells if the enviroment is fully\
						 observable')
	parser.add_argument('--pretrained_encoder', default = False, 
						action ='store_true', help=\
						'this paremeter tells if the NN loads a pretrained\
						 encoder for the Conv layer')
	parser.add_argument('--visual', default = False, 
						action ='store_true', help=\
						'this paremeter tells if the enviroment is\
						 fully observable')
	parser.add_argument('--mem', default = False, 
						action ='store_true', help=\
						'this paremeter tells if the agent use a\
						 Recurrent Network')
	parser.add_argument('--render', default = False, 
						action ='store_true',
						help='this paremeter tells if the map is rendered')
	parser.add_argument('--network', default='default', type=str, 
						help='This parameter indicates which kinf of NN to \
						use. The options are: ' + str(networks))
	parser.add_argument('--algorithm', default='a2c', type=str, 
						help='This parameter indicates which RL algorithm to \
						use. The options are: ' + str(algorithms))
	parser.add_argument('--random',default = False, action ='store_true',
						help='This parameters tells if use the random policy')
	parser.add_argument('--syntax', default='TTL2', type=str, 
						help='This parameter indicates the usage of a random \
						walker, intended for testing')
	parser.add_argument('--seedB', default=0, type=int,
						help='This parameter indicates seed bias to use.')
	parser.add_argument('--trainSz', default='large', type=str, 
						help='This parameter indicates the train vocab size \
						to use. The options are: ' + str(trainSzs))
	parser.add_argument('--rnn_type', default='lstm', type=str, 
						help='This parameter indicates the train vocab size \
						to use. The options are: ' + str(rnn_types))
	parser.add_argument('--complexT',  default = False, 
						action ='store_true',
						help='Selects the set of complex Tasks for the agent,\
						intended for testing a trained agent')
	parser.add_argument('--test',  default = False, 
						action ='store_true',
						help='loads and test a trained model')
	parser.add_argument('--multimodal',  default = False, 
						action ='store_true',
						help='This parameter tells if a MultiModal network is \
								used')
	parser.add_argument('--size', default=5, type=int,
						help='This parameter indicates the size of the training\
						map.')
	parser.add_argument('--tback', default=1, type=int,
						help='This parameter indicates the number of steps for the truncated BPTT.')
	parser.add_argument('--mapType', default='minigrid', type=str, 
						help='This parameter indicates which kinf of map to \
						use. The options are: ' + str(maps))
	parser.add_argument('--minigridType', default='MiniGrid-ttl1Reachability-7x7-3A-v1', type=str, 
						help='This parameter indicates which kinf of map to \
						use. The options are: ' + str(minigridTypes))
	parser.add_argument('--mem_type', default='lstm', type=str, 
						help='This parameter indicates which kinf of memory network to \
						use. The options are: ' + str(mem_types))
	parser.add_argument('--loss', default='cce', type=str, 
						help='This parameter indicates which kinf of loss is used by the \
						autoencoder. The options are: ' + str(losses))
	parser.add_argument('--learning_rate', default=8e-5, type=float, 
						help='This parameter indicates th3 learning rate. default is 8e-5')
	parser.add_argument('--loaded_point', default=0, type=int,
						help='Indicates if the training is restarting from an specific point.')
	parser.add_argument('--loaded_path', default='None', type=str, 
						help='Indicates the path to load a trained model ' + str(mem_types))
	parser.add_argument('--loaded_train_avg', default=0, type=float,
						help='Indicates the agerage performance of the loaded model.')
	parser.add_argument('--loaded_test_avg', default=0, type=float,
						help='Indicates the average test performance of the loaded model.')
	parser.add_argument('--loaded_best_test', default=0, type=float,
						help='Indicates the best test performance of the loaded model.')

	args = parser.parse_args()

	if args.algorithm not in algorithms: raise NotImplementedError("Algorithm "\
		+ str(args.algorithm) + " hasn't been implemented yet")
	if args.syntax not in operators: raise NotImplementedError(
		"Syntax " + str(args.syntax) + \
		" hasn't been defined yet")
	if args.minigridType not in minigridTypes: raise NotImplementedError(
		"Minigrid map " + str(args.minigridType) + \
		" hasn't been defined yet")
	if args.trainSz not in trainSzs: raise NotImplementedError(
		"Train Size " + str(args.trainSz) + \
		" hasn't been defined yet")
	if not (args.num_steps>0): raise NotImplementedError(
		"The number of steps should be a positive integer")
	if args.mem_type not in mem_types: raise NotImplementedError(
		"Mem net " + str(args.mem_type) + \
		" hasn't been defined yet")
	if args.loss not in losses: raise NotImplementedError(
		"Loss " + str(args.loss) + \
		" hasn't been defined yet")

	run_experiment(alg_name = args.algorithm, specs_type = args.syntax, 
					max_steps = int(args.num_steps), fully_obs = args.fobs,
					pretrained_encoder = args.pretrained_encoder, 
					show_print = args.show_Print, random_walker = args.random,
					render =args.render, test = args.test, visual= args.visual,
					mem = args.mem, trainSz= args.trainSz, seedB = args.seedB,
					complexT = args.complexT, num_neurons = args.num_neurons,
					num_layers = args.num_layers, size = args.size,
					rnn_type = args.rnn_type, mem_type = args.mem_type,
					multiModal = args.multimodal, network= args.network,
					mapType = args.mapType, batch_size= args.batch_size,
					loss= args.loss, learning_rate = args.learning_rate,
					TBPTT= args.tback, minigridType = args.minigridType,
					text_mode = args.text_mode, loaded_point = args.loaded_point, 
					loaded_path = args.loaded_path, loaded_train_avg = args.loaded_train_avg,
					loaded_test_avg = args.loaded_test_avg, loaded_best_test = args.loaded_best_test)