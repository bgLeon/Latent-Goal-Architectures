import numpy as np
from utils.random_game import GameParams
import utils.specifications as specs
import os, json, argparse

class VisualDic:
	def __init__(self, resolution, test):
		file = './utils/Objects/ObjectsDicTTL2.npy'
		if not os.path.exists('./utils/Objects/'): os.mkdir('./utils/Objects')
		self.resolution = resolution
		if not os.path.exists(file):
			np.random.seed(1000)
			# population = "AX!VTabcdefghijklmnopqrstuvwxyz"
			population = "AESX!-VGNTFU+Babcdefghijklmnopqrstuvwxyz0123456789"
			self.class_ids = {}
			for e in population:
				self.class_ids[e] = A = np.random.randint(255, size=(resolution, resolution))

			self.class_ids[" "] = np.full((resolution, resolution), 0)
			self.class_ids["X"] = np.full((resolution, resolution), 200)
			np.save(file,self.class_ids)

		else:
			self.class_ids = np.load(file, allow_pickle=True).item()
			# if test:
			# 	file2 = './utils/Objects/ObjectsDicTTL2_Exp.npy' #Test
			# 	self.class_ids = np.load(file2, allow_pickle=True).item()

				# file3 = './utils/Objects/ObjectsDicTTL3_Exp.npy' #Test2
				# self.class_ids = np.load(file3, allow_pickle=True).item()
				# print("TEST")
			# if test:
				# np.random.seed(13)
				# np.random.seed(24)
				# np.random.seed(24)
				# population_2 ="abcdefghijklmnopqrstuvwxyz0123456789"
				# for e in population_2:
				# 	self.class_ids[e] = np.random.randint(255, size=(resolution, resolution))
				# file_2 = './utils/Objects/ObjectsDicTTL3_Exp.npy'
				# np.save(file_2, self.class_ids)


class Saver:

	def __init__(self, alg_name, tester, curriculum):
		folder = "./tmp/"
		exp_name = tester.experiment
		exp_dir = os.path.join(folder, exp_name)
		if not os.path.exists(exp_dir):
			os.makedirs(exp_dir)
		self.file_out = os.path.join(exp_dir, alg_name + ".json")
		self.tester = tester
		
	def save_results(self):
		results = {}
		results['specs'] = [str(t) for t in self.tester.specs]
		# results['optimal'] = dict([(str(t), self.tester.optimal[t]) for t in self.tester.optimal])
		results['steps'] = self.tester.steps      
		results['results'] = dict([(str(t), self.tester.results[t]) for t in self.tester.results])        
		# Saving results
		save_json(self.file_out, results)

class Tester:
	def __init__(self, training_params, testing_params, map_id, specs_id,
				fully_obs, file_results=None):
		if file_results is None:
			# setting the test attributes
			self.experiment = "spec_%d/map_%d"%(specs_id,map_id)
			self.training_params = training_params
			self.testing_params = testing_params
			self.map = './experiments/easy_maps/5/map_2.txt'#%map_id #%d
			self.consider_night = False
			self.fully_obs = fully_obs
			if specs_id == 0:
				self.specs = specs.get_sequence_of_subspecs()
			if specs_id == 1:
				self.specs = specs.get_interleaving_subspecs()
			if specs_id == 2:
				self.specs = specs.get_safety_constraints()
				self.consider_night = True
			# optimal_aux  = _get_optimal_values(
			#     './experiments/optimal_policies/multiA_map_%d.txt'%(map_id),
			#     specs_id)

			# I store the results here
			self.results = {}
			# self.optimal = {}
			self.steps = []
			for i in range(len(self.specs)):
			#     self.optimal[self.specs[i]] = training_params.gamma **\
			#         (float(optimal_aux[i]) - 1)
				self.results[self.specs[i]] = {}
		else:
			# Loading precomputed results
			data = read_json(file_results)
			self.results = dict([(eval(t), data['results'][t]) for t in data['results']])
			# self.optimal = dict([(eval(t), data['optimal'][t]) for t in data['optimal']])
			self.steps   = data['steps']
			self.specs   = [eval(t) for t in data['specs']]
			# obs: json transform the interger keys from 'results' into strings
			# so I'm changing the 'steps' to strings
			for i in range(len(self.steps)):
				self.steps[i] = str(self.steps[i])
			
	def get_LTL_specs(self):
		return self.specs

	def get_spec_params(self, spec):
		return GameParams(self.map, spec, self.consider_night, self.fully_obs)

	def run_test(self, step, test_function, *test_args):
		# 'test_function' parameters should be (spec_params, training_params, testing_params, *test_args)
		# and returns the reward
		# print("results",self.results)
		i=0
		for t in self.specs:
			spec_params = self.get_spec_params(t)
			reward, spec = test_function(spec_params, self.training_params,
									self.testing_params, *test_args)
			print("test reward:", reward, "last_spec", spec)
			if step not in self.results[t]:
				self.results[t][step] = []
			if len(self.steps) == 0 or self.steps[-1] < step:
				self.steps.append(step)
			self.results[t][step].append(reward)
			i+=1
			if i > 3: break


	def show_results(self):
		average_reward = {}
		
		# Computing average perfomance per spec
		for t in self.specs:
			for s in self.steps:
				rewards = [r for r in self.results[t][s]]
				a = np.array(rewards)
				if s not in average_reward: average_reward[s] = a
				else: average_reward[s] = a + average_reward[s]

		# Showing average perfomance across all the spec
		print("\nAverage discounted reward on this map --------------------")
		print("\tsteps\tP25\tP50\tP75")            
		num_specs = float(len(self.specs))
		for s in self.steps:
			rewards = average_reward[s] / num_specs
			p25, p50, p75 = get_precentiles_str(rewards)
			print("\t" + str(s) + "\t" + p25 + "\t" + p50 + "\t" + p75)

	def export_results(self):
		average_reward = {}
		
		# Showing perfomance per spec
		for t in self.specs:
			for s in self.steps:
				rewards = [r for r in self.results[t][s]]
				a = np.array(rewards)
				if s not in average_reward: average_reward[s] = a
				else: average_reward[s] = a + average_reward[s]

		# Computing average perfomance across all the spec\
		ret = []
		num_specs = float(len(self.specs))
		for s in self.steps:
			rewards = average_reward[s] / num_specs
			ret.append([s, rewards])
		return ret

def get_precentiles_str(a):
	p25 = "%0.2f"%float(np.percentile(a, 25))
	p50 = "%0.2f"%float(np.percentile(a, 50))
	p75 = "%0.2f"%float(np.percentile(a, 75))
	return p25, p50, p75

def export_results_old(algorithm, spec, spec_id):
	for map_type, maps in [("random",range(0,5)),("adversarial",range(5,10))]:
		# Computing the summary of the results
		rewards = None
		for map_id in maps:
			result = "./tmp/spec_%d/map_%d/%s.json"%(spec_id, map_id, algorithm)
			tester = Tester(None, None, None, None, result)
			ret = tester.export_results()
			if rewards is None:
				rewards = ret
			else:
				for j in range(len(rewards)):
					rewards[j][1] = np.append(rewards[j][1], ret[j][1])
		# Saving the results
		folders_out = "./results/%s/%s"%(spec, map_type)
		if not os.path.exists(folders_out): os.makedirs(folders_out)
		file_out = "%s/%s.txt"%(folders_out, algorithm)
		f_out = open(file_out,"w")
		for j in range(len(rewards)):
			p25, p50, p75 = get_precentiles_str(rewards[j][1])
			f_out.write(str(normalized_rewards[j][0]) + "\t" + p25 + "\t" + p50 + "\t" + p75 + "\n")
		f_out.close()

def export_results(algorithm, spec, spec_id):
	# for map_id in range(0,3):
		# Computing the summary of the results
		rewards = None
		for map_id in range(0,3):
			print("map_id", map_id)
			result = "./tmp/spec_%d/map_%d/%s.json"%(spec_id, map_id, algorithm)
			tester = Tester(None, None, None, None, result)
			ret = tester.export_results()
			if rewards is None:
				rewards = ret
			else:
				for j in range(len(rewards)):
					rewards[j][1] = np.append(rewards[j][1], ret[j][1])
		# Saving the results
		folders_out = "./results/%s"%(spec)
		if not os.path.exists(folders_out): os.makedirs(folders_out)
		file_out = "%s/%s.txt"%(folders_out, algorithm)
		f_out = open(file_out,"w")
		for j in range(len(rewards)):
			p25, p50, p75 = get_precentiles_str(rewards[j][1])
			f_out.write(str(normalized_rewards[j][0]) + "\t" + p25 + "\t" + p50 + "\t" + p75 + "\n")
		f_out.close()

def save_json(file, data):
	with open(file, 'w') as outfile:
		json.dump(data, outfile)

def read_json(file):
	with open(file) as data_file:
		data = json.load(data_file)
	return data
	
def clear_screen():
	os.system('cls' if os.name == 'nt' else 'clear')


if __name__ == "__main__":

	# EXAMPLE: python3 test_utils.py --algorithm="lpopl" --specs="sequence"

	# Getting params
	algorithms = ["idqn-l", "hrl-e", "hrl-l", "lpopl"]
	specs      = ["sequence", "interleaving", "safety"]

	parser = argparse.ArgumentParser(prog="run_experiments", description='Runs a multi-spec RL experiment over a gridworld domain that is inspired by Minecraft.')
	parser.add_argument('--algorithm', default='idqn-l', type=str, 
						help='This parameter indicated which RL algorithm to use. The options are: ' + str(algorithms))
	parser.add_argument('--specs', default='sequence', type=str, 
						help='This parameter indicated which specs to solve. The options are: ' + str(specs))
	
	args = parser.parse_args()
	if args.algorithm not in algorithms: raise NotImplementedError("Algorithm " + str(args.algorithm) + " hasn't been implemented yet")
	if args.specs not in specs: raise NotImplementedError("specs " + str(args.specs) + " hasn't been defined yet")

	export_results(args.algorithm, args.specs, specs.index(args.specs))