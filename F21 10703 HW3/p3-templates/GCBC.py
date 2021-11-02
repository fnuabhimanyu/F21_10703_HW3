from collections import OrderedDict 
import gym
from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import random
import pdb

import djikstra as dji
import bfs
import model_pytorch
import copy
import tqdm
import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.autograd import Variable
# Import make_model here from the approptiate model_*.py file
# This model should be the same as problem 2

### 2.1 Build Goal-Conditioned Task
class FourRooms:
	def __init__(self, l=5, T=30):
		'''
		FourRooms Environment for pedagogic purposes
		Each room is a l*l square gridworld, 
		connected by four narrow corridors,
		the center is at (l+1, l+1).
		There are two kinds of walls:
		- borders: x = 0 and 2*l+2 and y = 0 and 2*l+2 
		- central walls
		T: maximum horizion of one episode
			should be larger than O(4*l)
		'''
		assert l % 2 == 1 and l >= 5
		self.l = l
		self.total_l = 2 * l + 3
		self.T = T

		# create a map: zeros (walls) and ones (valid grids)
		self.map = np.ones((self.total_l, self.total_l), dtype=np.bool)
		# build walls
		self.map[0, :] = self.map[-1, :] = self.map[:, 0] = self.map[:, -1] = False
		self.map[l+1, [1,2,-3,-2]] = self.map[[1,2,-3,-2], l+1] = False
		self.map[l+1, l+1] = False

		# define action mapping (go right/up/left/down, counter-clockwise)
		# e.g [1, 0] means + 1 in x coordinate, no change in y coordinate hence
		# hence resulting in moving right
		self.act_set = np.array([
			[1, 0], [0, 1], [-1, 0], [0, -1] 
		], dtype=np.int)
		self.action_space = spaces.Discrete(4)

		# you may use self.act_map in search algorithm 
		self.act_map = {}
		self.act_map[(1, 0)] = 0
		self.act_map[(0, 1)] = 1
		self.act_map[(-1, 0)] = 2
		self.act_map[(0, -1)] = 3

		# self.t = 0

	def get_key(self,set_value):

		for action,value in self.act_map.items():
			if value == set_value:
				return action

	def render_map(self):
		plt.imshow(self.map)
		plt.xlabel('y')
		plt.ylabel('x')
		plt.savefig('p2_map.png', 
					bbox_inches='tight', pad_inches=0.1, dpi=300)
		plt.show()

	
	def sample_sg(self):
		# sample s
		while True:
			s = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[s[0], s[1]]:
				break

		# sample g
		while True:
			g = [np.random.randint(self.total_l), 
				np.random.randint(self.total_l)]
			if self.map[g[0], g[1]] and \
				(s[0] != g[0] or s[1] != g[1]):
				break
		return s, g

	def reset(self, s=None, g=None):
		'''
		s: starting position, np.array((2,))
		g: goal, np.array((2,))
		return obs: np.cat(s, g)
		'''
		if s is None or g is None:
			s, g = self.sample_sg()
		else:
			assert 0 < s[0] < self.total_l - 1 and 0 < s[1] < self.total_l - 1
			assert 0 < g[0] < self.total_l - 1 and 0 < g[1] < self.total_l - 1
			assert (s[0] != g[0] or s[1] != g[1])
			assert self.map[s[0], s[1]] and self.map[g[0], g[1]]
		
		self.s = np.array(s)
		self.g = np.array(g)
		self.t = 1

		return self._obs()
	
	def step(self, a):
		'''
		a: action, a scalar
		return obs, reward, done, info
		- done: whether the state has reached the goal
		- info: succ if the state has reached the goal, fail otherwise 
		'''

		assert self.action_space.contains(a)
		# print("taking a step: ",self.t)
		# WRITE CODE HERE
		self.s_temp = self.s + np.array(self.get_key(a))
		# pdb.set_trace()
		if self.map[self.s_temp[0]][self.s_temp[1]]:
			self.s = self.s_temp

		self.t += 1
		try:
			# print("self.s: ",self.s)
			if np.linalg.norm(self.s-self.g) == 0 or self.t == self.T:
				done = 1
				if np.linalg.norm(self.s-self.g) == 0:
					info = "succ"
				else:
					info = "fail"
			else:
				done = 0
				info = "fail"
		except TypeError:
			pdb.set_trace()


		# END
		return self._obs(), 0.0, done, info

	def _obs(self):
		return np.concatenate([self.s, self.g])

# build env
l, T = 5, 30
env = FourRooms(l, T)
# pdb.set_trace()
### Visualize the map
env.render_map()


def plot_traj(env, ax, traj, goal=None):
	traj_map = env.map.copy().astype(np.float)
	# pdb.set_trace()
	traj_map[traj[:, 0], traj[:, 1]] = 2 # visited states
	traj_map[traj[0, 0], traj[0, 1]] = 1.5 # starting state
	traj_map[traj[-1, 0], traj[-1, 1]] = 2.5 # ending state
	if goal is not None:
		traj_map[goal[0], goal[1]] = 3 # goal
	ax.imshow(traj_map)
	ax.set_xlabel('y')
	ax.set_label('x')
	# pdb.set_trace()

### A uniformly random policy's trajectory
def test_step(env):
	s = np.array([1, 1])
	g = np.array([2*l+1, 2*l+1])
	s = env.reset(s, g)
	done = False
	traj = [s]
	while not done:
		a = env.action_space.sample()
		# pdb.set_trace()
		s, _, done, info = env.step(a)
		print("state and done criteria and action is {},{},{},{}".format(s,done,info,env.count))
		traj.append(s)
	# pdb.set_trace()
	traj = np.array(traj)

	ax = plt.subplot()
	plot_traj(env, ax, traj, g)
	plt.savefig('p2_random_traj.png', 
			bbox_inches='tight', pad_inches=0.1, dpi=300)
	plt.show()

def shortest_path_expert(env):
	from queue import Queue
	N = 1000
	expert_trajs = []
	expert_actions = []
	len_traj = []
	# WRITE CODE HERE
	# s = np.array([1, 1])
	# g = np.array([11, 11])
	# g = np.array([7,7])
	# dist = bfs.BFS(env.map,s)
	plot = 1
	for i in range(1000):
		s = env.reset()
		# print(s)
		expert_traj,expert_action_1 = dji.djikstra(env.map,s)
		expert_traj = np.array(expert_traj)
		expert_trajs.append(expert_traj)
		len_traj.append(expert_traj.shape[0])
		expert_action = []
		for ea in expert_action_1:
			# append(env.act_map[ea])
			expert_action.append(action_to_one_hot(env, env.act_map[ea]))
		expert_actions.append(np.array(expert_action))
	# pdb.set_trace()
	# print("minimum distance is :",dist)
	# END
	# You should obtain expert_trajs, expert_actions from search algorithm
	if plot==1:
		fig, axes = plt.subplots(5,5, figsize=(10,10))
		axes = axes.reshape(-1)
		# pdb.set_trace()
		for idx, ax in enumerate(axes):
			plot_traj(env, ax, expert_trajs[idx])

		# plt.savefig('p2_expert_trajs.png', 
		# 		bbox_inches='tight', pad_inches=0.1, dpi=300)
		plt.show()
		return expert_trajs,expert_actions,len_traj

	else:
		return expert_trajs,expert_actions,len_traj

def action_to_one_hot(env, action):
	action_vec = np.zeros(env.action_space.n)
	action_vec[action] = 1
	return action_vec  

# test_step(env)

class GCBC:

	def __init__(self, env, expert_trajs, expert_actions):
		self.env = env
		self.expert_trajs = expert_trajs
		self.expert_actions = expert_actions
		self.transition_num = sum(map(len, expert_actions))
		# self.model = model_pytorch.make_model()
		self.criterion = torch.nn.CrossEntropyLoss()


		# state_dim + goal_dim = 4
		# action_choices = 4
	
	def reset_model(self):
		self.model = model_pytorch.make_model(input_dim=4, output_dim=4)	

	def generate_behavior_cloning_data(self,expert_traj, expert_action):
		# 3 you will use action_to_one_hot() to convert scalar to vector
		# state should include goal
		self._train_states = []
		self._train_actions = []
		

		# WRITE CODE HERE
		# expert_traj, expert_action,_ = shortest_path_expert(self.env)
		# END
		pdb.set_trace()
		self._train_states = np.vstack(expert_traj) # size: (*, 4)
		self._train_actions = np.vstack(expert_action) # size: (*, 4)
		# pdb.set_trace()

		# self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		# self._train_actions = np.array(self._train_actions) # size: (*, 4)
		
	def generate_relabel_data(self,expert_traj, expert_action):
		# 4 apply expert relabelling trick
		self._train_states = []
		self._train_actions = []

		expert_traj_relabel_list = []
		expert_action_relabel_list = []

		# WRITE CODE HERE
		# expert_traj, expert_action, len_traj = shortest_path_expert(self.env)
		self._train_states = np.vstack(expert_traj) # size: (*, 4)
		self._train_actions = np.vstack(expert_action) # size: (*, 4)
		for i,traj in enumerate(expert_traj):
			len_traj = len(traj)
			# pdb.set_trace()
			# print("len_traj: ",len_traj)
			if len_traj>1:
				k = np.random.randint(0,len_traj-1)
			else:
				continue
			# print("k: ",k)
			expert_traj_relabel = copy.deepcopy(traj[0:k,:])
			expert_traj_relabel[:,2:4] = copy.deepcopy(traj[k,0:2])
			expert_traj_relabel = np.array(expert_traj_relabel)
			expert_action_relabel = np.array(expert_action[i][0:k,:])
			# pdb.set_trace()	
			expert_traj_relabel_list.append(expert_traj_relabel)
			expert_action_relabel_list.append(expert_action_relabel)
			# pdb.set_trace()
		# END
		self._train_states_label = np.vstack(expert_traj_relabel_list)
		self._train_actions_label = np.vstack(expert_action_relabel_list)

		# self._train_states = np.array(self._train_states).astype(np.float) # size: (*, 4)
		# self._train_actions = np.array(self._train_actions) # size: (*, 4)

		self._train_states = np.array(self._train_states_label).astype(np.float) # size: (*, 4)
		self._train_actions = np.array(self._train_actions_label) # size: (*, 4)

		# self._train_states = np.concatenate((self._train_states,self._train_states_label),axis=0)
		# self._train_actions = np.concatenate((self._train_actions,self._train_actions_label),axis=0)
		# pdb.set_trace()

	def train(self, num_epochs=20, batch_size=256):
		""" 3
		Trains the model on training data generated by the expert policy.
		Args:
			num_epochs: number of epochs to train on the data generated by the expert.
			batch_size
		Return:
			loss: (float) final loss of the trained policy.
			acc: (float) final accuracy of the trained policy
		"""
		# WRITE CODE HERE


		X = self._train_states  # some random data

		Y = self._train_actions

		Y = np.argmax(Y, axis=1) # Class labels
		
		train_set = TensorDataset(torch.Tensor(X), torch.Tensor(Y).type(torch.long))
		train_loader = DataLoader(dataset=train_set, batch_size = 256, shuffle=True)

		t = 1
		for epoch in range(10):
			running_loss = 0
			correct = 0
			for _,data in enumerate(train_loader, 0):
				x_batch, y_batch = data
				
				self.optimizer.zero_grad()
				yhat = self.model(x_batch)
				# pdb.set_trace()
				loss = self.criterion(yhat, y_batch)
				loss.backward()
				self.optimizer.step()
				
				correct += (torch.argmax(yhat, dim=1) == y_batch).float().sum()
				running_loss += loss.item()
		
		acc = correct / len(train_set)
		print('(%d) loss= %.3f; accuracy = %.1f%%' % (self.iter, loss, 100 * acc))

		return loss,acc
def evaluate_gc(env, policy, n_episodes=50):
	succs = 0
	for ne in range(n_episodes):
		info = generate_gc_episode(env, policy)
		# WRITE CODE HERE
		if info=="succ":
			succs += 1
		# END
	succs /= n_episodes
	print("Current episode {} and success till now {}: ".format(ne,succs))
	return succs

def generate_gc_episode(env, policy):
	"""Collects one rollout from the policy in an environment. The environment
	should implement the OpenAI Gym interface. A rollout ends when done=True. The
	number of states and actions should be the same, so you should not include
	the final state when done=True.
	Args:
		env: an OpenAI Gym environment.
		policy: a trained model
	Returns:
	"""
	done = False
	state = env.reset()

	# while not done:

	while not done:
		# a = env.action_space.sample()
		# pdb.set_trace()
		action = torch.argmax(policy.model(torch.from_numpy(state).type(torch.float))).item()
		state_next, _, done, info = env.step(action)
		# print("state and done criteria and action is {},{},{},{}".format(s,done,info,env.t))
		# state_next, _, done, info = env.step(action)
		# # pdb.set_trace()
		state = state_next
		# print("next state: ",action)


		# WRITE CODE HERE
		# END
	return info


def run_GCBC(expert_trajs, expert_actions, iterations=200):
	gcbc = GCBC(env, expert_trajs, expert_actions)
	# mode = 'vanilla'
	mode = 'relabel'

	if mode == 'vanilla':
		gcbc.generate_behavior_cloning_data(expert_trajs, expert_actions)
	else:
		gcbc.generate_relabel_data(expert_trajs, expert_actions)

	# pdb.set_trace()
	print(gcbc._train_states.shape)

	num_seeds = 5
	loss_vecs = []
	acc_vecs = []
	succ_vecs = []

	for i in range(num_seeds):
		print('*' * 50)
		print('seed: %d' % i)
		loss_vec = []
		acc_vec = []
		succ_vec = []
		gcbc.reset_model()
		gcbc.optimizer = torch.optim.Adam(gcbc.model.parameters(),lr=1e-4)
		for e in range(iterations):
			gcbc.iter = e
			loss, acc = gcbc.train(num_epochs=20)
			loss = loss.detach().numpy().item()
			acc = acc.numpy().item()
			# pdb.set_trace()
			succ = evaluate_gc(env, gcbc)
			loss_vec.append(loss)
			acc_vec.append(acc)
			succ_vec.append(succ)
			# print(e, round(loss,3), round(acc,3), succ)
		loss_vecs.append(loss_vec)
		acc_vecs.append(acc_vec)
		succ_vecs.append(succ_vec)

	loss_vec = np.mean(np.array(loss_vecs), axis = 0).tolist()
	acc_vec = np.mean(np.array(acc_vecs), axis = 0).tolist()
	succ_vec = np.mean(np.array(succ_vecs), axis = 0).tolist()
	# pdb.set_trace()
	### Plot the results
	from scipy.ndimage import uniform_filter
	# you may use uniform_filter(succ_vec, 5) to smooth succ_vec
	
	for mod in ["loss","acc","succ"]:

		plt.figure(figsize=(12, 3))
		if mod == "loss":
			plt.plot(loss_vec)
			plt.xlabel("iteration")
			plt.ylabel(mod)
		elif mod == "acc":
			plt.plot(acc_vec)
			plt.xlabel("iteration")
			plt.ylabel(mod)
		elif mod == "succ":
			succ_vec = uniform_filter(succ_vec, 5)
			plt.plot(succ_vec)
			plt.xlabel("iteration")
			plt.ylabel(mod)
		# WRITE CODE HERE
		# END
		plt.savefig('p2_gcbc_expert_relabel_%s.png' % mod, dpi=300)
		plt.show()

def generate_random_trajs(env):
	import random
	N = 1000
	random_trajs = []
	random_actions = []
	random_goals = []

	# WRITE CODE HERE
	for i in range(N):
		# print(i)
		action_list = []
		# traj = []
		done = False
		s = env.reset()
		traj = []
		while not done:
			traj.append(s)
			a = env.action_space.sample()
			# pdb.set_trace()
			a_one_vector = action_to_one_hot(env, a)
			action_list.append(a_one_vector)
			s, _, done, info = env.step(a)
			# print("state and done criteria and action is {},{},{}".format(s,done,info))
		traj = np.array(traj)
		action_list = np.array(action_list)
		random_trajs.append(traj)
		random_actions.append(action_list)

	random_trajs = np.array(random_trajs)
	random_actions = np.array(random_actions)
	# pdb.set_trace()


	return random_trajs,random_actions


	# END
	# You should obtain random_trajs, random_actions, random_goals from random policy

	# train GCBC based on the previous code
	# WRITE CODE HERE
	# run_GCBC(random_trajs,random_actions,iterations=50)
# generate_random_trajs(env)

is_random = 0

if not is_random:
	print("Using Djikstra search as an expert")
	expert_trajs,expert_actions,_ = shortest_path_expert(env)
	run_GCBC(expert_trajs,expert_actions,iterations=200)
else:
	print("Using random search as an expert")
	expert_trajs,expert_actions = generate_random_trajs(env)
	# pdb.set_trace()
	run_GCBC(expert_trajs,expert_actions,iterations=50)
