import numpy as np
import pdb
import copy

import matplotlib.pyplot as plt

def find_neighbors(env_map,u,x_dim,y_dim,map_dist):

	neighbors = []
	env_map_2D = env_map.reshape(x_dim,y_dim)
	map_dist_2D = map_dist.reshape(x_dim,y_dim)
	x = int(u/y_dim)
	y = u%x_dim

	if x-1>0:
		if env_map_2D[x-1,y] == True and map_dist_2D[x-1,y]!=1000:
			neighbors.append((x-1)*y_dim + y)

	if x+1<x_dim:
		if env_map_2D[x+1,y] == True and map_dist_2D[x+1,y]!=1000:
			neighbors.append((x+1)*y_dim + y)

	if y-1>0:
		if env_map_2D[x,y-1] == True and map_dist_2D[x,y-1]!=1000:
			neighbors.append((x*y_dim) + y-1)

	if y+1<y_dim:
		if env_map_2D[x,y+1] == True and map_dist_2D[x,y+1]!=1000:
			neighbors.append((x*y_dim) + y+1)
	# print("neighbors: ",neighbors)
	# pdb.set_trace()
	return np.array(neighbors)



def find_shortes_path(map,previous,state):
	
	start_node = state[0:2]
	goal_node = state[2:4]
	map_plot  = np.zeros(map.shape)
	start_node_1D = start_node[0]*map.shape[1]+start_node[1]
	goal_node_1D = goal_node[0]*map.shape[1]+goal_node[1]
	uu = goal_node_1D
	x_dim,y_dim = map.shape[0],map.shape[1]
	x_last = int(uu/y_dim)
	y_last = uu%x_dim
	s_path = []
	action = []

	for x in range(map.shape[0]):
		for y in range(map.shape[1]):
			# pdb.set_trace()
			if map[x,y] == True:
				map_plot[x,y] = 1
			elif map[x,y] == False:
				map_plot[x,y] = 0

			# print(map_plot)

	while uu != start_node_1D:
		# print("reached goal")
		# s_path.append(uu)
		uu = int(previous[uu])
		# print("shortest_path: ",action)
		x = int(uu/y_dim)
		y = uu%x_dim
		map_plot[x,y] = 2
		# plt.imshow(map_plot)
		# plt.show()
		action.append((x_last-x,y_last-y))
		s_path.append(np.array([x,y,goal_node[0],goal_node[1]]))
		x_last = x
		y_last = y
		# pdb.set_trace()
	return s_path[::-1],action[::-1]




def djikstra(map,state):

	map_flatten = map.flatten()
	map_dist = (np.ones(map.shape)*np.inf).flatten()
	start_node = state[0:2]
	goal_node = state[2:4]
	optimal_path = []
	map_dist[start_node[0]*map.shape[1]+start_node[1]] = 0
	previous = (np.zeros(map.shape)).flatten()
	x_dim,y_dim = map.shape[0],map.shape[1]
	map_plot  = np.zeros(map.shape)
	start_node_1D = start_node[0]*map.shape[1]+start_node[1]
	goal_node_1D = goal_node[0]*map.shape[1]+goal_node[1]

	for x in range(map.shape[0]):
		for y in range(map.shape[1]):
			# pdb.set_trace()
			if map[x,y] == True:
				map_plot[x,y] = 1
			elif map[x,y] == False:
				map_plot[x,y] = 0

	Q = np.arange(0,map_flatten.shape[0])
	count = 0
	u_last = start_node[0]*map.shape[1]+start_node[1]
	while Q.shape[0] is not 0:

		# print(map_dist)
		u = np.argmin(map_dist)
		# print("Getting into iteraion: ",u,map_dist[u])
		# pdb.set_trace()

		Q = np.delete(Q,np.where(Q==u))
		if u == goal_node[0]*map.shape[1]+goal_node[1]:
			# previous[u] = u
			s_path,action = find_shortes_path(map,previous,state)

			return s_path,action

		neighbors = find_neighbors(map_flatten,u,map.shape[0],map.shape[1],map_dist)

		for nb in neighbors:
			alt = map_dist[u] + 1
			if alt < map_dist[nb]:
				map_dist[nb] = alt
				previous[nb] = u


		map_dist[u] = 1000 #not deleting but adding big value
		count += 1
		if Q.shape[0] == 57:
			pdb.set_trace()
		map_plot[int(u_last/y_dim),u_last%x_dim] = 0.5
		u_last = u
		map_plot[int(u/y_dim),u%x_dim] = 2
		# plt.imshow(map_plot)
		# plt.show()
	pdb.set_trace()
	print("Reached goal")


	# while 	