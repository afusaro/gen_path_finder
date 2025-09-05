# genetic algorithm search of the one max optimization problem
from numpy import ceil, floor, log2
from numpy import random
from numpy.random import randint
from numpy.random import rand
import numpy as np


# random.seed(10)

max_value = 8
cant_bits = int(ceil(log2(max_value)))
mutations = []
n_cols=int(50)
n_rows=int(50)

MAX_COSTMAP_ELEMENT_COST = int(5)
DISTANCE_TO_END_COST = int(10)
OUTSIDE_ZONE_COST = int(50)
STEP_COST=(1)

costmap = np.zeros([n_rows,n_cols], dtype=int)

start_point = [0,0]
end_point = [n_rows,n_cols]


for i in range(0,cant_bits):
	value = 1 << i
	mutations.append(value)

# objective function
def onemax(x):
	return -sum(x)

def get_next_index(index, order):
	if order >8:
		order -=8
	step_cost = STEP_COST

	if order == 0:
		index[1]+=1 #go right
	elif order ==1:
		index[1]+=1 #go right
		index[0]-=1 #go up
	elif order==2:
		index[0]-=1 #go up
	elif order==3:
		index[1]-=1 #go left
		index[0]-=1 #go up
	elif order==4:
		index[1]-=1 #go left
	elif order ==5: 
		index[1]-=1 #go left
		index[0]+=1 #go down
	elif order ==6:
		index[0]+=1 #go down
	elif order ==7:
		index[1]+=1 #go right
		index[0]+=1 #go down
	elif order ==8: #stay right there
		step_cost = 0
		pass
	else:
		print("error indexing shouldnt be here")
	return index, step_cost

def check_index_inside_costmap(index):
	if index[0] < 0 or index[1] < 0 or index[0] >=n_rows or index[1] >=n_cols:
		return False
	return True 

def get_cost(index):
	return costmap[index[0],index[1]]

def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def calculate_distance_to_end_cost(index):
	return manhattan(index,end_point) * DISTANCE_TO_END_COST


def calculate_path_distance_to_end_cost(path):
	index = start_point.copy()
	for step in path:
		index, step_cost = get_next_index(index, step)
		
	cost = calculate_distance_to_end_cost(index)
	return cost

def calculate_cost(path):
	index = start_point.copy()
	cost = 0 
	for step in path:
		index, step_cost = get_next_index(index, step)
		if check_index_inside_costmap(index) is False:
			cost += OUTSIDE_ZONE_COST
		else:
			cost+=step_cost
			cost+=get_cost(index)
			

	cost += calculate_distance_to_end_cost(index)
	return cost
	
# tournament selection
def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better (e.g. perform a tournament)
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

# crossover two parents to create two children
def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

# mutation operator


def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			a = random.choice(mutations)
			bitstring[i] ^= a

# genetic algorithm
def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut, elitism_size):
	# initial population of random bitstring
	pop = [randint(0, max_value+1, n_bits).tolist() for _ in range(n_pop)]
	# keep track of best solution
	best, best_eval = 0, objective(pop[0])
	# enumerate generations
	for gen in range(n_iter):
		# evaluate all candidates in the population
		scores = [objective(c) for c in pop]
		
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))

		if elitism_size == 0:
			# select parents
			selected = [selection(pop, scores) for _ in range(n_pop)]
		else:

			zipped_lists = zip(scores, pop)
			sorted_pairs = sorted(zipped_lists)

			tuples = zip(*sorted_pairs)
			scores, pop = [ list(tuple) for tuple in  tuples]
			selected = pop[0: elitism_size]
			new_selection = [selection(pop[elitism_size:], scores[elitism_size:]) for _ in range(n_pop-elitism_size)]
			selected.extend(new_selection)



		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			# get selected parents in pairs
			p1, p2 = selected[i], selected[i+1]
			# crossover and mutation
			for c in crossover(p1, p2, r_cross):
				# mutation
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

# define the total iterations
n_iter = 150
# bits
n_steps = 70
# define the population size
n_pop = 100
# crossover rate
r_cross = 0.9
# mutation rate
r_mut = 1.0 / float(n_steps)
# r_mut=0.05
elitism_size = 10
# perform the genetic algorithm search
best, score = genetic_algorithm(calculate_cost, n_steps, n_iter, n_pop, r_cross, r_mut, elitism_size)
print('Done!')
print('f(%s) = %f' % (best, score))
print('distance to end cost: %f'%(calculate_path_distance_to_end_cost(best)))
