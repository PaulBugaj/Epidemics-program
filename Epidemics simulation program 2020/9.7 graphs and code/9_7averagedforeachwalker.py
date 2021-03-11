#Epidemic spread in a population of random walkers on a lattice
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import math
from scipy.stats import linregress
import statistics
import random

N = 64  #lattice size
M = 0 #number of random walkers
L = 20  #lifetime parameter
max_iter = 10000  #maximum number of iterations 
population_density = M/(N*N)
x_step = np.array([-1,0,1,0])  #template arrays
y_step = np.array([0,-1,0,1])
plt.figure(1)
for density_step_value in np.linspace(0.24,.49,3):
	multiple_list = []
	average_of_all_means_for_all_simulations = []
	for i in range (0,10):
		unrounded_M = density_step_value * (N*N)
		M = int(round((unrounded_M)))
		#print(M)

		infected_walkers = [0] * M

		population_density = M/(N*N)
		x,y = np.zeros(M),np.zeros(M)  #walker (x,y) coordinates
		infect = np.zeros(M)  #walker health status
		lifespan = np.zeros(M)
		ts_sick = np.zeros(max_iter)
		ts_population = np.zeros(max_iter)
		for j in range(M):
			x[j]=np.random.random_integers(0,N)
			y[j]=np.random.random_integers(0,N)
			lifespan[j]=L
		initial_random_walker = np.random.random_integers(0,M-1)  #Picking a random host for virus
		infect[initial_random_walker] = 1
		n_sick,n_dead,iterate = 1,0,0  #counters
		remaining_population = M
		while (n_sick > 0) and (iterate < max_iter):  #temporal iteration
			for j in range (0,M):  #loop over all walkers
				if infect[j] < 2:  #this walker is still alive
					pick_direction = np.random.choice([0,1,2,3])  
					random_percent_value = random.randint(1, 100)/100

					if random_percent_value <= 0.1:
						x[j] += 16*x_step[pick_direction]  #update walker coordinates
						y[j] += 16*y_step[pick_direction]
					else:
						x[j] += x_step[pick_direction]  #update walker coordinates
						y[j] += y_step[pick_direction]

					#Modulo function used in case of walkers overpassing the border, causes them to wrap round instead
					x[j] = x[j] % N
					y[j] = y[j] % N

					x[j] = min(N,max(x[j],1))  #bounding walls in x,y
					y[j] = min(N,max(y[j],1))
				if infect[j] == 1:  #this walker is sick
					lifespan[j] -= 1
					if lifespan[j] <= 0:
						infect[j] = 2
						n_sick -= 1
						n_dead += 1
						remaining_population -= 1
					for k in range(0,M):  #check for walkers on node
						if infect[k] == 0 and k != j:  #this walker is healthy
							if x[j] == x[k] and y[j] == y[k]:
								infect[k] = 1
								n_sick += 1
								#How many healthy walkers were infected by each already infected walkers
								#Afterwards, create another list that counts how many infected walkers infected 1 person, 2 people, ect. make histogram and then normalize it
								infected_walkers[j] += 1
			ts_population[iterate] = remaining_population
			ts_sick[iterate] = n_sick
			iterate += 1
			#print("Current iteration = {0}, healthy population = {1}, sick = {2}, dead = {3}.".format(iterate,remaining_population,n_sick,n_dead))
		death_rate = n_dead / M
		infected_counter_list = []
		#print(infected_walkers)
		count_of_how_many_infected = np.bincount(infected_walkers)
		#print(count_of_how_many_infected)
		for i in range (0,len(count_of_how_many_infected)):
			infected_counter_list.append(count_of_how_many_infected[i])
		#print(infected_counter_list)
		list_counter = list(range(len(count_of_how_many_infected)))
		#print(list_counter)
		#pdf = stats.norm.pdf(infected_walkers)

		multiple_list.append(infected_walkers)

		print("Current population density = {0}".format(population_density))
		print("Death rate for the simulation = {0}".format(death_rate))
	average_of_all_means_for_all_simulations = [statistics.mean(k) for k in zip(*multiple_list)]
	plt.hist(average_of_all_means_for_all_simulations, bins=len(list_counter)-1,histtype= 'step',fill = None,normed=True,label=str(density_step_value))


plt.grid(axis='y', alpha=0.4)
plt.grid(axis='x', alpha=0.4)
#plt.plot(infected_counter_list,list_counter)
plt.xlabel('Number of infected', fontsize = 12)
plt.ylabel('Number of occurences (PDF)', fontsize = 12)
#plt.title('Number of infected vs. occurrences (Normalized)')
plt.legend(loc='upper right',prop={'size': 16})
plt.savefig("PDF graph for 3 density values 10 simulations 0.png")
plt.show()

#plt.figure(2)
#plt.grid(axis='y', alpha=0.4)
#plt.grid(axis='x', alpha=0.4)
#plt.plot(list_counter,pdf,'r-', lw=5, label='norm pdf')
#plt.show()
#fig, ax1 = plt.subplots()

#color = 'tab:red'
#ax1.set_xlabel('Iteration')
#ax1.set_ylabel('Deaths', color=color)
#ax1.set_title('Virus spreading simulation')
#ax1.plot(range(0,iterate), ts_sick[0:iterate], color=color)
#ax1.tick_params(axis='y', labelcolor=color)

#ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

#color = 'tab:green'
#ax2.set_ylabel('Population', color=color)  # we already handled the x-label with ax1
#ax2.plot(range(0,iterate), ts_population[0:iterate], color=color)
#ax2.tick_params(axis='y', labelcolor=color)

#fig.tight_layout()  # otherwise the right y-label is slightly clipped
#plt.show()