#Epidemic spread in a population of random walkers on a lattice
import numpy as np
import matplotlib.pyplot as plt
import random
N = 64  #lattice size
M = 2007  #number of random walkers
L = 20  #lifetime parameter
max_iter = 10000  #maximum number of iterations 
population_density = M/(N*N)
x_step = np.array([-1,0,1,0])  #template arrays
y_step = np.array([0,-1,0,1])
print(population_density)
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
			x[j] += x_step[pick_direction]  #update walker coordinates
			y[j] += y_step[pick_direction]
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
	ts_population[iterate] = remaining_population
	ts_sick[iterate] = n_sick
	iterate += 1
	print("Current iteration = {0}, healthy population = {1}, sick = {2}, dead = {3}.".format(iterate,remaining_population,n_sick,n_dead))
	

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Iteration',fontsize = 14)
ax1.set_ylabel('Deaths', color=color,fontsize = 14)
#ax1.set_title('Virus spreading simulation')
ax1.plot(range(0,iterate), ts_sick[0:iterate], color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
ax2.set_ylabel('Population', color=color,fontsize = 14)  # we already handled the x-label with ax1
ax2.plot(range(0,iterate), ts_population[0:iterate], color=color)
ax2.tick_params(axis='y', labelcolor=color)
plt.savefig("Deaths and population graph.png")
#fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()