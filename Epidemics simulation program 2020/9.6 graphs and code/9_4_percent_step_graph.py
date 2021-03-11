#Epidemic spread in a population of random walkers on a lattice
import numpy as np
import matplotlib.pyplot as plt
import shapely.geometry as SG
import statistics
import random
from numba import jit


N = 64  #lattice size
M = 0  #number of random walkers
L = 20  #lifetime parameter
max_iter = 10000  #maximum number of iterations

@jit
#Calculate the average of a list
def Average(lst): 
    return sum(lst) / len(lst)

@jit
def algorithm(N,M,L,max_iter):
	plt.figure(1)
	percent_list = [0.01,0.05,0.1,0.15,0.2]
	step_list = [1,4,8,16]

	for current_step_value in step_list:

		population_density_value_at_50 = []

		for current_percent_value in percent_list:

			final_death_rate_average_list = []
			final_duration_average_list = []
			final_standard_deviation_death_rate_list = []
			final_standard_deviation_duration_list = []
			final_population_density_list = []

			for density_step_value in np.linspace(0.05,.3,10):
			#for density_step_value in np.arange(0.05,0.15,0.05):
				population_density_list = []
				death_rate_list = []
				duration_list = []

				for i in range (0,10):
					#Rounding the M value to correctly conduct simulation (Number of people can't be a decimal)
					unrounded_M = density_step_value * (N*N)
					M = int(round((unrounded_M)))
					#print(M)

					population_density = M/(N*N)
					x_step = np.array([-1,0,1,0])  #template arrays
					y_step = np.array([0,-1,0,1])

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
						#Add random larger step here for all walkers
							if infect[j] < 2:  #this walker is still alive					
								pick_direction = np.random.choice([0,1,2,3])
								random_percent_value = random.randint(1, 100)/100

								if random_percent_value <= current_percent_value:
									x[j] += current_step_value*x_step[pick_direction]  #update walker coordinates
									y[j] += current_step_value*y_step[pick_direction]
								else:
									x[j] += x_step[pick_direction]  #update walker coordinates
									y[j] += y_step[pick_direction]
								#x[j] += x_step[pick_direction]  #update walker coordinates
								#y[j] += y_step[pick_direction]
								#Modulo function used in case of walkers overpassing the border, causes them to wrap round instead
								x[j] = x[j] % N
								y[j] = y[j] % N

								#The bounds first get the value x[j], if its higher than 64, x[j] becomes the maximum value of 128 and prevents the walker from going any farther in that direction
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
						#print("Current iteration = {0}, healthy population = {1}, sick = {2}, dead = {3}.".format(iterate,remaining_population,n_sick,n_dead))
					
					#Calculate death rate
					death_rate = n_dead / M

					#Print Information for each simulation
					print()
					print("Current iteration = {0}".format(i+1))
					print("Current population density = {0}".format(population_density))
					print("Death rate for the simulation = {0}".format(death_rate))
					print("Final duration = {0}".format(iterate))

					#Adding the final values of each simulation into a list
					population_density_list.append(population_density)
					death_rate_list.append(death_rate)
					duration_list.append(iterate)

				#Calculating the average for each list
				average_population_density = Average(population_density_list)
				average_death_rate = Average(death_rate_list)
				average_duration = Average(duration_list)

				#Calculating the standard deviation of each list
				standard_deviation_death_rate = statistics.stdev(death_rate_list)
				standard_deviation_average_duration = statistics.stdev(duration_list)

				print()
				print("Average population density = {0}".format(average_population_density))
				print("Average death rate = {0}".format(average_death_rate))
				print("Average duration = {0}".format(average_duration))

				print()
				print("Standard deviation for death rate = {0}".format(standard_deviation_death_rate))
				print("Standard deviation duration = {0}".format(standard_deviation_average_duration))
				#fig.tight_layout()  # otherwise the right y-label is slightly clipped
				#plt.show()

				#Appending all the values to the final lists for the graphs
				final_population_density_list.append(density_step_value)
				final_death_rate_average_list.append(average_death_rate)
				final_duration_average_list.append(average_duration)
				final_standard_deviation_death_rate_list.append(standard_deviation_death_rate)
				final_standard_deviation_duration_list.append(standard_deviation_average_duration)

			#print(*final_death_rate_average_list, sep=',')
			#print(*final_duration_average_list, sep=',')

			#Find the population density value that corresponds with a death rate of 50% (0.5) 
			line = SG.LineString(list(zip(final_population_density_list,final_death_rate_average_list)))
			y0 = 0.5
			yline = SG.LineString([(min(final_population_density_list), y0), (max(final_population_density_list), y0)])
			#print(line.intersection(yline))
			coords = np.array(line.intersection(yline))
			x_value_at_50 = coords[0]
			#print(coords)
			#print(x_value_at_50)
			population_density_value_at_50.append(x_value_at_50)
			print("Average population density values at 0.5 death rate = {0}".format(population_density_value_at_50))

		plt.plot(percent_list,population_density_value_at_50,label=str(current_step_value))
		plt.scatter(percent_list,population_density_value_at_50)
		
	plt.grid(axis='y', alpha=0.4)
	plt.title("Population density where death rate is equal to 0.5 for different step sizes and percent chance")
	plt.xlabel("Chance of larger step size (%)",fontsize = 12)
	plt.ylabel("Population density",fontsize = 12)
	plt.legend(loc='upper right',prop={'size': 12})
	plt.savefig("All step size graphs for different step percentage chance compared to population density at 50%.png")
	plt.show()
			#label=str(density_step_value)
			#plt.figure(1)
			#plt.plot(final_population_density_list,final_death_rate_average_list)
			#plt.scatter(final_population_density_list,final_death_rate_average_list)
			#plt.errorbar(final_population_density_list,final_death_rate_average_list,final_standard_deviation_death_rate_list,capsize=5,linestyle='None')
			#plt.grid(axis='y', alpha=0.4)
			#plt.title("Population density vs. average death rate graph")
			#plt.xlabel("Population density",fontsize = 12)
			#plt.ylabel("Average death rate",fontsize = 12)
			#plt.savefig("Population density vs. average death rate graph 4 step (20%).png")

			#plt.figure(2)
			#plt.plot(final_population_density_list,final_duration_average_list)
			#plt.scatter(final_population_density_list,final_duration_average_list)
			#plt.errorbar(final_population_density_list,final_duration_average_list,final_standard_deviation_duration_list,capsize=5,linestyle='None')
			#plt.grid(axis='y', alpha=0.4)
			#plt.title("Population density vs. average duration graph")
			#plt.xlabel("Population density",fontsize = 12)
			#plt.ylabel("Average duration",fontsize = 12)
			#plt.savefig("Population density vs. average duration graph 4 step (20%).png")
			#plt.show()
algorithm(N,M,L,max_iter)
