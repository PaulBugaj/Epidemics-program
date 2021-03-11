#Epidemic spread in a population of random walkers on a lattice
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import linregress
import statistics
import random

def square(x): 
	return x*x 

def Average(lst): 
    return sum(lst) / len(lst)

N = 64  #lattice size
M = 0  #number of random walkers
L = 20  #lifetime parameter
max_iter = 10000  #maximum number of iterations 

percent_list = [0.01,0.05,0.1,0.15,0.2]
step_list = [1,4,8,16]

for current_step_value in step_list:
	slope_values = []
	for current_percent_value in percent_list:
		density_value = 0.49
		unrounded_M = density_value * (N*N)
		M = int(round((unrounded_M)))
		#print(M)
		population_density = M/(N*N)

		multiple_list = []
		#Start of simulations
		for i in range (0,10):		
			x_step = np.array([-1,0,1,0])  #template arrays
			y_step = np.array([0,-1,0,1])

			list_to_numpy = []

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

			#Find x and y value for first infected walker
			first_infected_walker_x = x[initial_random_walker]
			first_infected_walker_y = y[initial_random_walker]
			#print(first_infected_walker_x)
			#print(first_infected_walker_y)

			iteration_number = []
			final_mean_distance_for_iteration = []

			n_sick,n_dead,iterate = 1,0,0  #counters
			remaining_population = M
			while (n_sick > 0) and (iterate < max_iter):  #temporal iteration
				#Clear the list during each iteration
				distance_from_origin = []
				for j in range (0,M):  #loop over all walkers
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

						x[j] = min(N,max(x[j],1))  #bounding walls in x,y
						y[j] = min(N,max(y[j],1))
					if infect[j] == 1:  #this walker is sick
						lifespan[j] -= 1
						
						#Calculate distance from origin point
						distance = math.sqrt(square(x[j]-first_infected_walker_x) + square(y[j]-first_infected_walker_y))
						#print("Distance from origin for infected walker {0} = {1}".format(j,distance))

						#Add calculated distance from infected to origin to list
						distance_from_origin.append(distance)

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

				#Calculate the mean from all distances of infected from origin in the list					
				mean_distance_from_origin = Average(distance_from_origin)
				#print(mean_distance_from_origin)

				#Add the final mean distance value for that iteration to list
				final_mean_distance_for_iteration.append(mean_distance_from_origin)
				#print(final_mean_distance_for_iteration)

				ts_population[iterate] = remaining_population
				ts_sick[iterate] = n_sick
				iterate += 1
				iteration_number.append(iterate)
				#print(iteration_number)
				#print("Current iteration = {0}, healthy population = {1}, sick = {2}, dead = {3}.".format(iterate,remaining_population,n_sick,n_dead))

			list_to_numpy = final_mean_distance_for_iteration[0:120]
			#list_to_numpy = np.asarray(final_mean_distance_for_iteration, dtype = float)
			#print("list number {0}-{1}".format(i+1,list_to_numpy))
			#Append previous final mean distances to multiple simulation means list
			multiple_list.append(list_to_numpy)
			death_rate = n_dead / M

			print("Current iteration = {0}".format(i+1))
			print("Current population density = {0}".format(population_density))
			print("Death rate for the simulation = {0}".format(death_rate))

		#Used to show all lists combined into one
		#print(multiple_list)

		#Averaging all of the values in all of the lists one by one and then producing one final list of all the previous values from all simulations averaged
		average_of_all_means_for_all_simulations = [statistics.mean(k) for k in zip(*multiple_list)]
		#print()
		#print("The result of finding the averages of all 10 simulations is = {0}".format(average_of_all_means_for_all_simulations))
		final_iteration_number = iteration_number[0:len(average_of_all_means_for_all_simulations)]

		#slope, intercept, r_value, p_value, std_err = linregress(final_iteration_number, average_of_all_means_for_all_simulations)
		#slope_list = np.array(slope)

		#Generate graph for Average of all 10 simulations
		#plt.figure(1)
		#plt.grid(axis='y', alpha=0.4)
		#plt.plot(final_iteration_number,average_of_all_means_for_all_simulations,marker='o',markevery=12)
		#plt.scatter(final_iteration_number,average_of_all_means_for_all_simulations)
		#plt.text(20,10,'Slope= {0:.2f}'.format(slope_list),fontsize=12)
		#plt.title("Averaged mean distance value for each iteration of 20 simulations")
		#plt.xlabel("Iteration",fontsize = 14)
		#plt.ylabel("Mean distance",fontsize = 14)
		#plt.savefig("Averaged mean distance value for each iteration of 10 simulations 16 step (1%).png")

		#Set the graph parameters to the first 50 values of each list
		iteration_number_small_graph = final_iteration_number[0:50]
		final_mean_distance_for_iteration_small_graph = average_of_all_means_for_all_simulations[0:50]

		#Find the slope of the first 20 values
		slope2, intercept2, r_value2, p_value2, std_err2 = linregress(iteration_number_small_graph, final_mean_distance_for_iteration_small_graph)
		#slope_list2 = np.array(slope2)
		slope_values.append(slope2)
		print(slope_values)

		#Generate graph for first 20 iterations of all 10 simulations
		#plt.figure(2)

	plt.plot(percent_list,slope_values,label=str(current_step_value))
	plt.scatter(percent_list,slope_values)
		#plt.scatter(iteration_number,final_mean_distance_for_iteration)
		#plt.text(10,8,'Slope= {0:.2f}'.format(slope_list2),fontsize=12)
		#plt.title("Averaged mean distance value for each iteration of 20 simulations")
plt.grid(axis='y', alpha=0.4)
plt.xlabel("Percent chance of larger jump (%)",fontsize = 14)
plt.ylabel("Slope value (50 iterations)",fontsize = 14)
plt.legend(loc='upper right',prop={'size': 12})
plt.savefig("Percent chance of increased jump size compared to calculated slope values (first 50 iterations, 0.49 density).png")
plt.show()