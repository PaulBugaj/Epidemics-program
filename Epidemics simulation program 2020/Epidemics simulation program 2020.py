import numpy as np
import math
import shapely.geometry as SG
import statistics
import random
import argparse
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy import stats
from scipy.stats import norm
from scipy.stats import linregress
from numba import jit

parser = argparse.ArgumentParser(description='This program is used to produce different models for simulations tht analyze the spread of disease within a population (epidemic) program')
parser.add_argument('-n','--lattice_size', type=int, default = 64, help='Size of lattice for simulation (N * N), default value = 64')
parser.add_argument('-nrs','--nr_of_simulations', type=int, default = 10, help='The number of simulations that are to be conducted, default value = 10')
parser.add_argument('-pd','--population_density', type=float, default = 0.49, help='Initial population density value, default value = 0.49')
parser.add_argument('-npd','--population_density_points', type=int, default = 10, help='Number of population density points calculated within range, default value = 10')
parser.add_argument('-minpd','--minimum_population_density', type=float, default = 0.05, help='Minimum population density value used in ranges, default value = 0.05')
parser.add_argument('-maxpd','--maximum_population_density', type=float, default = 0.55, help='Maximum population density value used in ranges, default value = 0.55')
parser.add_argument('-s','--step_size', type=int, default = 1, help='The number of units a walker has a chance to move, default value = 1')
parser.add_argument('-pv','--percent_size', type=float, default = 0.01, help='Percent value that a certain step size will occur, default value = 0.01')
parser.add_argument('-f','--function', type=str, help='Choosing which function you want to use to generate a model (function_1 - function_7)', required=True)
args = parser.parse_args()

N = args.lattice_size  #lattice size
M = 0  #number of random walkers
L = 20  #lifetime parameter
max_iter = 10000  #maximum number of iterations 
population_density_value = args.population_density #population density value
min_population_density = args.minimum_population_density
max_population_density = args.maximum_population_density
number_of_densities = args.population_density_points
number_of_simulations = args.nr_of_simulations
step_value = args.step_size
percent_value = args.percent_size

@jit
#Calculate the average of a list
def Average(lst): 
    return sum(lst) / len(lst)

@jit
def square(x): 
	return x*x 

def first_function(N,M,L,max_iter,number_of_simulations,population_density_value,step_value,percent_value):
	unrounded_M = population_density_value * (N*N)
	M = int(round((unrounded_M)))
	#population_density = M/(N*N)
	final_ts_populations = []
	final_ts_sick = []

	for i in range (0,number_of_simulations):
		x_step = np.array([-1,0,1,0])  #template arrays
		y_step = np.array([0,-1,0,1])

		x,y = np.zeros(M),np.zeros(M)  #walker (x,y) coordinates
		infect = np.zeros(M)  #walker health status
		lifespan = np.zeros(M)
		ts_sick = []
		ts_population = []
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

					if random_percent_value <= percent_value:
						x[j] += step_value*x_step[pick_direction]  #update walker coordinates
						y[j] += step_value*y_step[pick_direction]
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
			ts_population.append(remaining_population)
			ts_sick.append(n_sick)
			iterate += 1
			print("Current iteration = {0}, healthy population = {1}, sick = {2}, dead = {3}.".format(iterate,remaining_population,n_sick,n_dead))
		final_ts_populations.append(ts_population)
		#print(final_ts_populations)
		final_ts_sick.append(ts_sick)

	average_of_all_populations = [statistics.mean(k) for k in zip(*final_ts_populations)]
	#print(average_of_all_populations)
	number_of_iterations = len(average_of_all_populations)
	average_of_all_sick = [statistics.mean(k) for k in zip(*final_ts_sick)]
	#print(final_ts_populations)

	fig, ax1 = plt.subplots()

	color = 'tab:red'
	ax1.set_xlabel('Iteration',fontsize = 14)
	ax1.set_ylabel('Deaths', color=color,fontsize = 14)
	#ax1.set_title('Virus spreading simulation')
	ax1.plot(range(0,len(average_of_all_populations)), average_of_all_sick, color=color)
	ax1.tick_params(axis='y', labelcolor=color)

	ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

	color = 'tab:green'
	ax2.set_ylabel('Population', color=color,fontsize = 14)  # we already handled the x-label with ax1
	ax2.plot(range(0,len(average_of_all_populations)), average_of_all_populations, color=color)
	ax2.tick_params(axis='y', labelcolor=color)
	plt.savefig("Infected versus healthy walker graph at each iteration for "+ str(number_of_simulations) +" simulation(s) for population density = "+ str(population_density_value) +" for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")
	fig.tight_layout()  # otherwise the right y-label is slightly clipped
	plt.show()

@jit
def second_function(N,M,L,max_iter,min_population_density,max_population_density,number_of_densities,step_value,percent_value):
	final_death_rate_average_list = []
	final_duration_average_list = []
	final_standard_deviation_death_rate_list = []
	final_standard_deviation_duration_list = []
	final_population_density_list = []

	for density_step_value in np.linspace(min_population_density,max_population_density,number_of_densities):
	#for density_step_value in np.arange(0.05,0.15,0.05):
		population_density_list = []
		death_rate_list = []
		duration_list = []

		for i in range (0,number_of_simulations):
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

						if random_percent_value <= percent_value:
							x[j] += step_value*x_step[pick_direction]  #update walker coordinates
							y[j] += step_value*y_step[pick_direction]
						else:
							x[j] += x_step[pick_direction]  #update walker coordinates
							y[j] += y_step[pick_direction]
						
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
	plt.figure(1)
	plt.plot(final_population_density_list,final_death_rate_average_list)
	plt.scatter(final_population_density_list,final_death_rate_average_list)
	plt.errorbar(final_population_density_list,final_death_rate_average_list,final_standard_deviation_death_rate_list,capsize=5,linestyle='None')
	plt.grid(axis='y', alpha=0.4)
	#plt.title("Population density vs. average death rate graph")
	plt.xlabel("Population density",fontsize = 12)
	plt.ylabel("Average death rate",fontsize = 12)
	plt.savefig("Population density vs. average death rate graph for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")

	plt.figure(2)
	plt.plot(final_population_density_list,final_duration_average_list)
	plt.scatter(final_population_density_list,final_duration_average_list)
	plt.errorbar(final_population_density_list,final_duration_average_list,final_standard_deviation_duration_list,capsize=5,linestyle='None')
	plt.grid(axis='y', alpha=0.4)
	#plt.title("Population density vs. average duration graph")
	plt.xlabel("Population density",fontsize = 12)
	plt.ylabel("Average duration",fontsize = 12)
	plt.savefig("Population density vs. average duration graph for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")
	plt.show()

def third_function(N,M,L,max_iter,number_of_simulations,population_density_value,step_value,percent_value):
	unrounded_M = population_density_value * (N*N)
	M = int(round((unrounded_M)))

	population_density = M/(N*N)

	multiple_list = []
	#Start of simulations
	for i in range (0,number_of_simulations):		
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

					if random_percent_value <= percent_value:
						x[j] += step_value*x_step[pick_direction]  #update walker coordinates
						y[j] += step_value*y_step[pick_direction]
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

	slope, intercept, r_value, p_value, std_err = linregress(final_iteration_number, average_of_all_means_for_all_simulations)
	slope_list = np.array(slope)

	#Generate graph for Average of all 10 simulations
	plt.figure(1)
	plt.grid(axis='y', alpha=0.4)
	plt.plot(final_iteration_number,average_of_all_means_for_all_simulations,marker='o',markevery=12)
	#plt.scatter(final_iteration_number,average_of_all_means_for_all_simulations)
	plt.text(20,10,'Slope= {0:.2f}'.format(slope_list),fontsize=12)
	#plt.title("Averaged mean distance value for each iteration of 20 simulations")
	plt.xlabel("Iteration",fontsize = 14)
	plt.ylabel("Mean distance",fontsize = 14)
	plt.savefig("Averaged mean distance value for each iteration of "+ str(number_of_simulations) +" simulations for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")

	#Set the graph parameters to the first 20 values of each list
	iteration_number_small_graph = final_iteration_number[0:20]
	final_mean_distance_for_iteration_small_graph = average_of_all_means_for_all_simulations[0:20]

	#Find the slope of the first 20 values
	slope2, intercept2, r_value2, p_value2, std_err2 = linregress(iteration_number_small_graph, final_mean_distance_for_iteration_small_graph)
	slope_list2 = np.array(slope2)

	#Generate graph for first 20 iterations of all 10 simulations
	plt.figure(2)
	plt.grid(axis='y', alpha=0.4)
	plt.plot(iteration_number_small_graph,final_mean_distance_for_iteration_small_graph,marker='o',markevery=12)
	#plt.scatter(iteration_number,final_mean_distance_for_iteration)
	plt.text(10,8,'Slope= {0:.2f}'.format(slope_list2),fontsize=12)
	#plt.title("Averaged mean distance value for each iteration of 20 simulations")
	plt.xlabel("Iteration",fontsize = 14)
	plt.ylabel("Mean distance",fontsize = 14)
	plt.savefig("Averaged mean distance value for first 20 iterations of "+ str(number_of_simulations) +" simulations for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")


def fourth_function(N,M,L,max_iter,number_of_simulations,min_population_density,max_population_density,number_of_densities,step_value,percent_value):
	x_step = np.array([-1,0,1,0])  #template arrays
	y_step = np.array([0,-1,0,1])
	plt.figure(1)
	number_of_bins = []
	
	for density_step_value in np.linspace(min_population_density,max_population_density,number_of_densities):

		print("Current population density = {0}".format(density_step_value))
		multiple_list = []
		average_of_all_means_for_all_simulations = []
		for i in range (0,number_of_simulations):
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

						if random_percent_value <= percent_value:
							x[j] += step_value*x_step[pick_direction]  #update walker coordinates
							y[j] += step_value*y_step[pick_direction]
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
									infected_walkers[j] += 1.0
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
			
			float_infected_counter_list = list(map(float, infected_counter_list))
			#print(float_infected_counter_list)
			#print(list_counter)
			multiple_list.append(float_infected_counter_list)
			#print("Current population density = {0}".format(population_density))
			print("Death rate for the simulation = {0}".format(death_rate))
		#print(multiple_list)
		longest_length = len(max(multiple_list,key=len))
		#print(longest_length)

		for q in range(0,longest_length-1):
			for row in multiple_list:
				if len(row) < longest_length:
					row.append(0)
		#print(multiple_list)

		average_of_all_means_for_all_simulations = [statistics.mean(k) for k in zip(*multiple_list)]
		#print(average_of_all_means_for_all_simulations)
		round_to_whole = [round(num) for num in average_of_all_means_for_all_simulations]
		#print(round_to_whole)
		sort_into_list = np.array(round_to_whole)
		#print(sort_into_list)
		list_counter = list(range(len(sort_into_list)))
		#print(list_counter)
		number_of_bins.append(len(list_counter))
		#print(number_of_bins)

		sort_into_list_final = np.repeat(np.arange(sort_into_list.size),sort_into_list)

		plt.hist(sort_into_list_final, bins=number_of_bins[0]+1, range=[0, number_of_bins[0]+1],histtype= 'step',fill = None,normed=True,label=str(density_step_value))


	plt.yscale('log')
	plt.tick_params(axis='y', which='minor', labelsize = 6)
	plt.tick_params(axis='y', which='major', labelsize = 9)
	plt.gca().yaxis.set_minor_formatter(FormatStrFormatter("%.4f"))
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
	plt.grid(axis='y', alpha=0.4)
	plt.grid(axis='x', alpha=0.4)
	#plt.plot(infected_counter_list,list_counter)
	plt.xlabel('Number of infected', fontsize = 12)
	plt.ylabel('Number of occurences (PDF)', fontsize = 12)
	#plt.title('Number of infected vs. occurrences (Normalized)')
	plt.legend(loc='upper right',prop={'size': 16})
	plt.savefig("PDF graph for 3 density values for "+ str(number_of_simulations) +" simulations for step = "+ str(step_value) +" and percent chance = "+ str(percent_value) +".png")
	plt.show()

@jit
def fifth_function(N,M,L,max_iter,number_of_simulations,min_population_density,max_population_density,number_of_densities):
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

			for density_step_value in np.linspace(min_population_density,max_population_density,number_of_densities):
			#for density_step_value in np.arange(0.05,0.15,0.05):
				population_density_list = []
				death_rate_list = []
				duration_list = []

				for i in range (0,number_of_simulations):
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
	plt.xlabel("Chance of larger step size (%)",fontsize = 12)
	plt.ylabel("Population density",fontsize = 12)
	plt.legend(loc='upper right',prop={'size': 12})
	plt.savefig("All step size graphs for different step percentage chance compared to population density at 50%.png")
	plt.show()

def sixth_function(N,M,L,max_iter,number_of_simulations,population_density_value):
	percent_list = [0.01,0.05,0.1,0.15,0.2]
	step_list = [1,4,8,16]

	for current_step_value in step_list:
		slope_values = []
		for current_percent_value in percent_list:
			unrounded_M = population_density_value * (N*N)
			M = int(round((unrounded_M)))
			#print(M)
			population_density = M/(N*N)

			multiple_list = []
			#Start of simulations
			for i in range (0,number_of_simulations):		
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

			iteration_number_small_graph = final_iteration_number[0:50]
			final_mean_distance_for_iteration_small_graph = average_of_all_means_for_all_simulations[0:50]

			slope2, intercept2, r_value2, p_value2, std_err2 = linregress(iteration_number_small_graph, final_mean_distance_for_iteration_small_graph)

			slope_values.append(slope2)
			print(slope_values)

			#Generate graph for first 20 iterations of all 10 simulations
			#plt.figure(2)

		plt.plot(percent_list,slope_values,label=str(current_step_value))
		plt.scatter(percent_list,slope_values)

	plt.grid(axis='y', alpha=0.4)
	plt.xlabel("Percent chance of larger jump (%)",fontsize = 14)
	plt.ylabel("Slope value (50 iterations)",fontsize = 14)
	plt.legend(loc='upper right',prop={'size': 12})
	plt.savefig("Percent chance of increased jump size compared to calculated slope values (iterations = 50, population density = " + str(population_density_value) + ").png")
	plt.show()

def seventh_function(N,M,L,max_iter,number_of_simulations,population_density_value):
	percent_list = [0.01,0.05,0.1,0.15,0.2]
	step_list = [1,4,8,16]

	for current_percent_value in percent_list:

		slope_values = []
		
		for current_step_value in step_list:
			unrounded_M = population_density_value * (N*N)
			M = int(round((unrounded_M)))
			#print(M)
			population_density = M/(N*N)

			multiple_list = []
			#Start of simulations
			for i in range (0,number_of_simulations):		
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

			#Set the graph parameters to the first 50 values of each list
			iteration_number_small_graph = final_iteration_number[0:50]
			final_mean_distance_for_iteration_small_graph = average_of_all_means_for_all_simulations[0:50]

			slope2, intercept2, r_value2, p_value2, std_err2 = linregress(iteration_number_small_graph, final_mean_distance_for_iteration_small_graph)

			slope_values.append(slope2)
			print(slope_values)

		plt.plot(step_list,slope_values,label=str(current_percent_value))
		plt.scatter(step_list,slope_values)

	plt.grid(axis='y', alpha=0.4)
	plt.xlabel("Step size",fontsize = 14)
	plt.ylabel("Slope value (50 iterations)",fontsize = 14)
	plt.legend(loc='upper right',prop={'size': 12})
	plt.savefig("Step compared to calculated slope values (iterations = 50, population density = " + str(population_density_value) + ").png")
	plt.show()

if args.function == 'function_1':
	first_function(N,M,L,max_iter,number_of_simulations,population_density_value,step_value,percent_value)

if args.function == 'function_2':
	second_function(N,M,L,max_iter,min_population_density,max_population_density,number_of_densities,step_value,percent_value)

if args.function == 'function_3':
	third_function(N,M,L,max_iter,number_of_simulations,population_density_value,step_value,percent_value)

if args.function == 'function_4':
	fourth_function(N,M,L,max_iter,number_of_simulations,min_population_density,max_population_density,number_of_densities,step_value,percent_value)

if args.function == 'function_5':
	fifth_function(N,M,L,max_iter,number_of_simulations,min_population_density,max_population_density,number_of_densities)

if args.function == 'function_6':
	sixth_function(N,M,L,max_iter,number_of_simulations,population_density_value)

if args.function == 'function_7':
	seventh_function(N,M,L,max_iter,number_of_simulations,population_density_value)