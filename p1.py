import numpy as np
import matplotlib.pyplot as plt
import time

# Fitness funcion Schwefel
def schwefel(x):
    n = len(x)
    sum_part = np.sum(x * np.sin(np.sqrt(np.abs(x))))
    return 418.9829 * n - sum_part

# The dimensions for initial population
dimensions = 2

# The limits of the search space are defined.
t = np.array([-500, 500])  # For Schwefel function
f_range = np.tile(t, (dimensions, 1))

# The maximum number of iterations is established.
max_iter = 100

# The population size is defined, as well as the variable
# to hold the population elements.
num_agents = 10
agents = np.zeros((num_agents, dimensions))

# Initialization process for the initial population.
for i in range(dimensions):
    dim_f_range = f_range[i, 1] - f_range[i, 0]
    agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

best_position = np.zeros(dimensions)
best_fitness = np.nan
fitness = np.empty(num_agents)


initialPop = agents.copy()
initialFitness = fitness.copy()

# The iteration counter is defined.
iter = 0

aux_selector = np.arange(num_agents)

# The scaling factor of the algorithm is established.
m = 0.5

# The cross factor of the algorithm is established.
cross_p = 0.2

# Main loop process for the optimization process.
while iter < max_iter:
    for i in range(agents.shape[0]):
        # Three different individuals are chosen.
        indexes = aux_selector[aux_selector != i]
        indexes = np.random.choice(indexes, 3, replace=False)
        agents_selected = agents[indexes]
        # The crossover operation is performed to obtain the mutant vector.
        mut = agents_selected[0] + m * (agents_selected[1] - agents_selected[2])
        # The differential mutation of the DE algorithm is performed.
        prob_vector = np.random.rand(dimensions) <= cross_p
        mut = agents[i] * prob_vector + mut * np.logical_not(prob_vector)

        # It is verified that the generated vector is
        # within the search space defined by the upper and lower limits.
        for j in range(dimensions):
            upper_limit = f_range[j, 1]
            lower_limit = f_range[j, 0]

            if mut[j] < lower_limit:
                mut[j] = lower_limit
            elif mut[j] > upper_limit:
                mut[j] = upper_limit

        # The fitness value of the mutant vector is obtained.
        fitness_mut = schwefel(np.array([mut]))

        # The replacement mechanism is then performed.
        if fitness_mut < fitness[i]:
            agents[i] = mut
            fitness[i] = fitness_mut
            if fitness[i] < best_fitness:
                best_position = agents[i]
                best_fitness = fitness[i]

    print("Iteration: " + str(iter))
    iter += 1

# Now we find the best solution for Schwefel function
best_position_schwefel = np.zeros(dimensions)
best_fitness_schwefel = np.nan
for i in range(num_agents):
    fitness_schwefel = schwefel(np.array([agents[i]]))
    if i == 0:
        best_position_schwefel = agents[i]
        best_fitness_schwefel = fitness_schwefel
    elif fitness_schwefel < best_fitness_schwefel:
        best_position_schwefel = agents[i]
        best_fitness_schwefel = fitness_schwefel

print("Best solution (Schwefel): " + str(best_position_schwefel))
print("Best fitness (Schwefel): " + str(best_fitness_schwefel))

# Function Graphs
xGraph = np.linspace(-500, 500, 25)
yGraph = np.linspace(-500, 500, 25)
xv, yv = np.meshgrid(xGraph, yGraph)

fitnessSchwefelGraph = np.zeros((25, 25))

for i in range(25):
    for j in range(25):
        arr = [[xv[i, j], yv[i, j]]]
        fitnessSchwefelGraph[i, j] = schwefel(np.asarray(arr))

plt.ion()
fig = plt.figure(figsize=(12, 6))

# Plot Schwefel Function
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Schwefel Function', fontsize=16)
ax2.plot_surface(xv, yv, fitnessSchwefelGraph, cmap='viridis', alpha=0.6)
ax2.scatter(initialPop[:, 0], initialPop[:, 1], initialFitness[:], c='green', s=10, marker="x")
ax2.scatter(agents[:, 0], agents[:, 1], fitness[:], c='red', s=10, marker="x")

plt.tight_layout()
plt.show()
