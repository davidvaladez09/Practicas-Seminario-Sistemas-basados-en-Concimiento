import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Function to calculate Mean Squared Error (MSE) for Linear Regression
def mse(x, y, coefficients):
    # Add a column of ones to the feature matrix to represent the bias term (intercept)
    x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))
    predicted = np.dot(x_with_bias, coefficients)
    mse = np.mean((predicted - y) ** 2)
    return mse

# Normalize the data
def normalize_data(x):
    x_mean = np.mean(x, axis=0)
    x_std = np.std(x, axis=0)
    x_normalized = (x - x_mean) / x_std
    return x_normalized, x_mean, x_std

archivo = 'D://Documentos//8vo//Sem Sistemas Inteligentes//Datos.csv'

# Load your dataset
dataset = pd.read_csv(archivo)  # Reemplaza 'tu_archivo.csv' con el nombre de tu archivo CSV

# Extract features (X) and labels (y)
x_train = dataset.iloc[:, :-1].values
y_train = dataset.iloc[:, -1].values.reshape(-1, 1)  # Reshape to ensure it's a column vector

# Normalize the features
x_train_normalized, x_mean, x_std = normalize_data(x_train)

# The dimensions for initial population
dimensions = x_train_normalized.shape[1] + 1  # Plus 1 for bias term

# The limits of the search space are defined.
t = np.array([-5, 5])  # For Linear Regression weights
f_range = np.tile(t, (dimensions, 1))

# The maximum number of iterations is established.
max_iter = 2000  # Increased number of iterations

# The population size is defined, as well as the variable
# to hold the population elements.
num_agents = 20  # Increased number of agents
agents = np.zeros((num_agents, dimensions))

# Initialization process for the initial population.
for i in range(dimensions):
    dim_f_range = f_range[i, 1] - f_range[i, 0]
    agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

best_position = np.zeros(dimensions)
best_fitness = np.inf
fitness = np.empty(num_agents)

initialPop = agents.copy()
initialFitness = fitness.copy()

# The iteration counter is defined.
iter = 0

aux_selector = np.arange(num_agents)

# The scaling factor of the algorithm is established.
m = 0.8  # Adjusted scaling factor

# The cross factor of the algorithm is established.
cross_p = 0.9  # Adjusted crossover factor

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
        fitness_mut = mse(x_train_normalized, y_train, mut)

        # The replacement mechanism is then performed.
        if fitness_mut < fitness[i]:
            agents[i] = mut
            fitness[i] = fitness_mut
            if fitness[i] < best_fitness:
                best_position = agents[i]
                best_fitness = fitness[i]

    print("Iteration: " + str(iter) + ", Best Fitness: " + str(best_fitness))
    iter += 1

# Denormalize the best solution
best_position_denormalized = np.zeros(dimensions)
best_position_denormalized[1:] = best_position[1:] * x_std / y_train.std()
best_position_denormalized[0] = best_position[0] * y_train.std() + np.dot(x_mean, best_position[1:] * x_std)

print("Best solution (Linear Regression Coefficients): " + str(best_position_denormalized))
print("Best fitness (MSE): " + str(best_fitness))

# Function Graphs
xGraph = np.linspace(-5, 5, 25)
yGraph = np.linspace(-5, 5, 25)
xv, yv = np.meshgrid(xGraph, yGraph)

fitnessMSEGraph = np.zeros((25, 25))

for i in range(25):
    for j in range(25):
        arr = np.array([xv[i, j], yv[i, j]])
        fitnessMSEGraph[i, j] = mse(x_train_normalized, y_train, arr)

plt.ion()
fig = plt.figure(figsize=(12, 6))

# Plot MSE Function
ax2 = fig.add_subplot(122, projection='3d')
ax2.set_xlabel('Weight 1')
ax2.set_ylabel('Weight 2')
ax2.set_title('Mean Squared Error (MSE) Function', fontsize=16)
ax2.plot_surface(xv, yv, fitnessMSEGraph, cmap='viridis', alpha=0.6)
ax2.scatter(initialPop[:, 0], initialPop[:, 1], initialFitness[:], c='green', s=10, marker="x")
ax2.scatter(agents[:, 0], agents[:, 1], fitness[:], c='red', s=10, marker="x")

plt.tight_layout()
plt.show()
