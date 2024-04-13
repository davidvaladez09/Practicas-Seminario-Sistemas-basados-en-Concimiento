import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Función de probabilidad (likelihood)
def likelihood(x, y, coefficients):
    x_with_bias = np.hstack((np.ones((x.shape[0], 1)), x))
    logits = np.dot(x_with_bias, coefficients)
    probabilities = 1 / (1 + np.exp(-logits))  # Función sigmoide
    likelihoods = np.where(y == 1, probabilities, 1 - probabilities)
    return np.prod(likelihoods)

# Número de dimensiones y límites del espacio de búsqueda
dimensions = 3
f_range = np.array([[-5, 5], [-5, 5], [-5, 5]])

# Número máximo de iteraciones
max_iter = 100

# Tamaño de la población y matriz para almacenar los individuos
num_agents = 10
agents = np.zeros((num_agents, dimensions))

# Inicialización de la población inicial
for i in range(dimensions):
    dim_f_range = f_range[i, 1] - f_range[i, 0]
    agents[:, i] = np.random.rand(num_agents) * dim_f_range + f_range[i, 0]

# Mejor solución y likelihood
best_position = np.zeros(dimensions)
best_likelihood = -np.inf
likelihoods = np.empty(num_agents)

# Generación de datos aleatorios para la compuerta OR de 2 bits
np.random.seed(0)
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([0, 1, 1, 1])  # Valores de salida de la compuerta OR

# Bucle principal para el proceso de optimización
for iter in range(max_iter):
    for i in range(num_agents):
        # Realizar mutación y obtener el likelihood
        mut = agents[i] + np.random.uniform(-0.5, 0.5, size=dimensions)
        likelihood_mut = likelihood(x_train, y_train, mut)

        # Actualizar si se encuentra un mejor likelihood
        if likelihood_mut > likelihoods[i]:
            agents[i] = mut
            likelihoods[i] = likelihood_mut
            if likelihood_mut > best_likelihood:
                best_position = agents[i]
                best_likelihood = likelihood_mut

# Imprimir la mejor solución y likelihood
print("Mejor solución (Pesos):", best_position)
print("Mejor likelihood:", best_likelihood)

# Gráfico en 3D de la función de likelihood
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Generar valores para los ejes x e y
x = np.linspace(f_range[0, 0], f_range[0, 1], 50)
y = np.linspace(f_range[1, 0], f_range[1, 1], 50)
x, y = np.meshgrid(x, y)
z = np.zeros_like(x)

# Calcular la función de likelihood para cada punto en el espacio
for i in range(50):
    for j in range(50):
        z[i, j] = likelihood(x_train, y_train, [1, x[i, j], y[i, j]])

# Graficar la superficie
ax.plot_surface(x, y, z, cmap='viridis')

# Etiquetas de los ejes
ax.set_xlabel('Peso 1')
ax.set_ylabel('Peso 2')
ax.set_zlabel('Likelihood')

# Mostrar el gráfico
plt.show()
