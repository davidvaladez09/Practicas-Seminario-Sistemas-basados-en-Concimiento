import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos
X = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
Y = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

# Normalización opcional
# X = (X - np.mean(X)) / np.std(X)
# Y = (Y - np.mean(Y)) / np.std(Y)

# Modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1], activation='linear')
])

# Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento
print("Entrenamiento")
historial = modelo.fit(X, Y, epochs=100, verbose=0)

# Visualización de la pérdida
plt.plot(historial.history["loss"])
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.show()

# Predicción
print("\nPrediciendo valores")
resultado = modelo.predict([-40])
print("El resultado es:", resultado)

# Pesos del modelo
print("\nPesos del modelo:")
for i, layer in enumerate(modelo.layers):
    print("Capa", i+1)
    print("Pesos:", layer.get_weights()[0])
    print("Bias:", layer.get_weights()[1])

print("Error del modelo")
loss_values = historial.history["loss"]
loss_values_float = [float(loss) for loss in loss_values]
for loss in loss_values_float:
    print(loss)