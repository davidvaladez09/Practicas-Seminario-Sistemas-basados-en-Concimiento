import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd

# Cargar el dataset desde el archivo Excel
df = pd.read_excel("D:/Documentos/8vo/Sem Sistemas Inteligentes/Practicas/Datos_Practica5.xlsx")

# Dividir el dataset en características (X) y etiquetas (Y)
X = df.iloc[:, :-1].values  # Todas las filas, todas las columnas excepto la última
Y = df.iloc[:, -1].values   # Todas las filas, sólo la última columna

# Normalización opcional
# X = (X - np.mean(X)) / np.std(X)
# Y = (Y - np.mean(Y)) / np.std(Y)

# Dividir los datos en conjunto de entrenamiento y conjunto de prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Definir el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=[X_train.shape[1]]),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='linear')
])

# Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='mean_squared_error'
)

# Entrenamiento
print("Entrenamiento")
historial = modelo.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=1)

# Visualización de la pérdida
plt.plot(historial.history["loss"], label="Training Loss")
plt.plot(historial.history["val_loss"], label="Validation Loss")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.show()

# Evaluación del modelo en el conjunto de prueba
loss = modelo.evaluate(X_test, Y_test)
print("Pérdida en el conjunto de prueba:", loss)

# Predicción
print("\nPrediciendo valores")
resultado = modelo.predict(X_test[:5])  # Predice los primeros 5 ejemplos del conjunto de prueba
print("Valores reales:", Y_test[:5])
print("Valores predichos:", resultado.flatten())

# Pesos del modelo
print("\nPesos del modelo:")
for i, layer in enumerate(modelo.layers):
    print("Capa", i+1)
    print("Pesos:", layer.get_weights()[0])
    print("Bias:", layer.get_weights()[1])

# Guardar el modelo
modelo.save("modelo_perceptron_multicapa.h5")

print("Error del modelo")
loss_values = historial.history["loss"]
for loss in loss_values:
    print(loss)
