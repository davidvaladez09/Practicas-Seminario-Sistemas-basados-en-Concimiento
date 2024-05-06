import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
import numpy as np
import idx2numpy

# Cargar los archivos IDX1-UBYTE
train_images = idx2numpy.convert_from_file('/content/train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('/content/train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('/content/t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('/content/t10k-labels.idx1-ubyte')

# Normalizar los valores de píxeles al rango [0, 1]
train_images = train_images.astype('float32') / 255.0
test_images = test_images.astype('float32') / 255.0

# Reshape para que tengan la forma adecuada para un modelo convolucional
train_images = np.expand_dims(train_images, axis=-1)  # Agregar una dimensión para los canales
test_images = np.expand_dims(test_images, axis=-1)  # Agregar una dimensión para los canales

# Construir el modelo CNN
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 clases en MNIST

model.summary()

# Compilar el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
history = model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluar el modelo en el conjunto de test
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test Accuracy:", test_accuracy)

# Plot de la precisión y la pérdida durante el entrenamiento
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.legend()
plt.show()

import cv2
import matplotlib.pyplot as plt

# Cargar las imágenes de gato y perro
test_img1 = cv2.imread('/content/n1.png')
test_img2 = cv2.imread('/content/n6.png')

# Convertir las imágenes a escala de grises
test_img1_gray = cv2.cvtColor(test_img1, cv2.COLOR_BGR2GRAY)
test_img2_gray = cv2.cvtColor(test_img2, cv2.COLOR_BGR2GRAY)

# Mostrar las imágenes
plt.figure(1)
plt.imshow(test_img1)
plt.figure(2)
plt.imshow(test_img2)

# Redimensionar las imágenes a 28x28 píxeles
test_img1_resized = cv2.resize(test_img1_gray, (28, 28))
test_img2_resized = cv2.resize(test_img2_gray, (28, 28))

# Normalizar los valores de píxeles al rango [0, 1]
test_img1_normalized = test_img1_resized.astype('float32') / 255.0
test_img2_normalized = test_img2_resized.astype('float32') / 255.0

# Agregar una dimensión para los canales (escala de grises)
test_input1 = test_img1_normalized.reshape((1, 28, 28, 1))
test_input2 = test_img2_normalized.reshape((1, 28, 28, 1))

# Realizar la predicción con el modelo
prediction1 = model.predict(test_input1)
prediction2 = model.predict(test_input2)

# Mostrar los resultados de la predicción
print("Predicción de la imagen 1:", prediction1)
print("Predicción de la imagen 2:", prediction2)