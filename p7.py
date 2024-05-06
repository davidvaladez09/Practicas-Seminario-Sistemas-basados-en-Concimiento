import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Reshape

import numpy as np
import idx2numpy

import cv2
import matplotlib.pyplot as plt
from keras.layers import UpSampling2D

# Función para agregar ruido sal y pimienta a una imagen
def add_salt_pepper_noise(image, amount=0.05):
    row, col = image.shape
    s_vs_p = 0.5  # Equilibrio entre sal y pimienta
    noisy_image = np.copy(image)

    # Sal
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords] = 1

    # Pimienta
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords] = 0
    return noisy_image

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

# Agregar ruido sal y pimienta a las imágenes de prueba
noisy_test_images = np.array([add_salt_pepper_noise(image.squeeze()) for image in test_images])

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
plt.plot(history.history['accuracy'], color='red', label='train')
plt.plot(history.history['val_accuracy'], color='blue', label='validation')
plt.title('Model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot de la pérdida
plt.plot(history.history['loss'], color='red', label='train')
plt.plot(history.history['val_loss'], color='blue', label='validation')
plt.title('Model loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Construir el modelo CNN para el autoencoder
autoencoder = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    MaxPooling2D((2, 2), padding='same'),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'),
    UpSampling2D((2, 2)),
    Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')
])

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Entrenar el autoencoder
autoencoder.fit(noisy_test_images, test_images, epochs=10)

# Probar el autoencoder en las imágenes con ruido
denoised_images = autoencoder.predict(noisy_test_images)

# Mostrar algunas imágenes originales, ruidosas y denoised para comparar
n = 5
plt.figure(figsize=(10, 4))
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_images[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title('Original')

    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(noisy_test_images[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title('Noisy')

    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(denoised_images[i].squeeze())
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    if i == n // 2:
        ax.set_title('Denoised')
plt.show()