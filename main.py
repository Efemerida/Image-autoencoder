import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils

data, label = utils.load_dataset()
print(data.shape)
data = np.reshape(data, (60000, 28, 28, 1))
new_data = utils.load_image_jpg()
print(new_data)
print("load")
#new_data = [tf.convert_to_tensor(x, dtype=tf.float32) for x in new_data]
print(new_data.shape)

# Генерация случайных данных для обучения
# data = np.random.rand(100, 28, 28, 1)


# Создание сверточного автокодировщика
input_layer = tf.keras.layers.Input(shape=(1024, 1024, 3))
encoded = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_layer)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
encoded = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)
encoded = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(encoded)


decoded = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
decoded = tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
decoded = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(decoded)
decoded = tf.keras.layers.UpSampling2D((2, 2))(decoded)
decoded = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(decoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Обучение сверточного автокодировщика
autoencoder.fit(new_data, new_data, epochs=5, batch_size=32)



# Сжатие изображения
encoded_imgs = autoencoder.predict(new_data)


# Визуализация сжатого изображения
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(new_data[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(encoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()