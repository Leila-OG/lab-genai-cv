import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt

from decoder import Decoder
import os

# -----------------------------
# 1) On crée un 'decoder' et on charge ses poids
# -----------------------------
checkpoint_dir = "../checkpoints"
decoder = Decoder()
# On suppose que vous avez sauvegardé précédemment :
# "vae.save_weights('vae_weights.h5')"
# Dans ce cas, le plus simple est de recharger dans le VAE complet
# ou juste la partie decoder si vous avez un checkpoint distinct.
# Par exemple:
# decoder.load_weights(os.path.join(checkpoint_dir, "decoder_weights.h5"))
decoder.load_weights(os.path.join(checkpoint_dir, "decoder_weights.h5"))  # si vous avez une sauvegarde spécifique

# -----------------------------
# 2) Définition du Discriminateur
# -----------------------------
class Discriminator(Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.flatten = layers.Flatten()
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(1, activation='sigmoid')

    def call(self, x):
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)

discriminator = Discriminator()

# -----------------------------
# 3) Construction du GAN
#    - On réutilise le 'decoder' comme generateur
# -----------------------------
latent_dim = 2  # Assurez-vous d'utiliser la même dim que dans le VAE

z_input = tf.keras.Input(shape=(latent_dim,))
img = decoder(z_input)           # Génération d'images
validity = discriminator(img)    # Probabilité de "réel"

gan_model = tf.keras.Model(z_input, validity, name="GAN")

# -----------------------------
# 4) Génération d'images (pas d'entraînement)
# -----------------------------
z_samples = np.random.normal(0, 1, size=(5, latent_dim)).astype("float32")
generated_images = decoder(z_samples)

plt.figure(figsize=(10, 2))
for i in range(5):
    plt.subplot(1, 5, i + 1)
    plt.imshow(generated_images[i].numpy().reshape(28, 28), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Ceci crée un "GAN" conceptuellement, mais sans entraînement,
# vous pouvez juste vérifier la sortie du 'decoder'.
