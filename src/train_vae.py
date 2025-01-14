import numpy as np
import tensorflow as tf
from tensorflow.keras import losses
import matplotlib.pyplot as plt

# On importe nos classes/fonctions
from encoder import Encoder
from decoder import Decoder
from vae import VAE
import os

############################
# 1) Paramètres
############################
latent_dim = 2
epochs = 10
batch_size = 128

############################
# 2) Chargement du dataset MNIST
############################
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Ajout d'une dimension de canal
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

############################
# 3) Instanciation du VAE (encodeur + décodeur)
############################
encoder = Encoder(latent_dim)
decoder = Decoder()
vae = VAE(encoder, decoder)

############################
# 4) Fonction de perte
############################
def vae_loss(x, reconstruction, mean, log_var):
    # Binary cross-entropy
    reconstruction_loss = tf.reduce_mean(
        losses.binary_crossentropy(x, reconstruction)
    )
    reconstruction_loss *= 28 * 28  # remise à l'échelle

    # KL divergence
    # kl = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
    kl_divergence = -0.5 * tf.reduce_sum(
        1 + log_var - tf.square(mean) - tf.exp(log_var), 
        axis=1
    )
    kl_divergence = tf.reduce_mean(kl_divergence)  # moyenne sur le batch
    
    return reconstruction_loss + kl_divergence

############################
# 5) Optimiseur
############################
optimizer = tf.keras.optimizers.Adam()

############################
# 6) Préparation des batches
############################
train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(buffer_size=60000)
    .batch(batch_size)
)

############################
# 7) Training loop
############################
@tf.function
def train_step(x):
    with tf.GradientTape() as tape:
        reconstruction, mean, log_var = vae(x)
        loss = vae_loss(x, reconstruction, mean, log_var)
    grads = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(grads, vae.trainable_variables))
    return loss

for epoch in range(epochs):
    for step, x_batch in enumerate(train_dataset):
        loss = train_step(x_batch)
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss.numpy():.4f}")

############################
# 8) Visualisation
############################
# On teste sur quelques images
x_test_batch = x_test[:10]
reconstructions, _, _ = vae(x_test_batch)

plt.figure(figsize=(10, 3))
for i in range(10):
    # Image originale
    plt.subplot(2, 10, i + 1)
    plt.imshow(x_test_batch[i].reshape(28, 28), cmap='gray')
    plt.axis("off")
    # Reconstruction
    plt.subplot(2, 10, 10 + i + 1)
    plt.imshow(reconstructions[i].numpy().reshape(28, 28), cmap='gray')
    plt.axis("off")

plt.tight_layout()
plt.show()

############################
# 9) Sauvegarde (optionnel)
############################
# Si on veut sauvegarder le modèle pour le réutiliser dans GAN
checkpoint_dir = "../checkpoints"  # Chemin relatif ou absolu selon votre arborescence
os.makedirs(checkpoint_dir, exist_ok=True)

vae.save_weights(os.path.join(checkpoint_dir, "vae_weights.h5"))

# Sauvegarde séparé de l'encodeur et décodeur 
encoder.save_weights(os.path.join(checkpoint_dir, "encoder_weights.h5"))
decoder.save_weights(os.path.join(checkpoint_dir, "decoder_weights.h5"))