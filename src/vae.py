import tensorflow as tf
from tensorflow.keras import Model

# Reparameterization trick
def sample_z(mean, log_var):
    """
    Echantillonne z = mean + sigma * epsilon, 
    où epsilon ~ N(0, 1) et sigma = exp(0.5 * log_var).
    """
    epsilon = tf.random.normal(shape=tf.shape(mean))
    return mean + tf.exp(0.5 * log_var) * epsilon

class VAE(Model):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def call(self, x):
        mean, log_var = self.encoder(x)
        z = sample_z(mean, log_var)
        reconstruction = self.decoder(z)
        return reconstruction, mean, log_var
