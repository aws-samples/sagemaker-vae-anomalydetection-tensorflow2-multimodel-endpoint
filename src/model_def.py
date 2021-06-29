import tensorflow                as tf
from   tensorflow                import keras
import tensorflow.keras.backend  as K
from   tensorflow.keras          import layers

K.clear_session()

def encode(x):
    x = layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), padding='same', activation='relu')(x)
    x = layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), padding='same', activation='relu')(x)
    x = layers.Flatten()(x)
    
    return x

def sample_z(args):
    latent_dim, mu, log_sigma = args
    eps=keras.backend.random_normal(shape=(latent_dim,), mean=0., stddev=1.)
    return mu + keras.backend.exp(log_sigma/2) * eps

def sampler(latent_dim, x):
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    z = layers.Lambda(sample_z)([latent_dim, z_mean, z_log_var])
    return z_mean, z_log_var, z

def decode(x):
    x = layers.Dense(units=7*7*32, activation=tf.nn.relu)(x)
    x = layers.Reshape(target_shape=(7, 7, 32))(x)
    x = layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    
    return x
    
@tf.function
def compute_vae_loss(encoder_mean, encoder_lgvar, vae, x):
    """Compute the loss function of Variational Autoencoders
        
    PARAMERTERS
    -----------
    input: encoder_mean - model part to output means in the hidden layer 
           encoder_lgvar - model part to output vars in the hidden layer
           vae - Variational Autoencoders
           x - input data
    
    RETURNS
    ------
    Variational Autoencoders loss
            = Reconstruction Loss + KL loss for each data in minibatch
    """
    z_mean = encoder_mean(x)
    z_lgvar = encoder_lgvar(x)
    x_pred = vae(x)
    
    #E(log P(X/z))
    cross_ent = K.binary_crossentropy(x, x_pred)
    recon = tf.reduce_sum(cross_ent, axis=[1,2,3]) #consolidate at each instance

    #KL divergence 
    kl = 0.5 * K.sum(K.exp(z_lgvar) + K.square(z_mean) - 1. - z_lgvar, axis=1)
    
    return recon, recon + kl
    
@tf.function
def compute_apply_gradients(encoder_mean, encoder_lgvar, vae, x, optimizer):
    """Compute the gradient and apply gradient to optimizer
    
    PARAMETERS
    ----------
    encoder_mean: model part to output means in the hidden layer 
    encoder_lgvar: model part to output vars in the hidden layer
    vae: Variational Autoencoders
    x : tensors
    optimizer : tensorflow optimizer object
    
    RETURNS
    -------
    None, but weights are updated
    """
    with tf.GradientTape() as tape:
        recon_loss, loss = compute_vae_loss(encoder_mean, encoder_lgvar, vae, x)
    gradients = tape.gradient(loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))