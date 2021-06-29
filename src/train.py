import os

import argparse
import numpy      as np
import pandas     as pd
import tensorflow as tf
from   tensorflow                import keras
import tensorflow.keras.backend  as K
from   tensorflow.keras          import layers
import time

import model_def
import config

def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    
    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    
    return parser.parse_known_args()

def get_train_data(train, train_buff, batch_size, anomalyNumber, validNumber):
    
    train_x = np.load(train + '/train_x.npy')
    train_y = np.load(train + '/train_y.npy')
    
    train_x = train_x.astype('float32')
    train_x = train_x / 255
    train_y_one_hot = tf.keras.utils.to_categorical(train_y)

    train_validIdxs = np.where(np.isin(train_y, validNumber))[0]
    train_anomalyIdxs = np.where(train_y==anomalyNumber)[0]

    train_x_normal = train_x[train_validIdxs]
    train_y_normal = train_y[train_validIdxs]

    train_x_anomaly = train_x[train_anomalyIdxs]
    train_y_anomaly = train_y[train_anomalyIdxs]

    print('train normal x: ', np.shape(train_x_normal))
    print('train normal y: ', np.shape(train_y_normal))
    print('train anomaly x: ', np.shape(train_x_anomaly))
    print('train anomaly y: ', np.shape(train_y_anomaly))
    
    train_x_normal_dataset = (tf.data.Dataset.from_tensor_slices(train_x_normal)).shuffle(train_buff).batch(batch_size)
    train_x_anomaly_dataset = (tf.data.Dataset.from_tensor_slices(train_x_anomaly)).shuffle(train_buff).batch(batch_size)
    
    return train_x_normal, train_x_normal_dataset, train_y_normal, train_x_anomaly, train_x_anomaly_dataset, train_y_anomaly

def get_test_data(test_dir, test_buff, batch_size, anomalyNumber, validNumber):
    
    test_x = np.load(test_dir + '/test_x.npy')
    test_y = np.load(test_dir + '/test_y.npy')
    
    test_x = test_x.astype('float32')
    test_x = test_x / 255
    test_y_one_hot = tf.keras.utils.to_categorical(test_y)

    test_validIdxs = np.where(np.isin(test_y, validNumber))[0] 
    test_anomalyIdxs = np.where(test_y==anomalyNumber)[0]

    test_x_normal = test_x[test_validIdxs]
    test_y_normal = test_y[test_validIdxs]

    test_x_anomaly = test_x[test_anomalyIdxs]
    test_y_anomaly = test_y[test_anomalyIdxs]

    print('test normal x: ', np.shape(test_x_normal))
    print('test normal y: ', np.shape(test_y_normal))
    print('test anomaly x: ', np.shape(test_x_anomaly))
    print('test anomaly y: ', np.shape(test_y_anomaly))
    
    test_x_normal_dataset = (tf.data.Dataset.from_tensor_slices(test_x_normal)).shuffle(test_buff).batch(batch_size)
    test_x_anomaly_dataset = (tf.data.Dataset.from_tensor_slices(test_x_anomaly)).shuffle(test_buff).batch(batch_size)    
    return test_x_normal, test_x_normal_dataset, test_y_normal, test_x_anomaly, test_x_anomaly_dataset, test_y_anomaly

if __name__ == '__main__':
    anomalyNumber = 5
    validNumber = [1,4]
    latent_dim = 16

    args, _ = parse_args()

    batch_size    = args.batch_size
    epochs        = args.epochs
    l_r           = args.learning_rate

    train_x_normal, train_x_normal_dataset, train_y_normal, train_x_anomaly, train_x_anomaly_dataset, train_y_anomaly = get_train_data(args.train, config.TRAIN_BUFF, batch_size, anomalyNumber, validNumber)
    
    test_x_normal, test_x_normal_dataset, test_y_normal, test_x_anomaly, test_x_anomaly_dataset, test_y_anomaly = get_test_data(args.test, config.TEST_BUFF, batch_size, anomalyNumber, validNumber)
    
    #Define models
    # Define Encoder model
    encoder_input = keras.Input(shape=[28, 28, 1], name='encoder_input')
    encoder_output = model_def.encode(encoder_input)
    z_mean, z_log_var, z = model_def.sampler(latent_dim, encoder_output)

    encoder_mean = keras.Model(encoder_input, z_mean, name = 'EncoderMean')
    encoder_lgvar = keras.Model(encoder_input, z_log_var, name = 'EncoderLogVar')
    encoder_sampler = keras.Model(encoder_input, z, name = 'EncoderSampler')
    encoder = keras.Model(encoder_input, z, name = 'Encoder')
    
    # Define Decoder Model
    decoder_input = keras.Input(shape=(latent_dim,), name='z_sampling')
    decoder_output = model_def.decode(decoder_input)
    decoder = keras.Model(decoder_input, decoder_output, name='Decoder')
    
    # Define VAE Model
    outputs = decoder(z)
    vae = keras.Model(encoder_input, outputs, name='VAE')

    device = '/CPU:0'
    print(f'Using device: {device}')
    print('batch_size = {}, epochs = {}, learning rate = {}'.format(batch_size, epochs, l_r))

    test_loss_all = []
    train_loss_all = []

    with tf.device(device):
        optimizer = tf.keras.optimizers.Adam(learning_rate=l_r)
        for epoch in range(1, epochs+1):
            start_time = time.time()
            for train_x in train_x_normal_dataset:
                model_def.compute_apply_gradients(encoder_mean, encoder_lgvar, vae, train_x, optimizer)
            end_time = time.time()

            if epoch % 1 == 0:
                test_loss = keras.metrics.Mean()
                for test_x in test_x_normal_dataset:
                    test_recon_batch_loss, test_batch_loss = model_def.compute_vae_loss(encoder_mean, encoder_lgvar, vae, test_x)
                    test_loss(test_batch_loss)
                train_loss = keras.metrics.Mean()
                for train_x in train_x_normal_dataset:
                    train_recon_batch_loss, train_batch_loss = model_def.compute_vae_loss(encoder_mean, encoder_lgvar, vae, train_x)
                    train_loss(train_batch_loss)

                test_loss_all.append(test_loss.result().numpy())
                train_loss_all.append(train_loss.result().numpy())
    #             display.clear_output(wait=False)
                print(f'Epoch: {epoch}, Train set loss: {train_loss.result().numpy()}, Test set loss: {test_loss.result().numpy()}, time elapse for current epoch: {end_time - start_time}')
    
    # Save model and data
    tf.keras.models.save_model(vae, args.model_dir + '/vae/1')
    tf.keras.models.save_model(encoder_mean, args.model_dir + '/encoder_mean/2')
    tf.keras.models.save_model(encoder_lgvar, args.model_dir + '/encoder_lgvar/3')
    tf.keras.models.save_model(encoder_sampler, args.model_dir + '/encoder_sampler/4')
    tf.keras.models.save_model(decoder, args.model_dir + '/decoder/5')
    np.save(args.model_dir + '/train_loss.npy', train_loss)
    np.save(args.model_dir + '/test_loss.npy', test_loss)

        
    
      
    