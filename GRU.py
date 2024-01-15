import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Reshape, ConvLSTM2D, BatchNormalization, Conv2DTranspose, Conv2D, Lambda
import random
import numpy as np


def create_conv_lstm_model(N, D,seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    input_shape =(D, N+1, N,1)  # Assuming a single channel


    model = Sequential()
    model.add(Reshape((D, N+1, N, 1), input_shape=input_shape))  # Reshape layer
    # First ConvLSTM layer
    model.add(ConvLSTM2D(filters=32, kernel_size=(3, 3), padding='same', return_sequences=True))
    model.add(BatchNormalization())

    # Second ConvLSTM layer
    model.add(ConvLSTM2D(filters=64, kernel_size=(3, 3), padding='same', return_sequences=False))
    model.add(BatchNormalization())

    # Conv2D Transpose layers to reshape the output to N x N
    model.add(Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same'))


    # Final Conv2D layer to get the desired output shape of N x N
    model.add(Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same'))

    # Custom Lambda layer to keep only the upper triangular part
    #model.add(Lambda(lambda x: x * tf.linalg.band_part(tf.ones_like(x), 0, -1)))
    #tf.linalg.band_part(input, 0, -1)
    #model.add(UpperTriangularLayer())
    #round to 1 or 0
    model.add(Lambda(lambda x: tf.round(x)))
    return model
