# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 11:23:13 2017
Updated on Nov 14 2017
@author: Zain
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Reshape, Permute
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.recurrent import GRU, LSTM


def CRNN2D(X_shape, nb_classes):
    '''
    Model used for evaluation in paper. Inspired by K. Choi model in:
    https://github.com/keunwoochoi/music-auto_tagging-keras/blob/master/music_tagger_crnn.py
    '''

    nb_layers = 4  # number of convolutional layers
    nb_filters = [64, 128, 128, 128]  # filter sizes
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (4, 2), (4, 2), (4, 2),
                 (4, 2)]  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model and normalize along frequency axis
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer specifies shape
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(axis=channel_axis))
    model.add(MaxPooling2D(pool_size=pool_size[0], strides=pool_size[0]))
    model.add(Dropout(0.1))

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))

        # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


###############################################################################
'''
Models below this point were only pre-tested and were not presented in the paper
'''


###############################################################################

def CRNN2DLarger(X_shape, nb_classes):
    '''
    Making the previous model larger and deeper
    '''
    nb_layers = 5  # number of convolutional layers
    nb_filters = [64, 128, 256, 512, 512]
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (2, 2), (2, 2), (4, 1),
                 (4, 1)]  # # size of pooling area
    # pool_size = [(4,2), (4,2), (4,1), (2,1)] this worked well

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting
    model.add(MaxPooling2D(pool_size=pool_size[0],
                           strides=pool_size[0]))  # Max pooling
    model.add(Dropout(0.1))  # 0.2

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting
        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))  # 0.2

    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def CRNN2DVGG(X_shape, nb_classes):
    '''
    Based on VGG-16 Architecture
    '''
    nb_layers = 5  # number of convolutional layers
    nb_filters = [64, 128, 256, 512, 512]
    kernel_size = (3, 3)  # convolution kernel size
    activation = 'elu'  # activation function to use after each layer
    pool_size = [(2, 2), (2, 2), (2, 2), (4, 1),
                 (4, 1)]  # # size of pooling area
    # pool_size = [(4,2), (4,2), (4,1), (2,1)] this worked well

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()
    model.add(BatchNormalization(axis=frequency_axis, input_shape=input_shape))

    # First convolution layer
    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting

    model.add(Conv2D(nb_filters[0], kernel_size=kernel_size, padding='same',
                     data_format="channels_last",
                     input_shape=input_shape))
    model.add(Activation(activation))
    model.add(BatchNormalization(
        axis=channel_axis))  # Improves overfitting/underfitting

    model.add(MaxPooling2D(pool_size=pool_size[0],
                           strides=pool_size[0]))  # Max pooling
    model.add(Dropout(0.1))  # 0.2

    # Add more convolutional layers
    for layer in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting

        model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                         padding='same'))
        model.add(Activation(activation))
        model.add(BatchNormalization(
            axis=channel_axis))  # Improves overfitting/underfitting

        if nb_filters[layer + 1] != 128:
            model.add(Conv2D(nb_filters[layer + 1], kernel_size=kernel_size,
                             padding='same'))
            model.add(Activation(activation))
            model.add(BatchNormalization(
                axis=channel_axis))  # Improves overfitting/underfitting

        model.add(MaxPooling2D(pool_size=pool_size[layer + 1],
                               strides=pool_size[layer + 1]))  # Max pooling
        model.add(Dropout(0.1))  # 0.2

    # Reshaping input for recurrent layer
    # (frequency, time, channels) --> (time, frequency, channel)
    model.add(Permute((time_axis, frequency_axis, channel_axis)))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # recurrent layer
    model.add(GRU(32, return_sequences=True))
    model.add(GRU(32, return_sequences=False))
    model.add(Dropout(0.2))

    # Output layer
    model.add(Dense(nb_classes))
    model.add(Activation("softmax"))
    return model


def CRNN1D(X_shape, nb_classes):
    '''
    Based on 1D convolution
    '''

    nb_layers = 3  # number of convolutional layers
    kernel_size = 5  # convolution kernel size
    activation = 'relu'  # activation function to use after each layer
    pool_size = 2  # size of pooling area

    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()

    model.add(Permute((time_axis, frequency_axis, channel_axis),
                      input_shape=input_shape))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    # First convolution layer
    model.add(Conv1D(64, kernel_size))
    model.add(Activation(activation))
    model.add(
        MaxPooling1D(pool_size=pool_size, strides=pool_size))  # Max pooling
    # model.add(Dropout(0.2))

    # Add more convolutional layers
    for _ in range(nb_layers - 1):
        # Convolutional layer
        model.add(Conv1D(128, kernel_size))
        model.add(Activation(activation))
        model.add(MaxPooling1D(pool_size=pool_size,
                               strides=pool_size))  # Max pooling

    model.add(GRU(64, return_sequences=True))
    model.add(GRU(64, return_sequences=False))

    model.add(Dense(nb_classes))  # note sure about this
    model.add(Activation('softmax'))

    # Output layer
    return model


def RNN(X_shape, nb_classes):
    '''
    Implementing only the RNN
    '''
    # shape of input data (frequency, time, channels)
    input_shape = (X_shape[1], X_shape[2], X_shape[3])
    frequency_axis = 1
    time_axis = 2
    channel_axis = 3

    # Create sequential model
    model = Sequential()

    model.add(Permute((time_axis, frequency_axis, channel_axis),
                      input_shape=input_shape))
    resize_shape = model.output_shape[2] * model.output_shape[3]
    model.add(Reshape((model.output_shape[1], resize_shape)))

    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=False))

    model.add(Dense(nb_classes))  # note sure about this
    model.add(Activation('softmax'))

    # Output layer
    return model
