"""
Deep Learning Architectures:
1. Deep Artificial Neural Network (ANN) - Section IV.E.1
2. 1D Convolutional Neural Network (CNN) - Section IV.E.2
3. Long Short-Term Memory (LSTM) - Section IV.E.3
4. Proposed Hybrid CNN-LSTM Network - Section IV.E.4
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout, Input


def build_ann(input_dim: int = 11) -> Sequential:
    """
    Builds the Deep ANN architecture:
    - Input: input_dim features
    - Hidden Layer 1: input_dim neurons, ReLU
    - Hidden Layer 2: 5 neurons, ReLU
    - Output: 2 neurons, Softmax
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(input_dim, activation='relu', name='ann_hidden_1'),
        Dense(5, activation='relu', name='ann_hidden_2'),
        Dense(2, activation='softmax', name='ann_output')
    ], name='ANN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_cnn(input_shape=(11, 1)) -> Sequential:
    """
    Builds the 1D-CNN architecture:
    - Input: (11, 1)
    - Conv1D (64 filters, kernel 3, same padding, ReLU)
    - MaxPooling1D (pool 2)
    - Dropout (0.2)
    - Flatten -> Dense (32, ReLU) -> Softmax (2)
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='cnn_conv1'),
        MaxPooling1D(pool_size=2, name='cnn_pool1'),
        Dropout(0.2, name='cnn_drop1'),
        Flatten(name='cnn_flatten'),
        Dense(32, activation='relu', name='cnn_dense'),
        Dense(2, activation='softmax', name='cnn_output')
    ], name='CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_lstm(input_shape=(11, 1)) -> Sequential:
    """
    Builds the LSTM architecture:
    - Input: (11, 1)
    - LSTM (64 units)
    - Dropout (0.2)
    - Dense (32, ReLU) -> Softmax (2)
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=False, name='lstm_layer'),
        Dropout(0.2, name='lstm_drop'),
        Dense(32, activation='relu', name='lstm_dense'),
        Dense(2, activation='softmax', name='lstm_output')
    ], name='LSTM')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_hybrid_cnn_lstm(input_shape=(11, 1)) -> Sequential:
    """
    Builds the Proposed Flagship Hybrid CNN-LSTM Architecture:
    - Input: (11, 1)
    - Conv1D Block 1 (32 filters, kernel 3, padding 'same', ReLU)
    - Conv1D Block 2 (64 filters, kernel 3, padding 'same', ReLU)
    - MaxPooling1D (pool size 2)
    - Dropout (0.2)
    - LSTM Layer (64 units) for sequential multivariate feature synthesis
    - Dropout (0.2)
    - Dense Layer (32 units, ReLU)
    - Softmax Output (2 units: HD Absence vs Presence)
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv1D(filters=32, kernel_size=3, padding='same', activation='relu', name='hybrid_conv1'),
        Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='hybrid_conv2'),
        MaxPooling1D(pool_size=2, name='hybrid_pool'),
        Dropout(0.2, name='hybrid_drop1'),
        LSTM(64, return_sequences=False, name='hybrid_lstm'),
        Dropout(0.2, name='hybrid_drop2'),
        Dense(32, activation='relu', name='hybrid_dense'),
        Dense(2, activation='softmax', name='hybrid_output')
    ], name='Hybrid_CNN_LSTM')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model
