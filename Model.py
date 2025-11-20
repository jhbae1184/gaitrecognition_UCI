from tensorflow.keras import layers, models
import numpy as np
import tensorflow as tf


def build_model_deep_1D(num_classes, input_shape):
    import keras.backend as K

    K.clear_session()

    model = models.Sequential()
    #model.add(layers.BatchNormalization())
    model.add(layers.Input(shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 3, activation='relu'))  # , input_shape=(8, 11, 1) # padding='same',
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 3, activation='relu'))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv1D(64, 1, activation='relu'))
    # model.add(layers.LocallyConnected2D(64, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(64, 1, activation='relu'))
    # model.add(layers.LocallyConnected2D(64, (1, 1), activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model



def build_model_1D(num_classes, input_shape):
    import keras.backend as K

    K.clear_session()

    model = models.Sequential()
    #model.add(layers.BatchNormalization())
    model.add(layers.Input(shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(32, 3, activation='relu'))  # , input_shape=(8, 11, 1) # padding='same',
    model.add(layers.BatchNormalization())
    model.add(layers.Conv1D(32, 3, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.25))

    model.add(layers.Dense(num_classes, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    return model



def build_emg_encoder(d_out):
    inputs = tf.keras.Input(shape=(5, 1))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(64, 1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(64, 1, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Dropout(0.25)(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    x = tf.keras.layers.Dense(d_out)(x)
    x = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=-1))(x)
    return tf.keras.Model(inputs, x)
