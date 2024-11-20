from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv3D,
    MaxPool3D,
    TimeDistributed,
    Flatten,
    Bidirectional,
    LSTM,
    Dense,
    Dropout,
)
import tensorflow as tf


class ModelBuilder:
    def __init__(self, input_shape=(75, 46, 140, 1), num_classes=41):
        """
        Initializes the model builder with input shape and number of classes.

        Args:
            input_shape (tuple): The shape of the input data.
            num_classes (int): The number of output classes for the Dense layer.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build(self):
        """
        Builds and returns the Sequential model.

        Returns:
            tf.keras.Model: The constructed model.
        """
        model = Sequential()
        model.add(tf.keras.Input(shape=self.input_shape))

        # 3D Convolutions
        model.add(Conv3D(128, 3, activation="relu", padding="same"))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(256, 3, activation="relu", padding="same"))
        model.add(MaxPool3D((1, 2, 2)))

        model.add(Conv3D(75, 3, activation="relu", padding="same"))
        model.add(MaxPool3D((1, 2, 2)))

        # TimeDistributed + Bidirectional LSTMs
        model.add(TimeDistributed(Flatten()))
        model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
        model.add(Dropout(0.5))

        model.add(Bidirectional(LSTM(128, kernel_initializer="Orthogonal", return_sequences=True)))
        model.add(Dropout(0.5))

        # Output layer
        model.add(Dense(self.num_classes, kernel_initializer="he_normal", activation="softmax"))
        return model
