from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

class ModelTrainer:
    def __init__(self, model, loss_fn, optimizer="adam"):
        """
        Initializes the ModelTrainer with the model, loss function, and optimizer.

        Args:
            model (tf.keras.Model): The model to train.
            loss_fn (callable): The custom loss function.
            optimizer (str or tf.keras.optimizers.Optimizer): The optimizer for training.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = tf.keras.optimizers.get(optimizer)
        self.callbacks = []

    def compile_model(self):
        """
        Compiles the model with the custom loss function and optimizer.
        """
        self.model.compile(optimizer=self.optimizer, loss=self.loss_fn)

    def add_callback(self, callback):
        """
        Adds a callback to the trainer.

        Args:
            callback (tf.keras.callbacks.Callback): The callback to add.
        """
        self.callbacks.append(callback)

    def train(self, train_data, val_data, epochs=10, batch_size=32):
        """
        Trains the model using the provided training and validation data.

        Args:
            train_data (tf.data.Dataset or np.ndarray): Training data.
            val_data (tf.data.Dataset or np.ndarray): Validation data.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.

        Returns:
            tf.keras.callbacks.History: Training history.
        """
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
        )
        return history
