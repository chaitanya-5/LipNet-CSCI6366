import tensorflow as tf
from tensorflow.keras.layers import StringLookup


class GenerateSamples(tf.keras.callbacks.Callback):
    def __init__(self, dataset, vocab):
        """
        Callback to generate predictions and compare with ground truth after each epoch.

        Args:
            dataset (tf.data.Dataset): The dataset for generating samples.
            vocab (list): The vocabulary list for decoding characters.
        """
        self.dataset = dataset.as_numpy_iterator()
        self.vocab = vocab

    def on_epoch_end(self, epoch, logs=None):
        data = self.dataset.next()
        pred = self.model.predict(data[0])

        numToChar = StringLookup(vocabulary=self.vocab, oov_token="", invert=True)
        
        decoded = tf.keras.backend.ctc_decode(pred, [75] * pred.shape[0], greedy=False)[0][0].numpy()
        for x in range(len(pred)):
            original = "".join([a.decode('UTF-8') for a in numToChar(data[1][x]).numpy()])
            prediction = "".join([numToChar(char).numpy().decode('UTF-8') for char in decoded[x] if char != -1])


            print(f"Original: {original}")
            print(f"Prediction: {prediction}")
            print("~" * 100)