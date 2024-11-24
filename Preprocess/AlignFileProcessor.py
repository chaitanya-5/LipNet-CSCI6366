import tensorflow as tf
from tensorflow.keras.layers import StringLookup

class AlignFileProcessor:
    def __init__(self, vocab=None):
        """
        Initializes the AlignFileProcessor with a specified or default vocabulary.

        Args:
            vocab (list, optional): List of characters for the vocabulary. Defaults to
                [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "].
        """
        self.vocab = vocab if vocab else [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        self.char_to_num = StringLookup(vocabulary=self.vocab, oov_token="")

    def read_align(self, file_path):
        """
        Reads and encodes an alignment file.

        Args:
            file_path (tf.Tensor or str): Path to the alignment file.

        Returns:
            tf.Tensor: Encoded alignment tokens as a tensor.
        """
        tokens = []

        # Convert tf.Tensor to string if necessary
        if isinstance(file_path, tf.Tensor):
            file_path = file_path.numpy().decode("utf-8")

        try:
            with open(file_path, 'r') as file:
                line = file.readline()
                while line:
                    line_data = line.split(" ")
                    if line_data[2][:-1] != "sil":  # Exclude silence tokens
                        tokens.append(" ")
                        tokens.append(line_data[2][:-1])
                    line = file.readline()
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None

        # Convert characters to numerical values using StringLookup
        encoded_tokens = self.char_to_num(
            tf.reshape(tf.strings.unicode_split(tokens, 'UTF-8'), (-1))
        )
        return encoded_tokens[1:]  # Skip the initial space token

