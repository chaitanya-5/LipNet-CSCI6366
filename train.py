import argparse
import os
from Preprocess.Preprocessor import Preprocessor
from Model.ModelBuilder import ModelBuilder
from Model.Trainer import ModelTrainer
from Model.Loss import CustomLoss
from Callbacks.Scheduler import scheduler
from Callbacks.GenerateSamples import GenerateSamples
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler


def parse_args():
    """
    Parse command-line arguments for various options.
    
    Returns:
        Namespace: A namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train the LipNet model on video data.")
    
    # Command line arguments for directories
    parser.add_argument('--root-dir', type=str, required=True, help="Root directory containing 'Videos' and 'Aligns' folders.")
    parser.add_argument('--videos-path', type=str, help="Path to the 'Videos' directory.", default=None)
    parser.add_argument('--align-path', type=str, help="Path to the 'Aligns' directory.", default=None)
    
    # Command line arguments for preprocessing
    parser.add_argument('--frame-slice', type=str, help="Frame slice for cropping (format: start:end).", default="190:236,80:220,:")
    
    # Options for dataset preprocessing
    parser.add_argument('--shuffle', type=bool, default=True, help="Whether to shuffle the dataset.")
    parser.add_argument('--cache', type=bool, default=True, help="Whether to cache the dataset.")
    parser.add_argument('--batch-size', type=int, default=2, help="Batch size for the dataset.")
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs for training.")
    parser.add_argument('--learning_rate', type=float, default=0.0001, help="Learning rate for training.")
    parser.add_argument('--model-save-dir', type=str, default=os.path.join(os.getcwd(), "Trained_Model"), help="Directory to save the model weights.")
    
    return parser.parse_args()

def preprocess_data(videos_path, align_path, frame_slice, shuffle, cache, batch_size):
    """
    Preprocess the video dataset using the Preprocessor class.

    Args:
        videos_path (str): Path to the video files.
        align_path (str): Path to the alignment files.
        frame_slice (tuple): Tuple defining the slice to extract from frames.
        shuffle (bool): Whether to shuffle the dataset.
        cache (bool): Whether to cache the dataset.
        batch_size (int): The batch size for dataset.

    Returns:
        tf.data.Dataset: Preprocessed dataset.
    """
    data_preprocessor = Preprocessor(videos_path, align_path, frame_slice)
    dataset = data_preprocessor.preprocess_dataset(shuffle,buffer_size=256, cache = cache, batch_size = batch_size, padded_shapes=([75, 46, 140, 1], [40]))
    return dataset

def build_model():
    """
    Build the LipNet model using the ModelBuilder class.

    Returns:
        tf.keras.Model: Compiled LipNet model.
    """
    builder = ModelBuilder()
    model = builder.build()
    loss_fn = CustomLoss.ctc_loss
    return model, loss_fn

def configure_callbacks(train_set, vocab, model_save_dir):
    """
    Configure the callbacks for training.

    Args:
        train_set (tf.data.Dataset): The training dataset.
        vocab (list): The vocabulary for decoding predictions.
        model_save_dir (str): The directory where model weights are saved.

    Returns:
        list: A list of configured callbacks.
    """
    sample_generator_callback = GenerateSamples(train_set, vocab)
    schedule_callback = LearningRateScheduler(scheduler)
    checkpoint_callback = ModelCheckpoint(
        filepath=os.path.join(model_save_dir, "checkpoints.weights.h5"),
        monitor="loss",
        save_weights_only=True,
    )
    return [sample_generator_callback, schedule_callback, checkpoint_callback]

def main():
    # Parse command-line arguments
    args = parse_args()

    # Set paths (use command-line arguments or defaults)
    root_dir = args.root_dir
    videos_path = args.videos_path or os.path.join(root_dir, 'Videos')
    align_path = args.align_path or os.path.join(root_dir, 'Aligns')
    learning_rate = args.learning_rate
    
    # Parse the frame slice argument
    frame_slice = tuple(
    slice(int(x.split(":")[0]), int(x.split(":")[1])) if ':' in x and x.split(":")[0] != "" and x.split(":")[1] != "" else slice(None) 
    for x in args.frame_slice.split(",")
    )
    
    # Preprocess the dataset
    dataset = preprocess_data(videos_path, align_path, frame_slice, args.shuffle, args.cache, args.batch_size)

    # Split the dataset into training and validation sets
    train_set = dataset.take(450)
    val_set = dataset.skip(450)

    # Build and compile the model
    model, loss_fn = build_model()

    # Initialize the trainer and compile the model
    trainer = ModelTrainer(model, loss_fn, learning_rate = learning_rate)
    trainer.compile_model()

    # Define vocabulary for predictions
    vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]

    # Configure the callbacks
    callbacks = configure_callbacks(train_set, vocab, args.model_save_dir)
    for callback in callbacks:
        trainer.add_callback(callback)

    # Train the model
    trainer.train(train_set, val_set, epochs=args.epochs)

if __name__ == '__main__':
    main()
