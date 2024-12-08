from Preprocess.FrameExtractor import FrameExtractor
from Model.ModelBuilder import ModelBuilder
import tensorflow as tf
from tensorflow.keras.layers import StringLookup
import tempfile
import ffmpeg
import os

# Function to extract frames from a video file
def extract_frames(file_path, frame_rate=1):
    """
    Extract frames from the video using a specific frame slice.

    Args:
        file_path (str): Path to the input video file.
        frame_rate (int): The rate at which frames should be sampled from the video.

    Returns:
        frames (tf.Tensor): A tensor of the extracted frames.
    """
    # Create an instance of FrameExtractor with a defined slice
    extractor = FrameExtractor((slice(190, 236), slice(80, 220), slice(None)))
    # Extract frames from the provided video file
    frames = extractor.extract_frames(file_path, frame_rate)
    return frames

# Function to load the pre-trained model from disk
def load_model(weights_path= os.path.join(os.getcwd(), 'StreamlitApp', 'Weights','checkpoints.weights.h5')):
    """
    Load the pre-trained model weights from the specified path.

    Args:
        weights_path (str): Path to the model weights file.

    Returns:
        model (tf.keras.Model): The loaded model ready for predictions.
    """
    # Create a new model using ModelBuilder
    builder = ModelBuilder()
    model = builder.build()  # Build the model
    model.load_weights(weights_path)  # Load the pre-trained weights into the model
    return model

# Function to save the uploaded file to a temporary location
def save_uploaded_file(uploaded_file, suffix):
    """
    Save the uploaded file to a temporary file.

    Args:
        uploaded_file (UploadedFile): The uploaded file object.
        suffix (str): The suffix to append to the temporary file (e.g., '.mpg').

    Returns:
        str: The path to the saved temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())  # Write the uploaded file's content to the temp file
        return tmp_file.name  # Return the path of the saved file
    
# Function to convert the video to MP4 format
def convert_video_to_mp4(input_path):
    """
    Convert the input video to MP4 format using FFmpeg.

    Args:
        input_path (str): Path to the input video file.

    Returns:
        str: Path to the converted MP4 video file.

    Raises:
        ValueError: If there's an error during video conversion.
    """
    # Create a temporary file to save the converted MP4 video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
        output_path = tmp_output.name  # The path where the MP4 video will be saved
        try:
            # Use FFmpeg to convert the video to MP4 format with specified codecs
            ffmpeg.input(input_path).output(output_path, vcodec="libx264", acodec="aac").run(overwrite_output=True)
        except Exception as e:
            # Raise an error if conversion fails
            raise ValueError(f"Error converting video: {e}")
    return output_path  # Return the path to the converted video

# Function to make predictions from the extracted frames
def make_predictions(frames):
    """
    Make predictions on the extracted frames using the trained LipNet model.

    Args:
        frames (tf.Tensor): The extracted frames to make predictions on.

    Returns:
        str: The predicted output (decoded text).
        
    Raises:
        ValueError: If an error occurs during the prediction process.
    """
    try:
        # Load the pre-trained model
        model = load_model()
        # Expand the frames tensor to match model input dimensions
        frames = tf.expand_dims(frames, axis=0)
        # Make predictions using the model
        pred = model.predict(frames)
        
        # Define the vocabulary for decoding predictions
        vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
        numToChar = StringLookup(vocabulary=vocab, oov_token="", invert=True)

        # Decode the predicted output using CTC decode
        decoded = tf.keras.backend.ctc_decode(pred, [75] * pred.shape[0], greedy=False)[0][0].numpy()
        
        # Join the decoded characters into a string
        for x in range(len(pred)):
            prediction = "".join([numToChar(char).numpy().decode('UTF-8') for char in decoded[x] if char != -1])

        return prediction  # Return the final decoded prediction as a string
    except Exception as e:
        # Raise an error if there's a failure in making predictions
        raise ValueError(f"Error making predictions: {e}")
    
# Function to read the align file.
def read_align_file(file_path):
    """
    Reads an align file and extracts the third word from each line, 
    excluding lines where the third word is 'sil'.

    Args:
        file_path (str): Path to the align file.

    Returns:
        str: Processed text with silence parts removed.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        
        # Extract the third word from each line if it's not 'sil'
        cleaned_text = " ".join(
            line.split()[2] for line in lines if len(line.split()) >= 3 and line.split()[2] != "sil"
        )
        return cleaned_text.strip()
    except FileNotFoundError:
        return "Error: File not found."
    except Exception as e:
        return f"Error: {e}"