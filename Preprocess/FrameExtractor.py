import cv2
import tensorflow as tf

class FrameExtractor:
    def __init__(self, frame_slice=None):
        """
        Initializes the FrameExtractor.

        Args:
            frame_slice (tuple, optional): A slice object or tuple specifying the section of the frame to extract.
                                            Defaults to None (entire frame is used).
        """
        self.frame_slice = frame_slice

    def extract_frames(self, file_path, frame_rate=1):
        """
        Extracts frames from a video file at a specified frame rate.

        Args:
            file_path (tf.Tensor or str): Path to the video file.
            frame_rate (int, optional): Frequency of frames to extract. Defaults to 1 (extract every frame).

        Returns:
            tf.Tensor: Normalized and sliced frames as a tensor.

        Raises:
            IOError: If the video file cannot be read.
            ValueError: If no frames are extracted from the video.
        """
        frames = []  # List to store extracted frames

        # Convert tf.Tensor to string if necessary
        if isinstance(file_path, tf.Tensor):
          file_path = file_path.numpy().decode("utf-8")

        # Open the video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise IOError("Error reading the file.")  # Raise an error if the file cannot be opened

        success = True  # Variable to track if frames are being read successfully
        count = 0       # Frame counter

        while success:
            success, frame = cap.read()  # Read the next frame
            if count % frame_rate == 0 and frame is not None:  # Process frames at the specified frame rate
                # Convert frame to grayscale
                frame = tf.image.rgb_to_grayscale(frame)

                # Apply frame slicing if specified
                if self.frame_slice:
                    frames.append(frame[self.frame_slice])
                else:
                    frames.append(frame)

            count += 1  # Increment the frame counter

        # Release the video capture object
        cap.release()

        # Raise an error if no frames were extracted
        if not frames:
            raise ValueError("No frames were extracted from the video.")

        # Normalize the frames
        mean = tf.math.reduce_mean(frames)  # Compute mean of the frames
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))  # Compute standard deviation of the frames
        frames = tf.cast((frames - mean), tf.float32) / std  # Normalize the frames using Z-score normalization

        return frames  # Return the normalized frames
