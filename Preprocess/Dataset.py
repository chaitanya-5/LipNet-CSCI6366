import tensorflow as tf
import os

class VideoDatasetManager:
    """
    A utility class to manage datasets of video and alignment files for training a model.
    """
    def __init__(self, video_dir, align_dir):
        """
        Initialize the dataset manager with directories for videos and alignments.

        Args:
            video_dir (str): Path to the directory containing video files (.mpg).
            align_dir (str): Path to the directory containing alignment files (.align).
        """
        self.video_dir = video_dir
        self.align_dir = align_dir

    def get_video_paths(self):
        """
        Retrieve a TensorFlow dataset of video file paths.

        Returns:
            tf.data.Dataset: A dataset containing paths to all video files in the specified directory.
        """
        return tf.data.Dataset.list_files(self.video_dir + '/*.mpg', shuffle=False)

    def get_align_paths(self):
        """
        Retrieve a TensorFlow dataset of alignment file paths.

        Returns:
            tf.data.Dataset: A dataset containing paths to all alignment files in the specified directory.
        """
        return tf.data.Dataset.list_files(self.align_dir + '/*.align', shuffle=False)

    def create_dataset(self):
        """
        Create a combined dataset of video and alignment file paths.

        Returns:
            tf.data.Dataset: A dataset where each element is a tuple containing a video path and 
                             its corresponding alignment file path.
        """
        video_paths = self.get_video_paths()
        align_paths = self.get_align_paths()
        # Combine the video and alignment file paths into a single dataset
        return tf.data.Dataset.zip((video_paths, align_paths))
