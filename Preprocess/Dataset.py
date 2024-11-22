import tensorflow as tf
import os

class VideoDatasetManager:
    def __init__(self, video_dir, align_dir):
        self.video_dir = video_dir
        self.align_dir = align_dir

    def get_video_paths(self):
        return [os.path.join(self.video_dir, f) for f in os.listdir(self.video_dir) if f.endswith('.mp4')]

    def get_align_paths(self):
        return [os.path.join(self.align_dir, f) for f in os.listdir(self.align_dir) if f.endswith('.align')]

    def create_dataset(self):
        video_paths = self.get_video_paths()
        align_paths = self.get_align_paths()
        return tf.data.Dataset.from_tensor_slices((video_paths, align_paths))
