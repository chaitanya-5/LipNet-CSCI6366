import tensorflow as tf
import os

class VideoDatasetManager:
    def __init__(self, video_dir, align_dir):
        self.video_dir = video_dir
        self.align_dir = align_dir

    def get_video_paths(self):
        return tf.data.Dataset.list_files(self.video_dir + '/*.mpg', shuffle = False)

    def get_align_paths(self):
        return tf.data.Dataset.list_files(self.align_dir + '/*.align', shuffle = False)

    def create_dataset(self):
        video_paths = self.get_video_paths()
        align_paths = self.get_align_paths()
        return tf.data.Dataset.zip((video_paths, align_paths))