from Dataset import VideoDatasetManager
from AlignFileProcessor import AlignFileProcessor
from FrameExtractor import FrameExtractor 

class Preprocessor:
    def __init__(self, video_dir, align_dir, frame_slice=None, vocab=None):
        """
        Initializes the Preprocessor with directories for videos and alignment files,
        and sets up frame extraction and alignment file processing.

        Args:
            video_dir (str): Path to the directory containing video files.
            align_dir (str): Path to the directory containing alignment files.
            frame_slice (tuple, optional): Slice for extracting specific frame sections. Defaults to None.
            vocab (list, optional): Vocabulary for alignment file processing. Defaults to the default vocabulary.
        """
        self.dataset_manager = VideoDatasetManager(video_dir, align_dir)
        self.frame_extractor = FrameExtractor(frame_slice=frame_slice)
        self.align_processor = AlignFileProcessor(vocab=vocab)

    def preprocess_video(self, video_path):
        """
        Preprocesses a single video file by extracting and normalizing frames.

        Args:
            video_path (str): Path to the video file.

        Returns:
            tf.Tensor: Processed frames.
        """
        return self.frame_extractor.extract_frames(video_path)

    def preprocess_align(self, align_path):
        """
        Preprocesses a single alignment file by encoding the alignment tokens.

        Args:
            align_path (str): Path to the alignment file.

        Returns:
            tf.Tensor: Encoded alignment tokens.
        """
        return self.align_processor.read_align(align_path)

    def preprocess_dataset(self, shuffle=False, buffer_size=500, cache=False, 
                           batch_size=None, padded_shapes=None, prefetch=True):
        """
        Creates a TensorFlow dataset from video and alignment files, applies preprocessing,
        and optionally applies transformations like shuffling, caching, and batching.

        Args:
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults to False.
            buffer_size (int, optional): Buffer size for shuffling. Defaults to 500.
            cache (bool, optional): Whether to cache the dataset. Defaults to False.
            batch_size (int, optional): Batch size for padded batching. Defaults to None.
            padded_shapes (tuple, optional): Shapes for padding the dataset. Defaults to None.
            prefetch (bool, optional): Whether to prefetch the dataset for better performance. Defaults to True.

        Returns:
            tf.data.Dataset: Preprocessed dataset with optional transformations.
        """
        dataset = self.dataset_manager.create_dataset()

        def preprocess_element(video_path, align_path):
            """
            Preprocesses a single dataset element by extracting frames and encoding alignment data.

            Args:
                video_path (tf.Tensor): Path to the video file.
                align_path (tf.Tensor): Path to the alignment file.

            Returns:
                tuple: Preprocessed video frames and alignment tokens.
            """
            # Use tf.py_function to call Python methods inside a TensorFlow map
            video_frames = tf.py_function(self.preprocess_video, [video_path], tf.float32)
            align_data = tf.py_function(self.preprocess_align, [align_path], tf.int64)
            return video_frames, align_data

        dataset = dataset.map(preprocess_element)

        # Apply optional transformations
        if shuffle:
            dataset = dataset.shuffle(buffer_size=buffer_size)
        if cache:
            dataset = dataset.cache()
        if batch_size and padded_shapes:
            dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes)
        if prefetch:
            dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset
