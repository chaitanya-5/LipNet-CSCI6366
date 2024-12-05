# LipNet Implementation - Final Project for CSCI 6366 Neural Networks and Deep Learning

This repository contains our implementation of the LipNet model, designed for end-to-end lipreading using deep learning. Developed as the final project for CSCI 6366: Neural Networks and Deep Learning, it demonstrates video-based speech recognition.  

### **Team Members**

- **Mayank Tiwari** - [mayankt28](https://github.com/mayankt28) | [Mayank.tiwari@gwu.edu](mailto:Mayank.tiwari@gwu.edu)  
- **Omkar Mane** - [Omkarbm03](https://github.com/Omkarbm03) | [omkarbalasaheb.mane@gwu.edu](mailto:omkarbalasaheb.mane@gwu.edu)  
- **Chaitanya Movva** - [chaitanya-5](https://github.com/chaitanya-5) | [movva.chaitanya@gwmail.gwu.edu](mailto:movva.chaitanya@gwmail.gwu.edu)  

### **References**

- Original Paper: [LipNet](https://arxiv.org/abs/1611.01599)  
- Dataset: [GRID Corpus](https://spandh.dcs.shef.ac.uk//gridcorpus/)  

## Requirements

- Python 3.x
- TensorFlow 2.x
- ffmpeg
- Other dependencies listed in `requirements.txt`

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/mayankt28/LipNet-CSCI6366.git
    cd LipNet-CSCI6366
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure you have ffmpeg installed on your system, as it is used to convert video formats. For installation instructions, visit the [ffmpeg website](https://ffmpeg.org/download.html).

## File Structure

- `train.py`: Script for training the LipNet model.
- `Preprocess/`: Contains the preprocessing pipeline for the video data.
- `Model/`: Contains the model architecture and training logic.
- `Callbacks/`: Contains callbacks for training, such as learning rate schedulers and sample generation.
- `StreamlitApp/`: Contains a streamlit based web app to generate predictions using the trained model.

## Running the Training Script

To train the LipNet model, you can run the `train.py` script from the command line. The script takes several arguments that allow you to configure the training process, such as paths to the video and align files, frame slice configuration, batch size, and the number of epochs.

### Command-Line Arguments

The following arguments can be provided when running the `train.py` script:

| Argument            | Description                                                                 | Default                          |
|---------------------|-----------------------------------------------------------------------------|----------------------------------|
| `--root-dir`         | Root directory containing 'Videos' and 'Aligns' folders.                    | (Required)                       |
| `--videos-path`      | Path to the 'Videos' directory.                                             | `<root-dir>/Videos`              |
| `--align-path`       | Path to the 'Aligns' directory.                                             | `<root-dir>/Aligns`              |
| `--frame-slice`      | Frame slice for cropping (format: start:end).                               | `"190:236,80:220,:"`             |
| `--shuffle`          | Whether to shuffle the dataset.                                             | `True`                           |
| `--cache`            | Whether to cache the dataset.                                               | `True`                           |
| `--batch-size`       | Batch size for training.                                                    | `2`                              |
| `--epochs`           | Number of epochs to train the model.                                        | `100`                            |
| `--model-save-dir`   | Directory to save the model weights.                                        | `./Model/`                       |

### Example Usage

To run the training script, use the following command:

```bash
python train.py --root-dir /path/to/root --videos-path /path/to/videos --align-path /path/to/aligns --frame-slice "190:236,80:220,:" --shuffle True --cache True --batch-size 2 --epochs 100
```

- Replace `/path/to/root`, `/path/to/videos`, and `/path/to/aligns` with the appropriate paths to your data.
- Adjust the `frame-slice`, `batch-size`, and `epochs` values based on your training preferences.

### Explanation of Arguments

- `--root-dir`: The root directory that contains the `Videos` and `Aligns` folders. This is a required argument.
- `--videos-path`: The path to the folder containing video files. Defaults to `<root-dir>/Videos` if not provided.
- `--align-path`: The path to the folder containing alignment files. Defaults to `<root-dir>/Aligns` if not provided.
- `--frame-slice`: Specifies the slice for cropping the video frames. The format is `start:end` for rows, columns, and channels (e.g., `"190:236,80:220,:"`). Defaults to `"190:236,80:220,:"`.
- `--shuffle`: Whether to shuffle the dataset during training. Defaults to `True`.
- `--cache`: Whether to cache the dataset in memory. Defaults to `True`.
- `--batch-size`: Batch size used during training. Defaults to `2`.
- `--epochs`: The number of epochs for training. Defaults to `100`.
- `--model-save-dir`: Directory where model weights will be saved. Defaults to `./Model/`.