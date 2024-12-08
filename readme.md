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
- `Sample_Model_Prediction.ipynb`: Contains a sample prediction generated using the model trained for 100 epochs.
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

## Testing the Trained Model with the Streamlit App

Once the model is trained, you can test it on video files using the interactive Streamlit app included in this repository. The app provides a user-friendly interface for uploading videos, running predictions, and visualizing results.

### Running the Streamlit App

The Streamlit app is located in the `StreamlitApp` directory. To run the app:

1. **Ensure Trained Weights Are Available**:  
   - The trained weights should be placed in the `StreamlitApp/Weights` directory.  
   - If you want to use a different path for the weights, update the `weights_path` in the `StreamlitApp/util.py` file:
     ```python
     def load_model(weights_path= os.path.join(os.getcwd(), 'StreamlitApp', 'Weights', 'checkpoints.weights.h5')):
     ```
You can download the trained weights for 100 epochs from the link below:

   - [Download Trained Weights](https://drive.google.com/file/d/1MGZXcl4gkZkDZJcOv5xvk1NGgap6jPHT/view?usp=share_link)

2. **Run the App**:  
   From the root directory of the repository, execute the following command:
   ```bash
   python3 -m streamlit run StreamlitApp/app.py
   ```

3. **Access the App**:  
   After running the command, Streamlit will display a link in the terminal, such as:
   ```
   Local URL: http://localhost:8501
   Network URL: http://<your-ip>:8501
   ```
   Open the link in your web browser to interact with the app.

### Using the App

1. **Upload a Video**:  
   - Use the file uploader in the app to upload a `.mpg` video file.  
   - The app will process the video and convert it to `.mp4` format for display.

2. **Generate Predictions**:  
   - After uploading the video, click the "Generate Predictions" button.  
   - The app will use the trained model to predict the text corresponding to the lip movements in the video.

3. **View Results**:  
   - The app will display the predictions alongside the uploaded video for easy comparison.