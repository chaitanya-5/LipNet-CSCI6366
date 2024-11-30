import streamlit as st
from util import extract_frames, make_predictions, convert_video_to_mp4, save_uploaded_file

# Set the title for the Streamlit app
st.title("LipNet - end-to-end lipreading", anchor=None)

# Create two columns for layout
left_column, right_column = st.columns(2)

# Create a file uploader for selecting a video file (.mpg)
uploaded_file = st.file_uploader(label="Select Video File", type='mpg', accept_multiple_files=False)
frames = None  # Initialize variable for storing extracted frames

# Left column: Handles video file upload and frame extraction
with left_column:
    # If a video file is uploaded
    if uploaded_file is not None:
        # Save the uploaded .mpg file to a temporary location
        input_path = save_uploaded_file(uploaded_file, suffix=".mpg")
        
        # Extract frames from the uploaded video
        frames = extract_frames(input_path)

        try:
            # Convert the uploaded video to MP4 format for compatibility
            output_path = convert_video_to_mp4(input_path)

            # Display the converted video in the app
            st.video(output_path)        
        except Exception as e:
            # If there's an error during conversion, display an error message
            st.error(f"Error converting video: {e}")

# Right column: Handles generating and displaying predictions from the extracted frames
with right_column:
    # If frames are extracted successfully
    if frames is not None:
        # Display the "Generate Predictions" button only if frames are available
        if st.button("Generate Predictions"):
            try:
                # Generate predictions based on the extracted frames
                predictions = make_predictions(frames)

                # Display the generated predictions in the app
                st.write("Predictions:")
                st.write(predictions)
            except Exception as e:
                # If there's an error while generating predictions, display an error message
                st.error(f"Error generating predictions: {e}")
