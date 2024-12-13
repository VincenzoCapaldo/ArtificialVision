import torch
from ultralytics import YOLO  # Ensure you have the Ultralytics YOLO library installed

def my_track(video_path, tracker, show=False):
    # Dynamically determine the best device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    # Load YOLO model with weights onto the selected device
    model = YOLO('models/yolov8m.pt')
    model.to(device)  # Move the model to the selected device

    # Confirm the device of the model
    print(f"The model is loaded on: {next(model.parameters()).device}")

    # Run tracking with the specified tracker configuration file
    model.track(
        source=video_path,  # The path to the input video file
        show=show,          # Boolean flag to display the processed video with tracked objects
        tracker=tracker,     # Path to the tracker configuration file (e.g., `botsort.yaml`)
        classes=[0]  # Filter to track only class 0 (people)
    )

video_path = 'videos/Atrio.mp4' # Path to the input video file (`video_fish.mp4`)
tracker='./confs/botsort.yaml' # Path to the tracker configuration file (`botsort.yaml`)
show=True # A boolean flag to display the processed video with tracked objects

my_track(video_path, tracker, show)