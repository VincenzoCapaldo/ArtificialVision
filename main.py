import torch
from tracking_BoTSort import start_track


def main():
    """
    Main function to initialize tracking process on a video file.
    This function checks if CUDA (GPU support) is available and selects the appropriate device (CPU or GPU).
    Then, it starts tracking on the video specified in the `video_path`.
    """
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Path to the input video file
    video_path = 'videos/Atrio.mp4'

    # Flag to control if the processed video should be displayed
    show = True

    # Start the tracking process
    start_track(device, video_path=video_path, show=show)


if __name__ == "__main__":
    main()