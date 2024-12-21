import torch
from tracking_BoTSort import start_track


def main():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    video_path = 'videos/Atrio.mp4'  # Path to the input video file
    show = True  # A boolean flag to display the processed video with tracked objects
    #model_path = "models/yolov8m.pt"
    start_track(device, video_path=video_path, show=show)

# Controllo per eseguire il codice solo quando il file è eseguito direttamente
if __name__ == "__main__":
    main()
