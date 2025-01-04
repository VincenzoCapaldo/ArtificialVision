import cv2
from PIL import Image
from ultralytics import YOLO
from OutputWriter import OutputWriter
from classification.nets import PARMultiTaskNet
from lines_utils import get_lines_info, check_crossed_lines
import gui_utils as gui
import torch
import torchvision.transforms as T
from collections import defaultdict


def start_track(device, model_path="models/yolo11m.pt", video_path="videos/video.mp4", show=False, tracker="confs/botsort.yaml"):
    """
    Main function to perform people tracking in a video using a pre-trained YOLO model.

    Parameters:
    device: Specifies the device to run the model on (e.g. 'cuda' for GPU or 'cpu').
    model_path: Path to the YOLO model file.
    video_path: Path to the video to analyze.
    show: Flag to display the results in real time.
    real_time: Flag to synchronize the processing in real time.
    tracker: Path to the tracker configuration file.
    """

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Caricamento del modello per la classificazione
    classification = PARMultiTaskNet(backbone='resnet50', pretrained=False, attention=True).to(device)
    checkpoint_path = './models/classification_model.pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classification.load_state_dict(checkpoint['model_state'])
    classification.eval()

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    probability_sum_gender = defaultdict(float)
    denominator_gender = defaultdict(float)
    probability_sum_bag = defaultdict(float)
    denominator_bag = defaultdict(float)
    probability_sum_hat = defaultdict(float)
    denominator_hat = defaultdict(float)
    pedestrian_attribute = []

    # Load lines info
    lines_info = get_lines_info()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS) + 1) # calcola gli fps del video

    # CONSTANTS
    PROCESSING_FRAME_RATE = 8  # Frame to process in 1 second
    FRAME_TO_SKIP = int(fps / PROCESSING_FRAME_RATE)
    INFERENCE_RATE = FRAME_TO_SKIP * 2
    frame_count = 0

    # Store the track history (for each ID, its trajectory)
    track_history = defaultdict(lambda: [])

    lista_attraversamenti = {}  # Stores the lines traversed by each ID
    lista_persone = []

    # Loop through the video frames
    while cap.isOpened():

        # Read a frame from the video
        success, frame = cap.read()

        # skippa i frames
        for _ in range(FRAME_TO_SKIP - 1):
            frame_count += FRAME_TO_SKIP
            cap.grab()  # Grab frames without decoding them

        if success:
            # Run YOLO tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker=tracker, classes=[0])
            if results[0] is not None and results[0].boxes is not None and results[0].boxes.id is not None:
                # Get the boxes and track IDs
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                # Visualize the results on the frame
                annotated_frame = frame.copy()

                # Plot the tracks
                for box, track_id in zip(boxes, track_ids):

                    # Add a new person (se non è già presente)
                    if track_id not in lista_persone:
                        lista_persone.append(track_id)

                    x, y, w, h = box
                    top_left_corner = (int(x - w / 2), int(y - h / 2))
                    bottom_right_corner = (int(x + w / 2), int(y + h / 2))
                    bottom_left_corner = (int(x - w / 2), int(y + h / 2))

                    # general information of the scene to display
                    text = [f"Total People: {len(track_ids)}"]
                    for line in lines_info:
                        text.append(f"Passages for line {line['line_id']}: {line['crossing_counting']}")

                    # Draw red bounding box
                    gui.add_bounding_box(annotated_frame, top_left_corner, bottom_right_corner)
                    # Draw the tracked people ID
                    gui.add_track_id(annotated_frame, track_id, top_left_corner)
                    # Draw general information about the scene
                    gui.add_info_scene(annotated_frame, text)
                    # Draw lines
                    annotated_frame = gui.draw_lines_on_frame(annotated_frame, lines_info)
                    # share bounding box

                    # FINE DISEGNI, INIZIO DISEGNI TRACCE
                    # Gestione delle traiettorie e disegno delle linee di tracciamento
                    track = track_history[track_id]
                    trajectory_point = 2  # Maintain up to 3 tracking points
                    track.append((float(x), float(y + h / 2)))  # x, y center point ''' (lower center of the bounding box) '''
                    if len(track) > trajectory_point:  # retain 5 tracks
                        track.pop(0)

                    # checking crossed lines
                    crossed_line_ids = check_crossed_lines(track, lines_info)
                    if len(crossed_line_ids) != 0:
                        if not (track_id in lista_attraversamenti):
                            lista_attraversamenti[track_id] = []
                        lista_attraversamenti[track_id].extend(crossed_line_ids)

                    # Inferenza
                    if frame_count % INFERENCE_RATE == 0:
                        screen = gui.get_bounding_box_image(frame, top_left_corner, bottom_right_corner)
                        image = transforms(Image.fromarray(screen).convert('RGB')).to(device)
                        image = image.unsqueeze(0)  # Aggiunge una dimensione batch
                        outputs = classification(image)

                        probability = {}
                        for task in ["gender", "bag", "hat"]:
                            probability[task] = torch.sigmoid(outputs[task]).item()

                        # Calcolo della media aritmetica
                        probability_sum_gender[track_id] += probability["gender"]
                        denominator_gender[track_id] += 1
                        probability_sum_bag[track_id] += probability["bag"]
                        denominator_bag[track_id] += 1
                        probability_sum_hat[track_id] += probability["hat"]
                        denominator_hat[track_id] += 1

                        gender = 1 if (probability_sum_gender[track_id] / denominator_gender[track_id]) > 0.5 else 0
                        bag = 1 if (probability_sum_bag[track_id] / denominator_bag[track_id]) > 0.5 else 0
                        hat = 1 if (probability_sum_hat[track_id] / denominator_hat[track_id]) > 0.5 else 0

                        # Costruisci la lista di attributi da mostrare
                        if gender:
                            pedestrian_attribute = ["Gender: F"]
                        else:
                            pedestrian_attribute = ["Gender: M"]
                        if not bag and not hat:
                            pedestrian_attribute.append("No Bag No Hat")
                        if bag and not hat:
                            pedestrian_attribute.append("Bag")
                        if not bag and hat:
                            pedestrian_attribute.append("Hat")
                        if bag and hat:
                            pedestrian_attribute.append("Bag Hat")

                        pedestrian_attribute.append(f"[{', '.join(map(str, lista_attraversamenti.get(track_id, [])))}]")

                    # Disegna gli attributi sotto il bounding box
                    gui.add_info_scene(annotated_frame, pedestrian_attribute, bottom_left_corner, 0.5, 2)

                # Display the annotated frame
                if show:
                    # Crea una finestra ridimensionabile
                    cv2.namedWindow("YOLO Tracking", cv2.WINDOW_NORMAL)
                    # Imposta la finestra a schermo intero
                    cv2.setWindowProperty("YOLO Tracking", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    cv2.imshow("YOLO Tracking", annotated_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    # Scrittura dei risultati su file
    # Create a result-writer object to write on a json file the results
    output_writer = OutputWriter()
    for track_id in lista_persone:
        gender = 1 if (probability_sum_gender[track_id] / denominator_gender[track_id]) > 0.5 else 0
        bag = 1 if (probability_sum_bag[track_id] / denominator_bag[track_id]) > 0.5 else 0
        hat = 1 if (probability_sum_hat[track_id] / denominator_hat[track_id]) > 0.5 else 0
        trajectory = lista_attraversamenti[track_id] if track_id in lista_attraversamenti else []
        output_writer.add_person(track_id, gender, bag, hat, trajectory)

    output_writer.write_output()
