from collections import defaultdict
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from OutputWriter import OutputWriter
from classification_andrea.nets import PARMultiTaskNet
from lines_utils import get_lines_info, check_crossed_line
import time
import gui_utils as gui
import torch
import torchvision.transforms as T

def start_track(device, model_path="models/yolo11m.pt", video_path="videos/Atrio.mp4", show=False, real_time=True, tracker="confs/botsort.yaml"):

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

    # Load lines info
    lines_info = get_lines_info()

    # Create an output-writer object to write on a json file the results
    output_writer = OutputWriter()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)+1)

    # Store the track history (for each ID, its trajectory)
    track_history = defaultdict(lambda: [])
    first_frame = True
    frame_count = 0

    # Caricamento del modello per la classificazione
    classification = PARMultiTaskNet(backbone='resnet50', pretrained=False).to(device)
    checkpoint_path = './models/resnet50 (modello GOAT senza crop).pth'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    classification.load_state_dict(checkpoint['model_state'])
    classification.eval()

    transforms = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    lista_attraversamenti = {}  # Stores the lines traversed by each ID
    lista_persone = []
    pedestrian_attributes = {}

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        # check the time for read one frame
        start_read_time = time.time()
        success, frame = cap.read()
        start_time = end_read_time = time.time()
        if success:
            # Run YOLO11 tracking on the frame, persisting tracks between frames
            results = model.track(frame, persist=True, tracker=tracker, classes=[0])

            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Visualize the results on the frame ( Bounding box rosso e id rosso in alto a sx)
            annotated_frame = frame.copy()

            # Plot the tracks
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                top_left_corner = (x1, y1)
                bottom_right_corner = (x2, y2)
                # starting point for display pedestrian attribute
                bottom_left_corner = (x1, y2)

                # general information of the scene to display
                text = []
                text.append(f"Total People: {len(track_ids)}")
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
                trajectory_point = 30  # Maintain up to 30 tracking points
                track.append((float(x), float(y+h/2)))  # x, y center point ''' (lower center of the bounding box) '''
                if len(track) > trajectory_point:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                #checking crossed lines
                crossed_line_id = check_crossed_line(track, lines_info)
                if(len(crossed_line_id)!=0):
                    if not(track_id in lista_attraversamenti):
                        lista_attraversamenti[track_id] = []
                    lista_attraversamenti[track_id].extend(crossed_line_id)

                if (frame_count % fps == 0): # ogni secondo
                    # Inferenza
                    screen = gui.screen(frame, top_left_corner, bottom_right_corner)
                    image = transforms(Image.fromarray(screen).convert('RGB')).to(device)
                    image = image.unsqueeze(0)  # Aggiunge una dimensione batch
                    outputs = classification(image)

                    prediction = {}
                    probability = {}
                    for task in ["gender", "bag", "hat"]:
                        probability[task] = torch.sigmoid(outputs[task])
                        prediction[task] = (probability[task] > 0.5).int().cpu().numpy()

                    # Aggiorna il dizionario degli attributi
                    pedestrian_attributes[(track_id, frame_count)] = {
                        "prob_gender": probability["gender"],
                        "prob_bag": probability["bag"],
                        "prob_hat": probability["hat"],
                        "gender": prediction["gender"],
                        "bag": prediction["bag"],
                        "hat": prediction["hat"]
                    }

                # Se track_id non è presente, viene creato un nuovo dizionario con i seguenti valori predefiniti
                attributes = pedestrian_attributes.get(track_id, {
                    "gender": False,
                    "bag": False,
                    "hat": False
                })

                # Costruisci la lista di attributi da mostrare
                if attributes["gender"]:
                    pedestrian_attribute = [f"Gender: F"]
                else:
                    pedestrian_attribute = [f"Gender: M"]
                if not attributes["bag"] and not attributes["hat"]:
                    pedestrian_attribute.append("No Bag No Hat")
                if attributes["bag"] and not attributes["hat"]:
                    pedestrian_attribute.append("Bag")
                if not attributes["bag"] and attributes["hat"]:
                    pedestrian_attribute.append("Hat")
                if attributes["bag"] and attributes["hat"]:
                    pedestrian_attribute.append("Bag Hat")

                pedestrian_attribute.append(f"[{', '.join(map(str, lista_attraversamenti.get(track_id, [])))}]")

                # Disegna gli attributi sotto il bounding box
                gui.add_info_scene(annotated_frame, pedestrian_attribute, bottom_left_corner, 0.5, 2)

                # Add a new person (se non è già presente)
                if track_id not in lista_persone:
                    lista_persone.append(track_id)

            frame_count += 1
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
        # Real-time synchronization management
        end_time = time.time()
        if real_time:
            ms_per_frame = 1 / fps
            elapsed_time = end_time - start_time
            discard_frames = int(elapsed_time/ms_per_frame)+1
            d2 = int(discard_frames + ((discard_frames * (end_read_time-start_read_time))/ms_per_frame))+1
            while d2 > 0 and not first_frame:
                cap.read()
                d2 -= 1
            first_frame = False

    # Calcolo della media ponderataper ogni persona
    probability_sum_gender = []
    denominator_gender = []
    probability_sum_bag = []
    denominator_bag = []
    probability_sum_hat = []
    denominator_hat = []
    for (id, frame), attributes in pedestrian_attributes.items():
        probability_sum_gender[id] += round(attributes["prob_gender"].item(), 2) * attributes["gender"]
        denominator_gender[id] += round(attributes["prob_gender"].item(), 2)
        probability_sum_bag[id] += round(attributes["prob_bag"].item(), 2) * attributes["bag"]
        denominator_bag[id] += round(attributes["prob_bag"].item(), 2)
        probability_sum_hat[id] += round(attributes["prob_hat"].item(), 2) * attributes["hat"]
        denominator_hat[id] += round(attributes["prob_hat"].item(), 2)

    # Scrittura dei risultati su file
    for id in lista_persone:
        output_writer.add_person(id)
        gender = 1 if (probability_sum_gender[id] / denominator_gender[id]) > 0.5 else 0
        output_writer.set_gender(id, gender)
        bag = 1 if (probability_sum_bag[id] / denominator_bag[id]) > 0.5 else 0
        output_writer.set_bag(id, bag)
        hat = 1 if (probability_sum_hat[id] / denominator_hat[id]) > 0.5 else 0
        output_writer.set_hat(id, hat)

    # Add trajectory for all the people
    for id in lista_attraversamenti:
        trajectory = lista_attraversamenti[id]
        output_writer.set_trajectory(id, trajectory)

    # Print people info on "./output/output.json" file
    output_writer.write_output()

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
