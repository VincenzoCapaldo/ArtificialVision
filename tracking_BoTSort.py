from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO
from lines_utils import draw_lines_on_frame, get_lines_info


def start_track(device, model_path="models/yolo11m.pt", video_path="videos/Atrio.mp4", show=False, tracker="confs/botsort.yaml"):

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Load lines info
    lines_info = get_lines_info()

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Store the track history
    track_history = defaultdict(lambda: [])

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

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

                # QUESTA PARTE DEI DISEGNI METTERLA IN UNA FUNZIONE
                # Draw red bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 4)  # Red color

                # Define the position for the text (id delle persone)
                text = f"{track_id}"
                font_scale = 0.6
                font_thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_x, text_y = x1 + 5, y1 + 20  # Slightly offset inside the BB
                text_width, text_height = text_size[0], text_size[1]
                background_start = (text_x - 2, text_y - text_height - 2)  # Add padding
                background_end = (text_x + text_width + 2, text_y + 2)

                # Draw white rectangle as background for text (id delle persone)
                cv2.rectangle(annotated_frame, background_start, background_end, (255, 255, 255), -1)  # White color

                # Draw track ID in red over the background
                cv2.putText(
                    annotated_frame,
                    text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (0, 0, 255),  # Red color
                    font_thickness
                )

                # Definizione del numero di persone rilevate
                num_people = len(track_ids)  # Supponendo che track_ids rappresenti gli ID delle persone rilevate
                text = f"Total People: {num_people}"

                # Parametri per il testo
                font_scale = 0.8
                font_thickness = 2
                font = cv2.FONT_HERSHEY_SIMPLEX

                # Calcolo della dimensione del testo
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_width, text_height = text_size[0], text_size[1]

                # Definizione della posizione del rettangolo
                padding = 10  # Spazio extra intorno al testo
                top_left_corner = (padding, padding)
                bottom_right_corner = (padding + text_width + 2 * padding, padding + text_height + 2 * padding)

                # Disegna il rettangolo bianco
                cv2.rectangle(annotated_frame, top_left_corner, bottom_right_corner, (255, 255, 255),
                              -1)  # Bianco riempito

                # Scrivi il testo sopra il rettangolo
                text_position = (top_left_corner[0] + padding, top_left_corner[1] + text_height + padding - 5)
                cv2.putText(
                    annotated_frame,
                    text,
                    text_position,
                    font,
                    font_scale,
                    (0, 0, 0),  # Colore nero per il testo
                    font_thickness
                )

                # Draw lines
                annotated_frame = draw_lines_on_frame(annotated_frame, lines_info)

                # FINE DISEGNI, INIZIO DISEGNI TRACCE
                track = track_history[track_id]
                track.append((float(x), float(y+h/2)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # Display the annotated frame
            if show:
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