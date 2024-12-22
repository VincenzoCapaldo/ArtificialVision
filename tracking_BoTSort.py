from collections import defaultdict
import cv2
import numpy as np
from ultralytics import YOLO

from OutputWriter import OutputWriter
from lines_utils import  get_lines_info, check_crossed_line
import time
import gui_utils as gui


def start_track(device, model_path="models/yolo11m.pt", video_path="videos/Atrio.mp4", show=False, real_time=True, tracker="confs/botsort.yaml"):

    # Load the YOLO model
    model = YOLO(model_path).to(device)

    # Load lines info
    lines_info = get_lines_info()
    number_of_lines = len(lines_info)

    # Create an output-writer object to write on a json file the results
    output_writer = OutputWriter()

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if real_time:
        fps = cap.get(cv2.CAP_PROP_FPS)
        ms_per_frame = 1/fps
    # Store the track history
    track_history = defaultdict(lambda: [])
    first_frame = True

    lista_attraversamenti = {}

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

                # general information of the scene to display
                # implementare il conteggio degli attraversamenti
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


                # FINE DISEGNI, INIZIO DISEGNI TRACCE
                track = track_history[track_id]
                track.append((float(x), float(y+h/2)))  # x, y center point
                if len(track) > 30:  # retain 90 tracks for 90 frames
                    track.pop(0)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

                crossed_line_id = check_crossed_line(track, lines_info)
                if(len(crossed_line_id)!=0):
                    if not(track_id in lista_attraversamenti):
                        lista_attraversamenti[track_id] = []
                    lista_attraversamenti[track_id].extend(crossed_line_id)

                # Add a new person
                output_writer.add_person(track_id)

            # Display the annotated frame
            if show:
                cv2.imshow("YOLO Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

        end_time = time.time()
        if real_time:
            elapsed_time = end_time - start_time
            discard_frames = int(elapsed_time/ms_per_frame)+1
            d2 = int(discard_frames + ((discard_frames * (end_read_time-start_read_time))/ms_per_frame))+1
            while d2 > 0 and not first_frame:
                cap.read()
                d2 -= 1
            first_frame = False

    print(lista_attraversamenti)

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
