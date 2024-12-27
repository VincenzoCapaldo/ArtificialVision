import cv2 as cv
import numpy as np


def add_bounding_box(frame, top_left_corner, bottom_right_corner, color=(0, 0, 255), thickness=4):
    """
    Add bounding box around detected people
    :param frame: frame on which to draw the bounding box
    :param top_left_corner: (x,y) coordinate of the top left corner of the bounding box
    :param bottom_right_corner: (x,y) coordinate of the bottom right corner of the bounding box
    :param color: (optional) border color of the bounding box
    :param thickness: (optional) thickness of the bounding box
    """

    cv.rectangle(frame, (top_left_corner[0], top_left_corner[1]),
                 (bottom_right_corner[0], bottom_right_corner[1]),
                 color, thickness)


def add_track_id(frame, track_id, top_left_corner, background_color=(255, 255, 255), text_color=(0, 0, 225)):
    """
    add the people id inside the bounding box
    :param frame: the frame on which to draw the given id
    :param track_id: the id to display
    :param top_left_corner: (x, y) coordinate
    :param background_color: (optional) background color of the rectangle
    :param text_color: (optional) text color for the track_id
    """
    text = f"{track_id}"
    font_scale = 0.6
    font_thickness = 2
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size = cv.getTextSize(text, font, font_scale, font_thickness)[0]
    text_x, text_y = top_left_corner[0] + 5, top_left_corner[1] + 20  # Slightly offset inside the BB
    text_width, text_height = text_size[0], text_size[1]
    background_start = (text_x - 2, text_y - text_height - 2)  # Add padding
    background_end = (text_x + text_width + 2, text_y + 2)

    # Draw white rectangle as background for the given id
    cv.rectangle(frame, background_start, background_end, background_color, -1)

    # Draw track ID in red over the background
    cv.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        text_color,  # Red color
        font_thickness)


def add_info_scene(frame, text):
    """
    Display the general information in the top left corner of the frame
    :param frame: the frame on which to draw the information
    :param text: vector which contains all the information to display
    """
    # Parametri per il testo
    font_scale = 0.8
    font_thickness = 2
    font = cv.FONT_HERSHEY_SIMPLEX
    text_size = []
    text_width = []
    text_height = []
    # Calcolo della dimensione del testo
    for t in text:
        text_size = cv.getTextSize(t, font, font_scale, font_thickness)[0]
        text_width.append(text_size[0])
        text_height.append(text_size[1])

    # Definizione della posizione del rettangolo
    padding = 10  # Spazio extra intorno al testo
    top_left_corner = (padding, padding)
    bottom_right_corner = (
    padding + max(text_width) + 2 * padding, padding + sum(text_height) + 2 * len(text) * padding)

    background_color = (255, 255, 255)
    # Disegna il rettangolo bianco
    cv.rectangle(frame, top_left_corner, bottom_right_corner, background_color, -1)

    # Scrivi il testo sopra il rettangolo
    total_height = 10
    for i in range(len(text)):
        text_position_x = top_left_corner[0] + padding
        text_position_y = total_height + text_height[0] + (2 * padding) - 5

        total_height = text_position_y
        text_position = (text_position_x, text_position_y)
        cv.putText(
            frame,
            text[i],
            text_position,
            font,
            font_scale,
            (0, 0, 0),  # Colore nero per il testo
            font_thickness
        )


# Funzione per disegnare le linee sui frame
def draw_lines_on_frame(frame, lines_info):
    """
    Draws blue lines given by lines_info
    :param frame: current frame, given in numpy array format
    :param lines_info: list of dictionaries with line information, generated by get_lines_info.
    """
    height, width = frame.shape[:2]
    for line in lines_info:
        # extract the information about the line
        line_id = line['line_id']
        start_point = line['start_point']
        end_point = line['end_point']
        text_position = line['text_position']
        arrow_start = line['arrow_start']
        arrow_end = line['arrow_end']

        # Clipping delle coordinate nei limiti dell'immagine, per evitare problemi al confine (forse da mettere nella funzione precedente per ridurre la computazione)
        # questa funzione si trovava nel file lines_utils tra la funzione intersaca e checked  cross line
        start_point = (np.clip(start_point[0], 0, width - 1), np.clip(start_point[1], 0, height - 1))
        end_point = (np.clip(end_point[0], 0, width - 1), np.clip(end_point[1], 0, height - 1))
        text_position = (np.clip(text_position[0], 0, width - 1), np.clip(text_position[1], 0, height - 1))
        arrow_start = (np.clip(arrow_start[0], 0, width - 1), np.clip(arrow_start[1], 0, height - 1))
        arrow_end = (np.clip(arrow_end[0], 0, width - 1), np.clip(arrow_end[1], 0, height - 1))

        # Disegna la linea blu
        cv.line(frame, start_point, end_point, (255, 0, 0), thickness=3)

        # Disegna l'ID in alto a sinistra della linea
        cv.putText(frame, str(line_id), text_position, cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), thickness=3)

        # Disegna la freccia
        cv.arrowedLine(frame, arrow_start, arrow_end, (255, 0, 0), thickness=3, tipLength=0.5)

    return frame

def screen_save(frame,top_left_corner,bottom_right_corner,id):
    x1,y1 = top_left_corner[0], top_left_corner[1]
    x2,y2 = bottom_right_corner[0], bottom_right_corner[1]
    cropped_image = frame[y1:y2, x1:x2]
    # Salva lo screenshot della bounding box
    output_path = f"./screen/bounding_box_screenshot{id}.jpg"
    cv.imwrite(output_path, cropped_image)
    #return cropped_box