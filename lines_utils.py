import numpy as np
from math import pi, cos, sin
import json
import cv2

def real_to_pixel(x_real, y_real, config_file="confs/camera_config.json"):
    """
    Converte le coordinate reali (x, y) in coordinate di pixel nell'immagine.

    Questa funzione applica una serie di trasformazioni (compreso yaw, pitch, e roll) per proiettare
    le coordinate reali (x, y) nello spazio 3D in coordinate in pixel nel piano 2D dell'immagine,
    utilizzando i parametri della fotocamera caricati da un file di configurazione JSON.

    Parametri:
    -----------
    x_real : numpy.ndarray
        Vettore contenente le coordinate x reali dei punti da proiettare.

    y_real : numpy.ndarray
        Vettore contenente le coordinate y reali dei punti da proiettare.

    config_file : str, opzionale
        Percorso del file di configurazione JSON che contiene i parametri della fotocamera (default è "confs/camera_config.json").

    Ritorna:
    --------
    u : numpy.ndarray
        Vettore delle coordinate x in pixel nel piano dell'immagine.

    v : numpy.ndarray
        Vettore delle coordinate y in pixel nel piano dell'immagine.
    """
    # Carica i parametri dal file di configurazione JSON
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Parametri della fotocamera dal file di configurazione
    f = config['focal_length']  # lunghezza focale in metri
    U = config['image_width']  # larghezza dell'immagine in pixel
    V = config['image_height']  # altezza dell'immagine in pixel
    yaw = config['yaw'] * pi / 180  # rotazione Yaw in radianti
    roll = config['roll'] * pi / 180  # rotazione Roll in radianti
    pitch = config['pitch'] * pi / 180  # rotazione Pitch in radianti
    xc, yc, zc = config['camera_position']  # posizione della fotocamera nel sistema mondo

    # Parametri del sensore
    s_w = config['sensor_width']  # larghezza sensore
    s_h = config['sensor_height']  # altezza sensore

    # Matrici di rotazione
    R_pitch = np.array([[1, 0, 0],
                        [0, cos(pitch), -sin(pitch)],
                        [0, sin(pitch), cos(pitch)]])

    R_roll = np.array([[cos(roll), 0, sin(roll)],
                       [0, 1, 0],
                       [-sin(roll), 0, cos(roll)]])

    R_yaw = np.array([[cos(yaw), -sin(yaw), 0],
                      [sin(yaw), cos(yaw), 0],
                      [0, 0, 1]])

    # Matrize di rotazione totale
    R = R_yaw @ R_roll @ R_pitch

    # Trasformazione delle coordinate reali nel sistema della fotocamera
    real_coordinates = np.vstack((x_real, y_real, np.zeros_like(x_real)))  # Aggiungi z=0 per i punti
    camera_position = np.array([xc, yc, zc]).reshape(3, 1)
    translated_coordinates = real_coordinates + camera_position  # Traslazione per portare il punto nell'origine della fotocamera

    # Applicazione della rotazione
    camera_coordinates = R @ translated_coordinates  # Trasformazione al sistema di coordinate della fotocamera

    # Estrazione delle coordinate x, y, z nel sistema della fotocamera
    dx = camera_coordinates[0, :]  # Coordinate x nel sistema fotocamera
    dy = camera_coordinates[1, :]  # Coordinate y nel sistema fotocamera
    dz = camera_coordinates[2, :]  # Coordinate z nel sistema fotocamera

    # Calcolo della lunghezza focale in pixel
    f_x = f / s_w * U
    f_y = f / s_h * V

    # Calcolare le coordinate in pixel (u, v)
    u = U / 2 + f_x * dx / dy  # Proiezione in x
    v = V / 2 + f_y * dz / dy  # Proiezione in y

    for i in range(len(u)):
        u[i], v[i] = int(u[i]), int(v[i])

    #GESTIONE ERRORI
    if np.any(u < 0) or np.any(u > U) or np.any(v < 0) or np.any(v > V):
        raise ValueError(f"Le coordinate in pixel (u, v) sono fuori dalla risoluzione dell'immagine: u={u}, v={v}")

    return u, v


def load_lines(config_file="confs/lines_config.json"):
    """
    Carica le linee da un file di configurazione JSON e converte le coordinate reali in coordinate pixel.

    Questa funzione carica le linee da un file JSON, estrae le loro coordinate reali e poi utilizza
    la funzione `real_to_pixel` per ottenere le coordinate in pixel nell'immagine. Le linee sono
    restituite come due liste separate: una con le linee in coordinate reali e l'altra con le linee
    in coordinate pixel.

    Parametri:
    -----------
    config_file : str, opzionale
        Percorso del file di configurazione JSON che contiene le linee e i parametri della fotocamera (default è "config/lines_config.json").

    Ritorna:
    --------
    lines_in_real : list
        Lista di dizionari contenente le linee con le coordinate reali (x1, y1, x2, y2).

    lines_in_pixel : list
        Lista di dizionari contenente le linee con le coordinate in pixel (x1, y1, x2, y2).
    """
    # Carica i parametri dal file di configurazione JSON
    with open(config_file, 'r') as f:
        config = json.load(f)

    # Liste per memorizzare le linee in coordinate reali e in pixel
    lines_in_real = []
    lines_in_pixel = []

    # Itera sulle linee nel file di configurazione
    for line in config['lines']:
        # Estrai le coordinate reali delle linee (x1, y1, x2, y2)
        id, x1_real, y1_real, x2_real, y2_real = line['id'], line['x1'], line['y1'], line['x2'], line['y2']

        # Converto le coordinate reali in coordinate in pixel
        try:
            x_pixel, y_pixel = real_to_pixel(np.array([x1_real, x2_real]), np.array([y1_real, y2_real]))  # Chiamata alla funzione real_to_pixel per ottenere le coordinate in pixel

            # Salvo le linee in coordinate reali nella lista `lines_in_real` solo se riesco a convertire
            lines_in_real.append({
                'id': id,  # ID della linea
                'x1': x1_real,  # Coordinate x del primo punto
                'y1': y1_real,  # Coordinate y del primo punto
                'x2': x2_real,  # Coordinate x del secondo punto
                'y2': y2_real  # Coordinate y del secondo punto
            })

            # Salvo le linee in coordinate in pixel nella lista `lines_in_pixel` solo se riesco a convertire
            lines_in_pixel.append({
                'id': id,  # ID della linea
                'x1': x_pixel[0],  # Coordinata x del primo punto in pixel
                'y1': y_pixel[0],  # Coordinata y del primo punto in pixel
                'x2': x_pixel[1],  # Coordinata x del secondo punto in pixel
                'y2': y_pixel[1]  # Coordinata y del secondo punto in pixel
            })
        except ValueError as e:
            print(e)

    return lines_in_real, lines_in_pixel  # Restituisce le due liste con le linee in coordinate reali e in pixel

def get_points_from_lines(lines_in_pixel):
    """
    Estrae una lista di ID e un vettore di punti (tuple x, y) dalle linee fornite.

    :param lines_in_pixel: Lista di linee, ciascuna rappresentata da un dizionario con i campi:
        'id': ID della linea,
        'x1': Coordinata x del primo punto,
        'y1': Coordinata y del primo punto,
        'x2': Coordinata x del secondo punto,
        'y2': Coordinata y del secondo punto.
    :return: Una tupla (ids, points) dove:
        - ids: Lista di ID delle linee.
        - points: Lista di tuple (x, y) con i punti delle linee.
    """
    ids = []
    points = []

    for line in lines_in_pixel:
        # Aggiungi l'ID della linea
        ids.append(line['id'])

        # Aggiungi i punti della linea
        points.append((line['x1'], line['y1']))
        points.append((line['x2'], line['y2']))
    return ids, points

def get_lines_info():
    """
    Genera informazioni dettagliate sulle linee, includendo ID, punti iniziali e finali,
    posizione del testo e direzione della freccia.

    :return: Una lista di dizionari contenenti:
        - 'line_id': ID della linea.
        - 'start_point': Punto iniziale della linea (x, y).
        - 'end_point': Punto finale della linea (x, y).
        - 'text_position': Coordinate per la posizione del testo (x, y).
        - 'arrow_start': Punto di partenza della freccia (x, y).
        - 'arrow_end': Punto di fine della freccia (x, y).
    """
    _, lines = load_lines()
    ids, points = get_points_from_lines(lines)
    info_lines = []
    for i in range(len(ids)):
        j = i * 2 # Per ogni id devo prendere 2 punti
        # Prendi i punti e l'ID associato
        start_point = points[j]
        end_point = points[j + 1]

        # Converti i punti in interi
        start_point = (int(start_point[0]), int(start_point[1]))
        end_point = (int(end_point[0]), int(end_point[1]))

        line_id = int(ids[i])

        # Disegna l'ID in alto a sinistra della linea
        text_position = (start_point[0] + 10, start_point[1] - 10)  # Posizione leggermente sopra l'inizio della linea

        # Calcola la direzione perpendicolare verso l'alto
        dx = start_point[0] - end_point[0]
        dy = start_point[1] - end_point[1]
        length = np.sqrt(dx ** 2 + dy ** 2)
        unit_dx = dx / length
        unit_dy = dy / length

        # Perpendicolare verso l'alto
        perp_dx = -unit_dy
        perp_dy = unit_dx

        # Punto medio della linea
        mid_x = (start_point[0] + end_point[0]) // 2
        mid_y = (start_point[1] + end_point[1]) // 2

        # Calcola la posizione della freccia
        arrow_start = (int(mid_x), int(mid_y))
        arrow_lenght = 25
        arrow_end = (int(mid_x + perp_dx * arrow_lenght), int(mid_y + perp_dy * arrow_lenght))

        info_lines.append({
            'line_id': line_id,     # ID della linea
            'start_point': start_point,  # Punto di inizio della linea
            'end_point': end_point,  # Punto di fine della linea
            'text_position': text_position,  # Coordinate del testo
            'arrow_start': arrow_start,  # Coordinate del punto dell'inizio della freccia
            'arrow_end': arrow_end   # Coordinate del punto della fine della freccia
        })

    return info_lines

# Funzione per disegnare le linee sui frame
def draw_lines_on_frame(frame, lines_info):
    """
    Disegna linee blu sui frame utilizzando le informazioni fornite da get_lines_info.
    :param frame: Frame corrente (immagine in formato numpy array).
    :param lines: Lista di dizionari con informazioni sulle linee, generata da get_lines_info.
    """
    height, width = frame.shape[:2]
    for line in lines_info:
        # Estrai le informazioni dalla linea
        line_id = line['line_id']
        start_point = line['start_point']
        end_point = line['end_point']
        text_position = line['text_position']
        arrow_start = line['arrow_start']
        arrow_end = line['arrow_end']

        # Clipping delle coordinate nei limiti dell'immagine, per evitare problemi al confine (forse da mettere nella funzione precedente per ridurre la computazione)
        start_point = (np.clip(start_point[0], 0, width - 1), np.clip(start_point[1], 0, height - 1))
        end_point = (np.clip(end_point[0], 0, width - 1), np.clip(end_point[1], 0, height - 1))
        text_position = (np.clip(text_position[0], 0, width - 1), np.clip(text_position[1], 0, height - 1))
        arrow_start = (np.clip(arrow_start[0], 0, width - 1), np.clip(arrow_start[1], 0, height - 1))
        arrow_end = (np.clip(arrow_end[0], 0, width - 1), np.clip(arrow_end[1], 0, height - 1))

        # Disegna la linea blu
        cv2.line(frame, start_point, end_point, (255, 0, 0), thickness=3)

        # Disegna l'ID in alto a sinistra della linea
        cv2.putText(frame, str(line_id), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), thickness=3)

        # Disegna la freccia
        cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 0, 0), thickness=3, tipLength=0.5)

    return frame

# Testing del codice
if __name__ == "__main__":
    lines_info = get_lines_info()
    print(lines_info)
    image = cv2.imread("videos/test.png")
    image = draw_lines_on_frame(image, lines_info)
    cv2.imshow("Immagine con linee disegnate", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
