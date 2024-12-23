import cv2


def screen_save(frame,top_left_corner,bottom_right_corner,id):
    x1,y1 = top_left_corner[0], top_left_corner[1]
    x2,y2 = bottom_right_corner[0], bottom_right_corner[1]
    cropped_image = frame[y1:y2, x1:x2]
    # Salva lo screenshot della bounding box
    output_path = f"./screen/bounding_box_screenshot{id}.jpg"
    cv2.imwrite(output_path, cropped_image)
    #return cropped_box



