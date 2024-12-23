import cv2


def screen_save(frame,x,y,w,h,id):
    cropped_box = frame[y:y + h, x:x + w]
    # Salva lo screenshot della bounding box
    output_path = "./test_sample/bounding_box_screenshot.jpg"
    print(": ")
    print(x)
    print(y)
    print(w)
    print(h)
    cv2.imwrite(output_path, cropped_box)
    #return cropped_box
