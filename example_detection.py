from ultralytics import YOLO
import torch


print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

if __name__ == '__main__':
    model = YOLO("models/yolov8m.pt")
    # model.info()
    # print(model.names)
    # ID = = (persone), ID=24 (zaino), non ho trovato il cappello
    classesID = [0, 24]
    source = "videos/Atrio.mp4"
    #source = "https://www.youtube.com/watch?v=jOb6eNN6m6M&pp=ygUOdmlkZW8gdHJhZmZpY28%3D"
    result = model(source, imgsz=(384, 640), show=True, device=device, max_det=50, classes=classesID, save=False,
                   vid_stride=1, iou=0.7, conf=0.4, agnostic_nms=True)

