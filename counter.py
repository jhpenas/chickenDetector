import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math
from utils import *
import argparse

# Lista de nomes das classes para a detecção de objetos
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
          "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
          "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
          "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
          "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
          "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
          "teddy bear", "hair drier", "toothbrush"
          ]


def main(video_path, yolo_model):
    mask = cv2.imread("mascara.png")
    cap = cv2.VideoCapture(video_path)
    model = YOLO(yolo_model)

    # Inicializando o objeto "tracker" usando a classe Sort com os seguintes parâmetros:
    # max_age: O número máximo de quadros (frames) que um objeto pode estar ausente antes de ser excluído
    # min_hits: O número mínimo de quadros (frames) que um objeto deve ser detectado para ser considerado rastreado
    # iou_threshold: O limiar de Intersecção sobre União (IoU) para determinar se duas bounding boxes se sobrepõem
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    # Lista para armazenar os IDs dos objetos contados
    counter = []

    while True:
        suc, img = cap.read()
        img_region = cv2.bitwise_and(img, mask)

        img_banner = cv2.imread("banner.png", cv2.IMREAD_UNCHANGED)
        img = cvzone.overlayPNG(img, img_banner, (0, 0))
        results = model(img_region, stream=True)

        detections = np.empty((0, 5))

        for res in results:
            boxes = res.boxes
            for box in boxes:
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                # cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
                w, h = x2 - x1, y2 - y1

                # Confidence
                conf = math.ceil((box.conf[0] * 100)) / 100
                # Class Name
                cls = int(box.cls[0])
                current_label = labels[cls]

                if current_label == "bird" and conf > 0.3:
                    # cvzone.putTextRect(img, f'{currentClass} {conf}', (max(0, x1), max(35, y1)),
                    #                    scale=0.6, thickness=1, offset=3)
                    # cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)
                    current_array = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, current_array))

        results_from_tracker = tracker.update(detections)

        for result in results_from_tracker:
            x1, y1, x2, y2, id_res = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            print(result)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f' {"Galinha"}', (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            if counter.count(id_res) == 0:
                counter.append(id_res)
                # cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        # cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))
        cv2.putText(img, str(len(counter)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='data/Chicken.mp4', help='path to the video file')
    parser.add_argument('--model', type=str, default='yolov8l.pt', help='path to the YOLO model')
    args = parser.parse_args()

    main(args.video, args.model)
