from ultralytics import YOLO
from predict import DetectionPredictor
import cv2

model = YOLO("yolov8n.pt") #I have used nano dataset, for more precise and vast detection, use l or x model.
while True:
    results = model.predict(source='0',show=True,save=True) #accepts all format : img, folder, videos
    print(results)
    

