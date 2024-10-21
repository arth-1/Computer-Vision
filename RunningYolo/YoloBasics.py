from ultralytics import YOLO
import cv2
 
model = YOLO('../Yolo-Weights/yolov8l.pt')
results = model("E:\ProjectsTest\Yolo\Yolofk\RunningYolo\Images/", show=True)
cv2.waitKey(0)