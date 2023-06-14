import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib_inline
from cvlib.object_detection import YOLO

cam =cv2.VideoCapture(0)
weights="yolov4-tiny-custom_8000.weights"
config="yolov4-tiny-custom.cfg"
labels="obj.names"
count=0

while True:
    ret,img=cam.read()
    count += 1
    if count % 10 != 0:
        continue

    img=cv2.resize(img,(680,460))
    yolo = YOLO(weights, config,labels)
    bbox, label, conf = yolo.detect_objects(img)
    img1=yolo.draw_bbox(img, bbox, label, conf)

    cv2.imshow("img1",img)

    if cv2.waitKey(1)&0xFF==ord("q"):
        break
cam.release() 
cv2.destroyAllWindows()