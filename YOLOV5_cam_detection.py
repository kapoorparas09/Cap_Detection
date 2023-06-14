import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import matplotlib_inline

model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Marj/Cap Detection/best.pt', force_reload=True)
model.conf = 0.6
model.iou = 0.45
cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()

    results = model(frame )
    # print(results)
    cv2.imshow('frame', np.squeeze(results.render()))

    if cv2.waitKey(1) & 0xFF == ord('q'): #Shutdown the camera by pressing Q
        break
    
cam.release()
cv2.destroyAllWindows()


# ,confidence=0.5, nms_thresh=0.3