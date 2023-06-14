import torch
from matplotlib import pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

model

img = 'highway.jpg'

results = model(img)
results.print()

# %matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()

%python train.py --img 320 --batch 16 --epochs 20 --data dataset.yaml --weights yolov5s.pt