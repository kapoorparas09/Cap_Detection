import torch
import matplotlib.pyplot as plt
import matplotlib_inline
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Marj/Cap Detection/best.pt', force_reload=True)

img = 'D:/Marj/Cap Detection/yolov5/highway.jpg'

results = model(img)

# %matplotlib inline 
plt.imshow(np.squeeze(results.render()))
plt.show()
print(results)
results.save("output")