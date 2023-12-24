import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import os
import glob

#%pip install ultralytics
import ultralytics
ultralytics.checks()

from ultralytics import YOLO
image_width = 676
image_height = 380
model = YOLO('yolov8n.pt')

model.predict(source='/kaggle/input/car-object-detection/data/testing_images', save=True, imgsz=(image_height,image_width), conf=0.5,classes=2,verbose=False)

from IPython.display import Image

files = glob.glob("./runs/detect/predict/*")
for i in range(0, 50, 5):
    img = Image(files[i])
    display(img)

model = YOLO('yolov8n-seg.pt')

results = model.predict(source='/kaggle/input/car-object-detection/data/training_images', 
                        hide_labels=True,boxes=False,classes=2, 
                        conf=0.5,verbose=False,show_labels=False,show_boxes=False)

for result in results:
    masks = result.masks

for i in range(100, 160, 2):
    if results[i].masks:
        results[i].boxes=0
        img = results[i].plot()
        plt.imshow(img)
        plt.show()