import numpy as np
import cv2 as c
import time
import ncnn
from ultralytics import YOLO

def loadImage(path):
    image = c.imread(path)
    image = c.cvtColor(image, c.COLOR_BGR2RGB)
    return image
def resizeWithPadding(image, target_size=(640, 640)):
    h, w, _ = image.shape
    scale = min(target_size[0] / w, target_size[1] / h)  # Maintain aspect ratio
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = c.resize(image, (new_w, new_h))
    
    # Create a new blank image of target size with padding
    padded_image = np.full((target_size[1], target_size[0], 3), 128, dtype=np.uint8)  # 128 is a neutral gray
    
    # Center the resized image
    pad_x = (target_size[0] - new_w) // 2
    pad_y = (target_size[1] - new_h) // 2
    padded_image[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    return padded_image

model = YOLO("model2stuff/best.pt")

image = loadImage("temp/IMG_2147.jpg")

result = model(image)

inference_time = result[0].speed['inference']

print(result)

print(inference_time)