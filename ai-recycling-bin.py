import numpy as np
import cv2 as c
import time
import ncnn

def getImage():
    #wait a few seconds
    time.sleep(3)
    #tell the system to use the default camera
    cap = c.VideoCapture(0)
    #let the camera warm up if it needs the time
    time.sleep(1)
    #read a singular frame
    ret, frame = cap.read()
    if(ret):
        c.imwrite("image.jpg", frame)
        print("yay")
    cap.release()
    c.destroyAllWindows()
    return frame
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
def getInference(frame):
    #Load the model
    net = ncnn.Net()
    net.load_param("model/model.ncnn.param")
    net.load_model("model/model.ncnn.bin")


    #preprocess the image
    image = resizeWithPadding(frame)
    while True:
        c.imshow("img",image)
        if c.waitKey(1) == ord('q'):
            break
    #image = c.resize(frame, (640, 640)) #resized to match the input size of the model
    image = image.astype(np.float32) / 255.0  # Normalize to [0,1]
    image = image.transpose(2, 0, 1)  # Convert to channel-first (C, H, W)

    # Convert to NCNN matrix
    ncnn_mat = ncnn.Mat.from_pixels(image.flatten(), ncnn.Mat.PixelType.PIXEL_RGB, 640, 640)

    # Create an extractor. the extractor is the thing that facilitates model inference
    ex = net.create_extractor()
    ex.input("in0", ncnn_mat) 

    # Run inference
    ret, output = ex.extract("out0")

    # Process the output
    output_np = np.array(output)
    #------------------------------------------------------------------------------------
    # something to consider: maybe add a feature later that
    #------------------------------------------------------------------------------------
    print(output_np)
    predicted_class = np.argmax(output_np)  # Get the class with the highest probability
    print(f"Predicted class: {predicted_class}")
    return predicted_class
def main():
    # image = getImage()
    image = loadImage("temp/IMG_3317.jpg")
    answer = getInference(image)

main()