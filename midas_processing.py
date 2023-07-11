import torch
import urllib.request
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)

class MiDaS:
    def __init__(self):
        self.model_type = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]
        self.model_index = 0
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type[self.model_index])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.depth_filter = None # need to create some [720x1280] array of 0-100 values
        
        if self.model_type[self.model_index] == "DPT_Large" or self.model_type[self.model_index] == "DPT_Hybrid":
            self.transform = self.midas_transforms.dpt_transform
        else:
            self.transform = self.midas_transforms.small_transform

    def predict(self, img):
        input_batch = self.transform(img).to(self.device)

        with torch.no_grad():
            prediction = self.midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()
        #print("Time elapsed: ", time.time() - start)
        return output
    
    # seperate methods to normalize and denormalize depth maps
    def normalize(self, img):
        # travis webcam is 1280x720
        maximum = np.amax(img)

        if maximum < 1200:
            img /= 1200
        else:
            img /= maximum
        return img
        
    # local depth map evaluation (test center third of image for depth values closer than XXXXX)
    def filter(self, img):
        # prioritize center of image
        #compress to 640 x 480

        print(img.shape)

        # Define the shape of the array
        height = 480
        width = 640

        # Calculate the center column
        center_column = width // 2

        # Create an array of zeros with the desired shape
        filter = np.zeros((height, width))

        # Generate the values using a Gaussian distribution
        for i in range(height):
            for j in range(width):
                filter[i, j] = np.exp(-0.5 * ((j - center_column) / (width / 6)) ** 2)
        
        priority_heatmap = img * filter

        if np.amax(priority_heatmap) > 0.5:
            print(np.amax(priority_heatmap))
            print("you are going to stub your toe")

    
if __name__ == "__main__":
    midas = MiDaS()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = midas.predict(img)
    plt.imshow(depth)
    plt.show()