import torch
import urllib.request
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
#urllib.request.urlretrieve(url, filename)

class MiDaS:
    def __init__(self, height=480, width=640):
        self.model_type = ["MiDaS_small", "DPT_Hybrid", "DPT_Large"]
        self.model_index = 0
        self.midas = torch.hub.load("intel-isl/MiDaS", self.model_type[self.model_index])
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.midas.to(self.device)
        self.midas.eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        self.height, self.width = height, width
        self.depth_filter = np.zeros((self.height, self.width)) # need to create some [720x1280] array of 0-100 values
        for i in range(self.height):
            for j in range(self.width):
                self.depth_filter[i, j] = np.exp( -0.5 * (((j - (self.width//2))) / (self.width / 6)) ** 2)
        
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
    def normalize(self, img, scale_factor=1):
        # travis webcam is 1280x720
        maximum = np.amax(img)

        if maximum < 1200:
            img /= 1200
        else:
            img /= maximum
        return img * scale_factor
        
    # local depth map evaluation (test center third of image for depth values closer than XXXXX)
    def filter(self, img, scale_factor=1):
        scale_image = img / scale_factor
        priority_heatmap = scale_image * self.depth_filter
        #return priority_heatmap
        if np.amax(priority_heatmap) > 0.6:
            # print("you are going to stub your toe")
            return True
        # print("ur fine lol")
        return False
    
    
if __name__ == "__main__":
    midas = MiDaS()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480))
    depth = midas.predict(img)
    depth = midas.filter(depth)
    plt.imshow(depth)
    plt.show()