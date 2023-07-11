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

        self.depth_filter = None # need to create it
        
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
        print(maximum)
        img /= maximum
        return img
        
    # local depth map evaluation (test center third of image for depth values closer than XXXXX)
    def filter(self, img):
        # prioritize center of image
        '''        
        priority_heatmap = img * self.depth_filter
        if np.amax(priority_heatmap) > 0.5:
            return True
        return False
        '''
        pass
    
        
    def alert(self, alert_flag):
        if alert_flag:
            print("ALERT YOU ARE ABOUT TO STUB YOUR TOE!")
    
if __name__ == "__main__":
    midas = MiDaS()
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    depth = midas.predict(img)
    plt.imshow(depth)
    plt.show()