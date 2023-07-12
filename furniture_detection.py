# all the code pointing to the roboflow inference API with our object detection model
from roboflow import Roboflow


rf = Roboflow(api_key="wJ1fFMldaUhe4Ni1w2Gx") # need to grab key

project = rf.workspace().project("MODEL_ENDPOINT") # need to grab endpoint
model = project.version(VERSION).model # need to grab version
#im adding stuff here

# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())
