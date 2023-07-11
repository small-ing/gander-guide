# all the code pointing to the roboflow inference API with our object detection model
from roboflow import Roboflow
#different comment
rf = Roboflow(api_key="API_KEY") # need to grab key
project = rf.workspace().project("MODEL_ENDPOINT") # need to grab endpoint
model = project.version(VERSION).model # need to grab version

# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())