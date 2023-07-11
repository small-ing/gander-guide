# all the code pointing to the roboflow inference API with our object detection model
from roboflow import Roboflow
<<<<<<< HEAD

rf = Roboflow(api_key="wJ1fFMldaUhe4Ni1w2Gx") # need to grab key
=======
rf = Roboflow(api_key="API_KEY") # need to grab key
>>>>>>> 2926f4d56d4290bc59c3a871df0d662073de6eb3
project = rf.workspace().project("MODEL_ENDPOINT") # need to grab endpoint
model = project.version(VERSION).model # need to grab version
#im adding stuff here
# infer on a local image
print(model.predict("your_image.jpg", confidence=40, overlap=30).json())