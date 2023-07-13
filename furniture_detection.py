from roboflow import Roboflow
rf = Roboflow(api_key="x85B1rA8t0ISXPx6cF4Z")
project = rf.workspace().project("furniture-identifier-u2tyo")
model = project.version(5).model

# infer on a local image

if __name__ == "__main__":
    print(model.predict("data/normal living rooms114.jpg", confidence=40, overlap=30).json())