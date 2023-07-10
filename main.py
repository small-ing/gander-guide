import cv2
import midas_processing as mp

# Path: main.py
camera = cv2.VideoCapture(0)
depth_model = mp.MiDaS()

while True:
    try:
        success, frame = camera.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here
    # model.predict(frame)
    frame = depth_model.predict(frame)
    #
    #
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()