import cv2
import midas_processing as mp
import time

# Path: main.py
camera = cv2.VideoCapture(0)

old_frame = None
depth_model = mp.MiDaS()
start = time.time()
frames = 1

while True:
    try:
        success, frame = camera.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here
    # model.predict(frame)
    frame = depth_model.predict(frame)
    fps = frames / (time.time() - start)
    cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    frames += 1
    #
    #
    
    try:
        cv2.imshow("Camera", frame - old_frame)
    except:
        pass
    
    old_frame = frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()