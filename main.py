import cv2

# Path: main.py
camera = cv2.VideoCapture(0)

while True:
    try:
        succss, frame = camera.read()
    except:
        print("There is no camera")
        break
    
    #insert any model calls here
    #
    #
    #
    
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()