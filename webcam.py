import numpy as np
import cv2

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
ret, frame = camera.read()
print('Starting webcam - press Q to quit')
while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    frame = cv2.rectangle(frame,(284,50),(510,303),(0,255,0),3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame,'label',(284,333), font, 1,(188,0,0),2,cv2.LINE_AA)
    cv2.imshow('Webcam',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
camera.release()
cv2.destroyAllWindows()