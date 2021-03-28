from imutils.video import VideoStream
from imutils.video import FPS
import cv2

fps = None
vga = (480,640)
hd = (1080,1920)
hd720 = (720,1280)
vs = VideoStream(src=1 + cv2.CAP_DSHOW,resolution=(480,640)).start()
fps = None

while True:
    #success,frame = cam.read()
    frame = vs.read()
    if frame is None:
        print('Failed to read frame from camera.')
        break
    if(fps is None):
        fps = FPS().start()
    else:        
        fps.update()
        fps.stop()
        print(int(fps.fps()))
        text = f'FPS: {int(fps.fps())}'
        cv2.putText(frame, text, (0,50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        

    cv2.imshow("Cam",frame)
    if cv2.waitKey(1) & 0xFF is ord('q'):
            break

cv2.destroyAllWindows()