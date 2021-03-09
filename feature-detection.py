# import the necessary packages
import cv2
import sys
import os
sys.path.append(os.path.abspath('./modules/'))
from detection import GolfBallDetection

image = cv2.imread('test-images/at-rest.png',1)
args = { 'weights': 'yolov3.weights' ,'classes':'yolo-classes.txt','config':'yolo.cfg'}
ball = GolfBallDetection(image,args)
if ball:
    (x,y,w,h) = ball
    x_plus_w = x+w
    y_plus_h = y+h
    color = (0,255,0)
    label = 'Ball'
    cv2.rectangle(image, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(image, label, (x-10,y_plus_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Detecting features",image)

cv2.waitKey(0)
cv2.destroyAllWindows()