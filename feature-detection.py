# import the necessary packages
import cv2
import sys
import os
import numpy as np
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
    section = image[y:y+h, x:x+w]
    cv2.imshow("Crop to ball",section)
        
    # generating the kernels
    kernel1 = np.array([[0, -1, -1], # kernel for embossing bottom left side
                        [1, 0, -1],
                        [1, 1, 0]])
                        
    # you can generate kernels for embossing top as well
    gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    embossdepth = np.ones((h, w), np.uint8) * 128
    embossed = cv2.add(cv2.filter2D(gray, -1, kernel1),embossdepth) # emboss on bottom left side
    cv2.imshow("Embossed",embossed)

    _, img_bin = cv2.threshold(cv2.cvtColor(section, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = 255 - img_bin
    cv2.imshow("Binary",img_bin)

    cv2.rectangle(image, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(image, label, (x-10,y_plus_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

cv2.imshow("Detecting features",image)

cv2.waitKey(0)
cv2.destroyAllWindows()