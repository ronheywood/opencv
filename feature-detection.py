# import the necessary packages
import cv2
import sys
import os
import numpy as np
sys.path.append(os.path.abspath('./modules/'))
import detection

image = cv2.imread('test-images/at-rest.png',1)
args = { 'weights': 'yolov3.weights' ,'classes':'yolo-classes.txt','config':'yolo.cfg'}
ball = detection.GolfBallDetection(image,args)
if ball:
    (x,y,w,h) = ball
    x_plus_w = x+w
    y_plus_h = y+h

    section = image.copy()[y:y+h, x:x+w]

    detection.draw_boundaries_and_label(image,(x,y),(w,h),(0,255,0),'Ball')
    
    cv2.imshow("Crop to ball",section)
        
    # generating the kernels
    kernel1 = np.array([[0, -1, -1], # kernel for embossing bottom left side
                        [1, 0, -1],
                        [1, 1, 0]])
                        
    # you can generate kernels for embossing top as well
    gray = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray",gray)

    embossdepth = np.ones((h, w), np.uint8) * 128
    embossed = cv2.add(cv2.filter2D(gray, -1, kernel1),embossdepth) # emboss on bottom left side
    cv2.imshow("Embossed",embossed)

    _, binary = cv2.threshold(cv2.cvtColor(section, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    binary = 255 - binary
    cv2.imshow("Binary",binary)

    edges = cv2.Canny(gray,100,200)
    cv2.imshow("Edges",edges)

    embossed_edges = cv2.Canny(embossed,100,200)
    cv2.imshow("Embossed Edges",embossed_edges)
    
    contours,heirarchy = cv2.findContours(embossed_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    im2 = cv2.cvtColor(section, cv2.COLOR_BGR2GRAY)
    dimples = []
    for contour in contours:
        approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
        area = cv2.contourArea(contour)
        if ((len(approx) > 8) & (area > 30) ):
            dimples.append(contour)

    cv2.drawContours(im2, dimples, -1, (0,255,0), 3)
    cv2.imshow("Contours",im2)

cv2.imshow("Detecting features",image)

cv2.waitKey(0)
cv2.destroyAllWindows()