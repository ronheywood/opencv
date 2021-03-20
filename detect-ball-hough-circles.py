# import the necessary packages
import cv2
import sys
import os
import numpy as np
import argparse
import time
sys.path.append(os.path.abspath('./modules/'))
import detection
import helpers
from matplotlib import pyplot as plt


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--debug", type=str, 
    default = None,
    help="Show each mask detection for Debugging")
ap.add_argument("-i", "--imagepath", type=str, 
    default = None,
    help="path to test image")

args = ap.parse_args()
if(args.imagepath is None):
    image_path = helpers.get_random_test_image()
else:
    image_path = args.imagepath

image = cv2.imread(image_path,1)
ball = detection.GolfBallDetection(image)

if ball:
    (x,y,w,h) = ball
    output = image.copy()
    #The boundaries are often not very accurate
    #so we can expand on them a little to make the mask better
    x -= 10
    y -= 10
    w += 30
    h += 30

    x_plus_w = x+w
    y_plus_h = y+h

    
    # show the output image
    circle = detection.get_ball_circle(image,x,y,w,h)
    if circle is not None:
        (cx, cy, r) = circle
        cx +=x 
        cy +=y
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (cx, cy), r, (0, 255, 0), 4)
        cv2.rectangle(output, (cx - 5, cy - 5), (cx + 5, cy + 5), (0, 128, 255), -1)
        
        cv2.imshow("Output", output)
    else:
        print ('No circles found')
        detection.draw_boundaries_and_label(output,(x,y),(w,h),(0,255,0),"No circle radius found")
        cv2.imshow('Failed to identify center of the ball',output)

cv2.waitKey(0)    
cv2.destroyAllWindows()