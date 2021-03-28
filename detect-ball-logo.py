# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import sys
import os
sys.path.append(os.path.abspath('./modules/'))
import detection
import helpers


def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

# construct the argument parse and parse command line arguments
ap = argparse.ArgumentParser()

ap.add_argument('-i', '--image', required=False,
                    help='Path to the image')

args = ap.parse_args()
if(args.image is None):
    image_path = helpers.get_random_test_image()
else:
    image_path = args.image

image = cv2.imread(image_path)

ball = detection.GolfBallDetection(image)
if ball is not None:
    (x,y,w,h) = ball
    cropped = image[y:y+h,x:x+w]
    min_hsv = (0, 0, 0)
    max_hsv = (152, 255, 154)
    mask = cv2.inRange(cropped, min_hsv , max_hsv)
    
    inverted = cv2.bitwise_not(mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
    dilated = cv2.erode(inverted.copy(),kernel,iterations=5)

    edges = cv2.Canny(dilated,100,200)
    

    cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    for c in cnts:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)

        if (len(approx) > 6):
            cv2.drawContours(cropped, [c], -1, (240, 0, 159), 3)
            break

    cv2.imshow("Logo detected",cropped)

cv2.imshow("Test image",image)
cv2.waitKey(0)
