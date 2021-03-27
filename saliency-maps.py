# import the necessary packages
from imutils import paths
import argparse
import imutils
import cv2
import os
import sys
sys.path.append(os.path.abspath('./modules/'))
import detection
import helpers

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--imagepath", type=str, 
    default = None,
    help="path to test image")
args = ap.parse_args()
if(args.imagepath is None):
    image_path = helpers.get_random_test_image()
else:
    image_path = args.imagepath

image = cv2.imread(image_path)

# initialize OpenCV's static fine grained saliency detector and
# compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(image)

threshMap = cv2.threshold((saliencyMap * 255).astype("uint8"), 100, 200, cv2.THRESH_BINARY)[1]
edges = cv2.Canny(threshMap,100,200)
cnts = cv2.findContours(edges, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
contourimage = image.copy()
print(f'[INFO] {len(cnts)} countours')
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)
    if(len(approx) <6):
        continue
    # compute the convex hull of the contour, then use the area of the
    # original contour and the area of the convex hull to compute the
    # solidity
    area = cv2.contourArea(c)
    if(area < 1):
        continue
    print(f'[INFO] area {area}')
    hull = cv2.convexHull(c)
    hullArea = cv2.contourArea(hull)
    if(hullArea < 30):
        continue
    print(f'[INFO] hullArea {hullArea}')

    cv2.drawContours(contourimage, [hull], -1, (255,0,0), thickness=1)
    cv2.drawContours(contourimage, [c], -1, (0, 255, 0), thickness=1)

# show the images
cv2.imshow("Image", image)
cv2.imshow("StaticSaliencyFineGrained", saliencyMap)
cv2.imshow("Mask", threshMap)
cv2.imshow("Mask", edges)
cv2.imshow("Contour image", contourimage)


cv2.waitKey(0)    