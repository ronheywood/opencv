# Connected components seems to offer a tool to filter interesting regions of the ball

# This example will load an image and identify conected components - but it doesn't seem to 
# help us fix on the logos very well,
# even the Bridgestone B1 logo - which I had expected to see better results with

# import the necessary packages
import numpy as np
import argparse
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=False,
	help="path to input image")
ap.add_argument("-t", "--threshold", type=int, default=100,
	help="binary image threshold between 0 and 255 (go lower for Hi vis golf balls")
ap.add_argument("-c", "--connectivity", type=int, default=4,
	help="connectivity for connected component analysis")
args = vars(ap.parse_args())

if(args["image"]):
    image = cv2.imread(args["image"])
else:
    image = cv2.imread('test-images/titleist/titleist-1.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, args["threshold"], 255,
	cv2.THRESH_BINARY)[1]

# apply connected component analysis to the thresholded image
output = cv2.connectedComponentsWithStats(
	thresh, args["connectivity"], cv2.CV_32S)
(numLabels, labels, stats, centroids) = output

# loop over the number of unique connected component labels
for i in range(0, numLabels):
    # if this is the first component then we examine the
    # *background* (typically we would just ignore this
    # component in our loop)
    if i == 0:
        text = f"examining component {i+1}/{numLabels} (background)"
    # otherwise, we are examining an actual connected component
    else:
        text = f"examining component {i+1}/{numLabels}"
    # print a status message update for the current connected
    # component
    print("[INFO] {}".format(text))
    # extract the connected component statistics and centroid for
    # the current label
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroids[i]

    # clone our original image (so we can draw on it) and then draw
    # a bounding box surrounding the connected component along with
    # a circle corresponding to the centroid
    output = image.copy()
    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(output, (int(cX), int(cY)), 4, (0, 0, 255), -1)    

    # construct a mask for the current connected component by
    # finding a pixels in the labels array that have the current
    # connected component ID
    componentMask = (labels == i).astype("uint8") * 255
    # show our output image and connected component mask
    cv2.imshow("Output", output)
    cv2.imshow("Connected Component", componentMask)
    cv2.waitKey(0)

cv2.destroyAllWindows()