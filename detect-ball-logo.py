# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from skimage import exposure
from skimage import feature
from imutils import paths
import argparse
import imutils
import cv2

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
ap.add_argument("-d", "--training", required=False, default = 'bridgestone', help="Name of the manufacturer for the logos training dataset inside test-images")
ap.add_argument("-t", "--test", required=False, help="Path to the test dataset")
args = vars(ap.parse_args())

# initialize the data matrix and labels
print("[INFO] extracting features...")
data = []
labels = []

# loop over the image paths in the training set
for imagePath in paths.list_images('test-images/' + args["training"]):
    # extract the manufacturer of the ball
    make = imagePath.replace('test-images/','').split('\\')[-2]
    print(f'[Info] Training manufacturer {make}')
    # load the image, convert it to grayscale, and detect edges
    #For now let us ask the user to draw the region where the logo is
    image = cv2.imread(imagePath)
    
    key = cv2.waitKey(1) & 0xFF
    
    # select the bounding box of the object we want to track (make
    # sure you press ENTER or SPACE after selecting the ROI)
    initBB = cv2.selectROI("Select the logo", image, fromCenter=False,
        showCrosshair=True)
    (x,y,w,h) =  initBB
    if(w==0 or h==0): 
        continue

    region = None
    region = image[y:y+h,x:x+w]

    #A golf balls logo is usually black
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    #ret,thresh1 = cv2.threshold(region,127,255,cv2.THRESH_BINARY)
    edged = imutils.auto_canny(gray)

    # find contours in the edge map, keeping only the largest one which
    # is presmumed to be the car logo
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    
    # extract the logo of the ball and resize it to a canonical width
    # and height
    (x, y, w, h) = cv2.boundingRect(c)
    logo = gray[y:y + h, x:x + w]
    logo = cv2.resize(logo, (200, 100))

    cv2.destroyAllWindows()
    cv2.imshow("Logo",edged)
    
    # extract Histogram of Oriented Gradients from the logo
    H = feature.hog(logo, orientations=9, pixels_per_cell=(10, 10),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")

    # # update the data and labels
    data.append(H)
    labels.append(make)

print('All training images processed')
cv2.waitKey(0)
cv2.destroyAllWindows()