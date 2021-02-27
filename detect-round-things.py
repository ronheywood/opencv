import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('test-images/at-rest.png',1)
#image = cv2.imread('test-images/at-rest-teed-up.png',1) 
#image = cv2.imread('test-images/high-spin-lob-wedge.png',1)
#image = cv2.imread('test-images/close-up-outdoor--no-club-grass.png',1)

output = image.copy()
height, width = image.shape[:2]
maxRadius = int(2.1*(width/12)/2)
minRadius = int(0.9*(width/12)/2)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#gray = cv2.GaussianBlur(gray,(5,5),0)
frame_HSV = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)

# Define range of white color in HSV
lower_white = np.array([0, 0, 80])
upper_white = np.array([131, 255, 255])
# Threshold the HSV image
mask = cv2.inRange(frame_HSV, lower_white, upper_white)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_TOZERO)
edges = cv2.Canny(thresh,100,200)

plt.subplot(121),plt.imshow(thresh,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()

# circles = cv2.HoughCircles(image=thresh, 
#                            method=cv2.HOUGH_GRADIENT, 
#                            dp=1.9, 
#                            minDist=5,
#                            param1=10,
#                            param2=10,
#                            minRadius=50,
#                            maxRadius=0                           
#                           )
# circles = None
# if circles is not None:
#     # convert the (x, y) coordinates and radius of the circles to integers
#     circlesRound = np.round(circles[0, :]).astype("int")
#     # loop over the (x, y) coordinates and radius of the circles
#     for (x, y, r) in circlesRound:
#         cv2.circle(output, (x, y), r, (0, 255, 0), 4)

#     cv2.imshow('detected circles',output)
# else:
#     print ('No circles found')
#     cv2.imshow('Copy',output)
    
cv2.waitKey(0)
cv2.destroyAllWindows()