import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('Trackbars')

# create trackbars for color change
cv2.createTrackbar('R','Trackbars',0,255,nothing)
cv2.createTrackbar('G','Trackbars',0,255,nothing)
cv2.createTrackbar('B','Trackbars',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'Trackbars',0,1,nothing)

print('Pess ESC to quit')
while(1):
    cv2.imshow('Trackbars',img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    r = cv2.getTrackbarPos('R','Trackbars')
    g = cv2.getTrackbarPos('G','Trackbars')
    b = cv2.getTrackbarPos('B','Trackbars')
    s = cv2.getTrackbarPos(switch,'Trackbars')

    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()