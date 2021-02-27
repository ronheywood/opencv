import numpy as np
import cv2

image_path = 'C:\\Users\\ron_h\\Pictures\\Holiday\\PICT0016.JPG'
img = cv2.imread(image_path,0)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()