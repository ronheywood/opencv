import cv2
import numpy as np
import matplotlib.pyplot as plt

launchMonitor = cv2.imread('images/launch-monitor.jpg')

gray = cv2.cvtColor(launchMonitor,cv2.COLOR_BGR2GRAY)



# adaptive threshold
adaptive = cv2.adaptiveThreshold(gray, 
255, 
cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
cv2.THRESH_BINARY, 175, 1)

retval2,otsu = cv2.threshold(gray,
125,
255,
cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow('Grey',gray)
cv2.imshow('Adaptive',adaptive)
cv2.imshow('OTSU',otsu)
#cv2.imshow('Inverse',mask_inv)

cv2.waitKey(0)
cv2.destroyAllWindows()