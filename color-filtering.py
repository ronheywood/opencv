import cv2
import numpy as np
import matplotlib.pyplot as plt

launchMonitor = cv2.imread('images/launch-monitor.jpg')

hsv = cv2.cvtColor(launchMonitor,cv2.COLOR_BGR2HSV)

lower_black = np.array([10,10,10])
upper_black = np.array([180,180,180])

mask = cv2.inRange(hsv,lower_black,upper_black)
res = cv2.bitwise_and(launchMonitor,launchMonitor, mask=mask)

cv2.imshow('original',launchMonitor)
cv2.imshow('mask',mask)
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllWindows()