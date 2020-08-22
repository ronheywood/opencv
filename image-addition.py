import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('images/3D-Matplotlib.png')
img2 = cv2.imread('images/mainsvmimage.png')

simpleAdd = img1+img2
cv2.imwrite('output/image-addition.png',simpleAdd)

cv2Add = cv2.add(img1,img2)
cv2.imwrite('output/cv2-image-addition.png',cv2Add)

weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
cv2.imwrite('output/cv2-weighted-addition.png',weighted)