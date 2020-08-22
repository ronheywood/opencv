import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('./images/watch.jpg')

extract = img[47:180,47:185]
cv2.imwrite('./output/extracted_image.png', extract)
region = cv2.imread('./output/extracted_image.png')

cv2.imshow('Original',img)
cv2.imshow('Extracted',region)

cv2.waitKey(0)
cv2.destroyAllWindows()