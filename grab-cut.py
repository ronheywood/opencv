# import the necessary packages
import cv2
import sys
import os
import time
import numpy as np
sys.path.append(os.path.abspath('./modules/'))
from detection import GolfBallDetection

image = cv2.imread('test-images/at-rest.png',1)
args = { 'weights': 'yolov3.weights' ,'classes':'yolo-classes.txt','config':'yolo.cfg'}
ball = GolfBallDetection(image,args)
if ball:
    (x,y,w,h) = ball
    x -= 10
    y -= 10
    
    x_plus_w = x+w
    y_plus_h = y+h

    x_plus_w += 20
    y_plus_h +=20

    color = (0,255,0)
    label = 'Ball'
    rect = (x,y,w,h)
    section = image[y:y_plus_h, x:x_plus_w]
    cv2.imshow("Crop to ball",section)
    mask = np.zeros(image.shape[:2], dtype='uint8') #empty image to be populated

    #alocate memory for mask array
    bg_model = np.zeros((1,65), dtype='float')
    fg_model = np.zeros((1,65), dtype='float')
    start = time.time()
    iter_count = 1
    (mask,bgModel,fgModel) = cv2.grabCut(image, mask, (x,y,x_plus_w,y_plus_h), bg_model, fg_model, iter_count,
     cv2.GC_INIT_WITH_RECT)
    end = time.time()
    print("[INFO] applying grabCut took {:.2f} seconds".format(end - start))

    values = (
        ("Definite Background", cv2.GC_BGD),
        ("Probable Background", cv2.GC_PR_BGD),
        ("Definite Foreground", cv2.GC_FGD),
        ("Probable Foreground", cv2.GC_PR_FGD)        
    )

    for (name,value) in values:
        print(f"[INFO] showing mask for {name}")
        valueMask = (mask == value).astype("uint8") * 255

        cv2.imshow(name,valueMask)
        cv2.waitKey(0)
    
    outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD),0,1)
    outputMask = (outputMask * 255).astype("uint8")

    output = cv2.bitwise_and(image, image, mask=outputMask)

    cv2.imshow("Grab cut filter",output)

cv2.waitKey(0)
cv2.destroyAllWindows()