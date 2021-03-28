import numpy as np
import cv2
import imutils
import time

golf_ball_detection_args = { 'weights': 'yolov3.weights' ,'classes':'yolo-classes.txt','config':'yolo.cfg'}
def GolfBallDetection(image,args = None):
    if args is None:
        args = golf_ball_detection_args
    start = time.time()
    print("Searching for a golf ball...")
    Height, Width, channels = image.shape
    scale = 0.00392
    forced_size = (416,416)

    classes = None

    weights_file = args['weights']
    config_file = args['config']
    with open(args['classes'], 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_file, config_file)

    blob = cv2.dnn.blobFromImage(image, scale, forced_size, (0,0,0), True, crop=False)
    
    net.setInput(blob)

    outs = net.forward(_get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.3
    nms_threshold = 0.4

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if classes[class_id] == 'golf ball' and confidence > conf_threshold:
                end = time.time()
                print(f'[INFO] Found {classes[class_id]} with confidence {(str(confidence))} after {(end - start)} seconds')
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                #The boundaries are often not very accurate
                #so we can expand on them a little to make the mask better
                #but it might be a good idea to look for the edges
                x -= 10
                y -= 10
                w += 30
                h += 40

                return(round(x), round(y), round(w), round(h))

    print("[INFO] YOLO detection failed to find a ball attempting HSV threshold detection")
    hsv = _hsv_contour_detection(image)
    if(hsv is not None):
        return cv2.boundingRect(hsv)

    return None

def _hsv_contour_detection(image):
    frame_to_thresh = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    min_hsv = (0, 0, 240)
    max_hsv = (255, 50, 255)
    mask = cv2.inRange(frame_to_thresh, min_hsv , max_hsv)
    inverted = cv2.bitwise_not(mask)
    #cv2.imshow("Inverted before erode",inverted)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
    dilated = cv2.dilate(inverted, element, iterations=1)

    # find contours in the thresholded image
    cnts = cv2.findContours(inverted.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    # safe helper for opencv version compatibility
    cnts = imutils.grab_contours(cnts)
    # loop over the contours
    for c in cnts:
        # compute the center of the contour
        M = cv2.moments(c)
        if(M["m00"]==0):
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

        # compute the convex hull of the contour, then use the area of the
        # original contour and the area of the convex hull to compute the
        # solidity
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hullArea = cv2.contourArea(hull)
        solidity = area / float(hullArea)

        # visualize the original contours and the convex hull and initialize
        # the name of the shape
        # cv2.drawContours(image, [hull], -1, 255, -1)
        # cv2.drawContours(image, [c], -1, (240, 0, 159), 3)
        shape = ""

        if(len(approx) <6 ): 
            continue #3 is a triangle, 4 is a rectangle, 5 is pentagon - not circles so can be ignored
        
        #A the countours bounding box aspect ratio should be close to 1.0
        # artifacts like shadows mess this up so we can refine it
        x,y,w,h = cv2.boundingRect(c)
        ar = min(w,h)/max(w,h)
        # don't want any tiny things
        if(w < int(image.shape[1]*0.25)):
            continue

        print(f'[INFO] found aspect ratio {ar}')
        #if(ar < 0.9):
            
        return c

    print("[INFO] HSV detection failed to find a ball")
    return None

def _get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_boundaries_and_label(image,xy:tuple, wh:tuple, color, label):
    thickness = 2
    x,y = (xy)
    w,h = (wh)
    x_plus_w = x+w
    y_plus_h = y+h
    center_x = int(w/2)
    center_y = int(h/2)

    cv2.rectangle(image, (x,y), (x_plus_w,y_plus_h), color, thickness)
    cv2.line(image,(x + center_x-10,y+center_y),( x + center_x+10,y+center_y),(255,0,0))
    cv2.line(image,(x +center_x,y+center_y-10),(x+center_x,y+center_y+10),(255,0,0))
    cv2.putText(image, label, (x-10,y_plus_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def get_ball_circle(image,x,y,w,h):
    circles = None
    start = time.time()
    
    hsv_circle = _hsv_contour_detection(image[y:y+h, x:x+w])
    if hsv_circle is not None:
        (xy,r) = cv2.minEnclosingCircle(hsv_circle)
        (x,y) = xy
        return (int(x),int(y),int(r))

    end = time.time()
    print(f'[INFO] HSV circle detection none found after {(end - start)} seconds - trying Hough circles')

    start = time.time()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[y:y+h, x:x+w]
    gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_CONSTANT)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,1.9,minDist=w)
    # ensure at least some circles were found
    if circles is not None:
        end = time.time()
        print(f'[INFO] Hough circle detection took {(end - start)} seconds')
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        return circles[0]

    print(f'[INFO] Hough circle detection none found after {(end - start)} seconds')    
    return None

def draw_circle_around_ball(image,circle:tuple,offset:tuple):
    (x,y) = offset
    (cx, cy, r) = circle
    cx +=x 
    cy +=y
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
    cv2.circle(image, (cx, cy), r, (0, 255, 0), 4)
    cv2.circle(image, (cx,cy), 5, (0, 0, 255), -1)