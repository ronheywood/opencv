# this implementation will require the tracking algorithm to
#  be run for each "frame"
# import the necessary packages
from centroidtracker import CentroidTracker
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-v', '--video', required=True,
                help = 'path to a test video stream')
ap.add_argument('-c', '--config', required=False,
                default='yolo.cfg',
                help = 'path to yolo config file, defaults to yolo.cgf')
ap.add_argument('-w', '--weights', required=False,
                default='yolov3.weights',
                help = 'path to yolo pre-trained weights, defaults to yolov3.weights.')
ap.add_argument('-cl', '--classes', required=False,
                default = 'yolo-classes.txt',
                help = 'path to text file containing class names, defaults to yolo-classes.txt')
ap.add_argument('-cf', '--confidence', required=False,
                default = 0.5,
                help = 'confidence threshold for object detection, defaults to 0.5')

args = ap.parse_args()


def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])
    if(label != 'golf ball'):
        return

    color = COLORS[class_id]        
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y_plus_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)  

image = cv2.imread('test-images/putter-launch.png')

# load our serialized model from disk
print("[INFO] loading model...")

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Limiting the console output to one object per category
class_ids = []

# loop over the frames from the video stream
while True:
    # Read the next frame from the video stream and resize it
    # (Testing with a static image for now)
    frame = vs.read()
    #frame = image

	#frame = imutils.resize(frame, width=400)
    scale = 0.00392 #The tutorial uses a scale of  1?

    # if the frame dimensions are None, grab them
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, scale, (W, H),
        (104.0, 177.0, 123.0))
    net.setInput(blob)

    # mean_subtract_from_channels = (104.0, 177.0, 123.0) #image yolo used (0,0,0)?
    # swap_rbg_for_bgr = True
    # blob = cv2.dnn.blobFromImage(frame, scale, (W,H), mean_subtract_from_channels, swap_rbg_for_bgr, crop=False)
    # net.setInput(blob)
    
    #detections = net.forward(get_output_layers(net))
    detections = net.forward(get_output_layers(net))
    rects = []
    # loop over the detections
    for out in detections:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            detect_confidence = scores[class_id]
            if(classes[class_id] != 'person' and detect_confidence > args.confidence ):
                box = detection[3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                    (0, 255, 0), 2)
                print(f'detection {classes[class_id]} confidence {detect_confidence} box {box.astype("int")}')

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.waitKey()
cv2.destroyAllWindows()
vs.stop()