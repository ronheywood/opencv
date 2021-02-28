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
ap.add_argument('-v', '--video', required=False,
                default = 0,
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

# load our serialized model from disk
print("[INFO] loading model...")

classes = None
with open(args.classes, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(args.weights, args.config)

# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
#vs = VideoStream(args.video).start()
if(args.video == 0):
    vs = VideoStream(0)
    time.sleep(2.0)
else: 
    vs = cv2.VideoCapture(args.video)
    if not vs.isOpened():
        print("Cannot open camera")
        exit()

#Limiting the console output to one object per category
class_ids = []

# loop over the frames from the video stream
while True:

    if(args.video == 0):
        image = vs.read()
    else:
        grabbed, image = vs.read()
        if not grabbed:
            break
    
    Height, Width, channels = image.shape

    scale = 0.00392

    classes = None

    weights_file = args.weights
    config_file = args.config
    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(args.weights, args.config)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

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
            
            if confidence > conf_threshold:
                print('Found ' + classes[class_id] +  ' with confidence ' + str(confidence))
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    cv2.imshow("Frame", image)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
if(args.video ==0):
        vs.stop()
else:
    vs.release()