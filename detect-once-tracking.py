# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import sys
import os
import numpy as np
import argparse
import imutils
import time
import cv2
sys.path.append(os.path.abspath('./modules/'))
import detection
import helpers

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, 
    default = 0,
    help="path to input video file")
ap.add_argument("-fv", "--flip_video", type=bool, default=False, help="Flip video orientation (if the camera is upside down)")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
    help="OpenCV object tracker type")
ap.add_argument("-o", "--output", type=str, default=None,
    help="Should the video capture write to an output file?")
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

# extract the OpenCV version info
(major, minor) = cv2.__version__.split(".")[:2]
# if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
# function to create our object tracker
if int(major) == 3 and int(minor) < 3:
    tracker = cv2.Tracker_create(args.tracker.upper())
# otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
# approrpiate object tracker constructor:
else:
    # initialize a dictionary that maps strings to their corresponding
    # OpenCV object tracker implementations
    OPENCV_OBJECT_TRACKERS = {
        "csrt": cv2.TrackerCSRT_create, # High tracking accuracy at the expense of FPS
        "kcf": cv2.TrackerKCF_create, # Lower tracking accuracy for higher FPS
        "mil": cv2.TrackerMIL_create
    }
    if(hasattr(cv2,'cv2.TrackerMOSSE_create')):
        OPENCV_OBJECT_TRACKERS["mosse"] = cv2.TrackerMOSSE_create
    if(hasattr(cv2,'cv2.TrackerMIL_create')):
        OPENCV_OBJECT_TRACKERS["mil"] = cv2.TrackerMIL_create
    if(hasattr(cv2,'cv2.TrackerBoosting_create')):
        OPENCV_OBJECT_TRACKERS["boosting"] = cv2.TrackerBoosting_create
    if(hasattr(cv2,'cv2.TrackerTLD_create')):
        OPENCV_OBJECT_TRACKERS["tld"] = cv2.TrackerTLD_create
    if(hasattr(cv2,'cv2.TrackerMedianFlow_create')):
        OPENCV_OBJECT_TRACKERS["medianflow"] = cv2.TrackerMedianFlow_create
    # grab the appropriate object tracker using our dictionary of
    # OpenCV object tracker objects
    tracker = OPENCV_OBJECT_TRACKERS[args.tracker]()

# initialize the bounding box coordinates of the object we are going
# to track
initBB = None
rotateCode = None

# if a video path was not supplied, grab the reference to the web cam
if not args.video:
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0,resolution=(480,640)).start()
    time.sleep(1.0)
# otherwise, grab a reference to the video file
else:
    vs = cv2.VideoCapture(args.video)
# initialize the FPS throughput estimator
fps = None

# If output writing we will need to initialise thiw
writer = None
zeros = None

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.video else frame
    # check to see if we have reached the end of the stream
    if frame is None:
        break
    # resize the frame (so we can process it faster) and grab the
    # frame dimensions
    frame = imutils.resize(frame, width=800)
    (H, W) = frame.shape[:2]
    # check if the frame needs to be rotated
    if args.flip_video:
        frame = cv2.flip(frame,-1)

    if(args.output and writer is None):
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*"DIVX")
        writer = cv2.VideoWriter(args.output + '.mp4',fourcc, 20.0, (W,H), True)
        zeros = np.zeros((H, W), dtype="uint8")

    # check to see if we are currently tracking an object
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(frame)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", args.tracker),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        initBB = detection.GolfBallDetection(frame)
        if(initBB):
            # start OpenCV object tracker using the supplied bounding box
            # coordinates, then start the FPS throughput estimator as well
            tracker.init(frame, initBB)
            fps = FPS().start()

    # show the output frame
    cv2.imshow("Frame", frame)
    if(writer):
        # write the output frame to file
        writer.write(frame)

    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
# if we are using a webcam, release the pointer
if not args.video:
    vs.stop()
# otherwise, release the file pointer
else:
    vs.release()

if(writer):
    writer.release()

# close all windows
cv2.destroyAllWindows()