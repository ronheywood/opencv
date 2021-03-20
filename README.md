# Open CV Experiments in object tracking

Experimenting with tracking a golf ball as it moves.

Practical uses will require high speed photography, but theoretically with suitable still images we should be able to determine launch angle, ball speed, spin and spin axis - which are the main components required for golf simulator software. 

I'm not anticipating trying to use live video for anything practical - I'll probably start with analyising two still frames - and to get round the high speed photography issue it'll probably be flash duration exposure
(I think the GC2 uses this technique)

# Setup

You will need the object detection trained model - which is too large for uploading to github, but can be downloaded here:

https://pjreddie.com/media/files/yolov3.weights

```
 wget https://pjreddie.com/media/files/yolov3.weights
```

I haven't written a configure script yet - You'll need to configure your machine for python, and install the pipenv dependencies

```
pipenv install
```

# Project Plan

Feel free to raise an issue if you want to get my attention on anything you find here, and if you're looking to do something similar.

You can follow some of the ideas I'm going for using the [project notes](https://github.com/ronheywood/opencv/projects/1) 

Currently I'm investigating Basic ball detection - I found the YOLO object detection algorithm is a good start, and while I'm currently assuming any "Sports ball" detected will be a golf ball, I suspect that a constraint that no other sports ball should be used, will be acceptable to users.

I was quite happy with the YOLO detection net I used against the images in the test-images folder, but I've since run it against some of the light box images from the specific manufactures ecommerce shots and it was not detecting them at all. It seems that it really only likes identifying objects in a real world environment

```
 py .\detect-ball-hough-circles.py
```

I started to understand how to track a detected object's movement in subsequent frames using the detect-once-tracking script:

```
py detect-once-tracking.py
```

Hold up a golf ball to your webcam - you'll see the frame rate is poor while it searches for the ball, 
but then improves once it starts to track using the object tracker algorithm.

I didn't investigate the issues around what to do if it loses sight of the golf ball - which happens if you move it quite fast, or obscure it.

# Current work in progress 
I'm investigating how to identify features of the golf ball so we can make decisions about how it's rotating - we'll need to do this to calculate spin and spin axis, as well as launch angle and ball speed

* Step 1 - Find the center of the ball
* Step 2 - Find a suitable feature of the ball that can be tracked so we can calculate center of rotation

Maybe we could do a UV map of the dimples, but it might be a good idea to start with tracking the logo, even though I can't assume it will always be visible in a functioning system...

I'm Currently (2021-03-20) iterating over the feature-detection script:

```
py feature-detection.py
```

This finds the ball, crops to it and runs a few transformations - I want to try and simplify it down to a few trackable features

To help zero in on the ball and to try and isolate features that will help
keep track of movement and orientation I'm feeling pleased with the behaviour in `detect-ball-hough-circles.py` and I am thinking about trying to combine this with picking out a logo using `detect-ball-logo.py`

