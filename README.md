# Open CV Experiments in object tracking

Experimenting with tracking a golf ball as it moves.

Practical uses will require high speed photogrqaphy, but theoritically with suitable still images we should be able to determine launch angle, ball speed, spin and spin axis - which are the main components required for golf simulator software.

You will need the object detection trained model - which is too large for uploading to github, but can be downloaded here:

https://pjreddie.com/media/files/yolov3.weights

```
 wget https://pjreddie.com/media/files/yolov3.weights
```

I haven't written a configure script yet - You'll need to configure python along with the opencv matplotlib and numpy packages

```
 py yolo-detection.py --image test-images/close-up-outdoor-address.png --config yolo.cfg --weights yolov3.weights --classes yolo-classes.tx
 ```
