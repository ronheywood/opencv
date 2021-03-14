import numpy as np
import cv2

golf_ball_detection_args = { 'weights': 'yolov3.weights' ,'classes':'yolo-classes.txt','config':'yolo.cfg'}
def GolfBallDetection(image,args = None):
    if args is None:
        args = golf_ball_detection_args
    print("Searching for a golf ball...")
    Height, Width, channels = image.shape

    scale = 0.00392

    classes = None

    weights_file = args['weights']
    config_file = args['config']
    with open(args['classes'], 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    net = cv2.dnn.readNet(weights_file, config_file)

    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

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
                print('Found ' + classes[class_id] +  ' with confidence ' + str(confidence))
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                return(round(x), round(y), round(w), round(h))
    print("no golf ball detected")
    return None

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

def _get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers
