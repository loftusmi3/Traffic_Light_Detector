import cv2
import numpy as np
import pdb

# Given a picture, get_box returns the bounding boxes on the traffic lights

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers


def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def create_det(weights,config):
    return cv2.dnn.readNet(weights, config)

def get_classes(classes_input):
    classes = None
    with open(classes_input, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes
    
def get_box(image, net, classes):
    
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
        
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.75
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # class_id == 9 so we only get the traffic lights
            if confidence > conf_threshold and class_id==9:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                boxes.append([x, y, w, h])
    
    return np.asarray(boxes)