#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
############################################


import cv2
import argparse
import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import join
import pdb

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--folder', required=True,
                help = 'path to input folder')
args = ap.parse_args()

config = "yolov3.cfg"
weights = "yolov3.weights"
classes_input = "yolov3.txt"

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


def print_image(image, name, output_boxes):
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    net.setInput(blob)

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
        
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_threshold:
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
    
    traffic_light_num = 0
    
    for i in indices:
        try:
            box = boxes[i]
        except:
            i = i[0]
            box = boxes[i]

        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        if class_ids[i] == 9:
            cropped = image[round(y):round(y+h), round(x):round(x+w)]
            cv2.imwrite(join(args.folder, "traffic_light" + str(traffic_light_num) + "_" + name), cropped)
            traffic_light_num += 1
            
        #draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

    #cv2.imshow("object detection", image)
    #cv2.waitKey()

    #cv2.imwrite(join(args.folder, "detected_" + name), image)
    #cv2.destroyAllWindows()

classes = None

with open(classes_input, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

net = cv2.dnn.readNet(weights, config)  
output_boxes = {}
        
for name in os.listdir(args.folder):
    if "detected" in name:
        continue
    if "traffic_light" in name:
        continue
    image = cv2.imread(join(args.folder, name))
    print_image(image, name, output_boxes)

with open("myfile.txt", 'w') as f: 
    for key, value in output_boxes.items(): 
        f.write('%s:%s\n' % (key, value))