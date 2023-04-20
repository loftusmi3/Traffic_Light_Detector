import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import pdb

"""
given a numpy array that contains image data of a traffic light,
detects the color that the traffic light is and returns a BGR tuple
for either red, yellow, or green.
"""
def get_color(data):

    if(data.size != 0):
    
        # convert to HSV
        hsvFrame = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)

        # Set range for red color and
        # define mask
        # red_lower = np.array([136, 87, 111], np.uint8)
        # red_upper = np.array([180, 255, 255], np.uint8)
        red_lower = np.array([116, 28, 111], np.uint8)
        red_upper = np.array([180, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

        # Set range for yellow color and
        # define mask
        yellow_lower = np.array([95, 190, 210], np.uint8)
        yellow_upper = np.array([115, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

        # Set range for green color and
        # define mask
        green_lower = np.array([25, 80, 40], np.uint8)
        green_upper = np.array([50, 255, 255], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

        # determine percentage that the light is red, yellow, or green.
        # ----mask values seem to be (0-255) where 255 is
        # ----the pixel that matches the color.
        red_percent = np.sum(red_mask) / (data.shape[0] * data.shape[1] * 255)
        #print(f'percentage light is red = {red_percent}')

        yellow_percent = np.sum(yellow_mask) / (data.shape[0] * data.shape[1] * 255)
        #print(f'percentage light is yellow = {yellow_percent}')

        green_percent = np.sum(green_mask) / (data.shape[0] * data.shape[1] * 255)
        #print(f'percentage light is green = {green_percent}')

        c = np.argmax([red_percent, yellow_percent, green_percent])

        #print(f'detected color is {color}')

        # BGR color for OpenCV
        if c == 0:
            return (0,0,255)
        elif c == 1:
            return (127,127,0)
        else:
            return (0,255,0)
        
    else:
        return (0,0,0)

if __name__ == "__main__":
    
    # Example for how to run from command line:
    # python color_lights.py -n 68
    # In this example, 68 is the image ID number
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number')
    args = parser.parse_args()
    while len(args.number) < 5:
        args.number = '0' + args.number
    imageFrame = Image.open(('./Valid_Traffic_Lights/traffic_light0_%s.jpg' % args.number))

    # # red light test
    # imageFrame = Image.open("Valid_Traffic_Lights/traffic_light0_00072.jpg")

    # # yellow light test
    # imageFrame = Image.open("./Valid_Traffic_Lights/traffic_light2_00476.jpg")
    
    # # green light test
    # imageFrame = Image.open("Valid_Traffic_Lights/traffic_light2_00012.jpg")

    
    # convert image to numpy array
    data = np.array(imageFrame)
    plt.imshow(imageFrame)
    plt.savefig('./light.png')
    color = get_color(data)
    print(color)
