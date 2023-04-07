import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

"""
given an image and location of traffic signal,
check what color light is currently active.
"""
def check_color(img):
  color = ['r', 'y', 'g']

  # # load the image
  # imageFrame = Image.open('/Valid_Traffic_Lights/traffic_light0_00001.jpg')

  # convert image to numpy array
  # data = np.array(imageFrame)
  data = np.array(img)
  
  # # git the detected slice of image
  # x1, y1, x2, y2 = coords
  # data = data[y1:y2,x1:x2]

  # convert to HSV
  hsvFrame = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
  plt.imsave("test-hsv.jpg", img)

  # Set range for red color and
  # define mask
  # red_lower = np.array([136, 87, 111], np.uint8)
  # red_upper = np.array([180, 255, 255], np.uint8)
  red_lower = np.array([100, 87, 111], np.uint8)
  red_upper = np.array([180, 255, 255], np.uint8)
  red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)

  # Set range for yellow color and
  # define mask
  yellow_lower = np.array([20, 100, 100], np.uint8)
  yellow_upper = np.array([30, 255, 255], np.uint8)
  yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)

  # Set range for green color and
  # define mask
  green_lower = np.array([25, 52, 72], np.uint8)
  green_upper = np.array([102, 255, 255], np.uint8)
  green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

  # determine percentage that the light is red, yellow, or green.
  # ----mask values seem to be (0-255) where 255 is
  # ----the pixel that matches the color.
  red_percent = np.sum(red_mask) / (data.shape[0] * data.shape[1] * 255)
  print(f'percentage light is red = {red_percent}')

  yellow_percent = np.sum(yellow_mask) / (data.shape[0] * data.shape[1] * 255)
  print(f'percentage light is yellow = {yellow_percent}')

  green_percent = np.sum(green_mask) / (data.shape[0] * data.shape[1] * 255)
  print(f'percentage light is green = {green_percent}')

  percents = [red_percent, yellow_percent, green_percent]

  color = color[np.argmax(percents)]

  print(f'detected color is {color}')
  return color

if __name__ == "__main__":
    imageFrame = Image.open('./Valid_Traffic_Lights/traffic_light0_00001.jpg')
    color = check_color(imageFrame)
    print(color)
