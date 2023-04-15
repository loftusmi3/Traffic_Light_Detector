import cv2
import sys
sys.path.append("../Detector")
import detector
import color_lights
import argparse
import pdb
import os

def KCF_tracker(detector,video_name):
    # Adapted from https://learnopencv.com/object-tracking-using-opencv-cpp-python/

    video = cv2.VideoCapture(video_name)
        # Exit if video not opened.
    if not video.isOpened():
        print("Video not found")
        sys.exit()
    
    tracker = cv2.TrackerKCF_create()
    
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    # Assume just one traffic light for now
    bbox = detector.get_box(frame, config, weights, classes_input)

    # Initialize tracker with first frame and bounding box
    tracker.init(frame, (int(bbox[0][0]),int(bbox[0][1]),int(bbox[0][2]),int(bbox[0][3])))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracked_video = cv2.VideoWriter("Results/tracked_"+os.path.basename(video_name), fourcc, 20, (frame.shape[1],frame.shape[0]))
    
    while ok:
        ok, frame = video.read()

        # Start timer
        timer = cv2.getTickCount()

        ok, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        # Draw bounding box
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            color = color_lights.get_color(frame[p1[0]:p2[0],p1[1]:p2[1]])
            cv2.rectangle(frame, p1, p2, color, 2)
        else :
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        # Display tracker type on frame
        cv2.putText(frame, "KCFTracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

        # Save video
        tracked_video.write(frame)
        cv2.imshow("Tracker",frame)

        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break

        # Read next frame for next iteration of the while loop
        ok, frame = video.read()
        
    cv2.destroyAllWindows()
    tracked_video.release()
    video.release()
    print("Object tracking completed")
    
    
if __name__ == "__main__":
    
    # Example for how to run from command line:
    # python color_lights.py -n 68
    # In this example, 68 is the image ID number
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--number')
    args = parser.parse_args()
    while len(args.number) < 5:
        args.number = '0' + args.number
    
    config = "../Detector/yolov3.cfg"
    weights = "../Detector/yolov3.weights"
    classes_input = "../Detector/yolov3.txt"
        
    KCF_tracker(detector,str('Videos/tl_vid%s.mp4' % args.number))
    
    