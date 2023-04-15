import cv2
import sys
sys.path.append("../Detector")
import detector
import color_lights
import argparse
import pdb
import os

def KCF_tracker(config,weights,classes_input,video_name):
    # Adapted from https://learnopencv.com/object-tracking-using-opencv-cpp-python/ and
    # https://learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/

    video = cv2.VideoCapture(video_name)
        # Exit if video not opened.
    if not video.isOpened():
        print("Video not found")
        sys.exit()
    
    ok, frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    
    net = detector.create_det(weights,config)
    
    # Assume just one traffic light for now
    bboxes = detector.get_box(frame, net, classes_input)
    
    multiTracker = cv2.MultiTracker_create()
    for i in range(0,len(bboxes)):
        multiTracker.add(cv2.TrackerKCF_create(), frame, (int(bboxes[i][0]),int(bboxes[i][1]),int(bboxes[i][2]),int(bboxes[i][3])))
    #for i in range(0,len(bboxes)):
    #    trackers[i].init(frame, (int(bboxes[i][0]),int(bboxes[i][1]),int(bboxes[i][2]),int(bboxes[i][3])))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tracked_video = cv2.VideoWriter("Results/tracked_"+os.path.basename(video_name), fourcc, 10, (frame.shape[1],frame.shape[0]))
    
    while ok:
        ok, frame = video.read()

        # Start timer
        #timer = cv2.getTickCount()

        #fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        
        ok, bboxes = multiTracker.update(frame)

        # Draw bounding box 
        if ok:
            for bbox in bboxes:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                color = color_lights.get_color(frame[p1[1]:p2[1],p1[0]:p2[0]])
                cv2.rectangle(frame, p1, p2, color, 2)
        # If the object tracker can't keep tracking, redetect and reinitialize object tracker
        elif len(bboxes) > 0:
            bboxes = detector.get_box(frame, net, classes_input)
            multiTracker = cv2.MultiTracker_create()
            for i in range(0,len(bboxes)):
                multiTracker.add(cv2.TrackerKCF_create(), frame,
                                 (int(bboxes[i][0]),int(bboxes[i][1]),int(bboxes[i][2]),int(bboxes[i][3])))

        # Display tracker type on frame
        cv2.putText(frame, "KCFTracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

        # Display FPS on frame
        #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

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
        
    KCF_tracker(config,weights,classes_input,str('Videos/tl_vid%s.mp4' % args.number))
    
    