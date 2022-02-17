from turtle import distance, width
import numpy as np
from tracker import *
import cv2
import math

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Video1.mp4")

# Tracker
tracker = EuclideanDistTracker()

#Nilai Kernel
kernel = np.ones((5,5), np.uint8)

# Subtractor
# mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=80)

# Keeps track of what frame we're on
count = 0
center_points_prev_frame =[]
tracking_objects = {}
track_id = 0


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    count += 1 
    #  Check if a current frame actually exist
    if not ret:
        break

    # Point Current Frame
    center_points_cur_frame = []
  


    # Enchancement brightness and contrast if needed
    # cv2.normalize(frame, frame, 0.8*255 , 50 , cv2.NORM_MINMAX)
    # cv2.normalize(frame , frame , alpha=-2 , beta=1.5*255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Extract Region of Interest
    roi = frame[200:600,650:1200]

    # 1. Object detection
    
    # 1.1. Background Subtraction
    mog2Mmask = mog2Subtractor.apply(roi)
    _, mog2Mmask = cv2.threshold(mog2Mmask, 254, 255, cv2.THRESH_BINARY )
    
    # 1.2. Fill Holes
    # frm_dilate = cv2.dilate(mog2Mmask, kernel, iterations=1)
    # frm_erode = cv2.erode(mog2Mmask, kernel, iterations=1)
    # frm_close = cv2.erode(frm_dilate, kernel, iterations=1)
    # frm_open = cv2.morphologyEx(mog2Mmask, cv2.MORPH_OPEN, kernel)
    # frm_close = cv2.morphologyEx(mog2Mmask, cv2.MORPH_CLOSE, kernel)
    frm_close = cv2.morphologyEx(mog2Mmask, cv2.MORPH_CLOSE, kernel)
    # opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
 
    # 1.3. Deteksi dengan Contour
    # contours, _ =cv2.findContours(frm_erode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ =cv2.findContours(frm_dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # contours, _ =cv2.findContours(mog2Mmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours, _ =cv2.findContours(frm_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 3000:
            # cv2.drawContours(resizedFrame, [cnt], -1, (0 ,255, 0), 2)
            # cv2.drawContours(roi, [cnt], -1, (0 ,255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cx = int ((x + x + w)/2)
            cy = int ((y + y + h)/2)
            center_points_cur_frame.append((cx,cy))
            # print("frame N" , count , " ", x,y,w,h)
            # cv2.circle (roi, (cx,cy), 5, (0,0,255),-1)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0, 255, 0), 3)
            detections.append ([x, y, w, h]) 

    # 2. Object Tracker
    boxes_ids = tracker.update(detections)
    for box_id in boxes_ids: 
        x , y , w , h , id =  box_id
        cv2.putText(roi, str(id), (x,y-15), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 2)
        cv2.rectangle(roi, (x,y), (x+w, y+h), (0, 255, 0), 3) 

    # 3. Speed Detection
    
    # print("TRACKING OBJECTS")
    # print(tracking_objects)
    
    print("CUR FRAME")
    print(center_points_cur_frame)

    print("PREV FRAME")
    print(center_points_prev_frame)
    # Count all the non zero pixels within the masks
    # height, width, _ = resizedFrame.shape
    # print(height, width)
    print (detections)
    cv2.imshow("ROI",roi)
    cv2.imshow('Normalisasi', frame)
    cv2.imshow('MOG2', mog2Mmask)

    # make a copy of current frame as prev frame
    center_points_prev_frame = list(center_points_cur_frame)

    key = cv2.waitKey(0)
    if key == 27:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()