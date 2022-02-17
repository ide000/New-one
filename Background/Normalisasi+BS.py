from turtle import width
import numpy as np
from tracker import *
import cv2

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("Video1.mp4")

# Tracker
tracker = EuclideanDistTracker()

# Subtractor
# mog2Subtractor = cv2.createBackgroundSubtractorMOG2(300, 400, True)
mog2Subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=60)

# Keeps track of what frame we're on


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    #  Check if a current frame actually exist
    if not ret:
        break

	# Resize the frame
    # resizedFrame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # resizedFrame1 = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # Enchancement brightness and contrast
    # cv2.normalize(resizedFrame, resizedFrame, 0.5*255 , 50 , cv2.NORM_MINMAX)
    # cv2.normalize(frame, frame, 0.8*255 , 50 , cv2.NORM_MINMAX)
    # cv2.normalize(frame , frame , alpha=-2 , beta=1.5*255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Extract Region of Interest
    roi = frame[200:600,600:1400]

    # Get the foreground masks using all of the subtractors
    # mog2Mmask = mog2Subtractor.apply(resizedFrame)
    mog2Mmask = mog2Subtractor.apply(roi)
    _, mog2Mmask = cv2.threshold(mog2Mmask, 254, 255, cv2.THRESH_BINARY)

    # Deteksi dengan Contour
    contours, _ =cv2.findContours(mog2Mmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            # cv2.drawContours(resizedFrame, [cnt], -1, (0 ,255, 0), 2)
            # cv2.drawContours(roi, [cnt], -1, (0 ,255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi, (x,y), (x+w, y+h), (0, 255, 0), 3)
            # print(x, y, w, h)
            detections.append ([x, y, w, h])    
    
    
    # Count all the non zero pixels within the masks
    # mog2MCount = np.count_nonzero(mog2Mmask)
    # height, width, _ = resizedFrame.shape
    # print(height, width)
    print (detections)
    cv2.imshow("ROI",roi)
    cv2.imshow('Normalisasi', frame)
    # cv2.imshow('Normalisasi', resizedFrame)
    cv2.imshow('MOG2', mog2Mmask)
    # cv2.imshow('Original',frame)

    key = cv2.waitKey(30)
    if key == 27:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()