#coding=utf-8  
import cv2
import dlib
import numpy as np
from math import hypot


class Gaze:

    status =""

    detector=dlib.get_frontal_face_detector()

    predictors= dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    #--------------function for gaze_ratio calculation---------------------------

    def get_gaze_ratio(eyepoints, facial_landmarks, frame, gray):
        eye_region = np.array([(facial_landmarks.part(eyepoints[0]).x, facial_landmarks.part(eyepoints[0]).y),
                                    (facial_landmarks.part(eyepoints[1]).x, facial_landmarks.part(eyepoints[1]).y),
                                    (facial_landmarks.part(eyepoints[2]).x, facial_landmarks.part(eyepoints[2]).y),
                                    (facial_landmarks.part(eyepoints[3]).x, facial_landmarks.part(eyepoints[4]).y),
                                    (facial_landmarks.part(eyepoints[4]).x, facial_landmarks.part(eyepoints[4]).y),
                                    (facial_landmarks.part(eyepoints[5]).x, facial_landmarks.part(eyepoints[5]).y)
                                    ])

        #creat a mask
        height, width, _ = frame.shape
        mask = np.zeros((height, width), np.int8)

        # cv2.imshow("mask",mask)

        # draw the ROI
        cv2.polylines(mask, [eye_region], True, 255, 2)
        cv2.fillPoly(mask, [eye_region], 255)
        # cv2.imshow("mask",mask)

        eye = cv2.bitwise_and(gray, gray, mask=mask)
     
        #find the boudary point of ROI
        min_x = np.min(eye_region[:, 0]) # min x point of eye
        max_x = np.max(eye_region[:, 0]) # max x point of eye
        min_y = np.min(eye_region[:, 1])
        max_y = np.max(eye_region[:, 1])

        gray_eye = eye[min_y:max_y, min_x:max_x]

        _, threshold_eye = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY)
        threshold_eye= cv2.resize(threshold_eye, None, fx=5, fy=5)

        height, width = threshold_eye.shape

        left_side_threshold = threshold_eye[0: height, 0: int(width / 2)]
        left_side_white = cv2.countNonZero(left_side_threshold)

        right_side_threshold = threshold_eye[0: height, int(width / 2): width]
        right_side_white = cv2.countNonZero(right_side_threshold)

        if left_side_white==0:
            gaze_ratio=1

        elif right_side_white==0:
            gaze_ratio=1.4

        else:
            gaze_ratio = left_side_white / right_side_white

        return gaze_ratio

    #--------------function for gaze_aversion detection--------------------

    def gaze_aversion(frame):

        height = frame.shape[0]
        width = frame.shape[1]
        new_frame = np.zeros((height//5, width//2, 3), np.uint8)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=Gaze.detector(gray)

        for face in faces:

            landmarks= Gaze.predictors(gray,face)

            gaze_ratio_left_eye = Gaze.get_gaze_ratio([36,37,38,39,40,41], landmarks, frame, gray)
            gaze_ratio_right_eye= Gaze.get_gaze_ratio([42,43,44,45,46,47], landmarks, frame, gray)
            
            gaze_ratio = (gaze_ratio_left_eye + gaze_ratio_right_eye)/2

            if gaze_ratio <= 1.3:
                Gaze.status = "Right"
                new_frame[:] = (80,130,255)

            elif 1.3 < gaze_ratio < 1.6:
                Gaze.status = "Center"
                new_frame[:] = (120,225,115)

            else:
                Gaze.status = "Left"
                new_frame[:] = (235,130,100)

        return Gaze.status, new_frame

# # ------------separate test-------------------

cap= cv2.VideoCapture(0)
while True:
    ret, frame= cap.read()

    gaze_status, new_frame = Gaze.gaze_aversion(frame)

    cv2.putText(frame, gaze_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("New Frame", new_frame)
    key=cv2.waitKey(1)

    if key==27:
        break
cap.release()
cv2.destroyAllWindows()