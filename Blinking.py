#coding=utf-8  
import numpy as np 
import cv2
import dlib
from scipy.spatial import distance
from imutils import face_utils

class Blink:

    # Numbers of corresponding feature points
    RIGHT_EYE_START = 37 - 1
    RIGHT_EYE_END = 42 - 1
    LEFT_EYE_START = 43 - 1
    LEFT_EYE_END = 48 - 1

    frame_counter = 0 # Continuous frame count
    blink_counter = 0 # Blinking count
    # str_counter = ""

    EYE_AR_THRESH = 0.30 # Threshold of EAR
    EYE_AR_CONSEC_FRAMES = 3 # Blinking will happen when EAR smaller than the threshold in continuous frames over this number
    detector = dlib.get_frontal_face_detector() # Human face detector
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat') # Facial feature points detector
    
    #------------function to calculate EAR----------------

    def eye_aspect_ratio(eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        ear = (A + B) / (2.0 * C)
        return ear

    #-----------function to detect blinking---------------

    def blink(frame):

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = Blink.detector(gray, 0) # facial detection

        for face in faces: # for every face
            shape = Blink.predictor(gray, face) # detect feature points
            points = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array

            leftEye = points[Blink.LEFT_EYE_START:Blink.LEFT_EYE_END + 1] # feature points of left eye
            rightEye = points[Blink.RIGHT_EYE_START:Blink.RIGHT_EYE_END + 1] # feature points of right eye

            # text_eye = points[Blink.LEFT_EYE_START]
            
            # print(text_eye[0], text_eye[1])

            leftEAR = Blink.eye_aspect_ratio(leftEye) # calculate the EAR of left eye
            rightEAR = Blink.eye_aspect_ratio(rightEye) # calculate the EAR of right eye
            ear = (leftEAR + rightEAR) / 2.0 # calculate the mean of both left and right eyes

            leftEyeHull = cv2.convexHull(leftEye) # the contour of left eye
            rightEyeHull = cv2.convexHull(rightEye) # the contour of right eye
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1) # draw the contour of left eye
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1) # draw the contour of right eye

            if ear < Blink.EYE_AR_THRESH:
                Blink.frame_counter += 1
            else:
                if Blink.frame_counter >= Blink.EYE_AR_CONSEC_FRAMES:
                    Blink.blink_counter += 1
                Blink.frame_counter = 0
            #Blink.str_counter = str(Blink.frame_counter)

        return Blink.blink_counter

# #---------------separate test-------------------

# cap= cv2.VideoCapture(0)
# while True:
#     ret, frame= cap.read()

#     counter = Blink.blink(frame)

#     cv2.putText(frame, "Blinks:" + str(counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

#     cv2.imshow("Frame", frame)

#     key=cv2.waitKey(1)

#     if key==27:
#         break
# cap.release()
# cv2.destroyAllWindows()
