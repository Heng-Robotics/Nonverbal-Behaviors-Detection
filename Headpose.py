import cv2
import numpy as np
import dlib
from imutils import face_utils
from numpy.core.records import array

class HeadPose:

    x_movement = 0
    y_movement = 0

    gesture_show = 10 #number of frames a gesture is shown
    gesture_threshold = 40 #define movement threshodls
    error_threshold = 30
    
    gesture_flag = True
    gesture = ''

    face_found = False #judgement condition of finding the face in the image
    gray = np.zeros(())
    p0 = [0,0]
    p1 = [0,0]

    #use the key points of dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    #------------function to get coordinates------------------

    def get_coords(p):

        return int(p[0]), int(p[1])
    
    #-----------function to detect the first point------------------

    def first_frame(frame):

        while not HeadPose.face_found:
            HeadPose.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = HeadPose.detector(HeadPose.gray, 0)

            for face in faces:
                shape = HeadPose.predictor(HeadPose.gray, face) # detect feature points
                points = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array

                left_point  = points[42] #NO.43 is the point of right corner of left eye
                right_point = points[39] #NO.40 is the point of left corner of right eye

                HeadPose.p0 = [int((left_point[0]+right_point[0])/2), int((left_point[1]+right_point[1])/2)] 

            #     print("p00:", HeadPose.p0[0], HeadPose.p0[1])
            
            # print("p01:", HeadPose.p0[0], HeadPose.p0[1])
            HeadPose.face_found = True


    #-----------function to detect head pose------------------

    def headpose(frame):

        HeadPose.first_frame(frame)

        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # cv2.imshow("img", gray_img)
        # cv2.waitKey(10000)
            
        faces = HeadPose.detector(gray_img, 0)     

        for face in faces:
            shape = HeadPose.predictor(gray_img, face) # detect feature points
            points = face_utils.shape_to_np(shape) # convert the facial landmark (x, y)-coordinates to a NumPy array

            left_point = points[42]
            right_point = points[39]

            HeadPose.p1 = [int((left_point[0]+right_point[0])/2), int((left_point[1]+right_point[1])/2)]
      
        cv2.circle(frame, (HeadPose.p1[0], HeadPose.p1[1]), 4, (0,0,255), -1)
        cv2.circle(frame, (HeadPose.p0[0], HeadPose.p0[1]), 4, (255,0,0))

        # print("p0:",HeadPose.p0[0], HeadPose.p0[1])
        # print("p1:",HeadPose.p1[0], HeadPose.p1[1])

        HeadPose.x_movement += abs(HeadPose.p0[0]-HeadPose.p1[0])
        HeadPose.y_movement += abs(HeadPose.p0[1]-HeadPose.p1[1])

        # print(HeadPose.x_movement, HeadPose.y_movement)
        
        #judgement of yes or no
        # if HeadPose.gesture_flag and HeadPose.x_movement > HeadPose.gesture_threshold and HeadPose.y_movement < HeadPose.error_threshold:
        if HeadPose.gesture_flag and HeadPose.x_movement > HeadPose.gesture_threshold:

            HeadPose.gesture = 'No'
            HeadPose.gesture_flag = False

        if HeadPose.gesture_flag and HeadPose.y_movement > HeadPose.gesture_threshold:
            HeadPose.gesture = 'Yes'
            HeadPose.gesture_flag = False

        if not HeadPose.gesture_flag and HeadPose.gesture_show > 0:
            HeadPose.gesture_show -=1
        
        if HeadPose.gesture_show == 0:
            HeadPose.gesture_flag = True
            HeadPose.x_movement = 0
            HeadPose.y_movement = 0
            HeadPose.gesture_show = 10 #number of frames a gesture is shown
        
        # print(HeadPose.gesture_show)
        HeadPose.p0 = HeadPose.p1
        return HeadPose.gesture 


#----------------separate test-------------------

cap= cv2.VideoCapture(0)

while True:
    ret, frame= cap.read()

    label = HeadPose.headpose(frame)

    cv2.putText(frame, "Headpose:" + str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

    cv2.imshow("Frame", frame)

    key=cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()