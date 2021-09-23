#coding=utf-8  
import cv2

class Smile:

    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_detetcor = cv2.CascadeClassifier('haarcascade_smile.xml')

    label = ''

    def smile(frame):

        frame_gray =  cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = Smile.face_detector.detectMultiScale(frame_gray)
        
        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

            the_face = frame[y:y+h, x:x+w]

            face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

            smiles = Smile.smile_detetcor.detectMultiScale(face_grayscale, scaleFactor = 1.5, minNeighbors = 17)

            if len(smiles) > 0:
                Smile.label = 'smile'

            else:
                Smile.label =  'no smile'

        return Smile.label

#----------------seperate test-------------------

cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    label = Smile.smile(frame)

    cv2.putText(frame, 'smiling:' + str(label), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    
    cv2.imshow('Smile detector', frame)

    # cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()