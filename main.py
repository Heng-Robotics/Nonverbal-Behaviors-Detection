#coding=utf-8  
import numpy as np
import cv2
import Blinking
import Headpose
import Smile
import Gaze_aversion


cap = cv2.VideoCapture(0) # open camera
ret, frame_block = cap.read() # get the first frame

height = frame_block.shape[0]
width = frame_block.shape[1]

text_region = np.zeros((height//5, width//2, 3), np.uint8) # create the region of text
# text_region.fill(255)

gaze_show = np.zeros((height//5, width//2, 3), np.uint8) # create the region of gaze_zversion hint

count_img = np.zeros((height//5, width, 3), np.uint8) # create the region of up half

show_img = np.zeros((height+height//5, width, 3), np.uint8) # create the whole region

while True:
    ret, frame = cap.read()

    gaze, new_frame = Gaze_aversion.Gaze.gaze_aversion(frame) # function for gaze aversion detection

    blinking = Blinking.Blink.blink(frame) # function for blinking detection 

    headpose = Headpose.HeadPose.headpose(frame) # function for head pose  detecttion
    
    smile = Smile.Smile.smile(frame)# function for smile detection

    gaze_show = new_frame 

    count_img[0 : height//5, 0 : width//2] = text_region
    count_img[0 : height//5, width//2 : width] = gaze_show

    show_img[0 : height//5, 0 : width] = count_img
    show_img[height//5 : (height + height//5), 0 : width] = frame

    cv2.putText(show_img, "Gaze_aversion:" + gaze, (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.7,(102,204,205),1)
    cv2.putText(show_img, "Blinks:" + str(blinking), (20, 45), cv2.FONT_HERSHEY_DUPLEX, 0.7, (102,204,205), 1)
    cv2.putText(show_img, "Head pose:" + headpose,  (20, 70), cv2.FONT_HERSHEY_DUPLEX, 0.7,(102,204,205),1)
    cv2.putText(show_img, "Smiling:" + smile, (20, 95), cv2.FONT_HERSHEY_DUPLEX, 0.7,(102,204,205),1)
 
    # cv2.imshow("New Frame", new_frame)
    cv2.imshow("Frame", show_img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()