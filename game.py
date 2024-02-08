import os
import cv2
import numpy as np
import tensorflow as tf

from game import movenet, get_torso_centroid, draw_prediction_on_image, input_size

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    hstate = 0
    vstate = 0
    
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (input_size, input_size))[:, ::-1, :]
        pose = movenet(tf.expand_dims(frame, axis=0))
        torso_centroid = get_torso_centroid(pose)
        # frame = draw_prediction_on_image(frame, pose, torso_centroid=torso_centroid)
        if torso_centroid is not None:
            y, x = torso_centroid
            frame = cv2.circle(frame.copy(), (int(x * input_size), int(y * input_size)), 5, (0, 0, 255), -1)
            frame = cv2.line(frame, (int(0.4 * input_size), 0), (int(0.4 * input_size), input_size), (0, 255, 0), 1)
            frame = cv2.line(frame, (int(0.6 * input_size), 0), (int(0.6 * input_size), input_size), (0, 255, 0), 1)
            frame = cv2.line(frame, (0, int(0.5 * input_size)), (input_size, int(0.5 * input_size)), (0, 255, 0), 1)
            frame = cv2.line(frame, (0, int(0.7 * input_size)), (input_size, int(0.7 * input_size)), (0, 255, 0), 1)
            if x < 0.4:
                if hstate == 0:
                    os.system("wtype -k left")
                elif hstate == 1:
                    os.system("wtype -k left")
                    os.system("wtype -k left")
                hstate = -1
            elif x > 0.6:
                if hstate == 0:
                    os.system("wtype -k right")
                elif hstate == -1:
                    os.system("wtype -k right")
                    os.system("wtype -k right")
                hstate = 1
            else:
                if hstate == -1:
                    os.system("wtype -k right")
                elif hstate == 1:
                    os.system("wtype -k left")
                hstate = 0
            
            if y < 0.5:
                if vstate != 1:
                    os.system("wtype -k up") 
                vstate = 1
            elif y > 0.7:
                if vstate != -1:
                    os.system("wtype -k down")
                vstate = -1
            else:
                vstate = 0
            
                
        
        cv2.imshow("frame", frame)
        cv2.waitKey(1)