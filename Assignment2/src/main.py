#from google.colab import drive
#drive.mount('/content/drive')

#!pip install -q flatbuffers==2.0.0
#!pip install -q mediapipe==0.9.1

import tensorflow as tf
import os
import cv2
import numpy as np
import mediapipe as mp

data_dir = "../data"
model_dir = "../model"
train_dir = os.path.join(data_dir, "train")
label_dir = os.path.join(data_dir, "label")
results_dir = "../results/train/"


with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:


    for filename in os.listdir(train_dir):
        
        image = cv2.imread(os.path.join(train_dir, filename))
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = hands.process(image)

        h, w, _ = image.shape

        for hand_landmarks in results.multi_hand_landmarks:
            
            landmarks = [[int(l.x * w), int(l.y * h)] for l in hand_landmarks.landmark]
            
            fingers = {
                'thumb': [landmarks[2], landmarks[3], landmarks[4]],
                'index': [landmarks[5], landmarks[6], landmarks[7]],
                'middle': [landmarks[9], landmarks[10], landmarks[11]],
                'ring': [landmarks[13], landmarks[14], landmarks[15]],
                'pinky': [landmarks[17], landmarks[18], landmarks[19]]
            }

            for finger_points in fingers.values():
                

                fingertip = finger_points[2]
                

                bbox_size = 200
                bbox_half = int(bbox_size / 2)
                bbox_tl = (fingertip[0] - bbox_half, fingertip[1] - bbox_half - 200)  
                bbox_br = (fingertip[0] + bbox_half, fingertip[1] + bbox_half - 200)  
                
                cv2.rectangle(image, bbox_tl, bbox_br, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(results_dir, filename), image)

# do this for test images
test_dir = os.path.join(data_dir, "test")
results_dir = "../results/test/"

with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.7) as hands:
    for filename in os.listdir(test_dir):
        
        image = cv2.imread(os.path.join(test_dir, filename))
        
        try:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
         continue   

        results = hands.process(image)

        h, w, _ = image.shape

        if results.multi_hand_landmarks is None:
          continue
       
        for hand_landmarks in results.multi_hand_landmarks:
            
            landmarks = [[int(l.x * w), int(l.y * h)] for l in hand_landmarks.landmark]

            fingers = {
                'thumb': [landmarks[2], landmarks[3], landmarks[4]],
                'index': [landmarks[5], landmarks[6], landmarks[7]],
                'middle': [landmarks[9], landmarks[10], landmarks[11]],
                'ring': [landmarks[13], landmarks[14], landmarks[15]],
                'pinky': [landmarks[17], landmarks[18], landmarks[19]]
            }

            for finger_points in fingers.values():
                
                fingertip = finger_points[2]
                
                bbox_size = 200
                bbox_half = int(bbox_size / 2)
                bbox_tl = (fingertip[0] - bbox_half, fingertip[1] - bbox_half - 200)  
                bbox_br = (fingertip[0] + bbox_half, fingertip[1] + bbox_half - 200)  
                
                cv2.rectangle(image, bbox_tl, bbox_br, (0, 255, 0), 2)

        cv2.imwrite(os.path.join(results_dir, filename), image)
