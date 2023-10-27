# CSE 666 Programming Assignment 1
This script performs face, emotion, gender, pose detection and recognition on images using the following packages:

1. mtcnn
2. tensorflow
3. facenet-pytorch
4. pandas
5. cv2
6. json
7. numpy
8. torch
9. dlib

## Installation


You need to install the required packages by running the following commands:

	pip install mtcnn tensorflow facenet-pytorch pandas cv2 json numpy torch dlib

## Usage
To use the script, run the following command:

	src/main.py -d /path/to/assignment/
    
    
## Outputs
The script outputs five images with face, emotion, gender, pose detection and recognition results:

1. **marked_image.jpg**: the original image with bounding boxes drawn around each detected face
2. **marked_emotions_image.jpg**: the original image with bounding boxes and detected emotions labels drawn around each detected face
3. **marked_genders_image.jpg**: the original image with bounding boxes and detected genders labels drawn around each detected face
4. **marked_poses_image.jpg**: the original image with bounding boxes and estimated pose labels drawn around each detected face
5. **faces_detected_image.jpg**: the original image with bounding boxes and face names written on each detected face

There are 4 json files -
1. **bounding_boxes.json** : contains bounding box details (x,y,w,h)
2. **detected_emotions.json** : contains the probabilities for each emotion and for all the detected faces 
3. **detected_genders.json** : contains the probabilities for each gender and for all the detected faces 
4. **face_embeddings.json** : contains the face embeddings for all the detected faces

