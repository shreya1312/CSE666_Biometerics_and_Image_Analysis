#!/usr/bin/python3

import pandas as pd
import cv2
import json
from mtcnn import MTCNN
import numpy as np
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential, load_model
import torch
from facenet_pytorch import InceptionResnetV1
import dlib
import torchvision.transforms as transforms
import argparse

# change the root_dir
root_dir = '/content/drive/MyDrive/50471594_shreyadh_assignment01/'

argParser = argparse.ArgumentParser()
argParser.add_argument("-d", default=root_dir, help="assignment root directory")
args = argParser.parse_args()
root_dir = args.d

models_dir = root_dir + 'src/model_weights/'
results_dir = root_dir + 'results/'
data_dir = root_dir + 'data/'

# 7: Face recognition
data = pd.read_csv(data_dir + 'congress.tsv', delimiter='\t')

detector = MTCNN()
embedder = InceptionResnetV1(pretrained='vggface2').eval()

congresspeople_embeddings = {}

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

congresspeople_embeddings = {}

for index, row in data.iterrows():
    image_fn = data_dir + str(row[0])
    image = cv2.imread(image_fn.replace(" ", ""), cv2.IMREAD_UNCHANGED)
    if image.shape[2] > 3:
      image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    faces = detector.detect_faces(image)

    if faces is None:
        continue

    for face in faces:
        x, y, w, h = face["box"]
        face_image = image[y:y + h, x:x + w]
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image = cv2.resize(face_image, (160, 160))
        face_image_tensor = transforms.functional.to_tensor(face_image).unsqueeze(0)
        embedding = embedder(face_image_tensor).detach().numpy()[0]
        congressperson_name = row[1]
        if congressperson_name not in congresspeople_embeddings:
            congresspeople_embeddings[congressperson_name] = [embedding]
        else:
            congresspeople_embeddings[congressperson_name].append(embedding)


image_bb = cv2.imread(root_dir + "count_faces.jpg")
image_em = cv2.imread(root_dir + "count_faces.jpg")
image_gen = cv2.imread(root_dir + "count_faces.jpg")
image_pose = cv2.imread(root_dir + "count_faces.jpg")
image_det = cv2.imread(root_dir + "count_faces.jpg")

# 2
bounding_boxes = []
results = detector.detect_faces(image_bb)

# 3
emotion_model = load_model(models_dir + "fer2013_mini_XCEPTION.119-0.65.hdf5")
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
detected_emotions = []

# 4
gender_model = load_model(models_dir + "gender_mini_XCEPTION.21-0.95.hdf5")
detected_genders = []

# 5
pose_predictor = dlib.shape_predictor(models_dir +  'shape_predictor_68_face_landmarks.dat')

# 6
face_embeddings = []

for result in results:
    # bounding box
    x, y, w, h = result["box"]

    # draw bounding box on each image
    for img in (image_bb, image_em, image_gen, image_pose, image_det):
      cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # storing bounding box details
    bounding_boxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})

    ################## sentiment analysis ####################
    roi = image_bb[y:y + h, x:x + w]
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = cv2.resize(roi, (48, 48))
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    roi = roi / 255.0

    preds = emotion_model.predict(roi)[0]
    emotion_probabilities = dict(zip(emotion_labels, preds))
    emotion_probabilities = {k: float(v) for k, v in emotion_probabilities.items()}
    detected_emotions.append(emotion_probabilities)

    # get emotion with max probabilty and draw label
    max_emotion = max(emotion_probabilities, key=emotion_probabilities.get)
    label = f"{max_emotion}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = x + (w - label_size[0]) // 2
    label_y = y - label_size[1] - 5
    cv2.putText(image_em, label, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    ################## gender detection #################################
    roi_gender = image_bb[y:y + h, x:x + w]
    roi_gender = cv2.cvtColor(roi_gender, cv2.COLOR_BGR2GRAY)
    roi_gender = cv2.resize(roi_gender, (64, 64))
    roi_gender = np.expand_dims(roi_gender, axis=-1)
    roi_gender = np.expand_dims(roi_gender, axis=0)
    roi_gender = roi_gender / 255.0

    gender_pred = gender_model.predict(roi_gender)[0]
    gender_probabilities = {"Male": float(gender_pred[0]), "Female": float(gender_pred[1])}
    detected_genders.append(gender_probabilities)

    # get gender with max probabilty and draw label
    max_gender = max(gender_probabilities, key=gender_probabilities.get)
    label = f"{max_gender}"
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = x + (w - label_size[0]) // 2
    label_y = y - label_size[1] - 5
    cv2.putText(image_gen, label, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    ################### pose estimation ###########################
    roi_pose = image_bb[y:y + h, x:x + w]
    roi_pose = cv2.cvtColor(roi_pose, cv2.COLOR_BGR2GRAY)
    landmarks = pose_predictor(roi_pose, dlib.rectangle(0, 0, w, h))
    landmarks = np.array([(landmarks.part(i).x + x, landmarks.part(i).y + y) for i in range(68)])

    # get head pose
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ], dtype=np.float32)
    image_points = np.array([
        landmarks[30],
        landmarks[8],
        landmarks[36],
        landmarks[45],
        landmarks[48],
        landmarks[54]
    ], dtype=np.float32)

    focal_length = image_bb.shape[1]
    center = (image_bb.shape[1] / 2, image_bb.shape[0] / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.double)
    dist_coeffs = np.zeros((4, 1))
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = -cv2.decomposeProjectionMatrix(proj_matrix)[6]

    yaw = eulerAngles[1]
    if -30 < yaw < 30:
        pose_class="straight"
    elif yaw < -30:
        pose_class="left"
    else:
        pose_class="right"

    label = f"{pose_class}"
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = x + (w - label_size[0]) // 2
    label_y = y - label_size[1] - 5
    cv2.putText(image_pose, label, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    ################# feature extraction ########
    roi_feature = image_bb[y:y + h, x:x + w]
    roi_feature = cv2.cvtColor(roi_feature, cv2.COLOR_BGR2RGB)
    roi_feature = cv2.resize(roi_feature, (160, 160))
    roi_feature = transforms.functional.to_tensor(roi_feature).unsqueeze(0)
    embedding = embedder(roi_feature).detach().numpy()[0]

    # compare with congresspeople embeddings
    label="Unknown"
    for congressperson_name, congressperson_embeddings in congresspeople_embeddings.items():
      for congressperson_embedding in congressperson_embeddings:
        distance = np.linalg.norm(embedding - congressperson_embedding)
        if distance < 0.89:
          label=f"{congressperson_name}"
          break
      if label != "Unknown":
        break
    label_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    label_x = x + (w - label_size[0]) // 2
    label_y = y - label_size[1] - 5
    cv2.putText(image_det, label, (label_x, label_y), font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    face_embeddings.append(embedding.tolist())
    
# number of detected faces
num_faces = len(bounding_boxes)
print('Number of faces detected:', num_faces)

# 2
cv2.imwrite(results_dir + "marked_image.jpg", image_bb)

# 3
cv2.imwrite(results_dir + "marked_emotions_image.jpg", image_em)

# 4
cv2.imwrite(results_dir + "marked_genders_image.jpg", image_gen)

# 5
cv2.imwrite(results_dir + "marked_poses_image.jpg", image_pose)

# 7
cv2.imwrite(results_dir + "faces_detected_image.jpg", image_det)

# Write the bb details, emotions,gender probabilties,face embeddings to files
with open(results_dir + "bounding_boxes.json", "w") as f:
    json.dump(bounding_boxes, f)

with open(results_dir + "detected_emotions.json", "w") as f:
    json.dump(detected_emotions, f)

with open(results_dir + "detected_genders.json", "w") as f:
    json.dump(detected_genders, f)

with open(results_dir + "face_embeddings.json", "w") as f:
    json.dump(face_embeddings, f)

# Task-2 : Evaluating the model
gt_img = cv2.imread(root_dir + 'results/annotated_image.jpg')
output_img = cv2.imread(root_dir + 'results/marked_image.jpg')
gt_gray = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY)
output_gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
gt_mask = cv2.threshold(gt_gray, 50, 255, cv2.THRESH_BINARY)[1]
output_mask = cv2.threshold(output_gray, 50, 255, cv2.THRESH_BINARY)[1]

tp = np.logical_and(gt_mask, output_mask).sum()
fp = np.logical_and(np.logical_not(gt_mask), output_mask).sum()
fn = np.logical_and(gt_mask, np.logical_not(output_mask)).sum()

precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1_score = 2 * precision * recall / (precision + recall)

print('Precision:', precision)
print('Recall:', recall)
print('F1-score:', f1_score)
