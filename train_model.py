import cv2
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

dataset_dir = os.path.join(BASE_DIR, "dataset")
trainer_dir = os.path.join(BASE_DIR, "trainer")
os.makedirs(trainer_dir, exist_ok=True)

cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(cascade_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []

for user_folder in os.listdir(dataset_dir):
    user_path = os.path.join(dataset_dir, user_folder)
    if not os.path.isdir(user_path):
        continue

    label = int(user_folder.split("_")[1])

    for image_name in os.listdir(user_path):
        image_path = os.path.join(user_path, image_name)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        faces.append(img)
        labels.append(label)

if len(faces) == 0:
    print("ERROR: No training images found")
    input()
    exit()

recognizer.train(faces, np.array(labels))
recognizer.save(os.path.join(trainer_dir, "trainer.yml"))

print("Training completed")
input()
