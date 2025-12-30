import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

trainer_path = os.path.join(BASE_DIR, "trainer", "trainer.yml")
cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(trainer_path)

face_cascade = cv2.CascadeClassifier(cascade_path)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("ERROR: Camera not accessible")
    input()
    exit()

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if confidence < 70:
            text = f"User {id}"
        else:
            text = "Unknown"

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            img,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

    cv2.imshow("Face Recognition", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
