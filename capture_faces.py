import cv2
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cam.isOpened():
    print("ERROR: Camera not accessible")
    input()
    exit()

cascade_path = os.path.join(BASE_DIR, "haarcascade_frontalface_default.xml")
face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    print("ERROR: Haar cascade not loaded")
    input()
    exit()

face_id = input("Enter numeric user ID: ")
count = 0

dataset_path = os.path.join(BASE_DIR, "dataset", f"user_{face_id}")
os.makedirs(dataset_path, exist_ok=True)

while True:
    ret, img = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        minSize=(80, 80)
    )

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(
            os.path.join(dataset_path, f"face_{count}.jpg"),
            gray[y:y+h, x:x+w]
        )
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        img,
        f"Images: {count}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capturing Faces", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break
    if count >= 500:
        break

cam.release()
cv2.destroyAllWindows()

print("Saved images:", count)
input()
