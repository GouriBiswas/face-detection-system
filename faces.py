import cv2
import os
import time
import numpy as np

# Load Haarcascade Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Create LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Folder to store face images
save_folder = "faces"
os.makedirs(save_folder, exist_ok=True)

def capture_faces():
    """ Capture face images for training """
    person_name = input("Enter your name: ").strip()
    person_folder = os.path.join(save_folder, person_name)
    os.makedirs(person_folder, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Look at the camera... Capturing images!")

    image_count = 0
    while image_count < 100:  # Capture 100 images
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in detected_faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))

            img_path = os.path.join(person_folder, f"face_{image_count}.jpg")
            cv2.imwrite(img_path, face_resized)  # Save face image
            image_count += 1
            time.sleep(0.1)

        cv2.imshow("Capturing Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {image_count} images for {person_name}!")

    train_faces()  # Train the model after capturing images

def train_faces():
    """ Train the face recognition model """
    faces, labels = [], []
    label_dict = {}

    people = os.listdir(save_folder)
    for label, person_name in enumerate(people):
        person_folder = os.path.join(save_folder, person_name)
        if not os.path.isdir(person_folder):
            continue

        label_dict[label] = person_name  # Map label to name
        for img_name in os.listdir(person_folder):
            img_path = os.path.join(person_folder, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label)

    if len(faces) == 0:
        print("No face data found! Please capture images first.")
        return

    faces = np.array(faces, dtype=np.uint8)
    labels = np.array(labels, dtype=np.int32)

    recognizer.train(faces, labels)
    recognizer.save("trained_faces.yml")

    np.save("labels_dict.npy", label_dict)
    print("Training complete! Model saved.")

def recognize_face():
    """ Recognize face using trained model """
    if not os.path.exists("trained_faces.yml"):
        print("Error: No trained model found! Please train first.")
        return

    recognizer.read("trained_faces.yml")

    # Load label dictionary
    if os.path.exists("labels_dict.npy"):
        label_dict = np.load("labels_dict.npy", allow_pickle=True).item()
    else:
        label_dict = {}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("Starting face recognition...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))

        for (x, y, w, h) in detected_faces:
            face_img = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face_img, (100, 100))

            label, confidence = recognizer.predict(face_resized)

            name = label_dict.get(label, "Unknown") if confidence < 80 else "Unknown"

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Main Menu
print("1. Train New Face")
print("2. Recognize Face")
choice = input("Enter your choice: ")

if choice == "1":
    capture_faces()
elif choice == "2":
    recognize_face()
else:
    print("Invalid choice! Exiting.")