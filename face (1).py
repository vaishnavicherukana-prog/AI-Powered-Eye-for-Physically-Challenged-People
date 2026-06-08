import cv2
import os
import numpy as np
import subprocess  # For espeak
import time

# Create directory for faces if not exists
if not os.path.exists("faces"):
    os.makedirs("faces")

# Global variable to track last announced name
last_announced_name = ""
last_announce_time = 0
ANNOUNCE_COOLDOWN = 5  # seconds between announcements for the same person

# Function to announce name using espeak
def announce_name(name):
    global last_announced_name, last_announce_time
    
    current_time = time.time()
    
    # Only announce if it's a different name or enough time has passed
    if (name != last_announced_name or 
        (name == last_announced_name and current_time - last_announce_time > ANNOUNCE_COOLDOWN)):
        
        try:
            # Use espeak to announce the name
            subprocess.run(['espeak', '-s', '150', name], 
                         stdout=subprocess.DEVNULL, 
                         stderr=subprocess.DEVNULL)
            print(f"[ANNOUNCEMENT] Detected: {name}")
            
            # Update tracking variables
            last_announced_name = name
            last_announce_time = current_time
            
        except Exception as e:
            print(f"[ERROR] Could not announce name: {e}")

# Step 1: Capture training images
def capture_faces():
    person_name = input("Enter the name of the person: ").strip()
    person_path = os.path.join("faces", person_name)
    os.makedirs(person_path, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)
    count = 0

    print("\n[INFO] Capturing faces... Look at the camera. Press ESC to stop early.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            file_path = os.path.join(person_path, f"{count}.jpg")
            cv2.imwrite(file_path, face)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Images: {count}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Capture Faces", frame)

        if cv2.waitKey(1) & 0xFF == 27 or count >= 150:  
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Captured {count} face images for '{person_name}'.\n")

# Step 2: Train and recognize faces
def recognize_faces():
    global last_announced_name, last_announce_time
    
    # Reset announcement tracking
    last_announced_name = ""
    last_announce_time = 0
    
    train_dir = "faces/"
    faces = []
    labels = []
    label_dict = {}
    label_count = 0

    # Load training images
    for person_name in os.listdir(train_dir):
        person_path = os.path.join(train_dir, person_name)
        if not os.path.isdir(person_path):
            continue
        label_dict[label_count] = person_name
        for image_name in os.listdir(person_path):
            img_path = os.path.join(person_path, image_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (200, 200))
                faces.append(img)
                labels.append(label_count)
        label_count += 1

    if len(faces) == 0:
        print("[ERROR] No training images found. Please capture faces first.")
        return

    print(f"[INFO] Training model on {len(faces)} images of {len(label_dict)} people...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    print("[INFO] Training complete. Starting real-time recognition...\n")

    cap = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in detected_faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (200, 200))
            label, confidence = recognizer.predict(roi_gray)

            if confidence < 80:
                name = label_dict[label]
                text = f"{name} ({round(confidence, 1)})"
                color = (0, 255, 0)
                
                # Announce the detected person's name
                announce_name(name)
            else:
                name = "Unknown"
                text = f"{name} ({round(confidence, 1)})"
                color = (0, 0, 255)

            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()


# Main Menu
def main():
    print("""
==============================
 FACE RECOGNITION SYSTEM
==============================
1. Capture new faces
2. Train & Recognize faces
3. Exit
""")

    choice = input("Enter your choice (1/2/3): ").strip()

    if choice == "1":
        capture_faces()
        main()
    elif choice == "2":
        recognize_faces()
        main()
    elif choice == "3":
        print("Exiting...")
        exit()
    else:
        print("Invalid choice. Try again.\n")
        main()


if __name__ == "__main__":
    main()
