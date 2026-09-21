import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]     
    face_samples = []
    ids = []
    
    for image_path in image_paths:
        gray_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        try:
            # Safely extract user ID integer values out of the photo file name array
            user_id = int(os.path.split(image_path)[-1].split(".")[1])
        except (ValueError, IndexError):
            continue
            
        faces = face_detector.detectMultiScale(gray_img)
        for (x, y, w, h) in faces:
            face_samples.append(gray_img[y:y+h, x:x+w])
            ids.append(user_id)
            
    return face_samples, ids

print("[SYSTEM] Reading face sample dataset folder matrix...")
if not os.path.exists('dataset') or len(os.listdir('dataset')) == 0:
    print("[ERROR] No face data found! Please run register_face.py first.")
else:
    faces, ids = get_images_and_labels('dataset')
    print("[SYSTEM] Training high-security LBPH structural matrix model...")
    recognizer.train(faces, np.array(ids))
    
    # Bake your master database blueprint file onto your local disk drive
    recognizer.write('trainer.yml')
    print(f"\n[SUCCESS] {len(np.unique(ids))} user profiles trained! Saved database to 'trainer.yml'")
