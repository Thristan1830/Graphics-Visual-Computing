import cv2
import numpy as np
import os
import sys

# ---  EXECUTABLE-SAFE PATH ROOT ---
if getattr(sys, 'frozen', False):
    base_dir = os.path.dirname(os.path.abspath(sys.executable))
else:
    base_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("| CAMPUS BIOMETRIC MODEL TRAINER ENGINE                     |")
    print("=" * 60)

    dataset_path = os.path.join(base_dir, "dataset")
    model_path = os.path.join(base_dir, "trainer.yml")

    if not os.path.exists(dataset_path):
        print(f"[ERROR] Dataset directory not found at: {dataset_path}")
        print("[ACTION REQUIRED] Please register faces using Option 1 first.")
        return

    image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    if len(image_paths) == 0:
        print(f"[ERROR] No face images found in dataset directory: {dataset_path}")
        print("[ACTION REQUIRED] Run '1. REGISTER NEW FACE' to capture images before training.")
        return

    print(f"[SYSTEM] Found {len(image_paths)} face sample(s). Compiling LBPH biometric matrix...")

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if os.path.exists(os.path.join(base_dir, 'haarcascade_frontalface_default.xml')):
        cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')

    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty() and hasattr(sys, '_MEIPASS'):
        alt_path = os.path.join(sys._MEIPASS, 'cv2', 'data', 'haarcascade_frontalface_default.xml')
        if os.path.exists(alt_path):
            face_cascade = cv2.CascadeClassifier(alt_path)

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_samples = []
    ids = []

    for path in image_paths:
        try:
            filename = os.path.basename(path)
            parts = filename.split('.')
            if len(parts) >= 3 and parts[0] == "User":
                user_id = int(parts[1])
            else:
                continue

            # Read image directly as grayscale via OpenCV (No PIL needed)
            img_np = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img_np is None:
                continue

            faces = face_cascade.detectMultiScale(img_np)
            if len(faces) == 0:
                face_samples.append(img_np)
                ids.append(user_id)
            else:
                for (x, y, w, h) in faces:
                    face_samples.append(img_np[y:y+h, x:x+w])
                    ids.append(user_id)
        except Exception as e:
            print(f"[WARNING] Skipping image {path}: {e}")

    if len(face_samples) == 0:
        print("[ERROR] Could not extract valid biometric features from dataset.")
        return

    print(f"[SYSTEM] Training LBPH model on {len(face_samples)} face array(s)...")
    recognizer.train(face_samples, np.array(ids))
    recognizer.write(model_path)

    print("-" * 60)
    print(f"[SUCCESS] Biometric model weight file created successfully at:\n  {model_path}")
    print("-" * 60)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL ERROR] Trainer engine crashed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        input("\n[TERMINAL PAUSE] Press Enter to return to main terminal...")