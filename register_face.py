import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not os.path.exists('dataset'):
    os.makedirs('dataset')

print("--- SIMS SECURE ENROLLMENT TERMINAL ---")
user_id = input('Assign unique numeric ID (e.g., 1, 2, 3): ')
user_name = input('Enter person\'s full name: ')

# 📝 THE NAME LOGGER ADDITION: Records the typed name into a local text database instantly!
with open("students.txt", "a") as f:
    f.write(f"{user_id},{user_name}\n")

print(f"\n[SYSTEM] Capturing 50 brightness-normalized face frames for {user_name}...")
count = 0

while True:
    ret, cap_frame = cap.read()
    if not ret:
        break
        
    cap_frame = cv2.resize(cap_frame, (640, 480))
    gray = cv2.cvtColor(cap_frame, cv2.COLOR_BGR2GRAY)
    equalized_gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(equalized_gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
    
    for (x, y, w, h) in faces:
        count += 1
        file_path = f"dataset/User.{user_id}.{count}.jpg"
        cv2.imwrite(file_path, equalized_gray[y:y+h, x:x+w])
        
        cv2.rectangle(cap_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(cap_frame, f"Captured: {count}/50", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.waitKey(60)
        
    cv2.imshow('SIMS: Secure Enrollment Terminal', cap_frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n[SUCCESS] 50 images stored for {user_name} under ID: {user_id}!")
