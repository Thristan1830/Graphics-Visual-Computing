import cv2
import numpy as np
import datetime
import os
import time

# --- 📅 CONFIGURATION ZONE: CAMPUS RULES (MON-THU: REGULAR, FRI-SAT: MAROON) ---
current_day = datetime.datetime.now().strftime("%A")

role_color_rules = {
    "Monday":    {"Student": "Regular", "Teacher": "Regular", "Staff": "Regular"},
    "Tuesday":   {"Student": "Regular", "Teacher": "Regular", "Staff": "Regular"},
    "Wednesday": {"Student": "Regular", "Teacher": "Regular", "Staff": "Regular"},
    "Thursday":  {"Student": "Regular", "Teacher": "Regular", "Staff": "Regular"},
    "Friday":    {"Student": "Maroon",  "Teacher": "Maroon",  "Staff": "Maroon"},
    "Saturday":  {"Student": "Maroon",  "Teacher": "Maroon",  "Staff": "Maroon"},
    "Sunday":    {"Student": "Regular", "Teacher": "Regular", "Staff": "Regular"}
}

# --- 👤 SYSTEM USER REGISTRY MATRIX ---
names_database = {
    1: "Faissal",
    2: "Thristan",   # Assigned to your face profile ID
    3: "Member 3",
    4: "Member 4",
    5: "Member 5"
}

# --- 🚀 INITIALIZATION ZONE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists('trainer.yml'):
    recognizer.read('trainer.yml')
    print("[SYSTEM] Secure biometric face database loaded successfully!")
else:
    print("[WARNING] trainer.yml not found. Defaulting to GUEST mode.")

blink_tracking_database = {}
attendance_log = {}

# --- 🌐 5-CAMERA NETWORK SIMULATION ROUTING (ORGANIZED IN 2x3 GRID) ---
camera_sources = {
    "GATE 01 (Main)": 0,    # Your master laptop webcam channel
    "GATE 02 (Mobile)": 1,  # Your Iriun wireless phone node stream
    "GATE 03 (Library)": 2, # Offline fallback slots
    "GATE 04 (Gym)": 3,
    "GATE 05 (Exit)": 4
}

caps = {}
for name, src in camera_sources.items():
    cap = cv2.VideoCapture(src)
    time.sleep(0.5) # Hardware driver warm-up buffer delay
    if cap.isOpened():
        caps[name] = cap
        print(f"[NETWORK SYSTEM] Stream link securely bound to physical index: {name}")
    else:
        if cap is not None:
            cap.release()
        caps[name] = None
        print(f"[NETWORK INFRASTRUCTURE] {name} index initialized as virtual offline node matrix.")

print(f"\n[SIMS INITIALIZED] Active System Day: {current_day.upper()}")
print("Press 'q' in the window screen to safely close the terminal loop.\n")

while True:
    frames = {}
    for name, cap in caps.items():
        ret = False
        frame = None
        
        if cap is not None:
            ret, frame = cap.read()
            
        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype="uint8")
            cv2.rectangle(frame, (15, 15), (625, 465), (25, 25, 25), -1)
            cv2.putText(frame, f"{name}: OFFLINE", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "STANDBY CONNECTIVITY LOOP", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
        else:
            frame = cv2.resize(frame, (640, 480))
            
            # --- 🪖 CAMERA SENSITIVITY & MULTI-USER HUD ENGINE ---
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized_gray = cv2.equalizeHist(gray) # Brightness protection fix
            
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            # Scale-invariant face detection parameters
            faces = face_cascade.detectMultiScale(equalized_gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                roi_gray = equalized_gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                
                # Default identity parameters fresh per iteration pass loop
                user_identity = "GUEST"
                user_role = "Guest"
                system_status = "SCANNING CAMPUS ENTRANT..."
                hud_color = (0, 255, 255)
                
                if os.path.exists('trainer.yml'):
                    user_id, confidence = recognizer.predict(roi_gray)
                    if confidence < 65:
                        # 🛡️ THE SECURITY GUARD UPGRADE: Reads your dynamic student notepad natively!
                        dynamic_names = {}
                        if os.path.exists("students.txt"):
                            with open("students.txt", "r") as f:
                                for db_line in f:
                                    if "," in db_line:
                                        db_id, db_name = db_line.strip().split(",", 1)
                                        dynamic_names[int(db_id)] = db_name

                        user_identity = dynamic_names.get(user_id, f"ID: {user_id}")
                        user_role = "Student"
                        system_status = "ACCESS GRANTED"
                        hud_color = (0, 255, 0) # Solid Green Pass
                    else:
                        user_identity = "GUEST"
                        user_role = "Guest"
                        system_status = "GUEST DETECTED"
                        hud_color = (255, 165, 0) # Warning Orange Pass

                # --- 🎨 DYNAMIC FLOATING COORD OVERLAYS ANCHORED DIRECTLY TO FACE BOXES ---
                cv2.putText(frame, f"ID: {user_identity.upper()}", (x, y - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 2)
                cv2.putText(frame, f"ROLE: {user_role.upper()}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1)
                cv2.putText(frame, f"STATUS: {system_status}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1)


            # Global Node Identifier anchors top corner cleanly
            cv2.putText(frame, f"GATEWAY NODE: {name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        frames[name] = frame

    # --- 🖥️ SURVEILLANCE SECURITY CONTROL GRID ASSEMBLY ---
    cam1 = frames["GATE 01 (Main)"]
    cam2 = frames["GATE 02 (Mobile)"]
    cam3 = frames["GATE 03 (Library)"]
    cam4 = frames["GATE 04 (Gym)"]
    cam5 = frames["GATE 05 (Exit)"]

    blank_hub = np.zeros((480, 640, 3), dtype="uint8")
    cv2.rectangle(blank_hub, (15, 15), (625, 465), (35, 25, 15), -1)
    cv2.putText(blank_hub, "SIMS MATRIX SERVER", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(blank_hub, f"SYSTEM DAY: {current_day.upper()}", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(blank_hub, f"REQUIRED UNIFORM: {role_color_rules.get(current_day, {}).get('Student', 'Regular').upper()}", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    row1 = np.hstack((cam1, cam2, cam3))
    row2 = np.hstack((cam4, cam5, blank_hub))
    grid = np.vstack((row1, row2))

    scaled_display = cv2.resize(grid, (1280, 720))
    cv2.imshow('SIMS: Secure Campus Surveillance Terminal', scaled_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

for cap in caps.values():
    if cap is not None:
        cap.release()
cv2.destroyAllWindows()
