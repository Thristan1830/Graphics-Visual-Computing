import cv2
import numpy as np
import datetime
import os
import time

# --- 📅 CONFIGURATION ZONE: CAMPUS RULES (MON-SUN: REGULAR UNIFORM) ---
current_day = datetime.datetime.now().strftime("%A")

# --- 🚀 INITIALIZATION ZONE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

if face_cascade.empty() and os.path.exists('haarcascade_frontalface_default.xml'):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if eye_cascade.empty() and os.path.exists('haarcascade_eye.xml'):
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
# 🚀 Strict Subfolder Rule: Directs the tracker straight to your active trained brain weights!
model_path = 'trainer.yml'
if not os.path.exists(model_path) and os.path.exists('Graphics-Visual-Computing/trainer.yml'):
    model_path = 'Graphics-Visual-Computing/trainer.yml'

if os.path.exists(model_path):
    recognizer.read(model_path)
    print(f"[SYSTEM] Secure biometric face database loaded successfully from {model_path}!")
else:
    print("[WARNING] trainer.yml model weight file not found. Defaulting to GUEST mode.")

attendance_log = {}

# --- 🌐 5-CAMERA NETWORK SIMULATION ROUTING ---
camera_sources = {
    "GATE 01 (Main)": 0,
    "GATE 02 (Mobile)": 1,
    "GATE 03 (Library)": 2,
    "GATE 04 (Gym)": 3,
    "GATE 05 (Exit)": 4
}

caps = {}
for cam_name, src in camera_sources.items():
    cap = cv2.VideoCapture(src)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    time.sleep(0.3)
    if cap.isOpened():
        caps[cam_name] = cap
        print(f"[NETWORK SYSTEM] Secure stream connection bound to: {cam_name}")
    else:
        if cap is not None:
            cap.release()
        caps[cam_name] = None
        print(f"[NETWORK INFRASTRUCTURE] {cam_name} mapped as virtual standby node.")

cv2.namedWindow('SIMS: Secure Campus Surveillance Terminal', cv2.WINDOW_NORMAL)

while True:
    # 🔍 READ REGISTERED STUDENT DATABASE NAMES NATIVELY
    names_database = {}
    txt_path = 'students.txt'
    if not os.path.exists(txt_path) and os.path.exists('Graphics-Visual-Computing/students.txt'):
        txt_path = 'Graphics-Visual-Computing/students.txt'

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8-sig") as f:
            for db_line in f:
                if "," in db_line:
                    clean_line = db_line.strip().replace("\\", "")
                    if clean_line:
                        db_id, db_name = clean_line.split(",", 1)
                        try:
                            names_database[int(db_id.strip())] = db_name.strip()
                        except ValueError:
                            continue

    frames = {}
    for cam_name, cap in caps.items():
        ret = False
        frame = None

        if cap is not None:
            ret, frame = cap.read()

        if not ret or frame is None:
            frame = np.zeros((480, 640, 3), dtype="uint8")
            cv2.rectangle(frame, (15, 15), (625, 465), (20, 20, 20), -1)
            cv2.putText(frame, f"{cam_name}: OFFLINE", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 🚀 SHADOW & LIGHT PROOF ENGINE: Applies CLAHE grid blocks to eliminate environment glare fluctuations!
            clahe_engine = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(8, 8))
            equalized_gray = clahe_engine.apply(gray)

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            faces = face_cascade.detectMultiScale(equalized_gray, scaleFactor=1.12, minNeighbors=6, minSize=(50, 50))
            eye_pairs = eye_cascade.detectMultiScale(equalized_gray, scaleFactor=1.15, minNeighbors=18, minSize=(50, 25))

            detection_targets = []
            for (x, y, w, h) in faces:
                detection_targets.append((x, y, w, h, False))

            if len(faces) == 0:
                for (ex, ey, ew, eh) in eye_pairs:
                    fx = max(0, ex - int(ew * 0.2))
                    fy = max(0, ey - int(eh * 0.4))
                    fw = int(ew * 1.4)
                    fh = int(eh * 2.6)
                    detection_targets.append((fx, fy, fw, fh, True))

            # 🚨 Everything below is now INSIDE the per-face loop, so each detected
            # face gets its own identity/uniform/HUD pass instead of reusing
            # whatever x/y/w/h/roi_gray happened to be left over from elsewhere.
            for (x, y, w, h, is_masked) in detection_targets:
                # 🚨 PHANTOM BOX FILTER VALVE: Completely ignore tiny mouth/nose glare boxes!
                if w < 75 or h < 75:
                    continue

                box_color = (255, 255, 0) if is_masked else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                roi_gray = equalized_gray[y:y + h, x:x + w]

                # 🚀 SAFETY CORE RESET: Everything defaults to anonymous visitor state initially
                resolved_identity = "GUEST"
                current_face_identity = "GUEST"
                current_face_role = "Guest"
                system_status = "GUEST LOCKED OUT"
                hud_color = (255, 165, 0)
                uniform_status = "VIOLATION: NO UNIFORM"

                # --- 🧠 BIOMETRIC IDENTITY MATCHING ---
                if os.path.exists(model_path):
                    try:
                        user_id, confidence = recognizer.predict(roi_gray)
                        print(f"[DIAGNOSTIC] Detected ID: {user_id} | Raw Confidence Score: {confidence:.2f}")

                        # 👔 STEP 1: ADAPTIVE THRESHOLD BASED ON FACE FRAME WIDTH
                        face_pixel_width = w
                        if face_pixel_width > 70:
                            threshold_cutoff = 125
                        elif face_pixel_width < 65:
                            threshold_cutoff = 120
                        else:
                            threshold_cutoff = 105 if is_masked else 100

                        # 🔐 STEP 2: DUAL-CEILING IDENTITY LOCKDOWN (SCALE-AWARE MULTIVARIABLE BIOMETRIC FILTER)
                        general_ceiling = 110.00
                        clean_search_id = int(user_id)

                        # Scale-Aware Rule: If the bounding face width is large (w >= 110), keep a strict 62.00 cap to lock out close relatives.
                        # If the bounding face width is small (w < 110), expand the ceiling to 78.00 to capture your side views and far stances smoothly!
                        if clean_search_id == 1:
                            id_ceiling = 62.00 if w >= 110 else 78.00
                        else:
                            id_ceiling = general_ceiling

                        if confidence < threshold_cutoff and confidence < general_ceiling and confidence < id_ceiling:
                            lookup_key = str(clean_search_id)

                            if lookup_key in names_database:
                                matched_name = names_database[lookup_key]
                            elif clean_search_id in names_database:
                                matched_name = names_database[clean_search_id]
                            else:
                                matched_name = None

                            if matched_name is not None:
                                resolved_identity = matched_name
                                current_face_identity = matched_name

                                # 🚀 100% Dynamic: Roles are split natively without forcing hardcoded name overrides!
                                if clean_search_id == 999 or "guest" in matched_name.lower():
                                    current_face_role = "Guest"
                                else:
                                    current_face_role = "Student"
                        else:
                            # 🛡️ HARD LOCKOUT SAFETY VALVE: Discard loose predictions or unverified distant glitches completely
                            resolved_identity = "GUEST"
                            current_face_identity = "GUEST"
                            current_face_role = "Guest"
                    except cv2.error:
                        pass

                # --- 👕 UNIFORM COMPLIANCE SCANNER (safe against edge-frame shape errors) ---
                try:
                    shirt_y = int(y + h * 1.15)
                    shirt_h = min(479 - shirt_y, int(h * 0.45))
                    shirt_x = max(0, int(x + w * 0.15))
                    shirt_w = min(639 - shirt_x, int(w * 0.7))

                    if shirt_h > 5 and shirt_w > 5:
                        shirt_roi_hsv = hsv[shirt_y:shirt_y + shirt_h, shirt_x:shirt_x + shirt_w]
                        lower_white = np.array([0, 0, 100], dtype="uint8")
                        upper_white = np.array([180, 80, 255], dtype="uint8")

                        white_mask = cv2.inRange(shirt_roi_hsv, lower_white, upper_white)
                        white_pixels = cv2.countNonZero(white_mask)
                        total_roi_pixels = int(shirt_roi_hsv.shape[0] * shirt_roi_hsv.shape[1])

                        white_density_score = (white_pixels / total_roi_pixels) * 100 if total_roi_pixels > 0 else 0

                        if white_density_score > 8:
                            uniform_status = "PASSED: WHITE SHIRT"
                            cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 255, 0), 2)
                        else:
                            uniform_status = "VIOLATION: NO UNIFORM"
                            cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 0, 255), 2)
                except cv2.error:
                    pass

                # --- 🚨 ACCESS DECISION ROUTER ---
                if uniform_status.startswith("PASSED"):
                    if current_face_identity != "GUEST":
                        system_status = "ACCESS GRANTED | UNIFORM OK"
                        hud_color = (0, 255, 0)  # Solid Green for verified student with uniform

                        log_key = f"{resolved_identity}_{cam_name}"
                        if log_key not in attendance_log:
                            attendance_log[log_key] = "IN"
                            with open("attendance_log.txt", "a") as log_file:
                                log_file.write(
                                    f"{datetime.datetime.now().strftime('%Y-%m-%d,%H:%M:%S')},"
                                    f"{resolved_identity},{current_face_role},{cam_name},AUTHORIZED_ENTRY\n"
                                )
                    else:
                        system_status = "GUEST PASSED UNIFORM"
                        hud_color = (255, 165, 0)  # Orange for unknown guest with uniform
                else:
                    # 🛡️ THE STRICT UNIFORM & REGISTRATION LOCKOUT ENGINE
                    if current_face_identity != "GUEST":
                        # Force a hard RED alert status if an authorized student forgot their white shirt uniform!
                        system_status = "ACCESS DENIED | NO UNIFORM"
                        hud_color = (0, 0, 255)  # Vibrant Solid Red for absolute compliance violation
                    else:
                        # Standard fallback for completely unregistered/random photo test samples
                        system_status = "GUEST LOCKED OUT"
                        hud_color = (255, 165, 0)  # Standard Orange for anonymous guest layout tracker
                        resolved_identity = "GUEST"
                        current_face_role = "Guest"

                # --- 🖥️ SINGLE CLEAN HUD RENDER (one block, no duplicated text) ---
                cv2.putText(frame, f"ID: {resolved_identity.upper()}", (x, y - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 2)
                cv2.putText(frame, f"ROLE: {current_face_role.upper()}", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1)
                cv2.putText(frame, f"STATUS: {system_status}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1)
                cv2.putText(frame, f"UNIFORM: {uniform_status}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 2)

        cv2.putText(frame, f"GATEWAY NODE: {cam_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        frames[cam_name] = frame

    # --- 🖥️ surveillance matrix panels view compiler ---
    cam1 = frames["GATE 01 (Main)"]
    cam2 = frames["GATE 02 (Mobile)"]
    cam3 = frames["GATE 03 (Library)"]
    cam4 = frames["GATE 04 (Gym)"]
    cam5 = frames["GATE 05 (Exit)"]

    # Standby Logic: Render crisp, independent network node templates instead of cloning Camera 1
    if caps["GATE 03 (Library)"] is None:
        cam3 = np.zeros((480, 640, 3), dtype="uint8")
        cv2.rectangle(cam3, (15, 15), (625, 465), (20, 20, 20), -1)
        cv2.putText(cam3, "GATE 03 (LIBRARY): OFFLINE STANDBY", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    if caps["GATE 04 (Gym)"] is None:
        cam4 = np.zeros((480, 640, 3), dtype="uint8")
        cv2.rectangle(cam4, (15, 15), (625, 465), (20, 20, 20), -1)
        cv2.putText(cam4, "GATE 04 (GYM): OFFLINE STANDBY", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    if caps["GATE 05 (Exit)"] is None:
        cam5 = np.zeros((480, 640, 3), dtype="uint8")
        cv2.rectangle(cam5, (15, 15), (625, 465), (20, 20, 20), -1)
        cv2.putText(cam5, "GATE 05 (EXIT): OFFLINE STANDBY", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)

    # --- 🖥️ CENTRAL SERVER MONITORING PANEL HUB ---
    blank_hub = np.zeros((480, 640, 3), dtype="uint8")
    cv2.rectangle(blank_hub, (15, 15), (625, 465), (35, 25, 15), -1)
    cv2.putText(blank_hub, "SIMS MATRIX SERVER", (80, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(blank_hub, f"SYSTEM DAY: {current_day.upper()}", (80, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(blank_hub, "REQUIRED UNIFORM: WHITE SHIRT", (80, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    row1 = np.hstack((cam1, cam2, cam3))
    row2 = np.hstack((cam4, cam5, blank_hub))
    grid = np.vstack((row1, row2))

    scaled_display = cv2.resize(grid, (1280, 720))
    cv2.imshow('SIMS: Secure Campus Surveillance Terminal', scaled_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- 🔌 CLEAN HARDWARE DECOUPLING SHUTDOWN ENGINE ---
for cap in caps.values():
    if cap is not None:
        cap.release()
cv2.destroyAllWindows()