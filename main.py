import cv2
import numpy as np
import datetime
import os
import sys
import subprocess
import time
from collections import Counter

# --- EXECUTABLE-SAFE PATH ROOT ---
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(os.path.abspath(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GVC_DIR = os.path.join(BASE_DIR, 'Graphics-Visual-Computing')

# --- CONFIGURATION ZONE ---
current_day = datetime.datetime.now().strftime("%A")

# --- INITIALIZATION ZONE ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')

if face_cascade.empty() and os.path.exists(os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml')):
    face_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_frontalface_default.xml'))
if eye_cascade.empty() and os.path.exists(os.path.join(BASE_DIR, 'haarcascade_eye.xml')):
    eye_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_eye.xml'))
if profile_cascade.empty() and os.path.exists(os.path.join(BASE_DIR, 'haarcascade_profileface.xml')):
    profile_cascade = cv2.CascadeClassifier(os.path.join(BASE_DIR, 'haarcascade_profileface.xml'))

recognizer = cv2.face.LBPHFaceRecognizer_create()
model_path = os.path.join(BASE_DIR, 'trainer.yml')
if not os.path.exists(model_path) and os.path.exists(os.path.join(GVC_DIR, 'trainer.yml')):
    model_path = os.path.join(GVC_DIR, 'trainer.yml')

if os.path.exists(model_path):
    try:
        recognizer.read(model_path)
        print(f"[SYSTEM] Secure biometric face database loaded successfully from {model_path}!")
    except Exception as e:
        print(f"[WARNING] Could not parse trainer.yml: {e}")
else:
    print("[WARNING] trainer.yml model weight file not found. Defaulting to GUEST mode.")

attendance_log = {}

# --- MULTI-CROP BIOMETRIC PREDICTION HELPER ---
def predict_face_multisample(recognizer_obj, gray_frame, bbox):
    """
    Evaluates 5 crop scale variations using histogram equalization to account 
    for varying webcam focal lengths, resolutions, and subject distances.
    Returns the prediction with the lowest distance score (best match).
    """
    x, y, w, h = bbox
    frame_h, frame_w = gray_frame.shape[:2]

    candidates = []
    # Test tighter (-5%), standard (0%), and expanded (+5%, +10%, +15%) bounding crops
    pad_ratios = [-0.05, 0.0, 0.05, 0.10, 0.15]

    for p in pad_ratios:
        pad_x = int(w * p)
        pad_y = int(h * p)

        x_start = max(0, x - pad_x)
        y_start = max(0, y - pad_y)
        w_crop = min(frame_w - x_start, w + 2 * pad_x)
        h_crop = min(frame_h - y_start, h + 2 * pad_y)

        if w_crop <= 10 or h_crop <= 10:
            continue

        crop = gray_frame[y_start:y_start + h_crop, x_start:x_start + w_crop]
        if crop.size == 0:
            continue

        resized = cv2.resize(crop, (200, 200))
        eq_tile = cv2.equalizeHist(resized)

        try:
            pred_id, dist = recognizer_obj.predict(eq_tile)
            candidates.append((pred_id, dist))
        except cv2.error:
            continue

    if not candidates:
        return None, 999.0

    best_candidate = min(candidates, key=lambda c: c[1])
    return best_candidate[0], best_candidate[1]

# --- CENTROID TRACKING & HYSTERESIS CONSENSUS ENGINE ---
class BiometricTrackerEngine:
    def __init__(self):
        self.tracks = {}

    def process_face(self, cam_name, bbox, raw_user_id, raw_conf, names_db, lock_thresh=68.0, retain_thresh=76.0):
        x, y, w, h = bbox
        cx, cy = x + w // 2, y + h // 2
        now = time.time()

        if cam_name not in self.tracks:
            self.tracks[cam_name] = []

        # Purge stale spatial tracks older than 1.2 seconds
        self.tracks[cam_name] = [t for t in self.tracks[cam_name] if now - t['last_seen'] < 1.2]

        best_track = None
        min_dist = 90.0

        for t in self.tracks[cam_name]:
            tcx, tcy = t['centroid']
            dist = np.hypot(cx - tcx, cy - tcy)
            if dist < min_dist:
                min_dist = dist
                best_track = t

        if best_track is None:
            best_track = {
                'centroid': (cx, cy),
                'locked_id': None,
                'id_history': [],
                'conf_history': [],
                'last_seen': now
            }
            self.tracks[cam_name].append(best_track)

        best_track['centroid'] = (cx, cy)
        best_track['last_seen'] = now

        # Check database validity
        is_registered_in_db = (raw_user_id in names_db) or (str(raw_user_id) in names_db)

        # State Machine Logic
        if best_track['locked_id'] is None:
            # UNLOCKED TRACK: Unlocks identity when score is < 68.0 and present in DB
            if is_registered_in_db and raw_conf < lock_thresh:
                best_track['id_history'].append(raw_user_id)
                best_track['conf_history'].append(raw_conf)

                valid_hits = [i for i in best_track['id_history'] if i == raw_user_id]
                if len(valid_hits) >= 1:
                    best_track['locked_id'] = raw_user_id
            else:
                best_track['id_history'].append("GUEST")
                best_track['conf_history'].append(raw_conf)
        else:
            # LOCKED TRACK: Retains registered identity while score is < 76.0
            if is_registered_in_db and (raw_user_id == best_track['locked_id']) and (raw_conf < retain_thresh):
                best_track['conf_history'].append(raw_conf)
            else:
                best_track['conf_history'].append(raw_conf)
                recent_confs = best_track['conf_history'][-4:]
                if len(recent_confs) >= 4 and all(c >= retain_thresh for c in recent_confs):
                    best_track['locked_id'] = None

        if len(best_track['id_history']) > 8:
            best_track['id_history'].pop(0)
        if len(best_track['conf_history']) > 8:
            best_track['conf_history'].pop(0)

        if best_track['locked_id'] is not None:
            valid_confs = [c for c in best_track['conf_history'] if c < retain_thresh]
            avg_conf = float(np.mean(valid_confs)) if valid_confs else raw_conf
            return best_track['locked_id'], avg_conf
        else:
            avg_conf = float(np.mean(best_track['conf_history'])) if best_track['conf_history'] else raw_conf
            return "GUEST", avg_conf

tracker_engine = BiometricTrackerEngine()

# ---  DYNAMIC MULTI-CAMERA HARDWARE BINDING ---
def open_hardware_cameras():
    caps = {}
    cam_names = [
        "GATE 01 (Main)",
        "GATE 02 (Exit)",
        "GATE 03 (Library)",
        "GATE 04 (Gym)",
        "GATE 05 (Exit)"
    ]
    
    backend = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY
    
    for idx, name in enumerate(cam_names):
        cap = cv2.VideoCapture(idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        if cap.isOpened():
            ret, test_frame = cap.read()
            if ret and test_frame is not None:
                caps[name] = cap
                print(f"[NETWORK SYSTEM] Secure stream connection bound to: {name} (Hardware Index {idx})")
            else:
                cap.release()
                caps[name] = None
                print(f"[NETWORK INFRASTRUCTURE] {name} (Index {idx}) offline - mapped as standby node.")
        else:
            caps[name] = None
            print(f"[NETWORK INFRASTRUCTURE] {name} (Index {idx}) mapped as virtual standby node.")

    return caps

caps = open_hardware_cameras()

cv2.namedWindow('SIMS: Secure Campus Surveillance Terminal', cv2.WINDOW_NORMAL)

# ---  MOUSE INTERACTION HANDLER ---
pending_action = None

def handle_matrix_click(event, x, y, flags, param):
    global pending_action
    if event == cv2.EVENT_LBUTTONDOWN:
        if 900 <= x <= 1230 and 550 <= y <= 610:
            print("[GUI HIT] Button 1: REGISTER NEW FACE Clicked!")
            pending_action = "REGISTER"
        elif 900 <= x <= 1230 and 625 <= y <= 685:
            print("[GUI HIT] Button 2: TRAIN BIOMETRIC MODEL Clicked!")
            pending_action = "TRAIN"

cv2.setMouseCallback('SIMS: Secure Campus Surveillance Terminal', handle_matrix_click)

while True:
    # ---  SUBPROCESS LAUNCHER WITH CLEAN ENVIRONMENT ---
    if pending_action is not None:
        action_type = pending_action
        pending_action = None

        print(f"[SYSTEM NODE] Executing launcher action: {action_type}. Releasing camera locks...")

        for cam_name, cap_obj in caps.items():
            if cap_obj is not None:
                cap_obj.release()
        caps = {}

        cv2.destroyAllWindows()
        cv2.waitKey(100)
        time.sleep(1.0)

        if action_type == "REGISTER":
            exe_name = "SIMS_Registration_Terminal.exe"
            script_name = "register_face.py"
        else:
            exe_name = "SIMS_Trainer.exe"
            script_name = "train_faces.py"

        exe_path = os.path.join(BASE_DIR, exe_name)
        script_path = os.path.join(BASE_DIR, script_name)

        clean_env = os.environ.copy()
        for key in ["_MEIPASS2", "PYTHONHOME", "PYTHONPATH"]:
            clean_env.pop(key, None)

        new_console_flag = subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0

        if os.path.exists(exe_path):
            subprocess.run([exe_path], creationflags=new_console_flag, cwd=BASE_DIR, env=clean_env)
        elif os.path.exists(script_path):
            subprocess.run([sys.executable, script_path], creationflags=new_console_flag, cwd=BASE_DIR, env=clean_env)
        else:
            print(f"[ERROR] Neither {exe_name} nor {script_name} was found in {BASE_DIR}")

        if os.path.exists(model_path):
            try:
                recognizer.read(model_path)
                print(f"[SYSTEM] Reloaded updated LBPH face database from {model_path}!")
            except Exception as e:
                print(f"[ERROR] Model reload failed: {e}")

        caps = open_hardware_cameras()

        cv2.namedWindow('SIMS: Secure Campus Surveillance Terminal', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('SIMS: Secure Campus Surveillance Terminal', handle_matrix_click)
        continue

    # ---  READ REGISTERED STUDENT LEDGER ---
    names_database = {}
    txt_path = os.path.join(BASE_DIR, 'students.txt')
    if not os.path.exists(txt_path) and os.path.exists(os.path.join(GVC_DIR, 'students.txt')):
        txt_path = os.path.join(GVC_DIR, 'students.txt')

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
            cv2.putText(frame, f"{cam_name}: OFFLINE STANDBY", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
        else:
            frame = cv2.resize(frame, (640, 480))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            blurred = cv2.GaussianBlur(frame, (5, 5), 0)
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=6, minSize=(50, 50))
            eye_pairs = eye_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=18, minSize=(50, 25))

            detection_targets = []
            for (x, y, w, h) in faces:
                detection_targets.append((x, y, w, h, False))

            if len(faces) == 0:
                profiles = profile_cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=6, minSize=(50, 50))
                for (px, py, pw, ph) in profiles:
                    detection_targets.append((px, py, pw, ph, False))

            if len(faces) == 0 and len(detection_targets) == 0:
                for (ex, ey, ew, eh) in eye_pairs:
                    fx = max(0, ex - int(ew * 0.2))
                    fy = max(0, ey - int(eh * 0.4))
                    fw = int(ew * 1.4)
                    fh = int(eh * 2.6)
                    detection_targets.append((fx, fy, fw, fh, True))

            for (x, y, w, h, is_masked) in detection_targets:
                if w < 50 or h < 50:
                    continue

                resolved_identity = "GUEST"
                current_face_identity = "GUEST"
                current_face_role = "Guest"
                system_status = "GUEST LOCKED OUT"
                hud_color = (255, 165, 0)  # Orange default for Guests
                uniform_status = "VIOLATION: NO UNIFORM"
                conf_text = "CONF: N/A"

                box_color = (255, 255, 0) if is_masked else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)

                if os.path.exists(model_path):
                    raw_id, raw_confidence = predict_face_multisample(recognizer, gray, (x, y, w, h))

                    if raw_id is not None:
                        consensus_id, smoothed_conf = tracker_engine.process_face(
                            cam_name, (x, y, w, h), raw_id, raw_confidence, names_database, lock_thresh=68.0, retain_thresh=76.0
                        )

                        conf_text = f"CONF: {smoothed_conf:.1f}"

                        if consensus_id != "GUEST":
                            try:
                                clean_search_id = int(consensus_id)
                            except ValueError:
                                clean_search_id = None

                            if clean_search_id is not None and clean_search_id in names_database:
                                matched_name = names_database[clean_search_id]
                                resolved_identity = matched_name
                                current_face_identity = matched_name
                                current_face_role = "Student"

                # Uniform Analysis
                try:
                    shirt_y = int(y + h * 1.15)
                    shirt_h = min(479 - shirt_y, int(h * 0.45))
                    shirt_x = max(0, int(x + w * 0.15))
                    shirt_w = min(639 - shirt_x, int(w * 0.7))

                    if shirt_h > 5 and shirt_w > 5:
                        shirt_roi_hsv = hsv[shirt_y:shirt_y + shirt_h, shirt_x:shirt_x + shirt_w]
                        lower_white = np.array([0, 0, 140], dtype="uint8")
                        upper_white = np.array([180, 40, 255], dtype="uint8")

                        white_mask = cv2.inRange(shirt_roi_hsv, lower_white, upper_white)
                        white_pixels = cv2.countNonZero(white_mask)
                        total_roi_pixels = int(shirt_roi_hsv.shape[0] * shirt_roi_hsv.shape[1])
                        white_density_score = (white_pixels / total_roi_pixels) * 100 if total_roi_pixels > 0 else 0

                        lower_red = np.array([0, 0, 140], dtype="uint8")
                        upper_red = np.array([180, 40, 255], dtype="uint8")

                        red_mask = cv2.inRange(shirt_roi_hsv, lower_red, upper_red)
                        red_pixels = cv2.countNonZero(red_mask)
                        total_roi_pixels_R = int(shirt_roi_hsv.shape[0] * shirt_roi_hsv.shape[1])
                        red_density_score = (red_pixels / total_roi_pixels_R) * 100 if total_roi_pixels_R > 0 else 0

                        if white_density_score > 10:
                            uniform_status = "PASSED: WHITE SHIRT"
                            cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 255, 0), 2)

                        elif red_density_score > 10:
                            uniform_status = "PASSED: RED SHIRT"
                            cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 255, 0), 2)
                        else:
                            uniform_status = "VIOLATION: NO UNIFORM"
                            cv2.rectangle(frame, (shirt_x, shirt_y), (shirt_x + shirt_w, shirt_y + shirt_h), (0, 0, 255), 2)
                except cv2.error:
                    pass

                # ---  ROLE & HUD COLOR ENFORCEMENT ---
                is_registered_student = (
                    current_face_identity != "GUEST" and 
                    resolved_identity != "GUEST" and 
                    current_face_role == "Student"
                )

                if not is_registered_student:
                    hud_color = (255, 165, 0)
                    if uniform_status.startswith("PASSED"):
                        system_status = "GUEST PASSED UNIFORM"
                    else:
                        system_status = "GUEST LOCKED OUT"
                else:
                    if uniform_status.startswith("PASSED"):
                        system_status = "ACCESS GRANTED | UNIFORM OK"
                        hud_color = (0, 255, 0)

                        log_key = f"{resolved_identity}_{cam_name}"
                        if log_key not in attendance_log:
                            attendance_log[log_key] = "IN"
                            with open(os.path.join(BASE_DIR, "attendance_log.txt"), "a") as log_file:
                                log_file.write(
                                    f"{datetime.datetime.now().strftime('%Y-%m-%d,%H:%M:%S')},"
                                    f"{resolved_identity},{current_face_role},{cam_name},AUTHORIZED_PERSONEL\n"
                                )
                    else:
                        system_status = "ACCESS DENIED | NO UNIFORM"
                        hud_color = (0, 0, 255)

                cv2.putText(frame, f"ID: {resolved_identity.upper()} ({conf_text})", (x, y - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 2)
                cv2.putText(frame, f"ROLE: {current_face_role.upper()}", (x, y - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, hud_color, 1)
                cv2.putText(frame, f"STATUS: {system_status}", (x, y - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, hud_color, 1)
                cv2.putText(frame, f"UNIFORM: {uniform_status}", (x, y + h + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, hud_color, 2)

        cv2.putText(frame, f"GATEWAY NODE: {cam_name.upper()}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        frames[cam_name] = frame

    cam1 = frames["GATE 01 (Main)"]
    cam2 = frames["GATE 02 (Exit)"]
    cam3 = frames["GATE 03 (Library)"]
    cam4 = frames["GATE 04 (Gym)"]
    cam5 = frames["GATE 05 (Exit)"]

    blank_hub = np.zeros((480, 640, 3), dtype="uint8")
    cv2.rectangle(blank_hub, (15, 15), (625, 465), (35, 25, 15), -1)
    cv2.putText(blank_hub, "SIMS MATRIX SERVER", (80, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(blank_hub, f"SYSTEM DAY: {current_day.upper()}", (80, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(blank_hub, "REQUIRED UNIFORM: WHITE SHIRT", (80, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(blank_hub, "REQUIRED UNIFORM: RED SHIRT", (80, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    cv2.rectangle(blank_hub, (70, 253), (565, 333), (0, 165, 255), -1)
    cv2.putText(blank_hub, "1. REGISTER NEW FACE", (135, 303), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.rectangle(blank_hub, (70, 353), (565, 433), (0, 180, 0), -1)
    cv2.putText(blank_hub, "2. TRAIN BIOMETRIC MODEL", (105, 403), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

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