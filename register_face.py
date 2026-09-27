import cv2
import os
import re
import time

# --- 🎨 TERMINAL UI HELPERS ---
BORDER_WIDTH = 60


def print_frame_header(label):
    print("=" * BORDER_WIDTH)
    print(f"| {label:<{BORDER_WIDTH - 4}} |")
    print("=" * BORDER_WIDTH)


def print_frame_footer():
    print("-" * BORDER_WIDTH)


def status(tag, message):
    print(f"[{tag}] {message}")


# --- 🚀 INITIALIZE PATHS & CASCADES ---
# Path guard checks if we are running inside the repository subfolder
base_dir = "Graphics-Visual-Computing" if os.path.exists("Graphics-Visual-Computing") else "."

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
if os.path.exists(os.path.join(base_dir, 'haarcascade_frontalface_default.xml')):
    cascade_path = os.path.join(base_dir, 'haarcascade_frontalface_default.xml')

face_cascade = cv2.CascadeClassifier(cascade_path)
cap = cv2.VideoCapture(0)

dataset_path = os.path.join(base_dir, "dataset")
os.makedirs(dataset_path, exist_ok=True)

student_txt = os.path.join(base_dir, "students.txt")

print_frame_header("CAMPUS BIOMETRIC REGISTRATION TERMINAL")

# --- 📇 LOAD EXISTING LEDGER ---
existing_records = {}
if os.path.exists(student_txt):
    with open(student_txt, "r", encoding="utf-8") as f:
        for line in f:
            if "," in line:
                idx, nm = line.strip().split(",", 1)
                existing_records[idx.strip()] = nm.strip()

user_id = input("[PROMPT] Enter Student ID Number: ").strip()

# --- 🛡️ INTERACTIVE LEDGER OVERWRITE CHALLENGE ---
is_update_session = False
if user_id in existing_records:
    print(f"[ALERT] ID {user_id} is already registered to '{existing_records[user_id]}'.")
    choice = input("Would you like to add/update more profile photos for this user? (y/n): ").strip().lower()

    if choice != 'y':
        status("SYSTEM", "Registration cancelled by operator. Releasing hardware...")
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit(0)

    is_update_session = True
    user_name = existing_records[user_id]
    status("INPUT", f"Session locked to existing profile: '{user_name}' (ID {user_id})")
else:
    user_name = input("[PROMPT] Enter Student Full Name: ").strip()
    existing_records[user_id] = user_name

# --- 💾 WRITE LEDGER ROWS BACK TO students.txt ---
with open(student_txt, "w", encoding="utf-8") as f:
    for idx, nm in existing_records.items():
        f.write(f"{idx},{nm}\n")

status("SUCCESS", f"Ledger synced for ID {user_id} -> '{user_name}'")
print_frame_footer()

# --- 🔢 MULTI-SESSION SUFFIX COUNT POINTER MERGE ---
# Scan the dataset directory for any prior captures belonging to this exact
# student ID, so a second/third enrollment session appends instead of
# overwriting earlier image sequence numbers.
sequence_pattern = re.compile(rf"^User\.{re.escape(user_id)}\.(\d+)\.jpg$")
highest_existing_index = 0

if os.path.isdir(dataset_path):
    for existing_file in os.listdir(dataset_path):
        match = sequence_pattern.match(existing_file)
        if match:
            file_index = int(match.group(1))
            if file_index > highest_existing_index:
                highest_existing_index = file_index

total_captured = highest_existing_index

if is_update_session:
    status("SYSTEM", f"Resuming capture sequence from index {total_captured} (previous session detected).")
else:
    status("SYSTEM", f"Starting fresh capture sequence at index {total_captured}.")

# --- 🎯 REGISTRATION STAGES ---
registration_stages = [
    {"name": "FRONT VIEW", "desc": "Look straight into the lens", "count": 25},
    {"name": "LEFT SIDE VIEW", "desc": "Tilt your head slightly to the left", "count": 25},
    {"name": "RIGHT SIDE VIEW", "desc": "Tilt your head slightly to the right", "count": 25},
    {"name": "FAR AWAY STANCE", "desc": "Step back to your presentation distance", "count": 25}
]

for stage_idx, stage in enumerate(registration_stages):
    print_frame_header(f"STAGE {stage_idx + 1}/{len(registration_stages)}: {stage['name']}")
    status("PROMPT", f"{stage['desc']}. Starting in 3 seconds...")

    # Visual countdown loop sequence
    for countdown in range(3, 0, -1):
        ret, frame = cap.read()
        if ret:
            display_frame = frame.copy()
            cv2.putText(display_frame, f"NEXT ANGLE NA: {stage['name']}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(display_frame, f"Starting in: {countdown}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            cv2.imshow("Biometric Enrollment Console", display_frame)
            cv2.waitKey(1000)

    stage_captured = 0
    while stage_captured < stage['count']:
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(equalized_gray, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30))

        display_frame = frame.copy()
        # On-screen guidance indicators
        cv2.putText(display_frame, f"STAGE: {stage['name']}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(display_frame, stage['desc'], (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display_frame, f"Captured: {stage_captured}/{stage['count']}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        for (x, y, w, h) in faces:
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Save frame slice, continuing the sequence index across sessions
            total_captured += 1
            stage_captured += 1
            face_roi = gray[y:y + h, x:x + w]
            file_name = f"User.{user_id}.{total_captured}.jpg"
            cv2.imwrite(os.path.join(dataset_path, file_name), face_roi)
            time.sleep(0.05)  # Safe capture interval spacing
            break

        cv2.imshow("Biometric Enrollment Console", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    status("SUCCESS", f"Stage {stage['name']} complete!")
    print_frame_footer()

# --- ✅ FINAL CONFIRMATION ---
print_frame_header("SUCCESS: TAPOS NA!")
status("SUCCESS", f"All view angles registered for ID {user_id} ('{user_name}').")
status("SUCCESS", f"Dataset sequence for this student now ends at index {total_captured}.")
print_frame_footer()

cap.release()
cv2.destroyAllWindows()