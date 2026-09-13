import cv2
import numpy as np
import datetime

# --- CONFIGURATION ZONE (MEMBER 3: RESEARCHER/COLOR) ---
# Active day security rule tracking (Dynamic based on system day)
current_day = datetime.datetime.now().strftime("%A")

# 1. First, create the daily color assignment dictionary map
daily_colors = {
    "Monday": "Green", 
    "Tuesday": "Blue", 
    "Wednesday": "Red", 
    "Thursday": "Yellow", 
    "Friday": "Orange"
}

# 2. Next, fetch today's required pass target safely
required_color = daily_colors.get(current_day, "Red")

# --- INITIALIZATION ZONE (MEMBER 1: LEAD CODER) ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
attendance_log = {}

print(f"[SIMS INITIALIZED] Active Security Day: {current_day} | Target Pass: {required_color}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- ZONE A (MEMBER 2: FACIAL RECOGNITION MATCHING) ---
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    user_identity = "UNKNOWN ENTRANT"
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        user_identity = "VERIFIED_USER_01" # Placeholder for recognition loop

    # --- ZONE B (MEMBER 3: DYNAMIC COLOR PASS VALIDATION) ---
    if required_color == "Red":
        lower_bound = np.array([0, 120, 70])
        upper_bound = np.array([10, 255, 255])
    elif required_color == "Blue":
        lower_bound = np.array([90, 50, 50])
        upper_bound = np.array([130, 255, 255])
    else:  # Default Green
        lower_bound = np.array([36, 25, 25])
        upper_bound = np.array([86, 255, 255])

    # Isolate the target day's color channel pixels
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    color_pass_detected = False
    for cnt in contours:
        if cv2.contourArea(cnt) > 2000: 
            color_pass_detected = True
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 255, 255), 2)
            break

    # --- ZONE C (MEMBER 4: DATABASE & LOG WRITER) ---
    system_status = "SCANNING BADGE..."
    hud_color = (0, 255, 255) # Scanning default yellow

    if user_identity != "UNKNOWN ENTRANT" and color_pass_detected:
        current_time = datetime.datetime.now()
        timestamp = current_time.strftime("%H:%M:%S")
        date_stamp = current_time.strftime("%Y-%m-%d")
        
        system_status = f"ACCESS GRANTED | LOGGED: {timestamp}"
        hud_color = (0, 255, 0) # Clear pass green

        if user_identity not in attendance_log:
            attendance_log[user_identity] = "IN"
            with open("attendance_log.txt", "a") as log_file:
                log_file.write(f"{date_stamp},{timestamp},{user_identity},CLOCKED_IN\n")

    # --- ZONE D (MEMBER 5: DASHBOARD UI DESIGN & TEST overlays) ---
    cv2.putText(frame, f"IDENTITY: {user_identity}", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)
    cv2.putText(frame, f"STATUS: {system_status}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hud_color, 2)

    cv2.imshow("SIMS: Two-Factor Identity Matrix Terminal", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
