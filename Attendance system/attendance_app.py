import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime
import os
import time
import pandas as pd
from typing import List, Dict, Tuple, Any
import winsound # For audio alerts (Windows only, comment out or change if not on Windows)

# --- 1. CONFIGURATION ---
LOG_FILE = 'attendance_log.csv'
EMPLOYEE_PHOTO_PATH = 'neeraj.jpg' # CHANGE THIS to your photo file!
COOLDOWN_SECONDS = 5
FRAME_SCALE = 0.25
UNKNOWN_VISITOR_ID = "Unknown_Visitor" # ID used in the log for all unknown 
# DeepFace Configuration

TOLERANCE = 0.55 # Standard is 0.6, we use 0.5 for stricter access control
FRAME_THICKNESS = 2
FONT_THICKNESS = 2
MODEL = "hog" # "hog" or "cnn"
# --- 2. SETUP: KNOWN FACES ---

def load_known_faces() -> Tuple[List[Any], List[str]]:
    """Loads known face encodings from files."""
    known_face_encodings = []
    known_face_names = []
    
    # Example for one person (Neeraj)
    try:
        image = face_recognition.load_image_file(EMPLOYEE_PHOTO_PATH)
        encoding = face_recognition.face_encodings(image)
        
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append("Neeraj")
            print(f"Loaded encoding for Neeraj.")
        else:
            print(f"Error: No face found in {EMPLOYEE_PHOTO_PATH}. Check photo alignment.")
            
    except FileNotFoundError:
        print(f"Error: Employee photo not found at {EMPLOYEE_PHOTO_PATH}. Please check the path.")
    except Exception as e:
        print(f"Could not load or process employee photo: {e}")
    
    # Add more employees here if necessary
    
    return known_face_encodings, known_face_names

# --- 3. LOGGING FUNCTIONS (Robust) ---

def initialize_log():
    """Creates the log file with headers if it doesn't exist."""
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Person_ID", "Action"]) 
            print(f"Created new log file: {LOG_FILE}")

def get_last_action(name: str) -> str:
    """Reads the log to find the last recorded action for a person."""
    initialize_log()
    try:
        df = pd.read_csv(LOG_FILE)
        
        if df.empty or len(df) == 0:
             return "EXIT" # Default: Person is OUTSIDE
        
        # Ensures correct chronological order
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True) # Remove corrupt rows

        person_log = df[df['Person_ID'] == name]
        
        if person_log.empty:
            return "EXIT" # If person has no log entries, they are OUTSIDE
        
        # Get the last action chronologically
        last_action = person_log.sort_values(by='Timestamp', ascending=False)['Action'].iloc[0]
        
        return last_action
        
    except Exception as e:
        print(f"Warning: Error reading log for status ({e}). Assuming EXIT.")
        return "EXIT"

def log_attendance(name, action):
    """Appends a new entry/exit record to the CSV log."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S") 
    
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, name, action])
    
    print(f"[LOGGED] {timestamp} - {name} has {action}.")

# --- 4. MAIN EXECUTION ---
known_face_encodings, known_face_names = load_known_faces()
initialize_log()

# Initialize webcam
video_capture = cv2.VideoCapture(0)

# Variables to prevent logging the same person multiple times per second
last_logged_time: Dict[str, float] = {} 

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("Error: Could not read frame from camera.")
        break

    # Fix for potential 4-channel BGRA frame bug
    if frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    # Resize frame for faster processing and convert BGR to RGB
    small_frame = cv2.resize(frame, (0, 0), fx=FRAME_SCALE, fy=FRAME_SCALE)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"
        action = "None"
        
        # Match face
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        
        current_time = time.time()
        
        if matches[best_match_index]:
            # --- Known Person Logic ---
            name = known_face_names[best_match_index]
            log_id = name
            
            last_time = last_logged_time.get(log_id, 0)
            
            # Get the actual last action from the log file (the person's current status)
            current_status = get_last_action(log_id)
            action = current_status # Set initial action for display (current status)
            
            # 1. Check for cooldown
            if (current_time - last_time) > COOLDOWN_SECONDS:
                
                # 2. Determine the action to be logged (The opposite of the current status)
                if current_status == "ENTRY":
                    new_action = "EXIT"
                else: 
                    new_action = "ENTRY"
                
                # 3. Log the event
                log_attendance(log_id, new_action)
                last_logged_time[log_id] = current_time
                action = new_action # CRITICAL FIX: Update 'action' to the newly logged action for immediate display
            else:
                # If still in cooldown, action remains the current_status retrieved earlier
                pass
            
            # Set display colors based on the current determined status/action
            color = (0, 255, 0) if action == "ENTRY" else (0, 0, 255) # Green for ENTRY, Red for EXIT
            status_text = f"{name} ({action})"

        else:
            # --- Unknown Person Logic ---
            log_id = UNKNOWN_VISITOR_ID
            name = "Unknown"
            
            last_time = last_logged_time.get(log_id, 0)
            # Get the actual last action from the log file (the unknown visitor's current status)
            current_status = get_last_action(log_id)
            action = current_status # Set initial action for display (current status)

            # 1. Check for cooldown
            if (current_time - last_time) > COOLDOWN_SECONDS:
                
                # 2. Determine the action to be logged
                if current_status == "ENTRY":
                    new_action = "EXIT"
                else: 
                    new_action = "ENTRY"
                
                # 3. Log the event using the generic ID
                log_attendance(log_id, new_action)
                last_logged_time[log_id] = current_time
                action = new_action # CRITICAL FIX: Update 'action' to the newly logged action for immediate display
                
                # ===============================================
                # 🛑 WARNING SYSTEM: TRIGGER ALERT ON ENTRY
                # ===============================================
                if action == "ENTRY": # Check the NEWLY logged action
                    # Visual: Print a big warning to the console
                    print("\n" + "="*50)
                    print(f"!!! SECURITY ALERT !!! UNKNOWN VISITOR DETECTED AT {datetime.now().strftime('%H:%M:%S')}")
                    print("="*50 + "\n")
                    
                    # Audible: Play a simple sound (Windows-specific)
                    try:
                        winsound.Beep(1000, 500)
                    except Exception:
                        pass
                # ===============================================
            
            # Set display colors (Red for unknown alert)
            color = (0, 0, 255) 
            status_text = f"!!! UNKNOWN ({action}) !!!" 


        # Scale back up the face coordinates
        top, right, bottom, left = [int(v / FRAME_SCALE) for v in (top, right, bottom, left)]

        # --- DRAW VISUALS ---
        # Draw the bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw the label bar
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        
        # Draw the name/status text
        font = cv2.FONT_HERSHEY_DUPLEX
        # Use contrasting text color (black for known, white for unknown alert)
        text_color = (0, 0, 0) if name != "Unknown" else (255, 255, 255)
        cv2.putText(frame, status_text, (left + 6, bottom - 6), font, 1.0, text_color, 1)

    # Display the resulting image
    cv2.imshow('Office Attendance Tracker', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()