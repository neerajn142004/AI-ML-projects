from ultralytics import YOLO
import cv2
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import time
import os
import random 
import numpy as np 
from collections import deque, Counter

# ------------------------------------------------------------------
# --- Configuration ---
# ------------------------------------------------------------------

# --- WEBCAM SETTINGS ---
# Change this to 0 for Internal Laptop Cam, or 1 for External USB Cam
WEBCAM_ID = 0

# --- MODEL PATHS ---
# 'yolov8n.pt' will auto-download if missing
GENERIC_MODEL_PATH = 'yolov8m.pt' 
RECOGNITION_MODEL_PATH = r'bestR.pt' 
IDENTIFICATION_MODEL_PATH = r'bestI.pt'

# --- CRITICAL FILTERS ---
# 2 = Car (We ONLY look for this to trigger Mode 2 and 3)
TRIGGER_CLASS_ID = 2 

# Identification Confidence (Very High to prevent false positives)
CONFIDENCE_THRESHOLD = 0.5

# Smoothing (Prevents flickering between models)
SMOOTHING_HISTORY_LENGTH = 10

# ------------------------------------------------------------------

class YOLOTrackerApp:
    def __init__(self, master):
        self.master = master
        self.master.title("AI System - Strict Optimization")
        self.master.geometry("1100x750")
        self.master.config(bg="#f0f0f0")
        self.master.resizable(True, True) 

        # --- State ---
        self.current_mode = 0 
        self.is_running = True
        self.active_model = None
        self.active_label_text = "None"
        self.prediction_history = deque(maxlen=SMOOTHING_HISTORY_LENGTH)
        
        # --- Video ---
        self.current_source = WEBCAM_ID 
        self.cap = None 
        self.is_file_source_video = False 
        self.video_thread = None
        
        # --- Models ---
        self.model_generic = None
        self.model_recognition = None
        self.model_identification = None
        self.is_recognition_ready = False
        self.is_identification_ready = False

        # --- Colors ---
        self.generic_class_colors = {}
        self.generic_class_names = []
        self.recognition_class_colors = {}
        self.recognition_class_names = []
        self.identification_class_colors = {}
        self.identification_class_names = []
        
        # Initialize models (modified to handle Generic auto-download)
        self._init_resources()

        self._setup_layout()
        
        # Start with default webcam
        if self.is_running:
            self._set_video_source(WEBCAM_ID) 
            
        master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _get_absolute_path(self, model_path):
        """Helper to find files in the same directory if not found directly."""
        if os.path.exists(model_path): return model_path
        # If it's the standard YOLO model, return as is to let ultralytics handle download
        if 'yolov8' in model_path: return model_path
        
        try: return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), model_path))
        except: return model_path

    def _init_model_and_colors(self, path, name):
        """
        Loads a model. 
        Special logic: If it's the Generic model, we allow auto-download.
        For custom models (bestR, bestI), we strictly check if the file exists.
        """
        abs_path = self._get_absolute_path(path)
        
        # Check existence ONLY for custom models
        if name != "Generic Detection" and not os.path.exists(abs_path):
            print(f"WARNING: {name} model not found at {abs_path}")
            return None, False, [], {}

        try:
            print(f"Loading {name} model from: {abs_path}...")
            model = YOLO(abs_path) # This triggers download for yolov8n.pt if missing
            class_names = list(model.names.values())
            class_colors = {i: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)) for i in range(len(class_names))}
            print(f"SUCCESS: {name} loaded.")
            return model, True, class_names, class_colors
        except Exception as e:
            print(f"ERROR: Failed to load {name}. Reason: {e}")
            return None, False, [], {}

    def _init_resources(self):
        self.model_generic, _, self.generic_class_names, self.generic_class_colors = \
            self._init_model_and_colors(GENERIC_MODEL_PATH, "Generic Detection")
        
        # If generic fails, we can't really run, but we won't crash the UI immediately
        if not self.model_generic:
            print("CRITICAL: Generic YOLOv8 model failed to load.")

        self.model_recognition, self.is_recognition_ready, self.recognition_class_names, self.recognition_class_colors = \
            self._init_model_and_colors(RECOGNITION_MODEL_PATH, "Recognition")

        self.model_identification, self.is_identification_ready, self.identification_class_names, self.identification_class_colors = \
            self._init_model_and_colors(IDENTIFICATION_MODEL_PATH, "Identification")
        return True

    def _setup_layout(self):
        self.control_panel = tk.Frame(self.master, width=280, bg="#222", padx=15, pady=15)
        self.video_frame = tk.Frame(self.master, bg="#000", padx=5, pady=5, bd=2, relief=tk.SUNKEN)
        self.master.grid_columnconfigure(0, weight=0); self.master.grid_columnconfigure(1, weight=1); self.master.grid_rowconfigure(0, weight=1)
        self.control_panel.grid(row=0, column=0, sticky="nsew"); self.video_frame.grid(row=0, column=1, sticky="nsew")
        
        tk.Label(self.control_panel, text="STRICT AI CONTROL", bg="#222", fg="white", font=("Arial", 14, "bold")).pack(pady=15)
        
        # WEBCAM BUTTON USES THE CONFIG ID
        self.btn_webcam = tk.Button(self.control_panel, text=f"📷 Live Camera ({WEBCAM_ID})", command=lambda: self._set_video_source(WEBCAM_ID), bg="#f39c12", fg="white"); self.btn_webcam.pack(fill='x', pady=2)
        self.btn_upload = tk.Button(self.control_panel, text="📁 Open File", command=self._open_file_dialog, bg="#1abc9c", fg="white"); self.btn_upload.pack(fill='x', pady=2)
        
        tk.Label(self.control_panel, text="-----------------", bg="#222", fg="#555").pack(pady=10)

        # BUTTON 1: Generic
        state_1 = tk.NORMAL if self.model_generic else tk.DISABLED
        self.btn_detection = tk.Button(self.control_panel, text="1. Detect All Objects", command=lambda: self.set_mode(1), bg="#27ae60", fg="white", state=state_1); self.btn_detection.pack(fill='x', pady=5, ipady=5)
        
        # BUTTON 2: Recognition
        self.btn_recognition = tk.Button(self.control_panel, text="2. Car Recognition", command=lambda: self.set_mode(2), bg="#2980b9", fg="white", state=tk.NORMAL if self.is_recognition_ready else tk.DISABLED); self.btn_recognition.pack(fill='x', pady=5, ipady=5)
        
        # BUTTON 3: Identification
        self.btn_identification = tk.Button(self.control_panel, text="3. Exact Identification", command=lambda: self.set_mode(3), bg="#c0392b", fg="white", state=tk.NORMAL if self.is_identification_ready else tk.DISABLED); self.btn_identification.pack(fill='x', pady=5, ipady=5)
        
        self.btn_off = tk.Button(self.control_panel, text="STOP SYSTEM", command=lambda: self.set_mode(0), bg="#7f8c8d", fg="white"); self.btn_off.pack(fill='x', pady=15, ipady=5)

        self.status_label = tk.Label(self.control_panel, text="Status: IDLE", bg="#222", fg="#f1c40f", font=("Arial", 10)); self.status_label.pack(pady=10)
        self.video_label = tk.Label(self.video_frame, bg="#000"); self.video_label.pack(fill="both", expand=True)

    def _open_file_dialog(self):
        file_path = filedialog.askopenfilename(filetypes=[("Media", "*.mp4 *.avi *.jpg *.png"), ("All", "*.*")])
        if file_path: self._set_video_source(file_path)

    def _set_video_source(self, source):
        if self.cap: self.cap.release()
        self.current_source = source
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, str): self.is_file_source_video = any(source.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mkv'])
        else: self.is_file_source_video = False
        self._start_video_thread()

    def _start_video_thread(self):
        if self.video_thread and self.video_thread.is_alive(): pass
        elif self.is_running:
            self.video_thread = threading.Thread(target=self._video_loop, daemon=True)
            self.video_thread.start()

    def set_mode(self, mode):
        self.current_mode = mode
        self.prediction_history.clear()
        
        # Define configs (Model, Text, Color, Names, Colors)
        # Handle cases where model might be None gracefully
        mode_configs = {
            0: (None, "System OFF", "#95a5a6", None, None),
            1: (self.model_generic, "Detection Active", "#27ae60", self.generic_class_names, self.generic_class_colors),
            2: (self.model_recognition, "Recognition Active", "#2980b9", self.recognition_class_names, self.recognition_class_colors), 
            3: (self.model_identification, "Identification Active", "#c0392b", self.identification_class_names, self.identification_class_colors) 
        }
        
        self.active_model, self.active_label_text, fg_color, self.active_class_names, self.active_class_colors = mode_configs[mode]
        self.status_label.config(fg=fg_color, text=f"Status: {self.active_label_text}")

    def _get_stable_prediction(self, current_class_name):
        if current_class_name is None: return None
        self.prediction_history.append(current_class_name)
        if len(self.prediction_history) < 3: return current_class_name
        return Counter(self.prediction_history).most_common(1)[0][0]

    def _video_loop(self):
        while self.is_running and self.cap and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                if self.is_file_source_video: self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0); success, frame = self.cap.read()
                else: continue

            # Mirror effect only for webcam
            if isinstance(self.current_source, int): 
                frame = cv2.flip(frame, 1) 
            
            annotated_frame = frame.copy()
            
            run_refined_detection = False
            target_box = None
            
            # =================================================
            # 1. GATEKEEPER: STRICT CAR TRIGGER (Mode 2 & 3)
            # =================================================
            # Now applied to Mode 2 (Recognition) AND Mode 3 (Identification)
            if self.current_mode in [2, 3] and self.model_generic:
                # Scan for CARS only (Class 2)
                results = self.model_generic.predict(source=frame, conf=0.7, classes=[TRIGGER_CLASS_ID], verbose=False)
                
                if results and len(results) > 0 and results[0].boxes:
                    # We found a car!
                    best_box = results[0].boxes[results[0].boxes.conf.argmax()]
                    target_box = best_box.xyxy[0].cpu().numpy().astype(int)
                    run_refined_detection = True
                else:
                    run_refined_detection = False

            # =================================================
            # 2. INFERENCE
            # =================================================
            should_run_inference = False
            if self.current_mode == 1:
                should_run_inference = True
            elif self.current_mode in [2, 3] and run_refined_detection:
                should_run_inference = True

            if self.active_model and should_run_inference:
                
                input_frame = frame.copy()
                x_offset, y_offset = 0, 0

                # Crop Logic (Mode 2 & 3)
                if self.current_mode in [2, 3] and target_box is not None:
                    x1, y1, x2, y2 = target_box
                    x1, y1 = max(0, x1 - 30), max(0, y1 - 30)
                    x2, y2 = min(frame.shape[1], x2 + 30), min(frame.shape[0], y2 + 30)
                    x_offset, y_offset = x1, y1
                    input_frame = frame[y1:y2, x1:x2]
                
                try:
                    results = self.active_model.predict(
                        source=input_frame,
                        conf=CONFIDENCE_THRESHOLD, 
                        agnostic_nms=True, 
                        max_det=1 if self.current_mode in [2, 3] else 300,
                        verbose=False
                    )
                    
                    if results and len(results) > 0:
                        for box in results[0].boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                            cls = int(box.cls[0].item())
                            conf = box.conf[0].item()
                            
                            bx1 += x_offset; by1 += y_offset
                            bx2 += x_offset; by2 += y_offset

                            # Smoothing Logic & Display
                            if self.current_mode in [2, 3]:
                                raw_label = self.active_class_names[cls] if cls < len(self.active_class_names) else str(cls)
                                stable_label = self._get_stable_prediction(raw_label)
                                
                                # Color lookup
                                try:
                                    stable_idx = self.active_class_names.index(stable_label)
                                    color = self.active_class_colors.get(stable_idx, (0, 255, 0))
                                except: color = (0, 255, 0)
                                
                                label_text = f"{stable_label} ({conf:.2f})"
                            else:
                                color = self.active_class_colors.get(cls, (0, 255, 0))
                                raw_label = self.active_class_names[cls] if cls < len(self.active_class_names) else str(cls)
                                label_text = f"{raw_label} ({conf:.2f})"
                            
                            cv2.rectangle(annotated_frame, (bx1, by1), (bx2, by2), color, 2)
                            cv2.putText(annotated_frame, label_text, (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                except Exception as e: print(f"Error: {e}")

            cv2image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            self.master.after(0, lambda img=cv2image: self._update_video_frame(img))
            if self.is_file_source_video: time.sleep(0.03)

    def _update_video_frame(self, cv2image):
        w = self.video_label.winfo_width(); h = self.video_label.winfo_height()
        if w > 10 and h > 10:
            img = Image.fromarray(cv2image).resize((w, h), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=self.tk_img)

    def on_closing(self):
        self.is_running = False; 
        if self.cap: self.cap.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YOLOTrackerApp(root)
    if app.is_running: root.mainloop()