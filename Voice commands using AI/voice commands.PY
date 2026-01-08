import os

# --- SILENCE LOGS ---
os.environ['GLOG_minloglevel'] = '3' 
os.environ['GRPC_VERBOSITY'] = 'NONE'
os.environ['ABSL_LOG_MIN_LEVEL'] = '3'
# --------------------

import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import colorsys
import threading
import time
from queue import Queue, Empty 
import speech_recognition as sr
import pyttsx3

# --- Configuration ---
MODEL_NAME = 'yolov8m.pt' 
BASE_WIDTH = 1280 
BASE_HEIGHT = 720
CONFIDENCE_THRESHOLD = 0.4
RECORDING_FPS = 30.0 

# !!! SET YOUR CAMERA ID HERE !!!
# 0 = Laptop Internal Camera
# 1 = USB External Camera
WEBCAM_ID = 0

# --- Colors ---
BG_DEEP_CHARCOAL = "#1e1e1e"        
BG_CONTROL_DARK = "#252526"         
FG_HIGH_CONTRAST = "#ffffff"        
FG_MUTED = "#cccccc"                
FRAME_BORDER = "#555555"            
BUTTON_NORMAL_BG = "#333333"        
BUTTON_NORMAL_FG = "#cccccc"        
BUTTON_ACTIVE_BG = "#007ACC"        
BUTTON_ACTIVE_FG = "#ffffff"        
BUTTON_RAISED_HOVER = "#444444"      
BUTTON_RECORD_ON = "#e74c3c"        
BUTTON_RECORD_OFF = "#f1c40f"       
ASSISTANT_BG = "#000000"
MIC_IDLE = "#ffffff"
MIC_LISTENING = "#00ff00" 
MIC_PROCESSING = "#00ccff" 
MIC_ERROR = "#ff0000" 

# ----------------------------------------
# DATA STRUCTURES
# ----------------------------------------
CLASSIFIED_COCO_CLASSES = {
    "People": { 0: 'person' },
    "Animals": { 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe' },
    "Vehicles": { 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat' },
    "Outdoors": { 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench' },
    "Sports": { 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket' },
    "Kitchen": { 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake' },
    "Furniture": { 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush' },
    "Electronics": { 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 67: 'cell phone', 68: 'microwave', 66: 'keyboard', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator' }
}

NAME_TO_ID = {} 
for group in CLASSIFIED_COCO_CLASSES.values():
    for cid, name in group.items():
        NAME_TO_ID[name] = cid

# --- SMART KEYWORD MAP ---
KEYWORD_MAP = {
    # APP COMMANDS
    "pause": ("CMD", "PAUSE"), "freeze": ("CMD", "PAUSE"),
    "continue": ("CMD", "CONTINUE"), "resume": ("CMD", "CONTINUE"), "play": ("CMD", "CONTINUE"),
    "record": ("CMD", "REC_START"), "recording": ("CMD", "REC_START"),
    "stop record": ("CMD", "REC_STOP"), "save": ("CMD", "REC_STOP"), "cut": ("CMD", "REC_STOP"),
    "quit": ("CMD", "QUIT"), "exit": ("CMD", "QUIT"), "close": ("CMD", "QUIT"),
    "status": ("CMD", "STATUS"), "count": ("CMD", "STATUS"), "how many": ("CMD", "STATUS"),
    "all": ("CMD", "ALL"), "everything": ("CMD", "ALL"),

    # CATEGORIES
    "vehicle": ("CAT", "Vehicles"), "transport": ("CAT", "Vehicles"),
    "animal": ("CAT", "Animals"), "creature": ("CAT", "Animals"),
    "food": ("CAT", "Kitchen"), "kitchen": ("CAT", "Kitchen"),
    "sport": ("CAT", "Sports"), 
    "electronic": ("CAT", "Electronics"), "device": ("CAT", "Electronics"),
    "furniture": ("CAT", "Furniture"), 
    
    # OBJECT SYNONYMS
    "human": ("OBJ", "person"), "humans": ("OBJ", "person"), "guy": ("OBJ", "person"), "girl": ("OBJ", "person"), "man": ("OBJ", "person"), "woman": ("OBJ", "person"), "people": ("OBJ", "person"),
    "auto": ("OBJ", "car"), "cab": ("OBJ", "car"), "taxi": ("OBJ", "car"),
    "bike": ("OBJ", "motorcycle"), "scooter": ("OBJ", "motorcycle"),
    "plane": ("OBJ", "airplane"), "jet": ("OBJ", "airplane"),
    "screen": ("OBJ", "tv"), "monitor": ("OBJ", "tv"), "telly": ("OBJ", "tv"),
    "phone": ("OBJ", "cell phone"), "mobile": ("OBJ", "cell phone"),
    "puppy": ("OBJ", "dog"), "kitty": ("OBJ", "cat"),
}

# Auto-add standard YOLO names
for name in NAME_TO_ID.keys():
    KEYWORD_MAP[name] = ("OBJ", name)
    KEYWORD_MAP[name + "s"] = ("OBJ", name) 

# --- SPEECH ENGINE ---
class SpeechEngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.queue = Queue()
        self.daemon = True 
        self.start()

    def run(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            while True:
                text = self.queue.get()
                if text is None: break 
                try:
                    engine.say(text)
                    engine.runAndWait()
                except: pass
                self.queue.task_done()
        except: pass

    def say(self, text):
        self.queue.put(text)

# --- VIDEO RECORDER ---
class VideoRecorder(threading.Thread):
    def __init__(self, frame_queue, width, height, fps, filepath): 
        super().__init__()
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.fps = fps
        self.filepath = filepath
        self.stop_event = threading.Event()
        self.is_paused = False
        self.video_writer = None
        
    def run(self):
        if not self.filepath: return
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        try:
            self.video_writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))
        except: return
        while not self.stop_event.is_set():
            try: frame = self.frame_queue.get(timeout=0.01) 
            except: continue
            if not self.is_paused: self.video_writer.write(frame)
                
    def stop(self, save_file=True):
        self.stop_event.set()
        self.join() 
        if self.video_writer:
            self.video_writer.release()
            if not save_file and os.path.exists(self.filepath):
                try: os.remove(self.filepath)
                except: pass

# --- LOCAL AI ASSISTANT ---
class AIAssistant:
    def __init__(self, master_app):
        self.app = master_app
        self.window = tk.Toplevel(master_app.master)
        self.window.title("AI")
        self.window.geometry("150x180+50+50") 
        self.window.configure(bg=ASSISTANT_BG)
        self.window.attributes('-topmost', True)
        self.window.overrideredirect(True) 
        
        self.speech_engine = SpeechEngine()
        try: self.recognizer = sr.Recognizer()
        except: pass

        # GUI
        self.status_label = tk.Label(self.window, text="LOCAL AI", fg="white", bg=ASSISTANT_BG, font=("Arial", 8, "bold"))
        self.status_label.pack(pady=5)
        self.canvas = tk.Canvas(self.window, width=80, height=80, bg=ASSISTANT_BG, highlightthickness=0)
        self.canvas.pack(pady=5)
        self.mic_circle = self.canvas.create_oval(10, 10, 70, 70, fill=BUTTON_NORMAL_BG, outline=MIC_IDLE, width=3)
        self.canvas.create_rectangle(35, 25, 45, 45, fill="white", outline="")
        self.canvas.create_line(40, 45, 40, 55, fill="white", width=2)
        self.canvas.create_line(30, 55, 50, 55, fill="white", width=2)
        self.canvas.bind("<Button-1>", self.start_listening_thread)
        self.instruction_label = tk.Label(self.window, text="Click & Speak", fg="#888888", bg=ASSISTANT_BG, font=("Arial", 8))
        self.instruction_label.pack(pady=5)
        tk.Button(self.window, text="X", command=self.window.destroy, bg="#330000", fg="white", bd=0, font=("Arial", 8)).place(x=130, y=0)

    def update_visuals(self, state):
        # We queue this update to the main thread too, just in case
        self.app.main_queue.put(("VISUAL", state))

    def _real_update_visuals(self, state):
        # This is called by the main thread
        color = MIC_IDLE; text = "READY"
        if state == "listening": color = MIC_LISTENING; text = "LISTENING..."
        elif state == "processing": color = MIC_PROCESSING; text = "THINKING..."
        elif state == "error": color = MIC_ERROR; text = "ERROR"
        self.canvas.itemconfig(self.mic_circle, outline=color)
        self.status_label.config(text=text, fg=color)
        self.window.update()

    def speak(self, text):
        print(f"AI: {text}")
        self.speech_engine.say(text)

    def start_listening_thread(self, event=None):
        threading.Thread(target=self.listen_and_process).start()

    def listen_and_process(self):
        self.update_visuals("listening")
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
            try:
                print("Listening...")
                audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=5)
                self.update_visuals("processing")
                
                try:
                    user_text = self.recognizer.recognize_google(audio).lower()
                    print(f"I Heard: '{user_text}'")
                    self.process_local_brain(user_text)
                except sr.UnknownValueError:
                    print("Speech unintelligible")
                    self.update_visuals("error")
                    self.speak("I didn't understand.")
                    time.sleep(1); self.update_visuals("idle")
                except sr.RequestError:
                    print("Internet connection failed for speech rec")
                    self.update_visuals("error")
                    self.speak("I need internet to hear you.")
                    time.sleep(1); self.update_visuals("idle")
                    
            except sr.WaitTimeoutError:
                self.update_visuals("idle")

    def process_local_brain(self, text):
        # Determine Intent
        is_negative = "stop" in text or "don't" in text or "remove" in text or "undetect" in text
        is_additive = "add" in text or "also" in text or "and" in text

        found_objects = []
        found_categories = []
        found_command = None

        for key, (type, value) in KEYWORD_MAP.items():
            if key in text: 
                if type == "CMD": found_command = value
                elif type == "CAT": found_categories.append(value)
                elif type == "OBJ": found_objects.append(value)

        # --- QUEUE ACTIONS FOR MAIN THREAD (NO DIRECT CALLS) ---
        
        # 1. APP COMMANDS
        if found_command:
            # We must map "STATUS" command here to speak
            if found_command == "STATUS":
                self.app.main_queue.put(("SPEAK_STATUS", None))
            else:
                self.app.main_queue.put(("CMD", found_command))
            self.update_visuals("idle")
            return

        # 2. DETECT ALL
        if "everything" in text or "all" in text:
            self.speak("Activating full detection.")
            self.app.main_queue.put(("ALL", True))
            self.update_visuals("idle")
            return

        # 3. CATEGORIES
        if found_categories:
            cat = found_categories[0]
            self.speak(f"Detecting {cat}.")
            reset = not is_additive and not is_negative
            self.app.main_queue.put(("CATEGORY", (cat, not is_negative, reset)))
            self.update_visuals("idle")
            return

        # 4. OBJECTS
        if found_objects:
            found_objects = list(set(found_objects))
            obj_str = ", ".join(found_objects)
            
            if is_negative:
                self.speak(f"Stopping {obj_str}.")
                self.app.main_queue.put(("OBJECTS", (found_objects, False, False)))
            elif is_additive:
                self.speak(f"Adding {obj_str}.")
                self.app.main_queue.put(("OBJECTS", (found_objects, True, False)))
            else:
                self.speak(f"Detecting {obj_str}.")
                self.app.main_queue.put(("OBJECTS", (found_objects, True, True))) # Reset=True
            self.update_visuals("idle")
            return

        self.speak("I heard you, but I don't know that command.")
        self.update_visuals("idle")

# --- APP CLASS ---
class YoloWebcamFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Video Feed")
        master.configure(bg=BG_DEEP_CHARCOAL) 
        
        # --- THREAD SAFE MESSAGE QUEUE ---
        self.main_queue = Queue() 
        # ---------------------------------

        self.is_running = True
        self.video_loop_id = None
        self.is_fullscreen = False
        self.is_controls_hidden_manual = False 
        self.latest_detection_count = 0 
        self.current_camera_index = WEBCAM_ID # Set initial ID from Config
        
        # Monitoring
        self.monitoring_active = False
        self.last_spoken_time = 0
        self.last_count = -1
        
        self.is_recording = False
        self.frame_queue = Queue(maxsize=30) 
        self.recorder_thread = None
        self.record_btn = None
        self.stop_record_btn = None

        self.control_window = None 
        self.category_buttons = {} 
        self.all_toggle_button = None    
        self.group_toggle_button = None  
        self.class_vars = {}
        self.checkbox_widgets = {}
        self.single_class_message = None 
        
        self.model = self._load_model()
        self.video_capture = self._init_webcam_auto()
        self.current_tk_image = None
        self.current_category_name = None
        self.class_colors = self._generate_colors()

        self._setup_video_layout()
        self._setup_control_window() 
        self._initial_geometry() 
        self._initialize_class_display()
        
        self.master.bind('l', self._toggle_fullscreen_event)
        self.master.bind('<Escape>', self._exit_fullscreen_event)
        self.master.bind('h', self._toggle_controls_visibility_event) 
        
        self.ai_assistant = AIAssistant(self)
        
        # Start the Queue Processor Loop
        self.process_queue_loop()
        self.video_loop()

    # --- MAIN THREAD QUEUE PROCESSOR (THE FIX) ---
    def process_queue_loop(self):
        try:
            while True:
                msg_type, data = self.main_queue.get_nowait()
                
                if msg_type == "VISUAL":
                    self.ai_assistant._real_update_visuals(data)
                elif msg_type == "CMD":
                    self.execute_app_command(data)
                elif msg_type == "ALL":
                    for var in self.class_vars.values(): var.set(data)
                    self.refresh_gui()
                    self.monitoring_active = True
                elif msg_type == "CATEGORY":
                    cat, state, reset = data
                    if reset: 
                        for var in self.class_vars.values(): var.set(False)
                    for cid in CLASSIFIED_COCO_CLASSES[cat]: self.class_vars[cid].set(state)
                    self.show_category_classes(cat)
                    self.refresh_gui()
                    self.monitoring_active = True
                elif msg_type == "OBJECTS":
                    objs, state, reset = data
                    if reset: 
                        for var in self.class_vars.values(): var.set(False)
                    for name in objs:
                        if name in NAME_TO_ID:
                            self.class_vars[NAME_TO_ID[name]].set(state)
                    self.refresh_gui()
                    self.monitoring_active = True
                elif msg_type == "SPEAK_STATUS":
                    count = self.latest_detection_count
                    self.ai_assistant.speak(f"I see {count} items.")

        except Empty:
            pass
        self.master.after(100, self.process_queue_loop)

    def execute_app_command(self, cmd):
        if cmd == "APP_PAUSE": self._pause_webcam()
        elif cmd == "APP_CONTINUE": self._continue_webcam()
        elif cmd == "APP_QUIT": self.on_closing()
        elif cmd == "APP_REC_START": self.start_recording_auto()
        elif cmd == "APP_REC_STOP": self.stop_recording_auto()
        elif cmd == "APP_STATUS": self.monitoring_active = True

    # --- WEBCAM ---
    def _init_webcam_auto(self):
        print(f"Attempting to open camera index {self.current_camera_index}...")
        # Priority 1: Try the configured WEBCAM_ID
        cap = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_HEIGHT)
                return cap
            else: cap.release()
        
        # Priority 2: Scan for any working camera
        print("Default camera failed. Scanning for others...")
        for i in range(5):
            if i == self.current_camera_index: continue
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    print(f"Found working camera at index {i}")
                    self.current_camera_index = i
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_WIDTH)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_HEIGHT)
                    return cap
                else: cap.release()
        messagebox.showerror("Error", "No working camera found.")
        return None

    def switch_camera(self):
        if self.video_capture: self.video_capture.release()
        self.current_camera_index += 1
        if self.current_camera_index > 3: self.current_camera_index = 0
        self.video_capture = cv2.VideoCapture(self.current_camera_index, cv2.CAP_DSHOW)
        if self.video_capture.isOpened():
            self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_WIDTH)
            self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_HEIGHT)

    # --- MONITORING ---
    def check_monitoring(self):
        if not self.monitoring_active: return
        current_time = time.time()
        if current_time - self.last_spoken_time > 5.0:
            count = self.latest_detection_count
            if count != self.last_count or count > 0:
                msg = "No objects." if count == 0 else f"I see {count} items."
                self.ai_assistant.speak(msg)
                self.last_spoken_time = current_time
                self.last_count = count

    # --- RECORDING ---
    def start_recording_manual(self):
        if self.is_recording: return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"yolo_rec_{timestamp}.avi"
        filepath = filedialog.asksaveasfilename(
            defaultextension=".avi", 
            initialfile=filename,
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")] 
        )
        if not filepath: return
        self._start_rec_internal(filepath)

    def start_recording_auto(self):
        if self.is_recording: self.ai_assistant.speak("Already recording."); return
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(os.getcwd(), f"AI_RECORDING_{timestamp}.avi")
        self.ai_assistant.speak("Recording started.")
        self._start_rec_internal(filepath)

    def _start_rec_internal(self, filepath):
        self.is_recording = True
        self.recorder_thread = VideoRecorder(self.frame_queue, BASE_WIDTH, BASE_HEIGHT, RECORDING_FPS, filepath)
        self.recorder_thread.start()
        self.record_btn.config(state=tk.DISABLED, bg=BUTTON_RECORD_ON)
        self.stop_record_btn.config(state=tk.NORMAL, bg=BUTTON_RECORD_OFF, fg=FG_HIGH_CONTRAST)

    def stop_recording_auto(self):
        if not self.is_recording: self.ai_assistant.speak("Not recording."); return
        self._stop_recording_action(save=True)
        self.ai_assistant.speak("Recording saved.")

    def stop_recording_prompt(self):
        if not self.is_recording: return
        if messagebox.askyesno("Stop", "Save video?"): self._stop_recording_action(save=True)
        else: self._stop_recording_action(save=False)

    def _stop_recording_action(self, save):
        if not self.recorder_thread: return
        with self.frame_queue.mutex: self.frame_queue.queue.clear()
        self.recorder_thread.stop(save_file=save)
        self.is_recording = False
        self.recorder_thread = None
        self.record_btn.config(state=tk.NORMAL, bg=BUTTON_NORMAL_BG)
        self.stop_record_btn.config(state=tk.DISABLED, bg=BUTTON_NORMAL_BG, fg=FG_MUTED)

    # --- CORE ---
    def _load_model(self):
        try: return YOLO(MODEL_NAME)
        except Exception as e: messagebox.showerror("Error", f"Failed to load YOLO model: {e}"); self.master.destroy(); return None

    def _generate_colors(self):
        colors = {}; num_classes = len(NAME_TO_ID)
        for i, class_id in enumerate(NAME_TO_ID.keys()):
            hue = i / float(num_classes); rgb_float = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors[class_id] = (int(rgb_float[2] * 255), int(rgb_float[1] * 255), int(rgb_float[0] * 255))
        return colors

    def display_frame(self, cv_image):
        try:
            label_w = self.image_label.winfo_width(); label_h = self.image_label.winfo_height()
            if label_w < 10 or label_h < 10: label_w, label_h = BASE_WIDTH, BASE_HEIGHT
            resized = cv2.resize(cv_image, (label_w, label_h))
            pil_image = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            self.current_tk_image = ImageTk.PhotoImage(pil_image)
            self.image_label.configure(image=self.current_tk_image, text="", anchor='center')
            self.image_label.image = self.current_tk_image
        except: pass
            
    def detect_and_filter(self, frame):
        target_class_ids = [cid for cid, var in self.class_vars.items() if var.get()]
        current_count = 0
        if not target_class_ids: self.latest_detection_count = 0; return frame
        annotated_frame = frame.copy() 
        results = self.model(annotated_frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=target_class_ids)
        for result in results:
            boxes = result.boxes; current_count += len(boxes)
            for box in boxes:
                class_id = int(box.cls.cpu().numpy()[0]); x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].cpu().numpy()]
                color = self.class_colors.get(class_id, (0, 255, 255))
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_frame, self.model.names[class_id], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        self.latest_detection_count = current_count
        return annotated_frame

    def video_loop(self):
        if not self.is_running: return
        if self.video_capture and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            if ret:
                annotated_frame = self.detect_and_filter(frame)
                self.display_frame(annotated_frame)
                self.check_monitoring()
                if self.is_recording:
                    try: self.frame_queue.put_nowait(cv2.resize(annotated_frame, (BASE_WIDTH, BASE_HEIGHT)))
                    except: pass 
        if self.is_running: self.video_loop_id = self.master.after(10, self.video_loop)

    def refresh_gui(self):
        self.update_all_toggle_button_color()
        if self.current_category_name:
            self.update_category_button_color(self.current_category_name)
            self.update_group_toggle_button_color(self.current_category_name)
        for cat in CLASSIFIED_COCO_CLASSES.keys():
            self.update_category_button_color(cat)

    # --- UI SETUP ---
    def _setup_video_layout(self):
        self.main_frame = tk.Frame(self.master, padx=0, pady=0, bg=BG_DEEP_CHARCOAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.main_frame, text="Initializing...", bg=BG_DEEP_CHARCOAL, fg=FG_MUTED)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=0) 

    def _setup_control_window(self):
        self.control_window = tk.Toplevel(self.master, bg=BG_CONTROL_DARK)
        self.control_window.title("Controls")
        self.control_window.resizable(True, True) 
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.control_frame = tk.LabelFrame(self.control_window, text="Filters", padx=10, pady=10, bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, bd=2, relief=tk.FLAT, highlightbackground=FRAME_BORDER, highlightcolor=FRAME_BORDER, font=('Inter', 10, 'bold'))
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.class_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        self._pre_generate_all_checkboxes() 
        self.single_class_message = tk.Label(self.class_frame, text="", bg=BG_CONTROL_DARK, fg="#2ecc71", font=('Inter', 10, 'bold'), pady=10)
        
        category_btn_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        self.all_toggle_button = tk.Button(category_btn_frame, text="All: Toggle", command=self.toggle_all_classes, bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER)
        self.all_toggle_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)
        for category_name in CLASSIFIED_COCO_CLASSES.keys():
            btn = tk.Button(category_btn_frame, text=category_name, command=lambda name=category_name: self.show_category_classes(name), bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9), fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.category_buttons[category_name] = btn 
        
        toggle_label_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        self.class_label = tk.Label(toggle_label_frame, text="Toggles:", bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, font=('Inter', 9))
        self.class_label.pack(side=tk.LEFT, anchor="w")
        control_btn_frame = tk.Frame(toggle_label_frame, bg=BG_CONTROL_DARK)
        control_btn_frame.pack(side=tk.RIGHT, padx=10)
        
        tk.Button(control_btn_frame, text="⏸︎", command=self._pause_webcam, bg="#e74c3c", relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG).pack(side=tk.LEFT, padx=2)
        tk.Button(control_btn_frame, text="▶︎", command=self._continue_webcam, bg="#2ecc71", relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG).pack(side=tk.LEFT, padx=2)
        tk.Button(control_btn_frame, text="📷", command=self.switch_camera, bg="#9b59b6", relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG).pack(side=tk.LEFT, padx=2)
        self.record_btn = tk.Button(control_btn_frame, text="🔴 Rec", command=self.start_recording_manual, bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        self.stop_record_btn = tk.Button(control_btn_frame, text="■ Stop", command=self.stop_recording_prompt, bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=FG_MUTED, state=tk.DISABLED)
        self.stop_record_btn.pack(side=tk.LEFT, padx=2)
        
        self.group_toggle_button = tk.Button(toggle_label_frame, text="Toggle Group", command=self.toggle_current_group, bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 8), activebackground=BUTTON_RAISED_HOVER, fg=BUTTON_NORMAL_FG)
        self.group_toggle_button.pack(side=tk.RIGHT, padx=5)
        
        category_btn_frame.pack(fill=tk.X, pady=(0, 10))
        toggle_label_frame.pack(fill=tk.X, pady=(0, 5))
        self.class_frame.pack(fill=tk.BOTH, expand=True) 
        self.control_window.update_idletasks()

    # --- HELPER METHODS ---
    def _initial_geometry(self):
        try: self.master.state('zoomed') 
        except: self.master.geometry(f"{self.master.winfo_screenwidth()}x{self.master.winfo_screenheight()}")
        self.control_window.update_idletasks()
        self.control_window.geometry(f"{self.control_window.winfo_reqwidth()}x{self.control_window.winfo_reqheight()}+{self.master.winfo_screenwidth()-self.control_window.winfo_reqwidth()-20}+20")

    def _initialize_class_display(self):
        self.current_category_name = None; self.update_all_toggle_button_color()
        self.group_toggle_button.config(state=tk.DISABLED, bg=BUTTON_NORMAL_BG, fg=FG_MUTED)
        for category_name in CLASSIFIED_COCO_CLASSES.keys(): self.update_category_button_color(category_name)

    def _toggle_controls_visibility_event(self, event=None):
        self.is_controls_hidden_manual = not self.is_controls_hidden_manual
        if self.is_controls_hidden_manual: self.control_window.withdraw()
        else: self.control_window.deiconify() 

    def _toggle_fullscreen_event(self, event=None):
        self.is_fullscreen = not self.is_fullscreen; self.master.attributes('-fullscreen', self.is_fullscreen)

    def _exit_fullscreen_event(self, event=None):
        self.is_fullscreen = False; self.master.attributes('-fullscreen', False)
                
    def on_checkbox_toggle(self, category_name):
        self.update_category_button_color(category_name)
        if category_name == self.current_category_name: self.update_group_toggle_button_color(category_name)
        self.update_all_toggle_button_color()

    def update_button_state(self, button, is_active):
        if button: button.config(bg=BUTTON_ACTIVE_BG if is_active else BUTTON_NORMAL_BG, fg=BUTTON_ACTIVE_FG if is_active else BUTTON_NORMAL_FG)

    def update_all_toggle_button_color(self):
        self.update_button_state(self.all_toggle_button, any(var.get() for var in self.class_vars.values()))

    def update_group_toggle_button_color(self, category_name):
        class_dict = CLASSIFIED_COCO_CLASSES.get(category_name, {})
        self.update_button_state(self.group_toggle_button, any(self.class_vars.get(cid).get() for cid in class_dict.keys() if cid in self.class_vars))

    def update_category_button_color(self, category_name):
        self.update_button_state(self.category_buttons.get(category_name), any(self.class_vars.get(cid).get() for cid in CLASSIFIED_COCO_CLASSES.get(category_name, {}).keys() if cid in self.class_vars))

    def toggle_current_group(self):
        if not self.current_category_name: return
        class_dict = CLASSIFIED_COCO_CLASSES.get(self.current_category_name)
        new_state = not all(self.class_vars.get(cid).get() for cid in class_dict.keys() if cid in self.class_vars)
        for cid in class_dict.keys(): 
            if cid in self.class_vars: self.class_vars[cid].set(new_state)
        self.on_checkbox_toggle(self.current_category_name)

    def toggle_all_classes(self):
        new_state = not all(var.get() for var in self.class_vars.values())
        for var in self.class_vars.values(): var.set(new_state)
        for category_name in CLASSIFIED_COCO_CLASSES.keys(): self.update_category_button_color(category_name)
        self.update_all_toggle_button_color() 
        if self.current_category_name: self.update_group_toggle_button_color(self.current_category_name)

    def show_category_classes(self, category_name):
        for widget in self.checkbox_widgets.values(): widget.grid_forget()
        self.single_class_message.grid_forget()
        class_dict = CLASSIFIED_COCO_CLASSES.get(category_name, {})
        if len(class_dict) == 1:
            cid = next(iter(class_dict.keys()))
            self.class_vars[cid].set(not self.class_vars[cid].get())
            self.on_checkbox_toggle(category_name) 
            self.single_class_message.grid(row=0, column=0, columnspan=8, sticky="n")
            self.group_toggle_button.config(state=tk.DISABLED)
        else:
            for i, cid in enumerate(class_dict.keys()):
                if cid in self.checkbox_widgets: self.checkbox_widgets[cid].grid(row=(i//8)+1, column=i%8, padx=5, pady=2, sticky="w")
            self.group_toggle_button.config(state=tk.NORMAL)
        self.current_category_name = category_name
        self.class_label.config(text=f"Toggles: {category_name}")
        self.update_group_toggle_button_color(category_name)
        self.control_window.update_idletasks()
        
    def _pre_generate_all_checkboxes(self):
        for category_name, class_dict in CLASSIFIED_COCO_CLASSES.items():
            for cid, name in class_dict.items():
                var = tk.BooleanVar(value=False)
                self.class_vars[cid] = var
                self.checkbox_widgets[cid] = tk.Checkbutton(self.class_frame, text=name, variable=var, bg=BG_CONTROL_DARK, fg=FG_MUTED, selectcolor=BG_CONTROL_DARK, font=('Inter', 9), anchor="w", command=lambda n=category_name: self.on_checkbox_toggle(n))

    def _pause_webcam(self):
        self.is_running = False
        if self.video_loop_id: self.master.after_cancel(self.video_loop_id)

    def _continue_webcam(self):
        if not self.is_running: self.is_running = True; self.video_loop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Quit?"):
            self.is_running = False
            if self.is_recording:
                self._stop_recording_action(save=True) # Auto save on force quit
            if self.video_capture: self.video_capture.release()
            self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloWebcamFilterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()