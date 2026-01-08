import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
import colorsys
import threading
import time
import os
from queue import Queue 

# --- Configuration ---
MODEL_NAME = 'yolov8m.pt' 

# Define a base resolution for the webcam capture
BASE_WIDTH = 1280 
BASE_HEIGHT = 720

# Webcam ID, Confidence, and FPS
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.4
RECORDING_FPS = 30.0 

# --- Professional Dark Theme Colors ---
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
# ----------------------------------------


# All 80 COCO classes
CLASSIFIED_COCO_CLASSES = {
    "People": { 0: 'person' },
    "Animals": {
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep',
        19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe'
    },
    "Vehicles & Transport": {
        1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
        6: 'train', 7: 'truck', 8: 'boat'
    },
    "Traffic & Outdoors": {
        9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter',
        13: 'bench'
    },
    "Sports & Accessories": {
        24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
        29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite',
        34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard',
        38: 'tennis racket'
    },
    "Kitchenware & Food": {
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife',
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich',
        49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza',
        54: 'donut', 55: 'cake'
    },
    "Furniture & Household": {
        56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
        61: 'toilet', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors',
        77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush' 
    },
    "Devices & Appliances": {
        62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 67: 'cell phone', 68: 'microwave', 
        66: 'keyboard', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator'
    }
}

# Flatten the categories for detection logic
SELECTABLE_CLASSES = {}
for group in CLASSIFIED_COCO_CLASSES.values():
    SELECTABLE_CLASSES.update(group)

# --- Video Recording Thread Class ---
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
        if not self.filepath:
            self.stop_event.set()
            return
            
        fourcc = cv2.VideoWriter_fourcc(*'XVID') 
        
        try:
            self.video_writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))
        except Exception as e:
            print(f"Failed to initialize VideoWriter: {e}")
            self.stop_event.set()
            return

        print(f"Recording started to: {self.filepath}")

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.01) 
            except Exception:
                continue

            if not self.is_paused:
                self.video_writer.write(frame)
                
    def stop(self, save_file=True): 
        self.stop_event.set()
        self.join() 
        if self.video_writer:
            self.video_writer.release()
            print("VideoWriter released.")
            
            if not save_file and os.path.exists(self.filepath):
                try:
                    os.remove(self.filepath)
                    print(f"Recording cancelled and temporary file deleted: {self.filepath}")
                except Exception as e:
                    print(f"Error deleting file {self.filepath}: {e}")
            elif save_file:
                 print("Recording stopped and file saved.")


# --- Application Class ---
class YoloWebcamFilterApp:
    def __init__(self, master):
        self.master = master
        master.title("Grayscale Video Feed") # Updated Title
        master.configure(bg=BG_DEEP_CHARCOAL) 

        # State Variables
        self.is_running = True
        self.video_loop_id = None
        self.is_fullscreen = False
        self.is_controls_hidden_manual = False 
        
        # Recording Variables
        self.is_recording = False
        self.frame_queue = Queue(maxsize=30) 
        self.recorder_thread = None
        self.current_recording_path = None 
        self.record_btn = None
        self.stop_record_btn = None

        # GUI Elements
        self.control_window = None 
        self.category_buttons = {} 
        self.all_toggle_button = None    
        self.group_toggle_button = None  
        self.class_vars = {}
        self.checkbox_widgets = {}
        self.single_class_message = None 
        
        # Initialization
        self.model = self._load_model()
        self.video_capture = self._init_webcam()
        self.current_tk_image = None
        self.current_category_name = None
        self.class_colors = self._generate_colors()

        self._setup_video_layout()
        self._setup_control_window() 
        
        self._initial_geometry() 
        self._initialize_class_display()
        
        # --- Keyboard Bindings ---
        self.master.bind('l', self._toggle_fullscreen_event)
        self.master.bind('<Escape>', self._exit_fullscreen_event)
        self.master.bind('h', self._toggle_controls_visibility_event) 
        # -----------------------------

        self.video_loop()

    # ----------------------------------------------------
    # --- Layout Setup for TWO WINDOWS ---
    # ----------------------------------------------------
    
    def _setup_video_layout(self):
        self.main_frame = tk.Frame(self.master, padx=0, pady=0, bg=BG_DEEP_CHARCOAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.main_frame, text="Initializing Webcam...", bg=BG_DEEP_CHARCOAL, fg=FG_MUTED)
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=0) 

    def _setup_control_window(self):
        self.control_window = tk.Toplevel(self.master, bg=BG_CONTROL_DARK)
        self.control_window.title("Class Filter Controls")
        self.control_window.resizable(True, True) 
        self.control_window.attributes('-topmost', True)
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.control_frame = tk.LabelFrame(self.control_window, text="Webcam Class Filtering", padx=10, pady=10, 
                                             bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, bd=2, relief=tk.FLAT, 
                                             highlightbackground=FRAME_BORDER, highlightcolor=FRAME_BORDER, 
                                             font=('Inter', 10, 'bold'))
        self.control_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.class_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        self._pre_generate_all_checkboxes() 
        self.single_class_message = tk.Label(self.class_frame, text="", bg=BG_CONTROL_DARK, fg="#2ecc71", font=('Inter', 10, 'bold'), pady=10)

        # A. Top Bar: All Toggle, Category Buttons
        category_btn_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        
        self.all_toggle_button = tk.Button(category_btn_frame, text="All: Toggle", command=self.toggle_all_classes,
                             bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'),
                             fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER)
        self.all_toggle_button.pack(side=tk.LEFT, padx=(0, 10), pady=5)
                             
        for category_name in CLASSIFIED_COCO_CLASSES.keys():
            btn = tk.Button(category_btn_frame, text=category_name,
                              command=lambda name=category_name: self.show_category_classes(name),
                              bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9),
                              fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER)
            btn.pack(side=tk.LEFT, padx=5, pady=5)
            self.category_buttons[category_name] = btn 

        # B. Second Bar: Labels and Control Buttons
        toggle_label_frame = tk.Frame(self.control_frame, bg=BG_CONTROL_DARK)
        self.class_label = tk.Label(toggle_label_frame, text="Individual Toggles:", bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, font=('Inter', 9))
        self.class_label.pack(side=tk.LEFT, anchor="w")
        control_btn_frame = tk.Frame(toggle_label_frame, bg=BG_CONTROL_DARK)
        control_btn_frame.pack(side=tk.RIGHT, padx=10)
        
        # Help Text
        tk.Label(toggle_label_frame, text="(H: Toggle Controls, L: Fullscreen, ESC: Exit)", bg=BG_CONTROL_DARK, font=('Inter', 8), fg=FG_MUTED).pack(side=tk.LEFT, padx=5)

        # Control Buttons
        tk.Button(control_btn_frame, text="⏸︎ Pause", command=self._pause_webcam, bg="#e74c3c", 
                             relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG, activebackground=BUTTON_RAISED_HOVER).pack(side=tk.LEFT, padx=2)
        tk.Button(control_btn_frame, text="▶︎ Continue", command=self._continue_webcam, bg="#2ecc71", 
                             relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG, activebackground=BUTTON_RAISED_HOVER).pack(side=tk.LEFT, padx=2)
        
        # --- RECORDING BUTTONS ---
        self.record_btn = tk.Button(control_btn_frame, text="🔴 Record", command=self.start_recording, 
                             bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG, activebackground=BUTTON_RAISED_HOVER)
        self.record_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_record_btn = tk.Button(control_btn_frame, text="■ Stop Rec", command=self.stop_recording_prompt, 
                             bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=FG_MUTED, state=tk.DISABLED, activebackground=BUTTON_RAISED_HOVER)
        self.stop_record_btn.pack(side=tk.LEFT, padx=2)
        
        tk.Button(control_btn_frame, text="■ Quit", command=self.on_closing, bg="#3498db", 
                             relief=tk.FLAT, font=('Inter', 9, 'bold'), fg=BUTTON_ACTIVE_FG, activebackground=BUTTON_RAISED_HOVER).pack(side=tk.LEFT, padx=2)
        
        # Toggle Current Group Button setup
        self.group_toggle_button = tk.Button(toggle_label_frame, text="Toggle Current Group", command=self.toggle_current_group,
                             bg=BUTTON_NORMAL_BG, relief=tk.FLAT, font=('Inter', 8),
                             activebackground=BUTTON_RAISED_HOVER, fg=BUTTON_NORMAL_FG)
        self.group_toggle_button.pack(side=tk.RIGHT, padx=5)

        # --- Packing Order inside control_frame ---
        category_btn_frame.pack(fill=tk.X, pady=(0, 10))
        toggle_label_frame.pack(fill=tk.X, pady=(0, 5))
        self.class_frame.pack(fill=tk.BOTH, expand=True) 
        
        self.control_window.update_idletasks()
        
    # ----------------------------------------------------
    # --- Recording Methods ---
    # ----------------------------------------------------

    def start_recording(self):
        if self.is_recording:
            messagebox.showinfo("Recording Status", "Recording is already in progress.")
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"yolo_recording_{timestamp}.avi"
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".avi",
            initialfile=filename,
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        
        if not filepath:
            messagebox.showinfo("Recording Status", "Recording startup cancelled.")
            return

        self.is_recording = True
        self.current_recording_path = filepath
        
        self.recorder_thread = VideoRecorder(
            self.frame_queue, BASE_WIDTH, BASE_HEIGHT, RECORDING_FPS, filepath
        )
        self.recorder_thread.start()
        
        self.record_btn.config(state=tk.DISABLED, bg=BUTTON_RECORD_ON)
        self.stop_record_btn.config(state=tk.NORMAL, bg=BUTTON_RECORD_OFF, fg=FG_HIGH_CONTRAST)


    def stop_recording_prompt(self):
        if not self.is_recording:
            messagebox.showinfo("Recording Status", "No recording is currently active.")
            return
            
        if messagebox.askyesno("Stop Recording", "Do you want to save the recorded video? Select 'No' to discard the file."):
            self._stop_recording_action(save=True)
            messagebox.showinfo("Recording Status", "Recording successfully saved!")
        else:
            self._stop_recording_action(save=False)
            messagebox.showinfo("Recording Status", "Recording discarded and temporary file deleted.")

    def _stop_recording_action(self, save):
        if not self.recorder_thread:
            return

        with self.frame_queue.mutex:
            self.frame_queue.queue.clear()
            
        self.recorder_thread.stop(save_file=save)
        
        self.is_recording = False
        self.recorder_thread = None
        self.current_recording_path = None 
        
        self.record_btn.config(state=tk.NORMAL, bg=BUTTON_NORMAL_BG)
        self.stop_record_btn.config(state=tk.DISABLED, bg=BUTTON_NORMAL_BG, fg=FG_MUTED)

    # ----------------------------------------------------
    # --- Existing Utility Methods ---
    # ----------------------------------------------------

    def _load_model(self):
        try:
            model = YOLO(MODEL_NAME)
            return model
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load YOLO model: {e}")
            self.master.destroy()
            return None

    def _init_webcam(self):
        cap = cv2.VideoCapture(WEBCAM_ID)
        if not cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam. Check WEBCAM_ID or permissions.")
            self.master.destroy()
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_HEIGHT)
        return cap
        
    def _generate_colors(self):
        colors = {}
        num_classes = len(SELECTABLE_CLASSES)
        for i, class_id in enumerate(SELECTABLE_CLASSES.keys()):
            hue = i / float(num_classes)
            saturation = 0.9 
            value = 0.9      
            rgb_float = colorsys.hsv_to_rgb(hue, saturation, value)
            b = int(rgb_float[2] * 255)
            g = int(rgb_float[1] * 255)
            r = int(rgb_float[0] * 255)
            colors[class_id] = (b, g, r)
        return colors

    def display_frame(self, cv_image):
        try:
            label_w = self.image_label.winfo_width()
            label_h = self.image_label.winfo_height()

            if label_w < 10 or label_h < 10:
                label_w, label_h = BASE_WIDTH, BASE_HEIGHT
                    
            resized_cv_image = cv2.resize(cv_image, (label_w, label_h))
            
            # Note: Since the input is Grayscale (but 3 channel), this 
            # BGR2RGB conversion just keeps it Grayscale, which is correct.
            rgb_image = cv2.cvtColor(resized_cv_image, cv2.COLOR_BGR2RGB)
            
            pil_image = Image.fromarray(rgb_image)
            self.current_tk_image = ImageTk.PhotoImage(pil_image)

            self.image_label.configure(image=self.current_tk_image, text="", anchor='center')
            self.image_label.image = self.current_tk_image

        except Exception:
            pass
            
    def detect_and_filter(self, frame):
        # ---------------------------------------------------------
        # GRAYSCALE MODIFICATION START
        # ---------------------------------------------------------
        # 1. Convert incoming Color frame to Grayscale (1 Channel)
        gray_single_channel = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 2. Convert Grayscale back to "Fake BGR" (3 Channels)
        # We assume R=G=B. This is necessary because YOLO expects 3 channels.
        frame_ready = cv2.cvtColor(gray_single_channel, cv2.COLOR_GRAY2BGR)
        # ---------------------------------------------------------
        
        target_class_ids = [cid for cid, var in self.class_vars.items() if var.get()]
        
        # If no classes selected, return the Grayscale frame
        if not target_class_ids: 
            return frame_ready
        
        # Use the Grayscale (3-channel) frame for annotation
        annotated_frame = frame_ready.copy() 
        
        results = self.model(annotated_frame, verbose=False, conf=CONFIDENCE_THRESHOLD, classes=target_class_ids)
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls.cpu().numpy()[0])
                x1, y1, x2, y2 = [int(x) for x in box.xyxy[0].cpu().numpy()]
                class_name = self.model.names[class_id]
                color = self.class_colors.get(class_id, (0, 255, 255))
                
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{class_name}"
                cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
        return annotated_frame

    def video_loop(self):
        if not self.is_running: return
        if self.video_capture is None or not self.video_capture.isOpened():
            self.is_running = False
            return

        ret, frame = self.video_capture.read()
        if ret:
            # 1. Process Detection (Now handles Grayscale conversion inside)
            annotated_frame = self.detect_and_filter(frame)
            
            # 2. Display Frame
            self.display_frame(annotated_frame)
            
            # 3. Handle Recording
            if self.is_recording:
                record_frame = cv2.resize(annotated_frame, (BASE_WIDTH, BASE_HEIGHT))
                try:
                    self.frame_queue.put_nowait(record_frame)
                except Exception:
                    pass 
        else:
            print("Error: Could not read frame from camera.")
            self._pause_webcam() 
            self.image_label.configure(text="WEBCAM STREAM LOST. Click 'Continue' or 'Quit'.", 
                                             fg="#e74c3c", font=('Inter', 12, 'bold'))

        if self.is_running:
              self.video_loop_id = self.master.after(10, self.video_loop)


    # --- Tkinter UI and State Methods (same as before) ---

    def _initial_geometry(self):
        try:
            self.master.state('zoomed') 
        except tk.TclError:
            screen_w = self.master.winfo_screenwidth()
            screen_h = self.master.winfo_screenheight()
            self.master.geometry(f"{screen_w}x{screen_h}")
        
        self.control_window.update_idletasks()
        screen_w = self.master.winfo_screenwidth()
        control_w = self.control_window.winfo_reqwidth()
        control_h = self.control_window.winfo_reqheight()
        
        x_pos = screen_w - control_w - 20 
        y_pos = 20                      
        self.control_window.geometry(f"{control_w}x{control_h}+{x_pos}+{y_pos}")

    def _initialize_class_display(self):
        self.current_category_name = None 
        self.update_all_toggle_button_color()
        self.group_toggle_button.config(state=tk.DISABLED, bg=BUTTON_NORMAL_BG, fg=FG_MUTED, text="Select Group")
        for category_name in CLASSIFIED_COCO_CLASSES.keys():
            self.update_category_button_color(category_name)

    def _toggle_controls_visibility_event(self, event=None):
        self.is_controls_hidden_manual = not self.is_controls_hidden_manual
        if self.is_controls_hidden_manual:
            self.control_window.withdraw()
        else:
            self.control_window.deiconify() 

    def _toggle_fullscreen_event(self, event=None):
        self.is_fullscreen = not self.is_fullscreen
        self.master.attributes('-fullscreen', self.is_fullscreen)
        if not self.is_fullscreen:
            try:
                self.master.state('zoomed')
            except tk.TclError:
                self.master.state('normal')

    def _exit_fullscreen_event(self, event=None):
        if self.is_fullscreen:
            self.master.attributes('-fullscreen', False)
            self.is_fullscreen = False
            try:
                self.master.state('zoomed')
            except tk.TclError:
                self.master.state('normal')
                
    def on_checkbox_toggle(self, category_name):
        self.update_category_button_color(category_name)
        if category_name == self.current_category_name:
            self.update_group_toggle_button_color(category_name)
        self.update_all_toggle_button_color()

    def update_button_state(self, button, is_active):
        if button:
            if is_active:
                button.config(bg=BUTTON_ACTIVE_BG, fg=BUTTON_ACTIVE_FG)
            else:
                button.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)

    def update_all_toggle_button_color(self):
        is_any_checked = any(var.get() for var in self.class_vars.values())
        self.update_button_state(self.all_toggle_button, is_any_checked)

    def update_group_toggle_button_color(self, category_name):
        class_dict = CLASSIFIED_COCO_CLASSES.get(category_name, {})
        is_any_checked_in_group = any(self.class_vars.get(class_id, tk.BooleanVar(value=False)).get() for class_id in class_dict.keys())
        self.update_button_state(self.group_toggle_button, is_any_checked_in_group)

    def update_category_button_color(self, category_name):
        class_dict = CLASSIFIED_COCO_CLASSES.get(category_name, {})
        is_any_checked = any(self.class_vars.get(class_id, tk.BooleanVar(value=False)).get() for class_id in class_dict.keys())
        button = self.category_buttons.get(category_name)
        self.update_button_state(button, is_any_checked)

    def toggle_current_group(self):
        if not self.current_category_name: return
        class_dict = CLASSIFIED_COCO_CLASSES.get(self.current_category_name)
        if not class_dict: return
        all_checked = all(self.class_vars.get(class_id, tk.BooleanVar(value=False)).get() for class_id in class_dict.keys())
        new_state = not all_checked
        if len(class_dict) > 1:
            for class_id in class_dict.keys():
                if class_id in self.class_vars: 
                    self.class_vars[class_id].set(new_state)
        self.update_category_button_color(self.current_category_name)
        self.update_group_toggle_button_color(self.current_category_name)
        self.update_all_toggle_button_color()

    def toggle_all_classes(self):
        all_are_checked = all(var.get() for var in self.class_vars.values())
        new_state = not all_are_checked
        for var in self.class_vars.values(): var.set(new_state)
        for category_name in CLASSIFIED_COCO_CLASSES.keys():
            self.update_category_button_color(category_name)
        self.update_all_toggle_button_color() 
        if self.current_category_name:
            self.update_group_toggle_button_color(self.current_category_name)
            class_dict = CLASSIFIED_COCO_CLASSES.get(self.current_category_name, {})
            if len(class_dict) == 1:
                class_id = next(iter(class_dict.keys()))
                class_name = class_dict[class_id]
                if self.single_class_message:
                    self.single_class_message.config(
                        text=f"Category '{self.current_category_name}' activates the single class: '{class_name}'.\nDetection is currently: {'ON' if new_state else 'OFF'}.",
                        fg="#2ecc71"
                    )

    def show_category_classes(self, category_name):
        for widget in self.checkbox_widgets.values(): widget.grid_forget()
        self.single_class_message.grid_forget()
        class_dict = CLASSIFIED_COCO_CLASSES.get(category_name, {})
        
        if len(class_dict) == 1:
            class_id = next(iter(class_dict.keys()))
            class_name = class_dict[class_id]
            current_state = self.class_vars.get(class_id, tk.BooleanVar(value=False)).get()
            new_state = not current_state
            self.class_vars[class_id].set(new_state)
            self.on_checkbox_toggle(category_name) 

            
            self.single_class_message.grid(row=0, column=0, columnspan=8, sticky="n")
            self.group_toggle_button.config(state=tk.DISABLED, text="Single Class Only")
        else:
            cols = 8
            for i, (class_id, class_name) in enumerate(class_dict.items()):
                row = (i // cols) + 1
                col = i % cols
                widget = self.checkbox_widgets.get(class_id)
                if widget: 
                    widget.grid(row=row, column=col, padx=5, pady=2, sticky="w")
            self.group_toggle_button.config(state=tk.NORMAL, text="Toggle Current Group")
                
        self.current_category_name = category_name
        self.class_label.config(text=f"Individual Toggles: {category_name}")
        self.control_window.update_idletasks() 
        self.update_group_toggle_button_color(category_name)
        
    def _pre_generate_all_checkboxes(self):
        for category_name, class_dict in CLASSIFIED_COCO_CLASSES.items():
            for class_id, class_name in class_dict.items():
                var = tk.BooleanVar(value=False)
                self.class_vars[class_id] = var
                chk = tk.Checkbutton(self.class_frame, text=class_name, variable=var,
                                     bg=BG_CONTROL_DARK, fg=FG_MUTED, selectcolor=BG_CONTROL_DARK,
                                     font=('Inter', 9), anchor="w",
                                     command=lambda name=category_name: self.on_checkbox_toggle(name))
                self.checkbox_widgets[class_id] = chk

    def _pause_webcam(self):
        self.is_running = False
        if self.video_loop_id: self.master.after_cancel(self.video_loop_id)

    def _continue_webcam(self):
        if not self.is_running:
            self.is_running = True
            self.image_label.configure(text="", compound="none", fg=FG_MUTED, font=('Inter', 9)) 
            self.video_loop()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit and stop the webcam?"):
            self.is_running = False
            if self.video_loop_id: self.master.after_cancel(self.video_loop_id)
            if self.is_recording:
                if messagebox.askyesno("Confirm Discard", "You are currently recording. Do you want to discard this video before closing? (Yes to discard, No to save and close)"):
                    self._stop_recording_action(save=False)
                else:
                    self._stop_recording_action(save=True)
                    
            if self.video_capture and self.video_capture.isOpened(): self.video_capture.release()
            
            if self.control_window and self.control_window.winfo_exists():
                self.control_window.destroy()
            self.master.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = YoloWebcamFilterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()