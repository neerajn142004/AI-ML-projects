import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
from ultralytics import YOLO
from queue import Queue
import time
import os
import colorsys
import threading

# --- Configuration ---
MODEL_NAME = 'yolov8m.pt'
BASE_WIDTH = 1280
BASE_HEIGHT = 720
WEBCAM_ID = 0
CONFIDENCE_THRESHOLD = 0.5
RECORDING_FPS = 30.0

# --- Colors ---
BG_DEEP_CHARCOAL = "#1e1e1e"
BG_CONTROL_DARK = "#252526"
FG_HIGH_CONTRAST = "#ffffff"
FG_MUTED = "#cccccc"
BUTTON_NORMAL_BG = "#333333"
BUTTON_NORMAL_FG = "#cccccc"
BUTTON_ACTIVE_BG = "#007ACC"
BUTTON_ACTIVE_FG = "#ffffff"
BUTTON_RECORD_ON = "#e74c3c"
BUTTON_RECORD_OFF = "#f1c40f"
BUTTON_COUNT_ACTIVE = "#2ecc71"
BUTTON_ALARM_ACTIVE = "#f39c12"

# COCO Classes
CLASSIFIED_COCO_CLASSES = {
    "People": { 0: 'person' },
    "Vehicles": { 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat' },
    "Animals": { 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe' },
    "Traffic": { 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench' },
    "Accessories": { 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase' },
    "Sports": { 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket' },
    "Kitchen": { 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake' },
    "Household": { 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush' },
    "Electronics": { 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 67: 'cell phone', 68: 'microwave', 66: 'keyboard', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator' }
}

SELECTABLE_CLASSES = {k: v for cat in CLASSIFIED_COCO_CLASSES.values() for k, v in cat.items()}

def get_category_and_name(class_id):
    name = SELECTABLE_CLASSES.get(class_id, "Unknown")
    for category, items in CLASSIFIED_COCO_CLASSES.items():
        if class_id in items:
            return category, name
    return "Other", name

# --- Helper Classes ---

class VideoRecorder(threading.Thread):
    def __init__(self, frame_queue, width, height, fps, filepath):
        super().__init__()
        self.frame_queue = frame_queue
        self.width = width
        self.height = height
        self.fps = fps
        self.filepath = filepath
        self.stop_event = threading.Event()
        self.video_writer = None
        
    def run(self):
        if not self.filepath: return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        try:
            self.video_writer = cv2.VideoWriter(self.filepath, fourcc, self.fps, (self.width, self.height))
        except Exception:
            return
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.01)
                self.video_writer.write(frame)
            except Exception:
                continue

    def stop(self, save_file=True):
        self.stop_event.set()
        self.join()
        if self.video_writer: self.video_writer.release()
        if not save_file and os.path.exists(self.filepath):
            try: os.remove(self.filepath)
            except: pass

class AlarmPanel:
    def __init__(self, master):
        self.top = tk.Toplevel(master, bg=BG_CONTROL_DARK)
        self.top.title("Alarm / Status Panel")
        self.top.withdraw()
        self.top.protocol("WM_DELETE_WINDOW", self.hide)
        
        style = ttk.Style()
        style.theme_use('default')
        style.configure("Treeview", background=BG_CONTROL_DARK, foreground=FG_HIGH_CONTRAST, fieldbackground=BG_CONTROL_DARK, rowheight=25)
        style.map('Treeview', background=[('selected', BUTTON_ACTIVE_BG)])
        
        columns = ('Category', 'Subclass', 'Live', 'Total')
        self.tree = ttk.Treeview(self.top, columns=columns, show='headings')
        self.tree.heading('Category', text='Class (Category)')
        self.tree.heading('Subclass', text='Subclass')
        self.tree.heading('Live', text='Live Count')
        self.tree.heading('Total', text='Total Count')
        
        self.tree.column('Category', width=100, anchor='center')
        self.tree.column('Subclass', width=100, anchor='center')
        self.tree.column('Live', width=80, anchor='center')
        self.tree.column('Total', width=80, anchor='center')

        self.tree.pack(fill='both', expand=True, padx=10, pady=10)
        self.items = {} 

    def show(self): self.top.deiconify()
    def hide(self): self.top.withdraw()

    def update_alarm_display(self, live_counts, total_unique_counts, alarm_classes):
        current_displayed_ids = set(self.items.keys())
        
        for cid in alarm_classes:
            cat, name = get_category_and_name(cid)
            live = live_counts.get(cid, 0)
            total = total_unique_counts.get(cid, 0)
            values = (cat, name, live, total)

            if cid not in self.items:
                self.items[cid] = self.tree.insert('', 'end', values=values)
            else:
                self.tree.item(self.items[cid], values=values)
            
            if cid in current_displayed_ids: current_displayed_ids.remove(cid)
        
        for cid in current_displayed_ids:
            if cid in self.items:
                self.tree.delete(self.items[cid])
                del self.items[cid]

# --- Main Application ---

class YoloCombinedApp:
    def __init__(self, master):
        self.master = master
        master.title("YOLO Combined System")
        master.configure(bg=BG_DEEP_CHARCOAL)

        # --- State ---
        self.is_running = True
        self.video_loop_id = None
        self.is_fullscreen = False
        self.is_controls_hidden = False
        self.setup_window_ref = None 
        
        # --- Recording ---
        self.is_recording = False
        self.frame_queue = Queue(maxsize=30)
        self.recorder_thread = None
        
        # --- Logic ---
        self.count_classes = set()
        self.alarm_classes = set()
        
        # Visibility Toggles
        self.class_vars = {} 
        
        # Counting Logic
        self.total_unique_counts = {}
        self.globally_counted_ids = {} 
        self.sequential_id_map = {} 
        self.sequential_counters = {}
        self._last_detected_counts = {}

        # --- Initialization ---
        self.model = self._load_model()
        self.video_capture = self._init_webcam()
        self.class_colors = self._generate_colors()
        self.current_tk_image = None
        
        # --- GUI Setup ---
        self.alarm_panel = AlarmPanel(master)
        self.checkbox_widgets = {}
        self.category_buttons = {}
        
        self._setup_video_layout()
        self._setup_control_window()
        self._initial_geometry()
        
        # Bindings
        self.master.bind('l', self._toggle_fullscreen)
        self.master.bind('<Escape>', self._exit_fullscreen)
        self.master.bind('h', self._toggle_controls)

        self.video_loop()

    # --- Core Logic ---

    def _load_model(self):
        try: return YOLO(MODEL_NAME)
        except: 
            messagebox.showerror("Error", "Model load failed")
            self.master.destroy()

    def _init_webcam(self):
        cap = cv2.VideoCapture(WEBCAM_ID)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, BASE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, BASE_HEIGHT)
        return cap

    def _generate_colors(self):
        colors = {}
        num = len(SELECTABLE_CLASSES)
        for i, cid in enumerate(SELECTABLE_CLASSES.keys()):
            hue = i / float(num)
            rgb = colorsys.hsv_to_rgb(hue, 0.9, 0.9)
            colors[cid] = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        return colors

    def detect_and_filter(self, frame):
        visible_ids = {cid for cid, var in self.class_vars.items() if var.get()}
        all_active_ids = visible_ids.union(self.count_classes).union(self.alarm_classes)
        
        if not all_active_ids:
            self._last_detected_counts = {}
            self.alarm_panel.update_alarm_display({}, self.total_unique_counts, self.alarm_classes)
            return frame

        annotated_frame = frame.copy()
        results = self.model.track(annotated_frame, verbose=False, conf=CONFIDENCE_THRESHOLD, 
                                   classes=list(all_active_ids), persist=True, tracker="bytetrack.yaml")

        live_counts = {}

        for result in results:
            boxes = result.boxes
            if result.boxes.id is not None:
                ids = result.boxes.id.cpu().numpy().astype(int)
            else:
                ids = [None] * len(boxes)

            for i, box in enumerate(boxes):
                class_id = int(box.cls.cpu().numpy()[0])
                track_id = ids[i]
                
                # --- LIVE COUNT ---
                if class_id in self.count_classes or class_id in self.alarm_classes:
                    live_counts[class_id] = live_counts.get(class_id, 0) + 1
                
                # --- TOTAL UNIQUE / SEQUENTIAL ID LOGIC ---
                # CHANGED: Now runs if class is in Count OR Alarm
                simple_seq_id = "?"
                
                if track_id is not None and (class_id in self.count_classes or class_id in self.alarm_classes):
                    if class_id not in self.sequential_id_map:
                        self.sequential_id_map[class_id] = {}
                        self.sequential_counters[class_id] = 0
                        self.globally_counted_ids[class_id] = set()
                        self.total_unique_counts[class_id] = 0
                    
                    # If this is a NEW track ID for this class, increment Total
                    if track_id not in self.sequential_id_map[class_id]:
                        self.sequential_counters[class_id] += 1
                        self.sequential_id_map[class_id][track_id] = self.sequential_counters[class_id]
                        self.globally_counted_ids[class_id].add(track_id)
                        self.total_unique_counts[class_id] += 1
                    
                    simple_seq_id = self.sequential_id_map[class_id][track_id]

                # --- Drawing Logic ---
                should_draw = (class_id in visible_ids) or (class_id in self.count_classes) or (class_id in self.alarm_classes)

                if should_draw:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    if class_id in self.alarm_classes: color = (0, 0, 255) 
                    elif class_id in self.count_classes: color = (255, 165, 0)
                    else: color = self.class_colors.get(class_id, (255, 255, 255))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{self.model.names[class_id]}"
                    if class_id in self.alarm_classes: label += f" [A] {simple_seq_id}"
                    if class_id in self.count_classes: label += f" [C]:{simple_seq_id}"
                    
                    cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self._last_detected_counts = live_counts
        if self.alarm_panel.top.winfo_exists():
            self.alarm_panel.update_alarm_display(live_counts, self.total_unique_counts, self.alarm_classes)
            
        return annotated_frame

    def video_loop(self):
        if not self.is_running: return
        ret, frame = self.video_capture.read()
        if ret:
            annotated = self.detect_and_filter(frame)
            self._display_frame(annotated)
            
            if self.is_recording:
                rec_frame = cv2.resize(annotated, (BASE_WIDTH, BASE_HEIGHT))
                try: self.frame_queue.put_nowait(rec_frame)
                except: pass
        
        self.video_loop_id = self.master.after(10, self.video_loop)

    def _display_frame(self, cv_image):
        try:
            w = self.image_label.winfo_width()
            h = self.image_label.winfo_height()
            if w < 10: w, h = BASE_WIDTH, BASE_HEIGHT
            
            resized = cv2.resize(cv_image, (w, h))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            self.current_tk_image = ImageTk.PhotoImage(Image.fromarray(rgb))
            self.image_label.config(image=self.current_tk_image)
        except: pass

    # --- UI Setup ---

    def _setup_video_layout(self):
        self.main_frame = tk.Frame(self.master, bg=BG_DEEP_CHARCOAL)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(self.main_frame, bg=BG_DEEP_CHARCOAL)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def _setup_control_window(self):
        self.control_window = tk.Toplevel(self.master, bg=BG_CONTROL_DARK)
        self.control_window.title("Combined Controls")
        self.control_window.attributes('-topmost', True)
        self.control_window.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Main Container
        main_container = tk.Frame(self.control_window, bg=BG_CONTROL_DARK, padx=10, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        # 1. Tools Section
        tools_frame = tk.LabelFrame(main_container, text="Tools & Recording", bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, font=('Inter', 10, 'bold'))
        tools_frame.pack(fill=tk.X, pady=(0, 10))
        
        feat_row = tk.Frame(tools_frame, bg=BG_CONTROL_DARK, pady=5)
        feat_row.pack(fill=tk.X)
        
        self.count_btn = tk.Button(feat_row, text="✅ Count Setup", command=lambda: self.open_selection_window("Count"), bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, relief=tk.FLAT)
        self.count_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.alarm_btn = tk.Button(feat_row, text="🔔 Alarm Setup", command=lambda: self.open_selection_window("Alarm"), bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, relief=tk.FLAT)
        self.alarm_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        self.show_alarm_panel_btn = tk.Button(feat_row, text="Show Panel", command=self.alarm_panel.show, bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, relief=tk.FLAT)
        self.show_alarm_panel_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)

        # Record Row
        rec_row = tk.Frame(tools_frame, bg=BG_CONTROL_DARK, pady=5)
        rec_row.pack(fill=tk.X)
        self.rec_btn = tk.Button(rec_row, text="🔴 Record", command=self.start_recording, bg=BUTTON_NORMAL_BG, fg=BUTTON_ACTIVE_FG, relief=tk.FLAT)
        self.rec_btn.pack(side=tk.LEFT, padx=5)
        self.stop_rec_btn = tk.Button(rec_row, text="■ Stop", command=self.stop_recording, bg=BUTTON_NORMAL_BG, fg=FG_MUTED, state=tk.DISABLED, relief=tk.FLAT)
        self.stop_rec_btn.pack(side=tk.LEFT, padx=5)
        tk.Button(rec_row, text="Quit", command=self.on_closing, bg="#e74c3c", fg="white", relief=tk.FLAT).pack(side=tk.RIGHT, padx=5)

        # 2. Filter Controls
        filter_frame = tk.LabelFrame(main_container, text="Class Visibility Filter", bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, font=('Inter', 10, 'bold'))
        filter_frame.pack(fill=tk.BOTH, expand=True)
        
        cat_frame = tk.Frame(filter_frame, bg=BG_CONTROL_DARK)
        cat_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(cat_frame, text="ALL", command=self.toggle_all, bg=BUTTON_NORMAL_BG, fg="white", width=4, relief=tk.FLAT).pack(side=tk.LEFT, padx=2)
        for cat in CLASSIFIED_COCO_CLASSES.keys():
            btn = tk.Button(cat_frame, text=cat, command=lambda c=cat: self.show_category(c), bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, relief=tk.FLAT, font=('Arial', 8))
            btn.pack(side=tk.LEFT, padx=2)
            self.category_buttons[cat] = btn

        self.chk_frame = tk.Frame(filter_frame, bg=BG_CONTROL_DARK)
        self.chk_frame.pack(fill=tk.BOTH, expand=True)
        
        for cat, classes in CLASSIFIED_COCO_CLASSES.items():
            for cid, name in classes.items():
                var = tk.BooleanVar(value=False)
                self.class_vars[cid] = var
                chk = tk.Checkbutton(self.chk_frame, text=name, variable=var, bg=BG_CONTROL_DARK, fg=FG_MUTED, selectcolor=BG_CONTROL_DARK, command=lambda c=cat: self.update_cat_color(c))
                self.checkbox_widgets[cid] = chk

    # --- Feature Selection Window ---
    def open_selection_window(self, mode):
        if self.setup_window_ref is not None and self.setup_window_ref.winfo_exists():
            self.setup_window_ref.destroy() 

        win = tk.Toplevel(self.control_window, bg=BG_CONTROL_DARK)
        win.title(f"Select for {mode}")
        self.setup_window_ref = win 
        win.grab_set() 
        
        nb = ttk.Notebook(win)
        nb.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        current_set = self.count_classes if mode == "Count" else self.alarm_classes
        temp_vars = {}

        for cat, classes in CLASSIFIED_COCO_CLASSES.items():
            f = tk.Frame(nb, bg=BG_CONTROL_DARK)
            nb.add(f, text=cat)
            for i, (cid, name) in enumerate(classes.items()):
                var = tk.BooleanVar(value=(cid in current_set))
                temp_vars[cid] = var
                tk.Checkbutton(f, text=name, variable=var, bg=BG_CONTROL_DARK, fg="white", selectcolor=BG_CONTROL_DARK).grid(row=i//4, column=i%4, sticky='w')

        def apply():
            new_set = {cid for cid, v in temp_vars.items() if v.get()}
            if mode == "Count":
                self.count_classes = new_set
                txt = f"✅ Count ({len(new_set)})" if new_set else "✅ Count Setup"
                bg = BUTTON_COUNT_ACTIVE if new_set else BUTTON_NORMAL_BG
                self.count_btn.config(text=txt, bg=bg)
            else:
                self.alarm_classes = new_set
                txt = f"🔔 Alarm ({len(new_set)})" if new_set else "🔔 Alarm Setup"
                bg = BUTTON_ALARM_ACTIVE if new_set else BUTTON_NORMAL_BG
                self.alarm_btn.config(text=txt, bg=bg)
                self.show_alarm_panel_btn.config(bg=bg)
            win.destroy()
            self.setup_window_ref = None
            
        tk.Button(win, text="APPLY", command=apply, bg=BUTTON_ACTIVE_BG, fg="white").pack(fill=tk.X, padx=10, pady=10)

    # --- UI Interaction Logic ---
    def show_category(self, category):
        for w in self.checkbox_widgets.values(): w.grid_forget()
        classes = CLASSIFIED_COCO_CLASSES[category]
        for i, cid in enumerate(classes.keys()):
            self.checkbox_widgets[cid].grid(row=i//6, column=i%6, sticky='w', padx=5)

    def update_cat_color(self, cat):
        classes = CLASSIFIED_COCO_CLASSES[cat]
        is_active = any(self.class_vars[cid].get() for cid in classes)
        self.category_buttons[cat].config(bg=BUTTON_ACTIVE_BG if is_active else BUTTON_NORMAL_BG, fg="white" if is_active else BUTTON_NORMAL_FG)

    def toggle_all(self):
        new_state = not any(v.get() for v in self.class_vars.values())
        for v in self.class_vars.values(): v.set(new_state)
        for cat in CLASSIFIED_COCO_CLASSES: self.update_cat_color(cat)

    # --- Recording Logic ---
    def start_recording(self):
        if self.is_recording: return
        path = filedialog.asksaveasfilename(defaultextension=".avi", filetypes=[("AVI", "*.avi")])
        if not path: return
        
        self.is_recording = True
        self.recorder_thread = VideoRecorder(self.frame_queue, BASE_WIDTH, BASE_HEIGHT, RECORDING_FPS, path)
        self.recorder_thread.start()
        self.rec_btn.config(state=tk.DISABLED, bg=BUTTON_RECORD_ON)
        self.stop_rec_btn.config(state=tk.NORMAL, bg=BUTTON_RECORD_OFF)

    def stop_recording(self):
        if not self.is_recording: return
        if messagebox.askyesno("Save?", "Save recording?"):
            self._finish_rec(True)
        else:
            self._finish_rec(False)

    def _finish_rec(self, save):
        self.is_recording = False
        if self.recorder_thread: self.recorder_thread.stop(save)
        self.rec_btn.config(state=tk.NORMAL, bg=BUTTON_NORMAL_BG)
        self.stop_rec_btn.config(state=tk.DISABLED, bg=BUTTON_NORMAL_BG)

    # --- Window Management ---
    def _initial_geometry(self):
        try: self.master.state('zoomed')
        except: self.master.geometry(f"{BASE_WIDTH}x{BASE_HEIGHT}")
        self.control_window.geometry("600x600+50+50")

    def _toggle_controls(self, e=None):
        self.is_controls_hidden = not self.is_controls_hidden
        if self.is_controls_hidden: self.control_window.withdraw()
        else: self.control_window.deiconify()

    def _toggle_fullscreen(self, e=None):
        self.is_fullscreen = not self.is_fullscreen
        self.master.attributes('-fullscreen', self.is_fullscreen)

    def _exit_fullscreen(self, e=None):
        self.is_fullscreen = False
        self.master.attributes('-fullscreen', False)

    def on_closing(self):
        if self.is_recording:
            if messagebox.askyesno("Recording", "Stop and save recording?"):
                self._finish_rec(True)
            else:
                self._finish_rec(False)
        self.is_running = False
        if self.video_capture: self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloCombinedApp(root)
    root.mainloop()