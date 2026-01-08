import tkinter as tk
from tkinter import messagebox, filedialog
import os

# --- Professional Dark Theme Colors ---
BG_CONTROL_DARK = "#E3E3EE"      # Left Panel background
BG_VIDEO_AREA = "#2b2b2b"        # Right Video background
FG_HIGH_CONTRAST = "#272323"     # Dark text
FG_MUTED = "#ba0c0c"             # Red accent text

BUTTON_NORMAL_BG = "#333333"      
BUTTON_NORMAL_FG = "#cccccc"      
BUTTON_ACTIVE_BG = "#007ACC"      
BUTTON_ACTIVE_FG = "#ffffff"      
BUTTON_RAISED_HOVER = "#444444"   
BUTTON_TOGGLE_ON = "#2ecc71"      # Green (Selection)
BUTTON_PLAY_BG = "#27ae60"        # Darker Green (Play Button)
BUTTON_EXIT_BG = "#e74c3c"        # Red

# --- Data ---
AVAILABLE_FILTERS = [
    "Rain Removal", "Fog Removal", "Dust Removal", 
    "Grayscale", "Contrast Enhancement"
]

CAMERA_OPTIONS = [
    "Cam 1", 
    "Cam 2", 
    "Cam 3", 
    "2 Stitching", 
    "3 Stitching"
]

class SurveillanceDashboard:
    def __init__(self, master):
        self.master = master
        master.title("Surveillance Control Panel")
        master.geometry("1100x800") 
        master.configure(bg=BG_VIDEO_AREA)
        master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # State Variables
        self.current_mode = None  
        self.active_filters = set() 
        self.selected_camera = None 
        self.video_path = None
        self.is_playing = False
        
        # UI Component Storage
        self.filter_buttons = {}
        self.camera_buttons = {}
        
        self._setup_layout()
        
        # Start in Live Mode by default
        self.set_mode('LIVE') 

    def _setup_layout(self):
        # =================================================
        # 1. LEFT PANEL (CONTROLS) - Fixed Width
        # =================================================
        self.left_panel = tk.Frame(self.master, bg=BG_CONTROL_DARK, width=320)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y)
        self.left_panel.pack_propagate(False) # Stable Width

        # Header
        header_frame = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK, pady=15)
        header_frame.pack(fill=tk.X)
        tk.Label(header_frame, text="CONTROL PANEL", bg=BG_CONTROL_DARK, fg=FG_MUTED, 
                 font=('Inter', 14, 'bold')).pack()

        # --- A. Source Selection ---
        self._add_section_header("1. Source Selection")
        
        mode_btn_container = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        mode_btn_container.pack(anchor="center", pady=5)

        self.live_btn = tk.Button(mode_btn_container, text="LIVE", command=lambda: self.set_mode('LIVE'),
                                  bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER, 
                                  relief=tk.FLAT, font=('Inter', 10, 'bold'), width=12)
        self.live_btn.pack(side=tk.LEFT, padx=5)

        self.video_btn = tk.Button(mode_btn_container, text="VIDEO", command=lambda: self.set_mode('VIDEO'),
                                   bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, activebackground=BUTTON_RAISED_HOVER, 
                                   relief=tk.FLAT, font=('Inter', 10, 'bold'), width=12)
        self.video_btn.pack(side=tk.LEFT, padx=5)
        
        # --- B. Camera Options ---
        self.lbl_cam_header = tk.Label(self.left_panel, text="2. Select Camera", bg=BG_CONTROL_DARK, 
                                       fg=FG_HIGH_CONTRAST, font=('Inter', 10, 'bold'), anchor="w")
        
        self.camera_container = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        
        self.camera_buttons = {}
        for cam_name in CAMERA_OPTIONS:
            cmd = lambda c=cam_name: self.select_camera(c)
            btn = tk.Button(self.camera_container, text=cam_name, command=cmd,
                            bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG,
                            activebackground=BUTTON_RAISED_HOVER, relief=tk.FLAT,
                            font=('Inter', 10), width=30) 
            btn.pack(pady=2)
            self.camera_buttons[cam_name] = btn

        # --- C. Filter Display ---
        self.filter_header_frame = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        self.filter_header_frame.pack(fill=tk.X, padx=15, pady=(20, 5))
        
        self.lbl_filter_title = tk.Label(self.filter_header_frame, text="3. Filters", 
                                         bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, 
                                         font=('Inter', 10, 'bold'))
        self.lbl_filter_title.pack(side=tk.LEFT)
        
        # SELECT ALL BUTTON
        self.btn_select_all = tk.Button(self.filter_header_frame, text="Select All", 
                                        command=self.toggle_all_filters,
                                        bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG,
                                        font=('Inter', 8), width=10, relief=tk.FLAT)
        self.btn_select_all.pack(side=tk.RIGHT)

        self.filter_container = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        self.filter_container.pack(fill=tk.X, pady=2)

        self.filter_buttons = {}
        for filter_name in AVAILABLE_FILTERS:
            cmd = lambda name=filter_name: self.select_filter(name)
            btn = tk.Button(self.filter_container, text=filter_name, command=cmd, 
                            bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG, 
                            activebackground=BUTTON_RAISED_HOVER, relief=tk.FLAT, 
                            font=('Inter', 10), width=30)
            btn.pack(pady=2)
            self.filter_buttons[filter_name] = btn

        # --- D. Upload Button (Video Mode Only) ---
        self.upload_container = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        self.upload_container.pack(fill=tk.X, pady=5)
        
        self.btn_upload = tk.Button(self.upload_container, text="UPLOAD", 
                                    command=self.upload_video_file,
                                    bg=BUTTON_ACTIVE_BG, fg=BUTTON_ACTIVE_FG,
                                    font=('Inter', 10, 'bold'), width=30, relief=tk.FLAT)
        # Initially Hidden

        # --- PLAY BUTTON ---
        self.play_container = tk.Frame(self.left_panel, bg=BG_CONTROL_DARK)
        self.play_container.pack(fill=tk.X, pady=(20, 10))
        
        self.btn_play = tk.Button(self.play_container, text="▶ PLAY", 
                                  command=self.on_play_press,
                                  bg=BUTTON_PLAY_BG, fg="white",
                                  activebackground="#27ae60", relief=tk.FLAT,
                                  font=('Inter', 12, 'bold'), width=30, height=2)
        self.btn_play.pack(pady=5)

        # --- E. Exit Button ---
        tk.Label(self.left_panel, text="", bg=BG_CONTROL_DARK).pack(fill=tk.Y, expand=True) # Spacer
        self.exit_btn = tk.Button(self.left_panel, text="EXIT SYSTEM", command=self.on_closing,
                                  bg=BUTTON_EXIT_BG, fg="white", 
                                  activebackground="#c0392b", relief=tk.FLAT, 
                                  font=('Inter', 10, 'bold'), width=20)
        self.exit_btn.pack(side=tk.BOTTOM, pady=20)

        # =================================================
        # 2. RIGHT PANEL (FEED AREA)
        # =================================================
        self.right_panel = tk.Frame(self.master, bg=BG_VIDEO_AREA)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(self.right_panel, text="NO SIGNAL", bg="black", fg="#555555",
                                    font=("Arial", 24, "bold"))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

    def _add_section_header(self, text):
        lbl = tk.Label(self.left_panel, text=text, bg=BG_CONTROL_DARK, fg=FG_HIGH_CONTRAST, 
                       font=('Inter', 10, 'bold'), anchor="w")
        lbl.pack(fill=tk.X, padx=15, pady=(15, 5))
        return lbl

    # =================================================
    # LOGIC
    # =================================================

    def set_mode(self, mode):
        self.current_mode = mode
        self.active_filters.clear()
        self.is_playing = False
        self._update_filter_visuals()
        
        # Reset Video/Upload state
        self.btn_upload.pack_forget()
        self.video_label.config(text="NO SIGNAL", fg="#555555")
        self.video_path = None

        if mode == 'LIVE':
            self.live_btn.config(bg=BUTTON_TOGGLE_ON, fg=BUTTON_ACTIVE_FG)
            self.video_btn.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)
            
            # Show Cameras
            self.lbl_cam_header.pack(fill=tk.X, padx=15, pady=(15, 5), after=self.live_btn.master)
            self.camera_container.pack(fill=tk.X, pady=2, after=self.lbl_cam_header)
            
            # Reset Camera Selection
            self.selected_camera = None
            for btn in self.camera_buttons.values():
                btn.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)
            
            # Show Select All
            self.btn_select_all.pack(side=tk.RIGHT)
            
        elif mode == 'VIDEO':
            self.video_btn.config(bg=BUTTON_TOGGLE_ON, fg=BUTTON_ACTIVE_FG)
            self.live_btn.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)
            
            # Hide Cameras
            self.camera_container.pack_forget()
            self.lbl_cam_header.pack_forget()
            
            # Hide Select All
            self.btn_select_all.pack_forget()
            
            self.video_label.config(text="NO SIGNAL\n(Select a Single Filter to Upload)")

    def select_camera(self, cam_name):
        self.selected_camera = cam_name
        self.is_playing = False # Reset play state when changing cam
        
        for name, btn in self.camera_buttons.items():
            if name == cam_name:
                btn.config(bg=BUTTON_TOGGLE_ON, fg=BUTTON_ACTIVE_FG)
            else:
                btn.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)
        
        # --- KEY CHANGE: Feed starts immediately (Raw) ---
        self.video_label.config(
            text=f"📹 {cam_name}: LIVE FEED (RAW)\n\n[Status: Running - Unfiltered]", 
            fg="#2ecc71" # Green Text
        )

    def select_filter(self, filter_name):
        if self.current_mode == 'LIVE':
            # Live: Toggle
            if filter_name in self.active_filters:
                self.active_filters.remove(filter_name)
            else:
                self.active_filters.add(filter_name)
                
        elif self.current_mode == 'VIDEO':
            # Video: Exclusive
            if filter_name in self.active_filters:
                self.active_filters.remove(filter_name) 
            else:
                self.active_filters.clear() 
                self.active_filters.add(filter_name)

        self._update_filter_visuals()
        
        # Video Upload Logic
        if self.current_mode == 'VIDEO':
            if len(self.active_filters) > 0:
                self.btn_upload.pack(pady=5) 
                self.video_label.config(text="FILTER SELECTED.\nPLEASE UPLOAD VIDEO.", fg="white")
            else:
                self.btn_upload.pack_forget()
                self.video_label.config(text="NO SIGNAL\n(Select a Single Filter to Upload)", fg="#555555")

    def toggle_all_filters(self):
        if len(self.active_filters) == len(AVAILABLE_FILTERS):
            self.active_filters.clear()
        else:
            self.active_filters.update(AVAILABLE_FILTERS)
        self._update_filter_visuals()

    def _update_filter_visuals(self):
        for name, btn in self.filter_buttons.items():
            if name in self.active_filters:
                btn.config(bg=BUTTON_TOGGLE_ON, fg=BUTTON_ACTIVE_FG)
            else:
                btn.config(bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)
        
        if len(self.active_filters) == len(AVAILABLE_FILTERS):
            self.btn_select_all.config(text="Clear All", bg=BUTTON_TOGGLE_ON, fg=BUTTON_ACTIVE_FG)
        else:
            self.btn_select_all.config(text="Select All", bg=BUTTON_NORMAL_BG, fg=BUTTON_NORMAL_FG)

    def upload_video_file(self):
        path = filedialog.askopenfilename(filetypes=[("Video", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = os.path.basename(path)
            self.btn_upload.config(bg=BUTTON_TOGGLE_ON) 
            self.video_label.config(text=f"FILE LOADED: {self.video_path}\nREADY TO PLAY.", fg="white")

    # --- PLAY BUTTON LOGIC ---
    def on_play_press(self):
        self.is_playing = True
        
        if self.current_mode == 'LIVE':
            if not self.selected_camera:
                messagebox.showwarning("Warning", "Please select a Camera first.")
                return
            
            filter_text = ", ".join(self.active_filters) if self.active_filters else "None (Raw)"
            
            # --- FEED WITH FILTERS APPLIED ---
            self.video_label.config(
                text=f"▶ PROCESSING LIVE FEED\n\nSOURCE: {self.selected_camera}\nAPPLIED AI: {filter_text}",
                fg="#3498db" # Blue Text for Processing
            )

        elif self.current_mode == 'VIDEO':
            if not self.video_path:
                messagebox.showwarning("Warning", "Please Upload a Video first.")
                return
            
            active_filter = list(self.active_filters)[0]
            
            self.video_label.config(
                text=f"▶ PLAYING VIDEO FILE\n\nFILE: {self.video_path}\nFILTER: {active_filter}",
                fg="#3498db"
            )

    def on_closing(self):
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SurveillanceDashboard(root) 
    root.mainloop()