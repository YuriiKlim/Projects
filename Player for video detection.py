import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
import torch
from PIL import Image, ImageTk, ImageOps
from ultralytics import YOLO
import time
import queue


class YOLOVideoDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title('YOLO Video Detector')

        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.model_file = None
        self.video_file = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.running = False
        self.paused = False
        self.detection_enabled = True

        self.cap = None
        self.frame = None
        self.total_frames = 0
        self.current_frame_number = 0
        self.fps = 0

        self.frame_queue = queue.Queue(maxsize=1)
        self.processed_frame = None

        self.conf = tk.DoubleVar(value=0.25)
        self.iou = tk.DoubleVar(value=0.7)

        self.container = tk.Frame(self.root)
        self.container.grid(row=0, column=0, sticky='nsew')

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.container.grid_rowconfigure(2, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        params_frame = tk.Frame(self.container)
        params_frame.grid(row=0, column=0, pady=5)

        tk.Label(params_frame, text='Confidence Threshold (conf):').grid(row=0, column=0, padx=5, sticky='e')
        self.conf_entry = tk.Entry(params_frame, textvariable=self.conf)
        self.conf_entry.grid(row=0, column=1, padx=5)

        tk.Label(params_frame, text='IoU Threshold (iou):').grid(row=0, column=2, padx=5, sticky='e')
        self.iou_entry = tk.Entry(params_frame, textvariable=self.iou)
        self.iou_entry.grid(row=0, column=3, padx=5)

        buttons_frame = tk.Frame(self.container)
        buttons_frame.grid(row=1, column=0, pady=5)

        self.select_model_button = tk.Button(buttons_frame, text='Select YOLO Model', command=self.select_model)
        self.select_model_button.grid(row=0, column=0, padx=5)

        self.select_video_button = tk.Button(buttons_frame, text='Select Video File', command=self.select_video)
        self.select_video_button.grid(row=0, column=1, padx=5)

        self.start_detection_button = tk.Button(buttons_frame, text='Start Detection', command=self.start_detection)
        self.start_detection_button.grid(row=0, column=2, padx=5)

        video_frame = tk.Frame(self.container)
        video_frame.grid(row=2, column=0, sticky='nsew')

        self.container.grid_rowconfigure(2, weight=1)

        self.video_label = tk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)

        controls_frame = tk.Frame(self.container)
        controls_frame.grid(row=3, column=0, pady=5)

        self.pause_button = tk.Button(controls_frame, text='Pause', command=self.toggle_pause)
        self.pause_button.grid(row=0, column=0, padx=5)
        self.pause_button.grid_remove()

        self.toggle_detection_button = tk.Button(controls_frame, text='Disable Detection', command=self.toggle_detection)
        self.toggle_detection_button.grid(row=0, column=1, padx=5)
        self.toggle_detection_button.grid_remove()

        self.stop_button = tk.Button(controls_frame, text='Stop', command=self.stop_video)
        self.stop_button.grid(row=0, column=2, padx=5)
        self.stop_button.grid_remove()

        playback_frame = tk.Frame(self.container)
        playback_frame.grid(row=4, column=0, pady=5)

        self.current_time_label = tk.Label(playback_frame, text='00:00')
        self.current_time_label.pack(side=tk.LEFT, padx=5)

        self.seek_bar = tk.Scale(
            playback_frame,
            from_=0,
            to=100,
            orient='horizontal',
            command=self.seek
        )
        self.seek_bar.pack(side=tk.LEFT, padx=5)
        self.seek_bar.configure(state='disabled')

        self.total_time_label = tk.Label(playback_frame, text='00:00')
        self.total_time_label.pack(side=tk.RIGHT, padx=5)

        self.root.bind('<Configure>', self.update_seek_bar_length)

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_seek_bar_length(self, event=None):
        window_width = self.root.winfo_width()
        new_length = int(window_width * 0.7)
        self.seek_bar.configure(length=new_length)

    def select_model(self):
        self.model_file = filedialog.askopenfilename(
            title='Select YOLO Model File',
            filetypes=(('PyTorch Model files', '*.pt'), ('All Files', '*.*'))
        )
        if self.model_file:
            try:
                self.model = YOLO(self.model_file)
                self.model.to(self.device)
                print(f"Model loaded successfully: {self.model_file}")
            except Exception as e:
                messagebox.showerror('Error', f"Error loading model: {e}")

    def select_video(self):
        self.video_file = filedialog.askopenfilename(
            title='Select Video File',
            filetypes=(('Video files', '*.mp4;*.avi;*.mov'), ('All Files', '*.*'))
        )

    def start_detection(self):
        if not self.model:
            messagebox.showerror('Error', 'Please select a YOLO model file first.')
            return
        if not self.video_file:
            messagebox.showerror('Error', 'Please select a video file.')
            return
        if self.running:
            messagebox.showinfo('Info', 'Detection is already running.')
            return

        try:
            conf_value = float(self.conf.get())
            iou_value = float(self.iou.get())
            if not (0 <= conf_value <= 1):
                raise ValueError('Confidence threshold must be between 0 and 1.')
            if not (0 <= iou_value <= 1):
                raise ValueError('IoU threshold must be between 0 and 1.')
        except ValueError as e:
            messagebox.showerror('Error', f"Invalid parameter value: {e}")
            return

        self.running = True
        self.paused = False
        self.detection_enabled = True

        self.conf_entry.configure(state='disabled')
        self.iou_entry.configure(state='disabled')

        self.root.resizable(True, True)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.maxsize(screen_width, screen_height)

        self.seek_bar.configure(state='normal')

        self.select_model_button.grid_remove()
        self.select_video_button.grid_remove()
        self.start_detection_button.grid_remove()

        self.pause_button.grid()
        self.toggle_detection_button.grid()
        self.stop_button.grid()

        self.update_seek_bar_length()

        self.cap = cv2.VideoCapture(self.video_file)
        if not self.cap.isOpened():
            messagebox.showerror('Error', 'Cannot open video file.')
            self.running = False
            return

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_duration = self.total_frames / self.fps if self.fps > 0 else 0

        self.total_time_label.config(text=self.format_time(self.total_duration))

        self.processing_thread = Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()

        self.update_frame()

    def toggle_pause(self):
        self.paused = not self.paused
        self.pause_button.config(text='Resume' if self.paused else 'Pause')

    def toggle_detection(self):
        self.detection_enabled = not self.detection_enabled
        self.toggle_detection_button.config(text='Enable Detection' if not self.detection_enabled else 'Disable Detection')

    def stop_video(self):
        self.running = False
        self.paused = False
        self.cleanup()

    def seek(self, value):
        if self.cap is not None:
            frame_number = int(int(value) * self.total_frames / 100)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            self.current_frame_number = frame_number

    def process_frames(self):
        while self.running:
            if not self.paused:
                if self.frame_queue.empty():
                    time.sleep(0.01)
                    continue

                frame_rgb = self.frame_queue.get()

                if self.detection_enabled:
                    try:
                        conf_value = float(self.conf.get())
                        iou_value = float(self.iou.get())
                        results = self.model(frame_rgb, conf=conf_value, iou=iou_value)
                        detections = results[0].boxes
                    except Exception as e:
                        messagebox.showerror('Error', f"Error during detection: {e}")
                        self.running = False
                        break

                    if detections is not None:
                        for det in detections:
                            x1, y1, x2, y2 = det.xyxy[0].cpu().numpy().astype(int)
                            confidence = det.conf[0].item()
                            cls_id = int(det.cls[0])
                            label = f"{self.model.names[cls_id]} {confidence:.2f}"
                            cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame_rgb, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                self.processed_frame = frame_rgb
            else:
                time.sleep(0.1)

    def update_frame(self):
        if self.running and not self.paused:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                self.cleanup()
                return

            self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if not self.frame_queue.full():
                self.frame_queue.put(frame_rgb)

            if self.processed_frame is not None:
                display_frame = self.processed_frame
            else:
                display_frame = frame_rgb

            self.root.update_idletasks()
            label_width = self.video_label.winfo_width()
            label_height = self.video_label.winfo_height()

            frame_pil = Image.fromarray(display_frame)
            if label_width > 1 and label_height > 1:
                frame_pil = ImageOps.contain(frame_pil, (label_width, label_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=frame_pil)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

            if self.total_frames > 0:
                position = int((self.current_frame_number / self.total_frames) * 100)
                self.seek_bar.set(position)

            current_time = self.current_frame_number / self.fps if self.fps > 0 else 0
            self.current_time_label.config(text=self.format_time(current_time))

        if self.running:
            self.root.after(10, self.update_frame)
        else:
            self.video_label.configure(image='')

    def format_time(self, seconds):
        seconds = int(seconds)
        if seconds < 60:
            return f"00:{seconds:02d}"
        elif seconds < 3600:
            minutes = seconds // 60
            seconds = seconds % 60
            return f"{minutes:02d}:{seconds:02d}"
        else:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours}:{minutes:02d}:{seconds:02d}"

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.running = False
        self.paused = False
        self.detection_enabled = True
        self.seek_bar.configure(state='disabled')
        self.pause_button.config(text='Pause')
        self.toggle_detection_button.config(text='Disable Detection')

        self.pause_button.grid_remove()
        self.toggle_detection_button.grid_remove()
        self.stop_button.grid_remove()

        self.seek_bar.configure(state='disabled')

        self.current_time_label.config(text='00:00')
        self.total_time_label.config(text='00:00')

        self.conf_entry.configure(state='normal')
        self.iou_entry.configure(state='normal')

        self.root.resizable(False, False)

        self.select_model_button.grid()
        self.select_video_button.grid()
        self.start_detection_button.grid()

    def on_closing(self):
        self.running = False
        self.paused = False
        self.cleanup()
        self.root.destroy()


if __name__ == '__main__':
    root = tk.Tk()
    app = YOLOVideoDetectorApp(root)
    root.mainloop()
