import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import requests
from io import BytesIO
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import cv2
from ultralytics import YOLO
from SegCloth import segment_clothing
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


class TransferLearningClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        resnet = models.resnet152(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad = False
        in_features = resnet.fc.in_features
        resnet.fc = nn.Identity()
        self.feature_extractor = resnet
        self.fc1 = nn.Linear(in_features, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = F.relu(self.fc1(out))
        out = self.dropout1(out)
        out = F.relu(self.fc2(out))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


class_dict = {'Dress': 0, 'Football Boots': 1, 'Football Sneakers': 2, 'Full-zip Hoodie': 3, 'Hoodie': 4,
              'Jacket': 5, 'Leggings': 6, 'Longsleeve': 7, 'Pants': 8, 'Parka': 9, 'Polo': 10, 'Puffer jacket': 11,
              'Shirt': 12, 'Shorts': 13, 'Skirt': 14, 'Slippers': 15, 'Sneakers': 16, 'Sweater': 17,
              'Sweatshirt': 18, 'T-shirt': 19, 'Tank Top': 20, 'Top': 21, 'Track Jacket': 22, 'Vest': 23}

inverse_class_dict = {v: k for k, v in class_dict.items()}


class PhotoProcessing1:
    def __init__(self, yolo_model_path='yolov8n.pt', background_color='white', device='cpu'):
        self.model = YOLO(yolo_model_path)
        self.device = device
        self.background_color = background_color.lower()
        self.model.to(self.device)
        self.processed_images = {}
        self.person_class_id = self._get_person_class_id()

    def _get_person_class_id(self):
        for class_id, class_name in self.model.names.items():
            if class_name == 'person':
                return class_id
        raise ValueError("Class 'person' not found in model classes")

    def process_image(self, image):
        width, height = image.size
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32
        image_resized = image.resize((new_width, new_height))

        output_image_rgb = np.array(image_resized)
        output_image_tensor = torch.from_numpy(output_image_rgb).permute(2, 0, 1).float() / 255.0
        output_image_tensor = output_image_tensor.unsqueeze(0).to(self.device)

        results = self.model(output_image_tensor)
        detections = results[0].boxes.data.cpu().numpy()
        person_detected = any(int(detection[5]) == self.person_class_id for detection in detections)

        if person_detected:
            processed_image = self.apply_segcloth(image_resized)
        else:
            processed_image = self.apply_background_color(image_resized)

        return processed_image

    def apply_segcloth(self, image):
        result = segment_clothing(img=image)
        return result

    def apply_background_color(self, image):
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        mask = np.zeros(image_cv.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, image_cv.shape[1] - 10, image_cv.shape[0] - 10)
        cv2.grabCut(image_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        output_image = image_cv.copy()

        if self.background_color == 'white':
            output_image[mask2 == 0] = [255, 255, 255]
        elif self.background_color == 'black':
            output_image[mask2 == 0] = [0, 0, 0]
        elif self.background_color == 'transparent':
            output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2BGRA)
            output_image[mask2 == 0] = [0, 0, 0, 0]
        else:
            raise ValueError("Unsupported background color")

        output_image_rgb = cv2.cvtColor(output_image,
                                        cv2.COLOR_BGR2RGB if self.background_color != 'transparent' else cv2.COLOR_BGRA2RGBA)
        return Image.fromarray(output_image_rgb)


class ColorAnalyzerHSV:
    def __init__(self):
        self.colors_ranges = {
            "Beige": ([20, 50, 100], [30, 150, 200]),
            "Black": ([0, 0, 0], [180, 255, 50]),
            "Blue": ([100, 150, 0], [140, 255, 255]),
            "Bordo": ([160, 100, 20], [180, 255, 255]),
            "Brown": ([10, 100, 20], [20, 255, 200]),
            "Green": ([35, 100, 20], [85, 255, 255]),
            "Grey": ([0, 0, 50], [180, 50, 200]),
            "Light-Blue": ([90, 50, 70], [110, 255, 255]),
            "Olive": ([25, 50, 50], [45, 255, 200]),
            "Orange": ([5, 100, 100], [15, 255, 255]),
            "Pink": ([140, 100, 100], [170, 255, 255]),
            "Red": ([0, 100, 100], [10, 255, 255]),
            "Turquoise": ([85, 100, 100], [100, 255, 255]),
            "Purple": ([130, 50, 50], [160, 255, 255]),
            "White": ([0, 0, 200], [180, 30, 255]),
            "Yellow": ([20, 100, 100], [30, 255, 255])
        }

    def find_dominant_colors(self, image, mask, min_area_threshold=0.15):
        image = np.array(image)
        image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        color_areas = {}
        for color_name, (lower, upper) in self.colors_ranges.items():
            lower_bound = np.array(lower, dtype="uint8")
            upper_bound = np.array(upper, dtype="uint8")
            mask_color = cv2.inRange(image_hsv, lower_bound, upper_bound)
            combined_mask = cv2.bitwise_and(mask_color, mask)
            color_area = cv2.countNonZero(combined_mask)
            total_area = cv2.countNonZero(mask)
            area_percentage = color_area / total_area
            if area_percentage > min_area_threshold:
                color_areas[color_name] = area_percentage * 100
        return color_areas


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Clothing Classifier")
        self.root.geometry("800x900")
        self.model = None
        self.processor = None
        self.image = None

        # Select Model Button
        self.model_button = tk.Button(root, text="Select Model", command=self.load_model)
        self.model_button.grid(row=0, column=0, padx=10, pady=10)

        # Image URL Entry
        self.url_label = tk.Label(root, text="Image URL:")
        self.url_label.grid(row=1, column=0, padx=10, pady=5)
        self.url_entry = tk.Entry(root, width=110)
        self.url_entry.grid(row=1, column=1, padx=10, pady=5)
        self.url_entry.bind("<KeyRelease>", self.check_predict_ready)

        # Select Image Button
        self.image_button = tk.Button(root, text="Select Image", command=self.load_image)
        self.image_button.grid(row=2, column=0, padx=10, pady=10)

        # Predict Button
        self.predict_button = tk.Button(root, text="Predict", command=self.predict, state=tk.DISABLED)
        self.predict_button.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        # Clear Image Button
        self.clear_button = tk.Button(root, text="Clear Image", command=self.clear_image)
        self.clear_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

        # Output Area
        self.output_text = tk.Text(root, height=7, width=97)
        self.output_text.grid(row=5, column=0, columnspan=3, padx=10, pady=10)

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model Files", "*.pth")])
        if model_path:
            self.model = TransferLearningClassifier(num_classes=len(class_dict))
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            self.model.eval()
            self.processor = PhotoProcessing1(background_color='white')
            messagebox.showinfo("Model Loaded", "Model loaded successfully!")
            self.check_predict_ready()

    def load_image(self):
        image_path = filedialog.askopenfilename(title="Select Image File",
                                                filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if image_path:
            self.image = Image.open(image_path).convert('RGB')
            self.show_image(self.image)
            self.check_predict_ready()

    def show_image(self, image):
        max_size = 500
        aspect_ratio = max(image.size) / max_size
        new_size = (int(image.size[0] / aspect_ratio), int(image.size[1] / aspect_ratio))
        image_resized = image.resize(new_size, Image.LANCZOS)

        photo = ImageTk.PhotoImage(image_resized)
        img_label = tk.Label(image=photo)
        img_label.image = photo
        img_label.grid(row=6, column=0, columnspan=2)

    def check_predict_ready(self, event=None):
        if self.model and (self.image or self.url_entry.get()):
            self.predict_button.config(state=tk.NORMAL)

    def clear_image(self):
        self.image = None
        self.url_entry.delete(0, tk.END)
        self.predict_button.config(state=tk.DISABLED)
        for widget in self.root.grid_slaves():
            if isinstance(widget, tk.Label):
                widget.destroy()
        self.output_text.delete("1.0", tk.END)

    def predict(self):
        self.output_text.delete("1.0", tk.END)
        if self.url_entry.get() and not self.image:
            response = requests.get(self.url_entry.get())
            self.image = Image.open(BytesIO(response.content)).convert('RGB')
            self.show_image(self.image)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.8329, 0.8211, 0.8254], std=[0.2497, 0.2639, 0.2580])
        ])

        processed_image = self.processor.process_image(self.image)
        image_tensor = transform(processed_image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        self.model = self.model.to(device)

        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = output.argmax(dim=1).item()

        class_name = inverse_class_dict[prediction]
        self.output_text.insert(tk.END, f"Predicted class: {class_name}\n")

        image_np = np.array(processed_image)
        mask = np.zeros(image_np.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        rect = (10, 10, image_np.shape[1] - 10, image_np.shape[0] - 10)
        cv2.grabCut(image_np, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

        analyzer = ColorAnalyzerHSV()
        dominant_colors = analyzer.find_dominant_colors(processed_image.convert("RGB"), mask2)

        if dominant_colors:
            self.output_text.insert(tk.END, "Dominant colors:\n")
            for color_name, percentage in dominant_colors.items():
                self.output_text.insert(tk.END, f"{color_name}: {percentage:.2f}%\n")
        else:
            self.output_text.insert(tk.END, "No dominant colors found above the threshold.\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
