import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QWidget, QScrollArea, QPushButton)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage
import torch
from PIL import Image
from ultralytics import YOLO
import numpy as np
from utils import get_device

class PredictionThread(QThread):
    """Separate thread for model prediction to keep UI responsive"""
    finished = pyqtSignal(list)  # Signal to emit results
    
    def __init__(self, model, image_path):
        super().__init__()
        self.model = model
        self.image_path = image_path
        
    def run(self):
        # Run prediction
        results = self.model.predict(self.image_path, device=get_device())
        # Get top 5 predictions with probabilities
        probs = results[0].probs.data.cpu().numpy()
        top5_idx = np.argsort(probs)[-5:][::-1]
        predictions = [(idx, float(probs[idx])) for idx in top5_idx]
        self.finished.emit(predictions)

class BirdClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Species Classifier")
        self.setMinimumSize(800, 600)
        
        # Load model
        model_path = "bird_classification/train/weights/best.pt"
        if not os.path.exists(model_path):
            model_path = "yolov8n.pt"  # Use base model if no fine-tuned model exists
        self.model = YOLO(model_path)
        
        # Load class names
        self.setup_class_names()
        
        # Setup UI
        self.setup_ui()
        
    def setup_class_names(self):
        """Load bird species names"""
        try:
            from dataset_utils import BirdDatasetProcessor
            processor = BirdDatasetProcessor()
            dataset_info = processor.get_dataset_info()
            if dataset_info:
                self.class_names = dataset_info['names']
            else:
                # Fallback to downloading class names
                import requests
                url = "https://huggingface.co/datasets/sasha/birdsnap/resolve/main/species.txt"
                response = requests.get(url)
                lines = response.text.split('\n')
                self.class_names = [line.split()[1].replace('_', ' ').title() 
                                  for line in lines if line.strip()]
        except Exception as e:
            print(f"Error loading class names: {e}")
            self.class_names = [f"Class {i}" for i in range(500)]  # Fallback
    
    def setup_ui(self):
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Create drop area
        self.drop_area = QLabel("Drag and drop bird images here\nor click to select files")
        self.drop_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_area.setMinimumSize(400, 300)
        self.drop_area.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 10px;
                background-color: #f0f0f0;
                padding: 20px;
            }
        """)
        self.drop_area.setAcceptDrops(True)
        layout.addWidget(self.drop_area)
        
        # Create results area
        self.results_label = QLabel("Results will appear here")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
            }
        """)
        
        # Add results to a scroll area
        scroll = QScrollArea()
        scroll.setWidget(self.results_label)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(200)
        layout.addWidget(scroll)
        
        # Event handling
        self.drop_area.dragEnterEvent = self.dragEnterEvent
        self.drop_area.dropEvent = self.dropEvent
        
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()
            
    def dropEvent(self, event: QDropEvent):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        for file_path in files:
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.process_image(file_path)
                break
    
    def process_image(self, image_path):
        # Display the image
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.drop_area.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.drop_area.setPixmap(scaled_pixmap)
        
        # Start prediction in separate thread
        self.prediction_thread = PredictionThread(self.model, image_path)
        self.prediction_thread.finished.connect(self.handle_prediction)
        self.prediction_thread.start()
        
        # Show loading message
        self.results_label.setText("Analyzing image...")
        
    def handle_prediction(self, predictions):
        # Format results
        results_text = "Top 5 Predictions:\n\n"
        for idx, prob in predictions:
            species_name = self.class_names[idx]
            results_text += f"{species_name}: {prob*100:.1f}%\n"
        
        # Update UI
        self.results_label.setText(results_text)

def main():
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show the main window
    window = BirdClassifierWindow()
    window.show()
    
    sys.exit(app.exec())

if __name__ == '__main__':
    main() 