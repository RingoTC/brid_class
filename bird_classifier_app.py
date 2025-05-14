import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QWidget, QScrollArea, QPushButton, QHBoxLayout)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QDragEnterEvent, QDropEvent, QImage
import torch
from PIL import Image
import numpy as np
from transformers import ViTImageProcessor, ViTForImageClassification
from utils import get_device

class PredictionThread(QThread):
    """Separate thread for model prediction to keep UI responsive"""
    finished = pyqtSignal(list)  # Signal to emit results
    
    def __init__(self, model, processor, image_path):
        super().__init__()
        self.model = model
        self.processor = processor
        self.image_path = image_path
        
    def run(self):
        try:
            # Load and preprocess image
            image = Image.open(self.image_path)
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(get_device()) for k, v in inputs.items()}
            
            # Run prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Get top 5 predictions
            top5_values, top5_indices = torch.topk(probs[0], 5)
            predictions = [(idx.item(), prob.item()) for idx, prob in zip(top5_indices, top5_values)]
            self.finished.emit(predictions)
        except Exception as e:
            print(f"Prediction error: {e}")
            self.finished.emit([])

class BirdClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Species Classifier")
        self.setMinimumSize(1200, 800)  # Increased window size for side-by-side display
        
        # Load pretrained model and processor
        model_name = "google/vit-base-patch16-224"
        self.pretrained_processor = ViTImageProcessor.from_pretrained(model_name)
        self.pretrained_model = ViTForImageClassification.from_pretrained(model_name)
        self.pretrained_model.to(get_device())
        self.pretrained_model.eval()
        
        # Load finetuned model if available
        self.finetuned_model = None
        self.finetuned_processor = None
        finetuned_path = "models/finetuned/best"
        if os.path.exists(finetuned_path):
            try:
                self.finetuned_model = ViTForImageClassification.from_pretrained(finetuned_path)
                self.finetuned_processor = ViTImageProcessor.from_pretrained(finetuned_path)
                self.finetuned_model.to(get_device())
                self.finetuned_model.eval()
                print("Loaded finetuned model successfully")
            except Exception as e:
                print(f"Error loading finetuned model: {e}")
        
        # Setup class names
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
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
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
        main_layout.addWidget(self.drop_area)
        
        # Create horizontal layout for results
        results_layout = QHBoxLayout()
        
        # Create results area for pretrained model
        self.pretrained_results_label = QLabel("Pretrained Model Results")
        self.pretrained_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pretrained_results_label.setWordWrap(True)
        self.pretrained_results_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                margin: 5px;
            }
        """)
        
        # Create results area for finetuned model
        self.finetuned_results_label = QLabel(
            "Finetuned Model Results\n(No finetuned model available)" 
            if self.finetuned_model is None else "Finetuned Model Results"
        )
        self.finetuned_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.finetuned_results_label.setWordWrap(True)
        self.finetuned_results_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                margin: 5px;
            }
        """)
        
        # Add results labels to scroll areas
        pretrained_scroll = QScrollArea()
        pretrained_scroll.setWidget(self.pretrained_results_label)
        pretrained_scroll.setWidgetResizable(True)
        pretrained_scroll.setMinimumHeight(200)
        pretrained_scroll.setMinimumWidth(400)
        
        finetuned_scroll = QScrollArea()
        finetuned_scroll.setWidget(self.finetuned_results_label)
        finetuned_scroll.setWidgetResizable(True)
        finetuned_scroll.setMinimumHeight(200)
        finetuned_scroll.setMinimumWidth(400)
        
        # Add scroll areas to results layout
        results_layout.addWidget(pretrained_scroll)
        results_layout.addWidget(finetuned_scroll)
        
        # Add results layout to main layout
        main_layout.addLayout(results_layout)
        
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
        
        # Start prediction with pretrained model
        self.pretrained_thread = PredictionThread(
            self.pretrained_model, 
            self.pretrained_processor, 
            image_path
        )
        self.pretrained_thread.finished.connect(self.handle_pretrained_prediction)
        self.pretrained_thread.start()
        
        # Start prediction with finetuned model if available
        if self.finetuned_model is not None:
            self.finetuned_thread = PredictionThread(
                self.finetuned_model,
                self.finetuned_processor,
                image_path
            )
            self.finetuned_thread.finished.connect(self.handle_finetuned_prediction)
            self.finetuned_thread.start()
        
        # Show loading messages
        self.pretrained_results_label.setText("Analyzing image...")
        if self.finetuned_model is not None:
            self.finetuned_results_label.setText("Analyzing image...")
        
    def handle_pretrained_prediction(self, predictions):
        if not predictions:
            self.pretrained_results_label.setText("Unable to classify the image. Please try another image.")
            return
            
        # Format results
        results_text = "Pretrained Model Predictions:\n\n"
        for idx, prob in predictions:
            species_name = self.class_names[idx]
            results_text += f"{species_name}: {prob*100:.1f}%\n"
        
        # Update UI
        self.pretrained_results_label.setText(results_text)
        
    def handle_finetuned_prediction(self, predictions):
        if not predictions:
            self.finetuned_results_label.setText("Unable to classify the image. Please try another image.")
            return
            
        # Format results
        results_text = "Finetuned Model Predictions:\n\n"
        for idx, prob in predictions:
            species_name = self.class_names[idx]
            results_text += f"{species_name}: {prob*100:.1f}%\n"
        
        # Update UI
        self.finetuned_results_label.setText(results_text)

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