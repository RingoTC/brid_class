import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                           QWidget, QScrollArea, QPushButton, QHBoxLayout,
                           QComboBox)
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
            # Convert to RGB mode to ensure 3 channels
            image = image.convert('RGB')
            
            # Print image information for debugging
            print(f"Image size: {image.size}")
            print(f"Image mode: {image.mode}")
            
            # Preprocess image with explicit format
            inputs = self.processor(
                images=image,
                return_tensors="pt",
                do_resize=True,
                do_normalize=True,
                size={"height": 224, "width": 224}  # ViT default size
            )
            
            # Move to device
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
            import traceback
            traceback.print_exc()  # Print full error traceback
            self.finished.emit([])

class BirdClassifierWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bird Species Classifier")
        self.setMinimumSize(1200, 800)
        
        # Initialize models dict
        self.models = {}
        self.current_model = None
        self.current_processor = None
        
        # Load available models
        self.load_available_models()
        
        # Setup class names
        self.setup_class_names()
        
        # Setup UI
        self.setup_ui()
    
    def load_available_models(self):
        """Load all available models"""
        # Check for CIFAR model
        cifar_path = "models/cifar10/best"  # Changed from cifar100 to cifar10
        print(f"Checking for CIFAR model at: {cifar_path}")
        if os.path.exists(cifar_path):
            try:
                model = ViTForImageClassification.from_pretrained(cifar_path)
                processor = ViTImageProcessor.from_pretrained(cifar_path)
                model.to(get_device())
                model.eval()
                self.models['cifar10'] = (model, processor)  # Use lowercase for consistency
                print("Loaded CIFAR-10 model successfully")
            except Exception as e:
                print(f"Error loading CIFAR-10 model: {e}")
        else:
            print(f"CIFAR model not found at {cifar_path}")
        
        # Check for BirdSnap model
        birdsnap_path = "models/birdsnap/best"
        print(f"Checking for BirdSnap model at: {birdsnap_path}")
        if os.path.exists(birdsnap_path):
            try:
                model = ViTForImageClassification.from_pretrained(birdsnap_path)
                processor = ViTImageProcessor.from_pretrained(birdsnap_path)
                model.to(get_device())
                model.eval()
                self.models['birdsnap'] = (model, processor)  # Use lowercase for consistency
                print("Loaded BirdSnap model successfully")
            except Exception as e:
                print(f"Error loading BirdSnap model: {e}")
        else:
            print(f"BirdSnap model not found at {birdsnap_path}")
            
        if not self.models:
            print("Warning: No models were loaded successfully")
        
    def setup_class_names(self):
        """Load class names for different models"""
        from dataset_utils import get_dataset_labels
        
        self.class_names = {}
        for model_name in self.models.keys():
            labels = get_dataset_labels(model_name)
            if labels:
                self.class_names[model_name] = labels
                print(f"Loaded {len(labels)} class names for {model_name}")
            else:
                print(f"Warning: Could not load class names for {model_name}")
            
    def setup_ui(self):
        # Create central widget and main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create model selection layout
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("Select Model:")
        self.model_combo = QComboBox()
        
        # Add models with display names
        model_display_names = {
            'cifar10': 'CIFAR-10',
            'birdsnap': 'BirdSnap'
        }
        for model_name in self.models.keys():
            display_name = model_display_names.get(model_name, model_name)
            self.model_combo.addItem(display_name, model_name)  # Store internal name as data
            
        self.model_combo.currentTextChanged.connect(self.on_model_changed)
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_combo)
        model_selection_layout.addStretch()
        main_layout.addLayout(model_selection_layout)
        
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
        
        # Create results area
        self.results_label = QLabel("Model Results")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setWordWrap(True)
        self.results_label.setStyleSheet("""
            QLabel {
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ddd;
                margin: 5px;
            }
        """)
        
        # Add results label to scroll area
        results_scroll = QScrollArea()
        results_scroll.setWidget(self.results_label)
        results_scroll.setWidgetResizable(True)
        results_scroll.setMinimumHeight(200)
        results_scroll.setMinimumWidth(800)
        main_layout.addWidget(results_scroll)
        
        # Set initial model if available
        if self.model_combo.count() > 0:
            self.on_model_changed(self.model_combo.currentText())
        
        # Event handling
        self.drop_area.dragEnterEvent = self.dragEnterEvent
        self.drop_area.dropEvent = self.dropEvent
    
    def on_model_changed(self, display_name):
        """Handle model selection change"""
        # Get the internal model name from the combobox data
        index = self.model_combo.currentIndex()
        model_name = self.model_combo.itemData(index)
        
        if model_name in self.models:
            self.current_model, self.current_processor = self.models[model_name]
            self.results_label.setText(f"Selected model: {display_name}\nDrag and drop an image to classify")
    
    def process_image(self, image_path):
        # Display the image
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(self.drop_area.size(), 
                                    Qt.AspectRatioMode.KeepAspectRatio,
                                    Qt.TransformationMode.SmoothTransformation)
        self.drop_area.setPixmap(scaled_pixmap)
        
        if self.current_model is None:
            self.results_label.setText("Please select a model first")
            return
        
        # Start prediction thread
        self.prediction_thread = PredictionThread(
            self.current_model,
            self.current_processor,
            image_path
        )
        self.prediction_thread.finished.connect(self.handle_prediction)
        self.prediction_thread.start()
        
        # Show loading message
        self.results_label.setText("Analyzing image...")
    
    def handle_prediction(self, predictions):
        if not predictions:
            self.results_label.setText("Unable to classify the image. Please try another image.")
            return
        
        # Get current model name and corresponding class names
        index = self.model_combo.currentIndex()
        model_name = self.model_combo.itemData(index)
        display_name = self.model_combo.currentText()
        
        if model_name not in self.class_names:
            self.results_label.setText(f"Error: No class names available for {display_name}")
            return
            
        # Format results
        results_text = f"{display_name} Model Predictions:\n\n"
        for idx, prob in predictions:
            class_name = self.class_names[model_name][idx]
            results_text += f"{class_name}: {prob*100:.1f}%\n"
        
        # Update UI
        self.results_label.setText(results_text)
    
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