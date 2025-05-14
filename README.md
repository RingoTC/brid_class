# Bird Classification Model

This project implements a bird classification model using YOLOv8 and the Birdsnap dataset from Hugging Face.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- MPS-enabled device (for Apple Silicon) or CPU

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset

The project uses the Birdsnap dataset from Hugging Face, which contains various bird species images. The dataset is automatically downloaded and processed when you run the training script.

The data is split into:
- Training set (70%)
- Validation set (15%)
- Test set (15%)

## Training

To train the model, simply run:
```bash
python train.py
```

The script will:
1. Download and process the Birdsnap dataset
2. Create the necessary directory structure
3. Split the data into train/val/test sets
4. Train the YOLOv8 model
5. Save the results in the `bird_classification` directory

## Model Configuration

The training uses the following configuration:
- Image size: 640x640
- Batch size: 16
- Number of epochs: 100
- Early stopping patience: 20
- Device: MPS (Apple Silicon) or CPU

## Results

Training results and model weights will be saved in the `bird_classification` directory. 