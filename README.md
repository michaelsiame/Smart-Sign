# Smart Sign - Sign Language Interpreter

A real-time sign language interpreter using hand gesture recognition with MediaPipe and TensorFlow. This system can detect and classify hand gestures, track finger movements, and continuously improve through dataset reinforcement and retraining.

## Features

- **Real-time Hand Gesture Recognition**: Detects and classifies hand signs using MediaPipe hand landmarks
- **Finger Movement Tracking**: Tracks and classifies dynamic finger gestures
- **Confidence Scores**: Displays prediction confidence for better accuracy assessment
- **Data Collection Mode**: Easily collect training data for new gestures
- **Model Training**: Complete training pipeline with data augmentation
- **Model Evaluation**: Comprehensive evaluation tools with visualization
- **Gesture Management**: Easy-to-use utility for managing gesture labels
- **Optimized for Performance**: Uses TensorFlow Lite for fast inference

## Supported Gestures

### Hand Signs (Keypoint Classifier)
- Open
- Close
- Pointer
- OK
- ASL A
- ASL B
- ASL C
- ASL D

### Finger Gestures (Point History Classifier)
- Clockwise
- Counter-Clockwise
- Move
- Stop

## Installation

### Prerequisites
- Python 3.8 or higher
- Webcam for real-time detection

### Setup

1. Clone the repository:
```bash
git clone https://github.com/michaelsiame/Smart-Sign.git
cd Smart-Sign
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Real-time Gesture Recognition

Run the main application to start real-time gesture recognition:

```bash
python app.py
```

**Keyboard Controls:**
- `ESC`: Exit the application
- `n`: Normal mode (recognition only)
- `k`: Keypoint logging mode (collect hand gesture data)
- `h`: History logging mode (collect finger movement data)
- `0-9`: Select gesture class when in logging mode

### 2. Collecting Training Data

To add a new gesture or improve existing ones:

1. Run the application:
```bash
python app.py
```

2. Press `k` to enter keypoint logging mode
3. Press a number key (0-9) to select the gesture class
4. Perform the gesture in front of the camera
5. Data will be automatically saved to `model/keypoint_classifier/keypoint.csv`

For finger movement gestures:
1. Press `h` to enter history logging mode
2. Press a number key (0-9) to select the gesture class
3. Make the finger movement
4. Data will be saved to `model/point_history_classifier/point_history.csv`

### 3. Managing Gestures

Use the gesture management utility to add, remove, or rename gestures:

```bash
# List all gestures
python manage_gestures.py list

# Add a new gesture
python manage_gestures.py add "Thumbs Up"

# Remove a gesture by name or index
python manage_gestures.py remove "Thumbs Up"
python manage_gestures.py remove 5

# Rename a gesture
python manage_gestures.py rename "Old Name" "New Name"

# Show dataset statistics
python manage_gestures.py stats

# Interactive mode
python manage_gestures.py interactive
```

### 4. Training Models

#### Train Keypoint Classifier (Hand Gestures)

```bash
# Basic training
python train_keypoint_classifier.py

# Training with data augmentation
python train_keypoint_classifier.py --augment

# Custom parameters
python train_keypoint_classifier.py \
    --dataset model/keypoint_classifier/keypoint.csv \
    --num_classes 8 \
    --epochs 1000 \
    --batch_size 128 \
    --augment
```

**Available options:**
- `--dataset`: Path to training dataset CSV
- `--num_classes`: Number of gesture classes
- `--epochs`: Number of training epochs
- `--batch_size`: Training batch size
- `--test_size`: Proportion of data for testing (default: 0.25)
- `--patience`: Early stopping patience (default: 20)
- `--augment`: Enable data augmentation

#### Train Point History Classifier (Finger Movements)

```bash
# Basic training
python train_point_history_classifier.py

# Training with data augmentation
python train_point_history_classifier.py --augment

# Custom parameters
python train_point_history_classifier.py \
    --dataset model/point_history_classifier/point_history.csv \
    --num_classes 4 \
    --epochs 1000 \
    --augment
```

### 5. Evaluating Models

Evaluate model performance and generate detailed reports:

```bash
# Evaluate keypoint classifier
python evaluate_model.py --model_type keypoint

# Evaluate point history classifier
python evaluate_model.py --model_type point_history

# Custom evaluation
python evaluate_model.py \
    --model_type keypoint \
    --keras_model model/keypoint_classifier/keypoint_classifier.keras \
    --tflite_model model/keypoint_classifier/keypoint_classifier.tflite \
    --output_dir evaluation_results
```

**Evaluation outputs:**
- Confusion matrix
- Per-class performance metrics
- Confidence score distribution
- Detailed classification report

## Project Structure

```
Smart-Sign/
├── app.py                              # Main application
├── requirements.txt                     # Python dependencies
├── README.md                           # This file
│
├── model/
│   ├── __init__.py
│   ├── keypoint_classifier/
│   │   ├── keypoint_classifier.py      # Keypoint classifier class
│   │   ├── keypoint_classifier.tflite  # Trained TFLite model
│   │   ├── keypoint_classifier.keras   # Trained Keras model
│   │   ├── keypoint.csv                # Training dataset
│   │   └── keypoint_classifier_label.csv # Gesture labels
│   │
│   └── point_history_classifier/
│       ├── point_history_classifier.py  # Point history classifier class
│       ├── point_history_classifier.tflite # Trained TFLite model
│       ├── point_history.csv           # Training dataset
│       └── point_history_classifier_label.csv # Gesture labels
│
├── utils/
│   ├── __init__.py
│   └── cvfpscalc.py                    # FPS calculation utility
│
├── train_keypoint_classifier.py        # Training script for hand gestures
├── train_point_history_classifier.py   # Training script for finger movements
├── evaluate_model.py                   # Model evaluation script
├── manage_gestures.py                  # Gesture management utility
│
├── keypoint_classification.ipynb       # Training notebook (alternative)
└── point_history_classification.ipynb  # Training notebook (alternative)
```

## Model Architecture

### Keypoint Classifier
- **Input**: 42 features (21 hand landmarks × 2 coordinates)
- **Architecture**:
  - Dropout (0.2)
  - Dense (20 units, ReLU)
  - Dropout (0.4)
  - Dense (10 units, ReLU)
  - Dense (num_classes, Softmax)
- **Output**: Gesture class probabilities

### Point History Classifier
- **Input**: 32 features (16 points × 2 coordinates)
- **Architecture**:
  - Dropout (0.2)
  - Dense (20 units, ReLU)
  - Dropout (0.5)
  - Dense (10 units, ReLU)
  - Dense (num_classes, Softmax)
- **Output**: Movement class probabilities

## Data Augmentation

Both training scripts support data augmentation to improve model robustness:

**Keypoint Classifier Augmentation:**
- Random noise addition (simulates hand tremor)
- Random scaling (accounts for hand size variations)

**Point History Classifier Augmentation:**
- Random noise addition
- Temporal shifting (accounts for timing variations)

## Training Tips

1. **Collect diverse data**: Record gestures from different angles, lighting conditions, and backgrounds
2. **Balance your dataset**: Aim for similar number of samples per gesture class
3. **Use data augmentation**: Enable `--augment` flag for better generalization
4. **Monitor training**: Check the generated plots for signs of overfitting
5. **Evaluate regularly**: Use the evaluation script to track model performance
6. **Iterate**: Collect more data for poorly performing gestures

## Performance Optimization

- Models are quantized to TensorFlow Lite format for fast inference
- Typical inference time: < 5ms per frame on modern CPUs
- Supports multi-threading for improved performance

## Troubleshooting

### Camera not detected
- Check if your webcam is properly connected
- Try changing the `--device` parameter: `python app.py --device 1`

### Low accuracy
- Collect more training data (aim for 500+ samples per gesture)
- Enable data augmentation during training
- Ensure good lighting conditions
- Check if gestures are too similar (consider merging similar classes)

### Import errors
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+ required)

## Advanced Usage

### Custom Camera Settings

```bash
python app.py --device 0 --width 1280 --height 720
```

### Adjusting Detection Confidence

Edit `app.py` and modify:
```python
min_detection_confidence = 0.7  # Increase for stricter detection
min_tracking_confidence = 0.5   # Increase for more stable tracking
```

### Adding More Gesture Classes

1. Update the number of classes in label files
2. Add gesture names using `manage_gestures.py`
3. Collect training data for new gestures
4. Retrain with updated `--num_classes` parameter

## Citation

If you use this project in your research, please cite:

```
@software{smart_sign,
  author = {Smart Sign Contributors},
  title = {Smart Sign - Sign Language Interpreter},
  year = {2024},
  url = {https://github.com/michaelsiame/Smart-Sign}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is open source and available for educational and research purposes.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for hand landmark detection
- [TensorFlow](https://www.tensorflow.org/) for model training and inference
- The sign language community for inspiration

## Contact

For questions and support, please open an issue on GitHub.

---

**Made with dedication to improving accessibility through technology**
