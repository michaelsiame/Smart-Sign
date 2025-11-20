#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for the point history classifier model.
This script trains a neural network to classify finger gestures based on movement history.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def get_args():
    parser = argparse.ArgumentParser(description='Train point history classifier')
    parser.add_argument('--dataset', type=str, default='model/point_history_classifier/point_history.csv',
                        help='Path to the training dataset CSV')
    parser.add_argument('--model_save_path', type=str, default='model/point_history_classifier/point_history_classifier.keras',
                        help='Path to save the trained model')
    parser.add_argument('--tflite_save_path', type=str, default='model/point_history_classifier/point_history_classifier.tflite',
                        help='Path to save the TFLite model')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='Number of gesture classes')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Training batch size')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Proportion of dataset for testing')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')

    return parser.parse_args()


def load_dataset(csv_path):
    """Load dataset from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    # Load features (point history, 16 points x 2 coordinates = 32 features)
    X = np.loadtxt(csv_path, delimiter=',', dtype='float32', usecols=list(range(1, (16 * 2) + 1)))
    # Load labels
    y = np.loadtxt(csv_path, delimiter=',', dtype='int32', usecols=(0))

    print(f"Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Class distribution: {np.bincount(y)}")

    return X, y


def augment_data(X, y, augmentation_factor=2):
    """
    Augment training data with small random variations.

    Args:
        X: Training features
        y: Training labels
        augmentation_factor: How many augmented copies to create per sample

    Returns:
        Augmented X and y
    """
    X_augmented = [X]
    y_augmented = [y]

    for _ in range(augmentation_factor):
        # Add small random noise to simulate hand tremor
        noise = np.random.normal(0, 0.015, X.shape)
        X_noisy = X + noise

        # Random time shifting (circular shift)
        shifts = np.random.randint(-2, 3, X.shape[0])
        X_shifted = np.array([np.roll(x.reshape(-1, 2), shift, axis=0).flatten()
                              for x, shift in zip(X, shifts)])

        X_augmented.extend([X_noisy, X_shifted])
        y_augmented.extend([y, y])

    X_final = np.vstack(X_augmented)
    y_final = np.hstack(y_augmented)

    print(f"Data augmented: {X.shape[0]} -> {X_final.shape[0]} samples")

    return X_final, y_final


def create_model(num_classes=4):
    """Create the point history classifier model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((16 * 2, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model


def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix."""
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)

    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_cmx, annot=True, fmt='g', square=False, cmap='Blues')
    plt.title('Confusion Matrix - Point History Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)

    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")
    plt.close()


def convert_to_tflite(model, save_path):
    """Convert Keras model to TensorFlow Lite format."""
    # Save model without optimizer for inference
    model.save(save_path.replace('.tflite', '_inference.keras'), include_optimizer=False)

    # Convert to TFLite with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    # Save TFLite model
    with open(save_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {save_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")


def main():
    args = get_args()

    print("=" * 60)
    print("Point History Classifier Training")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Model save path: {args.model_save_path}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Test size: {args.test_size}")
    print(f"Random seed: {args.random_seed}")
    print(f"Data augmentation: {args.augment}")
    print("=" * 60)

    # Load dataset
    X, y = load_dataset(args.dataset)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")

    # Apply data augmentation if enabled
    if args.augment:
        X_train, y_train = augment_data(X_train, y_train)

    # Create model
    model = create_model(num_classes=args.num_classes)
    model.summary()

    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Create callbacks
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        args.model_save_path,
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        verbose=1,
        restore_best_weights=True
    )

    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )

    # Train model
    print("\nTraining model...")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint_callback, early_stopping_callback, reduce_lr_callback],
        verbose=1
    )

    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=args.batch_size)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Generate predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot confusion matrix
    output_dir = os.path.dirname(args.model_save_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    confusion_matrix_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
    plot_confusion_matrix(y_test, y_pred, save_path=confusion_matrix_path)

    # Plot training history
    history_path = os.path.join(output_dir, f'training_history_{timestamp}.png')
    plot_training_history(history, save_path=history_path)

    # Convert to TFLite
    print("\nConverting to TensorFlow Lite...")
    convert_to_tflite(model, args.tflite_save_path)

    # Test TFLite model
    print("\nTesting TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=args.tflite_save_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test on a single sample
    interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]], dtype=np.float32))
    interpreter.invoke()
    tflite_result = interpreter.get_tensor(output_details[0]['index'])

    print(f"TFLite prediction: {np.argmax(tflite_result)}")
    print(f"Actual label: {y_test[0]}")

    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()
