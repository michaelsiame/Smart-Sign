#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Model Evaluation Script
Evaluate trained models and generate comprehensive performance reports.
"""

import argparse
import os
import csv
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score
)


def load_dataset(csv_path, num_features):
    """Load dataset from CSV file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    X = np.loadtxt(csv_path, delimiter=',', dtype='float32', usecols=list(range(1, num_features + 1)))
    y = np.loadtxt(csv_path, delimiter=',', dtype='int32', usecols=(0))

    return X, y


def load_labels(label_file):
    """Load gesture labels."""
    with open(label_file, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        labels = [row[0] for row in reader]
    return labels


def evaluate_keras_model(model_path, X_test, y_test, class_names):
    """Evaluate a Keras model."""
    print(f"\nEvaluating Keras model: {model_path}")

    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("Keras Model Performance")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print("=" * 60)

    return y_pred, y_pred_probs


def evaluate_tflite_model(model_path, X_test, y_test, class_names):
    """Evaluate a TFLite model."""
    print(f"\nEvaluating TFLite model: {model_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference on test set
    y_pred_probs = []
    for sample in X_test:
        interpreter.set_tensor(input_details[0]['index'], np.array([sample], dtype=np.float32))
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        y_pred_probs.append(output[0])

    y_pred_probs = np.array(y_pred_probs)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')

    print("\n" + "=" * 60)
    print("TFLite Model Performance")
    print("=" * 60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    # Model size
    model_size = os.path.getsize(model_path) / 1024
    print(f"Model Size: {model_size:.2f} KB")
    print("=" * 60)

    return y_pred, y_pred_probs


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix with labels."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path):
    """Plot per-class precision, recall, and F1 score."""
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)

    ax.set_xlabel('Gesture Class', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1.1])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Per-class metrics saved to {save_path}")
    plt.close()


def plot_confidence_distribution(y_pred_probs, y_true, save_path):
    """Plot confidence score distribution."""
    # Get confidence scores (max probability for each prediction)
    confidence_scores = np.max(y_pred_probs, axis=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    correct = (y_pred == y_true)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Overall confidence distribution
    ax1.hist(confidence_scores, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.axvline(np.mean(confidence_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(confidence_scores):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Correct vs Incorrect predictions
    ax2.hist(confidence_scores[correct], bins=30, alpha=0.7, label='Correct', edgecolor='black')
    ax2.hist(confidence_scores[~correct], bins=30, alpha=0.7, label='Incorrect', edgecolor='black')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title('Confidence: Correct vs Incorrect Predictions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confidence distribution saved to {save_path}")
    plt.close()


def generate_report(y_true, y_pred, class_names, save_path):
    """Generate and save detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    with open(save_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DETAILED CLASSIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(report)
        f.write("\n" + "=" * 80 + "\n")

        # Additional statistics
        accuracy = accuracy_score(y_true, y_pred)
        f.write(f"\nOverall Accuracy: {accuracy:.4f}\n")

        # Per-class accuracy
        f.write("\nPer-Class Accuracy:\n")
        f.write("-" * 40 + "\n")
        for i, class_name in enumerate(class_names):
            mask = (y_true == i)
            if np.sum(mask) > 0:
                class_acc = accuracy_score(y_true[mask], y_pred[mask])
                f.write(f"{class_name}: {class_acc:.4f} ({np.sum(mask)} samples)\n")

    print(f"Detailed report saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')

    parser.add_argument('--model_type', type=str, choices=['keypoint', 'point_history'],
                        default='keypoint', help='Type of model to evaluate')

    parser.add_argument('--keras_model', type=str,
                        help='Path to Keras model (optional)')

    parser.add_argument('--tflite_model', type=str,
                        help='Path to TFLite model (optional)')

    parser.add_argument('--dataset', type=str,
                        help='Path to dataset CSV (optional)')

    parser.add_argument('--labels', type=str,
                        help='Path to labels CSV (optional)')

    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation results')

    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Proportion of dataset for testing')

    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    # Set default paths based on model type
    if args.model_type == 'keypoint':
        keras_model = args.keras_model or 'model/keypoint_classifier/keypoint_classifier.keras'
        tflite_model = args.tflite_model or 'model/keypoint_classifier/keypoint_classifier.tflite'
        dataset = args.dataset or 'model/keypoint_classifier/keypoint.csv'
        labels_file = args.labels or 'model/keypoint_classifier/keypoint_classifier_label.csv'
        num_features = 21 * 2  # 21 landmarks x 2 coordinates
    else:
        keras_model = args.keras_model or 'model/point_history_classifier/point_history_classifier.keras'
        tflite_model = args.tflite_model or 'model/point_history_classifier/point_history_classifier.tflite'
        dataset = args.dataset or 'model/point_history_classifier/point_history.csv'
        labels_file = args.labels or 'model/point_history_classifier/point_history_classifier_label.csv'
        num_features = 16 * 2  # 16 points x 2 coordinates

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print(f"Model Evaluation - {args.model_type.upper()}")
    print("=" * 60)

    # Load dataset and labels
    print(f"Loading dataset from {dataset}...")
    X, y = load_dataset(dataset, num_features)

    print(f"Loading labels from {labels_file}...")
    class_names = load_labels(labels_file)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_seed, stratify=y
    )

    print(f"Test set size: {len(X_test)} samples")
    print(f"Number of classes: {len(class_names)}")

    # Evaluate models
    results = {}

    if os.path.exists(keras_model):
        y_pred_keras, y_pred_probs_keras = evaluate_keras_model(
            keras_model, X_test, y_test, class_names
        )
        results['keras'] = (y_pred_keras, y_pred_probs_keras)

        # Generate visualizations for Keras model
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_keras_{timestamp}.png')
        plot_confusion_matrix(y_test, y_pred_keras, class_names, cm_path)

        metrics_path = os.path.join(args.output_dir, f'per_class_metrics_keras_{timestamp}.png')
        plot_per_class_metrics(y_test, y_pred_keras, class_names, metrics_path)

        conf_path = os.path.join(args.output_dir, f'confidence_distribution_keras_{timestamp}.png')
        plot_confidence_distribution(y_pred_probs_keras, y_test, conf_path)

        report_path = os.path.join(args.output_dir, f'classification_report_keras_{timestamp}.txt')
        generate_report(y_test, y_pred_keras, class_names, report_path)

    if os.path.exists(tflite_model):
        y_pred_tflite, y_pred_probs_tflite = evaluate_tflite_model(
            tflite_model, X_test, y_test, class_names
        )
        results['tflite'] = (y_pred_tflite, y_pred_probs_tflite)

        # Generate visualizations for TFLite model
        cm_path = os.path.join(args.output_dir, f'confusion_matrix_tflite_{timestamp}.png')
        plot_confusion_matrix(y_test, y_pred_tflite, class_names, cm_path)

        metrics_path = os.path.join(args.output_dir, f'per_class_metrics_tflite_{timestamp}.png')
        plot_per_class_metrics(y_test, y_pred_tflite, class_names, metrics_path)

        conf_path = os.path.join(args.output_dir, f'confidence_distribution_tflite_{timestamp}.png')
        plot_confidence_distribution(y_pred_probs_tflite, y_test, conf_path)

        report_path = os.path.join(args.output_dir, f'classification_report_tflite_{timestamp}.txt')
        generate_report(y_test, y_pred_tflite, class_names, report_path)

    if not results:
        print("\nError: No models found to evaluate!")
        return

    print("\n" + "=" * 60)
    print("Evaluation completed successfully!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
