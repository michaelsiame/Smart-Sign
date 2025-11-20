#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gesture Management Utility
This script helps manage gesture labels and provides utilities for the sign language interpreter.
"""

import argparse
import csv
import os
import sys


def read_labels(label_file):
    """Read gesture labels from CSV file."""
    if not os.path.exists(label_file):
        print(f"Error: Label file not found at {label_file}")
        return []

    with open(label_file, encoding='utf-8-sig') as f:
        reader = csv.reader(f)
        labels = [row[0] for row in reader]

    return labels


def write_labels(label_file, labels):
    """Write gesture labels to CSV file."""
    with open(label_file, 'w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        for label in labels:
            writer.writerow([label])

    print(f"Labels saved to {label_file}")


def list_gestures(label_file):
    """List all gestures in the label file."""
    labels = read_labels(label_file)

    if not labels:
        print("No gestures found.")
        return

    print(f"\nGestures in {label_file}:")
    print("=" * 50)
    for idx, label in enumerate(labels):
        print(f"{idx}: {label}")
    print("=" * 50)
    print(f"Total: {len(labels)} gestures")


def add_gesture(label_file, gesture_name):
    """Add a new gesture to the label file."""
    labels = read_labels(label_file)

    if gesture_name in labels:
        print(f"Error: Gesture '{gesture_name}' already exists at index {labels.index(gesture_name)}")
        return False

    labels.append(gesture_name)
    write_labels(label_file, labels)
    print(f"Successfully added gesture '{gesture_name}' at index {len(labels) - 1}")

    return True


def remove_gesture(label_file, gesture_identifier):
    """Remove a gesture from the label file by name or index."""
    labels = read_labels(label_file)

    if not labels:
        print("No gestures to remove.")
        return False

    # Try to parse as index
    try:
        index = int(gesture_identifier)
        if 0 <= index < len(labels):
            removed = labels.pop(index)
            write_labels(label_file, labels)
            print(f"Successfully removed gesture '{removed}' at index {index}")
            print("Warning: You may need to update your training data and retrain the model!")
            return True
        else:
            print(f"Error: Index {index} out of range (0-{len(labels)-1})")
            return False
    except ValueError:
        # Treat as gesture name
        if gesture_identifier in labels:
            index = labels.index(gesture_identifier)
            labels.remove(gesture_identifier)
            write_labels(label_file, labels)
            print(f"Successfully removed gesture '{gesture_identifier}' at index {index}")
            print("Warning: You may need to update your training data and retrain the model!")
            return True
        else:
            print(f"Error: Gesture '{gesture_identifier}' not found")
            return False


def rename_gesture(label_file, old_name, new_name):
    """Rename a gesture in the label file."""
    labels = read_labels(label_file)

    if old_name not in labels:
        print(f"Error: Gesture '{old_name}' not found")
        return False

    if new_name in labels:
        print(f"Error: Gesture '{new_name}' already exists")
        return False

    index = labels.index(old_name)
    labels[index] = new_name
    write_labels(label_file, labels)
    print(f"Successfully renamed '{old_name}' to '{new_name}' at index {index}")

    return True


def count_samples(csv_file, num_classes):
    """Count the number of samples per class in the dataset."""
    if not os.path.exists(csv_file):
        print(f"Dataset file not found at {csv_file}")
        return

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        class_counts = [0] * num_classes

        for row in reader:
            try:
                label = int(row[0])
                if 0 <= label < num_classes:
                    class_counts[label] += 1
            except (ValueError, IndexError):
                continue

    return class_counts


def show_dataset_stats(label_file, data_file):
    """Show statistics about the dataset."""
    labels = read_labels(label_file)
    counts = count_samples(data_file, len(labels))

    if counts is None:
        return

    print(f"\nDataset Statistics for {data_file}:")
    print("=" * 60)
    print(f"{'Index':<8} {'Gesture':<25} {'Samples':<10}")
    print("-" * 60)

    total = 0
    for idx, (label, count) in enumerate(zip(labels, counts)):
        print(f"{idx:<8} {label:<25} {count:<10}")
        total += count

    print("-" * 60)
    print(f"{'Total:':<33} {total:<10}")
    print("=" * 60)

    # Check for imbalanced classes
    if total > 0:
        print("\nClass Balance:")
        for idx, (label, count) in enumerate(zip(labels, counts)):
            percentage = (count / total) * 100
            print(f"  {label}: {percentage:.1f}%")


def interactive_mode(label_file):
    """Interactive mode for managing gestures."""
    while True:
        print("\n" + "=" * 60)
        print("Gesture Management - Interactive Mode")
        print("=" * 60)
        print("1. List all gestures")
        print("2. Add a new gesture")
        print("3. Remove a gesture")
        print("4. Rename a gesture")
        print("5. Exit")
        print("=" * 60)

        choice = input("Enter your choice (1-5): ").strip()

        if choice == '1':
            list_gestures(label_file)
        elif choice == '2':
            name = input("Enter new gesture name: ").strip()
            if name:
                add_gesture(label_file, name)
            else:
                print("Error: Gesture name cannot be empty")
        elif choice == '3':
            identifier = input("Enter gesture name or index to remove: ").strip()
            if identifier:
                remove_gesture(label_file, identifier)
            else:
                print("Error: Please provide a gesture name or index")
        elif choice == '4':
            old_name = input("Enter current gesture name: ").strip()
            new_name = input("Enter new gesture name: ").strip()
            if old_name and new_name:
                rename_gesture(label_file, old_name, new_name)
            else:
                print("Error: Both names must be provided")
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")


def main():
    parser = argparse.ArgumentParser(description='Manage gesture labels for sign language interpreter')

    parser.add_argument('--label_file', type=str,
                        default='model/keypoint_classifier/keypoint_classifier_label.csv',
                        help='Path to the label CSV file')

    parser.add_argument('--data_file', type=str,
                        default='model/keypoint_classifier/keypoint.csv',
                        help='Path to the training data CSV file')

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # List command
    subparsers.add_parser('list', help='List all gestures')

    # Add command
    add_parser = subparsers.add_parser('add', help='Add a new gesture')
    add_parser.add_argument('name', type=str, help='Name of the gesture to add')

    # Remove command
    remove_parser = subparsers.add_parser('remove', help='Remove a gesture')
    remove_parser.add_argument('identifier', type=str, help='Gesture name or index to remove')

    # Rename command
    rename_parser = subparsers.add_parser('rename', help='Rename a gesture')
    rename_parser.add_argument('old_name', type=str, help='Current gesture name')
    rename_parser.add_argument('new_name', type=str, help='New gesture name')

    # Stats command
    subparsers.add_parser('stats', help='Show dataset statistics')

    # Interactive command
    subparsers.add_parser('interactive', help='Enter interactive mode')

    args = parser.parse_args()

    # If no command provided, show help
    if args.command is None:
        parser.print_help()
        return

    # Execute command
    if args.command == 'list':
        list_gestures(args.label_file)

    elif args.command == 'add':
        add_gesture(args.label_file, args.name)

    elif args.command == 'remove':
        remove_gesture(args.label_file, args.identifier)

    elif args.command == 'rename':
        rename_gesture(args.label_file, args.old_name, args.new_name)

    elif args.command == 'stats':
        show_dataset_stats(args.label_file, args.data_file)

    elif args.command == 'interactive':
        interactive_mode(args.label_file)


if __name__ == '__main__':
    main()
