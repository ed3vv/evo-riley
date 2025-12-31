#!/usr/bin/env python3
"""
Test the trained card classifier on sample images from each class.
"""

from ultralytics import YOLO
from pathlib import Path
import random


def test_classifier(classifier_path='models/card_classifier.pt', sorted_dir='sorted', samples_per_class=3):
    """
    Test classifier on random samples from each class.

    Args:
        classifier_path: Path to the trained classifier
        sorted_dir: Path to sorted dataset
        samples_per_class: Number of samples to test per class
    """

    print("=" * 80)
    print("CARD CLASSIFIER TEST")
    print("=" * 80)

    # Load classifier
    print(f"\nLoading classifier: {classifier_path}")
    model = YOLO(classifier_path)
    print("✅ Classifier loaded\n")

    # Get class names from model
    if hasattr(model, 'names'):
        model_classes = model.names
        print(f"Model trained on {len(model_classes)} classes:")
        for idx, name in model_classes.items():
            print(f"  {idx}: {name}")
        print()

    # Test on samples from sorted directory
    sorted_path = Path(sorted_dir)

    if not sorted_path.exists():
        print(f"Error: {sorted_dir} not found!")
        return

    print("=" * 80)
    print("TESTING ON RANDOM SAMPLES")
    print("=" * 80)

    total_tested = 0
    total_correct = 0

    for class_dir in sorted(sorted_path.iterdir()):
        if not class_dir.is_dir():
            continue

        true_class = class_dir.name

        # Get all images
        images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))

        if not images:
            continue

        # Sample random images
        num_samples = min(samples_per_class, len(images))
        samples = random.sample(images, num_samples)

        print(f"\n{true_class}:")
        print("-" * 60)

        class_correct = 0

        for img_path in samples:
            # Predict
            results = model.predict(source=img_path, verbose=False, imgsz=64)

            # Get top prediction
            probs = results[0].probs
            top1_idx = probs.top1
            top1_class = model.names[top1_idx]
            top1_conf = probs.top1conf.item()

            # Check if correct
            is_correct = (top1_class == true_class)
            symbol = "✅" if is_correct else "❌"

            print(f"  {symbol} {img_path.name[:30]:<30} → {top1_class:20} ({top1_conf:.1%})")

            if is_correct:
                class_correct += 1

            total_tested += 1

        total_correct += class_correct
        accuracy = (class_correct / num_samples) * 100
        print(f"  Class accuracy: {class_correct}/{num_samples} ({accuracy:.1f}%)")

    # Overall stats
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    overall_accuracy = (total_correct / total_tested) * 100
    print(f"Total samples tested: {total_tested}")
    print(f"Correct predictions:  {total_correct}")
    print(f"Accuracy:            {overall_accuracy:.1f}%")
    print("=" * 80)

    if overall_accuracy >= 90:
        print("\n🎉 Excellent! Classifier is working well!")
    elif overall_accuracy >= 75:
        print("\n✅ Good! Classifier is performing reasonably.")
    else:
        print("\n⚠️  Classifier needs more training or data.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test card classifier")
    parser.add_argument('--classifier', default='models/card_classifier.pt', help='Path to classifier')
    parser.add_argument('--sorted', default='sorted', help='Path to sorted dataset')
    parser.add_argument('--samples', type=int, default=3, help='Samples per class to test')

    args = parser.parse_args()

    test_classifier(args.classifier, args.sorted, args.samples)
