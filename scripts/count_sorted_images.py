#!/usr/bin/env python3
"""
Count images in each class of the sorted dataset.
Use this to verify dataset before/after Colab training.
"""

from pathlib import Path
import sys


def count_sorted_images(sorted_dir='sorted'):
    """Count images in each class directory."""
    sorted_path = Path(sorted_dir)

    if not sorted_path.exists():
        print(f"Error: Directory '{sorted_dir}' not found!")
        return

    print("=" * 60)
    print("SORTED DATASET IMAGE COUNT")
    print("=" * 60)
    print(f"{'Card Type':<30} {'Count':>10}")
    print("-" * 60)

    class_counts = {}

    for class_dir in sorted(sorted_path.iterdir()):
        if class_dir.is_dir():
            # Count PNG and JPG images
            png_count = len(list(class_dir.glob('*.png')))
            jpg_count = len(list(class_dir.glob('*.jpg')))
            total_count = png_count + jpg_count

            class_counts[class_dir.name] = total_count
            print(f"{class_dir.name:<30} {total_count:>10}")

    print("=" * 60)
    print(f"{'Total Classes:':<30} {len(class_counts):>10}")
    print(f"{'Total Images:':<30} {sum(class_counts.values()):>10}")
    print("=" * 60)

    return class_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Count images in sorted dataset"
    )
    parser.add_argument(
        '--sorted',
        default='sorted',
        help='Path to sorted directory (default: sorted)'
    )

    args = parser.parse_args()

    count_sorted_images(args.sorted)
