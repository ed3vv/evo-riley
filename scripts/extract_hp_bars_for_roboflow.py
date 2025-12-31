#!/usr/bin/env python3
"""
Extract HP bar regions for Roboflow labeling

Extracts color HP bar images (no preprocessing) with expanded coordinates
for digit bounding box annotation.

Usage:
    python3 scripts/extract_hp_bars_for_roboflow.py
"""

import cv2
import os
from pathlib import Path
import shutil


# Expand HP bar regions by 10px in all directions
HP_BAR_REGIONS = {
    # Enemy towers (top of screen) - expanded by 10px
    'enemy_left_princess': (135, 160, 212, 202),
    'enemy_king': (328, 9, 410, 55),
    'enemy_right_princess': (514, 160, 591, 202),

    # Ally towers (bottom of screen) - expanded by 10px
    'ally_left_princess': (135, 784, 212, 826),
    'ally_king': (328, 958, 410, 1005),
    'ally_right_princess': (514, 784, 591, 826),
}


def extract_hp_bars_for_labeling(
    training_data_dir: str = "training_data",
    output_dir: str = "datasets/tower_hp_bars_roboflow"
):
    """
    Extract color HP bar images for Roboflow digit labeling

    Args:
        training_data_dir: Directory with training screenshots
        output_dir: Output directory for extracted HP bars
    """
    training_path = Path(training_data_dir)
    output_path = Path(output_dir)

    # Clean output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Find all screenshots
    screenshots = list(training_path.glob("**/*.png"))

    if not screenshots:
        print(f"❌ No screenshots found in {training_data_dir}")
        return

    print(f"Found {len(screenshots)} screenshots")
    print(f"Extracting HP bars to: {output_path}\n")

    extract_count = 0

    for i, screenshot_path in enumerate(screenshots, 1):
        print(f"[{i}/{len(screenshots)}] Processing {screenshot_path.name}...")

        screenshot = cv2.imread(str(screenshot_path))
        if screenshot is None:
            print(f"  ⚠️  Could not load, skipping")
            continue

        # Create unique filename by including parent folder (session name)
        # This prevents overwrites when multiple sessions have same screenshot numbers
        session_name = screenshot_path.parent.name
        base_name = screenshot_path.stem

        # Extract each HP bar region
        for tower_name, (x1, y1, x2, y2) in HP_BAR_REGIONS.items():
            # Crop HP bar region (COLOR - no preprocessing!)
            hp_bar_crop = screenshot[y1:y2, x1:x2]

            # Save with descriptive filename including session
            filename = f"{session_name}_{base_name}_{tower_name}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), hp_bar_crop)

            extract_count += 1

    print(f"\n{'='*60}")
    print(f"✅ Extracted {extract_count} HP bar images")
    print(f"📁 Location: {output_path}")
    print(f"{'='*60}")
    print("\nNext steps:")
    print("1. Zip the dataset:")
    print(f"   cd {output_path.parent}")
    print(f"   zip -r {output_path.name}.zip {output_path.name}/")
    print("2. Upload to Roboflow")
    print("3. Draw bounding boxes around each digit (0-9)")
    print("4. Export in YOLOv8 format")
    print("5. Train YOLOv8 object detection model")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    extract_hp_bars_for_labeling()
