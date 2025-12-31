#!/usr/bin/env python3
"""
Extract tower HP bar regions from existing training screenshots

This script:
1. Uses existing training screenshots from training_data/
2. Extracts full HP bar regions (entire number, not individual digits)
3. Saves HP bar images for manual labeling (e.g., "2456", "1823", "0")

Usage:
    # Step 1: Adjust HP bar positions (edit HP_BAR_REGIONS below)
    python3 scripts/capture_tower_digits.py --test <screenshot.png>

    # Step 2: Extract HP bars from all training screenshots
    python3 scripts/capture_tower_digits.py --extract training_data/

Manual calibration:
    1. Use any battle screenshot
    2. Run --test to see annotated boxes
    3. Edit HP_BAR_REGIONS below with correct coordinates
    4. Run --extract to process all training screenshots
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime


# ============================================================================
# MANUALLY EDIT THESE COORDINATES TO MATCH YOUR SCREENSHOTS
# ============================================================================
# Each tower has ONE HP bar region: (x1, y1, x2, y2) for entire HP number
# Make the box wide enough to capture all 4 digits (0-9999 HP)
# Adjust these after taking a test screenshot
# ============================================================================

HP_BAR_REGIONS = {
    # Enemy towers (top of screen)
    'enemy_left_princess': (145, 170, 202, 192),   # Full HP bar region
    'enemy_king': (338, 19, 400, 45),
    'enemy_right_princess': (524, 170, 581, 192),

    # Ally towers (bottom of screen)
    'ally_left_princess': (145, 794, 202, 816),
    'ally_king': (338, 968, 400, 995),
    'ally_right_princess': (524, 794, 581, 816),
}


class TowerHPExtractor:
    """
    Extract tower HP bar regions from screenshots
    """

    def __init__(self):
        """Initialize HP bar extraction system"""
        self.hp_bar_regions = HP_BAR_REGIONS

        # Output directory
        self.output_dir = Path("datasets/tower_hp_bars")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Extraction tracking
        self.extract_count = 0

    def test_hp_bar_positions(self, screenshot_path: str):
        """
        Test HP bar positions on a screenshot file

        Args:
            screenshot_path: Path to test screenshot
        """
        print("\n" + "="*70)
        print("Testing Tower HP Bar Positions")
        print("="*70)
        print(f"\nLoading screenshot: {screenshot_path}")

        # Load screenshot
        screenshot = cv2.imread(screenshot_path)
        if screenshot is None:
            print(f"❌ Could not load screenshot: {screenshot_path}")
            return

        print(f"✅ Loaded screenshot: {screenshot.shape[1]}x{screenshot.shape[0]}")

        # Draw boxes on screenshot
        display = screenshot.copy()
        self._draw_hp_bar_boxes(display)

        # Save annotated version
        annotated_path = self.output_dir / "test_screenshot_annotated.png"
        cv2.imwrite(str(annotated_path), display)
        print(f"✅ Saved annotated screenshot: {annotated_path}")

        # Show preview at 2x zoom
        h, w = display.shape[:2]
        zoomed = cv2.resize(display, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("HP Bar Positions - 2x Zoom (press any key to close)", zoomed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Extract and show test crops
        self._show_test_crops(screenshot)

        print("\n" + "="*70)
        print("Next Steps:")
        print("="*70)
        print("1. Check the annotated screenshot and test crops")
        print("2. If positions are wrong, edit HP_BAR_REGIONS in this file")
        print("3. Re-run: python3 scripts/capture_tower_digits.py --test <screenshot>")
        print("4. When correct, extract: python3 scripts/capture_tower_digits.py --extract <folder>")
        print("="*70 + "\n")

    def _draw_hp_bar_boxes(self, image):
        """Draw HP bar position boxes on image"""
        colors = {
            'enemy_left_princess': (0, 255, 255),
            'enemy_king': (0, 255, 0),
            'enemy_right_princess': (255, 255, 0),
            'ally_left_princess': (255, 0, 255),
            'ally_king': (255, 0, 0),
            'ally_right_princess': (0, 0, 255),
        }

        for tower_name, (x1, y1, x2, y2) in self.hp_bar_regions.items():
            color = colors.get(tower_name, (255, 255, 255))

            # Draw box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # Label
            label = tower_name.replace('_', ' ').title()
            cv2.putText(image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    def _show_test_crops(self, screenshot):
        """Extract and save test HP bar crops"""
        print("\nExtracting test crops...")

        crops_dir = self.output_dir / "test_crops"
        crops_dir.mkdir(exist_ok=True)

        for tower_name, (x1, y1, x2, y2) in self.hp_bar_regions.items():
            # Crop HP bar
            hp_bar_crop = screenshot[y1:y2, x1:x2]

            # Preprocess
            gray = cv2.cvtColor(hp_bar_crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            processed = cv2.bitwise_not(binary)

            # Scale up
            scale = 6
            scaled = cv2.resize(processed, (processed.shape[1] * scale, processed.shape[0] * scale),
                               interpolation=cv2.INTER_NEAREST)

            # Save
            filename = f"{tower_name}.png"
            filepath = crops_dir / filename
            cv2.imwrite(str(filepath), scaled)

        print(f"✅ Test crops saved to: {crops_dir}")
        print(f"   Check these images to verify HP bar positions are correct")

    def extract_from_folder(self, folder_path: str):
        """
        Extract HP bars from all PNG screenshots in a folder

        Args:
            folder_path: Path to folder containing screenshots
        """
        folder = Path(folder_path)
        if not folder.exists():
            print(f"❌ Folder not found: {folder_path}")
            return

        print("\n" + "="*70)
        print("Batch HP Bar Extraction")
        print("="*70)
        print(f"\nSearching for screenshots in: {folder}")

        # Find all PNG files
        screenshots = list(folder.glob("**/*.png"))
        print(f"Found {len(screenshots)} screenshots\n")

        if not screenshots:
            print("❌ No PNG files found")
            return

        # Process each screenshot
        for i, screenshot_path in enumerate(screenshots, 1):
            print(f"[{i}/{len(screenshots)}] Processing {screenshot_path.name}...")
            screenshot = cv2.imread(str(screenshot_path))

            if screenshot is None:
                print(f"  ⚠️  Could not load, skipping")
                continue

            self._extract_hp_bars(screenshot, screenshot_path.stem)

        print("\n" + "="*70)
        print(f"✅ Extracted {self.extract_count} HP bar images")
        print(f"📁 Location: {self.output_dir}")
        print("="*70)
        print("\nNext steps:")
        print("1. Manually label images with HP values (e.g., rename to '2456.png', '1823.png', '0.png')")
        print("2. Organize by tower if desired")
        print("3. Train YOLOv8 classifier on Colab (classification or regression)")
        print("="*70 + "\n")

    def _extract_hp_bars(self, screenshot, prefix):
        """Extract all HP bar crops from a screenshot"""
        for tower_name, (x1, y1, x2, y2) in self.hp_bar_regions.items():
            tower_dir = self.output_dir / tower_name
            tower_dir.mkdir(parents=True, exist_ok=True)

            # Crop HP bar region
            hp_bar_crop = screenshot[y1:y2, x1:x2]

            # Preprocess
            gray = cv2.cvtColor(hp_bar_crop, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            processed = cv2.bitwise_not(binary)

            # Scale up for better visibility
            scale = 4
            scaled = cv2.resize(processed, (processed.shape[1] * scale, processed.shape[0] * scale),
                               interpolation=cv2.INTER_NEAREST)

            # Save
            filename = f"{prefix}_{tower_name}.png"
            filepath = tower_dir / filename
            cv2.imwrite(str(filepath), scaled)

            self.extract_count += 1


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Extract tower HP bars from screenshots")
    parser.add_argument('--test', metavar='SCREENSHOT', help='Test HP bar positions on a screenshot file')
    parser.add_argument('--extract', metavar='FOLDER', help='Extract HP bars from all screenshots in folder')
    args = parser.parse_args()

    extractor = TowerHPExtractor()

    if args.test:
        # Test mode: verify HP bar positions
        extractor.test_hp_bar_positions(args.test)

    elif args.extract:
        # Extract mode: batch process screenshots
        extractor.extract_from_folder(args.extract)

    else:
        # Show usage
        print("="*70)
        print("Tower HP Bar Extractor")
        print("="*70)
        print("\nUsage:")
        print("  python3 scripts/capture_tower_digits.py --test <screenshot.png>")
        print("      Test HP bar positions and save annotated version")
        print()
        print("  python3 scripts/capture_tower_digits.py --extract <folder>")
        print("      Extract HP bars from all screenshots in folder")
        print()
        print("Workflow:")
        print("  1. Find any battle screenshot")
        print("  2. Run --test to verify HP bar positions")
        print("  3. Edit HP_BAR_REGIONS in this file if needed")
        print("  4. Run --extract on training_data/ folder")
        print("  5. Manually label images with HP values")
        print()
        print("Example:")
        print("  python3 scripts/capture_tower_digits.py --extract training_data/")
        print("="*70)


if __name__ == "__main__":
    main()
