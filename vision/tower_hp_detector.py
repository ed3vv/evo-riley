#!/usr/bin/env python3
"""
Tower HP Detection using YOLOv8 digit detection

Detects individual digits (0-9) in tower HP bars and concatenates them
to form complete HP values. Handles edge cases:
- Destroyed towers (no detections for 2 consecutive frames)
- Unactivated king towers (only "levels" class detected)
- Variable-width HP numbers (1-4 digits)
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Optional, List, Tuple


# Tower HP bar regions (expanded by 10px from original)
HP_BAR_REGIONS = {
    'enemy_left_princess': (135, 160, 212, 202),
    'enemy_king': (328, 9, 410, 55),
    'enemy_right_princess': (514, 160, 591, 202),
    'ally_left_princess': (135, 784, 212, 826),
    'ally_king': (328, 958, 410, 1005),
    'ally_right_princess': (514, 784, 591, 826),
}

# Default HP values
DEFAULT_KING_HP = 3528
DEFAULT_PRINCESS_HP = 2184

# ===== MANUAL CONFIGURATION: Ally Tower Max HP =====
# Change these values to set the maximum HP for ally towers
# Any detected HP above these values will be capped (prevents false positive rewards)
ALLY_KING_MAX_HP = 3528
ALLY_PRINCESS_MAX_HP = 2184
# ===================================================


class TowerHPDetector:
    """
    Detect tower HP using YOLOv8 digit detection
    """

    def __init__(self, model_path: str = "models/tower_hp.pt", confidence: float = 0.3):
        """
        Initialize tower HP detector

        Args:
            model_path: Path to trained YOLOv8 model
            confidence: Confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.hp_bar_regions = HP_BAR_REGIONS

        # Track consecutive failed detections for each tower
        # Format: {tower_name: consecutive_failures}
        self.failed_detections = {tower: 0 for tower in HP_BAR_REGIONS.keys()}

        # Track HP history for auto-correction (prevents reward from flickering HP)
        # Format: {tower_name: [hp1, hp2, hp3]}
        self.hp_history = {tower: [] for tower in HP_BAR_REGIONS.keys()}
        self.HP_HISTORY_SIZE = 3  # Keep last 3 readings

        # Previous HP values for fallback
        self.previous_hp = {tower: None for tower in HP_BAR_REGIONS.keys()}

    def detect_tower_hp(self, screenshot: np.ndarray) -> Dict[str, int]:
        """
        Detect HP for all towers in screenshot

        Args:
            screenshot: Full game screenshot (720x1280)

        Returns:
            Dictionary mapping tower names to HP values
        """
        tower_hp = {}

        for tower_name, (x1, y1, x2, y2) in self.hp_bar_regions.items():
            # Crop HP bar region
            hp_bar_crop = screenshot[y1:y2, x1:x2]

            # Detect HP value
            hp_value = self._detect_single_tower_hp(hp_bar_crop, tower_name)

            tower_hp[tower_name] = hp_value

        return tower_hp

    def _detect_single_tower_hp(self, hp_bar_crop: np.ndarray, tower_name: str) -> int:
        """
        Detect HP value for a single tower

        Args:
            hp_bar_crop: Cropped HP bar region
            tower_name: Name of tower (for tracking failed detections)

        Returns:
            HP value (0 if destroyed, default if unactivated, or detected value)
        """
        # Run YOLOv8 detection
        results = self.model(hp_bar_crop, conf=self.confidence, verbose=False)[0]

        # Extract detections
        detections = []
        has_levels = False

        for box in results.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            x_center = float(box.xywh[0][0])

            # Class 10 = "levels" (unactivated king tower)
            if class_id == 10:
                has_levels = True
                continue

            # Classes 0-9 = digits
            if 0 <= class_id <= 9:
                detections.append({
                    'digit': class_id,
                    'confidence': confidence,
                    'x_center': x_center
                })

        # Case 1: "levels" detected → unactivated king tower
        if has_levels:
            self.failed_detections[tower_name] = 0  # Reset failure counter
            hp_value = self._get_default_hp(tower_name)
            self.previous_hp[tower_name] = hp_value
            return hp_value

        # Case 2: Digits detected → parse HP value
        if detections:
            self.failed_detections[tower_name] = 0  # Reset failure counter

            # Sort detections left-to-right by x_center
            detections.sort(key=lambda d: d['x_center'])

            # Concatenate digits to form HP value
            hp_string = ''.join(str(d['digit']) for d in detections)
            hp_value = int(hp_string)

            # Cap ally tower HP to configured max (prevents false positive rewards)
            if 'ally' in tower_name:
                if 'king' in tower_name and hp_value > ALLY_KING_MAX_HP:
                    print(f"[HP] {tower_name}: Capping {hp_value} → {ALLY_KING_MAX_HP} (over max)")
                    hp_value = ALLY_KING_MAX_HP
                elif 'princess' in tower_name and hp_value > ALLY_PRINCESS_MAX_HP:
                    print(f"[HP] {tower_name}: Capping {hp_value} → {ALLY_PRINCESS_MAX_HP} (over max)")
                    hp_value = ALLY_PRINCESS_MAX_HP

            # Auto-correction BEFORE adding to history: If HP fluctuates (e.g., 1000 → 100 → 1000), correct immediately
            if len(self.hp_history[tower_name]) >= 2:
                history = self.hp_history[tower_name]
                # Check if current reading breaks a stable pattern
                # Pattern: history[-2] (stable) → history[-1] (outlier) → hp_value (back to stable)
                if len(history) >= 2 and history[-2] == hp_value and history[-1] != hp_value:
                    # Also check that it's not a legitimate decrease (towers can only lose HP)
                    # If history[-2] == hp_value, this would be HP staying same or returning to previous
                    corrected_hp = history[-2]  # Use the stable value
                    if hp_value != corrected_hp:
                        print(f"[HP] {tower_name}: Auto-correcting {hp_value} → {corrected_hp} (flickering detected: last 2 = [{history[-2]}, {history[-1]}], current = {hp_value})")
                        hp_value = corrected_hp

            # Sanity check: Only reject HP increases that go above maximum possible HP
            # This catches misreads like "21847" when max is 2184, but allows corrections like 75 → 1890
            max_hp = ALLY_KING_MAX_HP if 'king' in tower_name else ALLY_PRINCESS_MAX_HP

            if self.previous_hp[tower_name] is not None and self.previous_hp[tower_name] > 0:
                # Only reject if new HP exceeds maximum AND is higher than previous
                # (allows corrections from misreads, but rejects impossible values)
                if hp_value > max_hp and hp_value > self.previous_hp[tower_name]:
                    print(f"[HP] {tower_name}: Rejecting HP {hp_value} (exceeds max {max_hp}), keeping previous value {self.previous_hp[tower_name]}")
                    hp_value = self.previous_hp[tower_name]

            # Special case: Allow resurrection from 0 (corrects false destructions)
            # If tower was at 0 and now has HP, it was likely a false destruction detection
            # We allow it to correct the error, but log it
            elif self.previous_hp[tower_name] == 0 and hp_value > 0:
                print(f"[HP] {tower_name}: Correcting false destruction, 0 → {hp_value} (detection recovered)")
                # hp_value is already set correctly, just allow it through

            # Add to HP history for tracking
            self.hp_history[tower_name].append(hp_value)
            if len(self.hp_history[tower_name]) > self.HP_HISTORY_SIZE:
                self.hp_history[tower_name].pop(0)

            # Print HP change if it changed (BEFORE updating previous_hp)
            if self.previous_hp[tower_name] is not None and self.previous_hp[tower_name] != hp_value:
                change = hp_value - self.previous_hp[tower_name]
                print(f"[HP] {tower_name}: {self.previous_hp[tower_name]} → {hp_value} ({change:+d})")

            # Update previous HP for next detection
            self.previous_hp[tower_name] = hp_value
            return hp_value

        # Case 3: No detections → increment failure counter
        self.failed_detections[tower_name] += 1

        # Only mark as destroyed after 2 consecutive failures
        if self.failed_detections[tower_name] >= 2:
            # Only print if this is the first time marking as destroyed
            if self.previous_hp[tower_name] != 0:
                print(f"[HP] {tower_name}: {self.previous_hp[tower_name]} → 0 (DESTROYED - {self.failed_detections[tower_name]} consecutive detection failures)")
            self.previous_hp[tower_name] = 0
            return 0

        # Otherwise, return previous HP (or default if no previous)
        if self.previous_hp[tower_name] is not None:
            return self.previous_hp[tower_name]
        else:
            # First detection failure - use default HP
            default_hp = self._get_default_hp(tower_name)
            return default_hp

    def _get_default_hp(self, tower_name: str) -> int:
        """
        Get default HP for a tower

        Args:
            tower_name: Name of tower

        Returns:
            Default HP value
        """
        if 'king' in tower_name:
            return DEFAULT_KING_HP
        else:
            return DEFAULT_PRINCESS_HP

    def reset_tracking(self):
        """Reset failed detection counters and HP history (call at start of new battle)"""
        self.failed_detections = {tower: 0 for tower in HP_BAR_REGIONS.keys()}
        self.previous_hp = {tower: None for tower in HP_BAR_REGIONS.keys()}
        self.hp_history = {tower: [] for tower in HP_BAR_REGIONS.keys()}
