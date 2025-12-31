# Scripts Directory

This directory contains utility scripts for setting up, testing, and maintaining the Clash Royale RL bot.

---

## Setup & Calibration (Run Once)

### `calibrate_arena.py`
Calibrate the arena boundaries for your BlueStacks instance.
- **When to use**: First time setup, or if you change screen resolution
- **Output**: `config/instance_X.json`
- **Usage**:
  ```bash
  python3 scripts/calibrate_arena.py
  ```

### `calibrate_tower_hp.py`
Calibrate tower HP bar detection regions.
- **When to use**: First time setup, or if UI changes
- **Output**: Tower HP region coordinates
- **Usage**:
  ```bash
  python3 scripts/calibrate_tower_hp.py
  ```

---

## Data Collection & Preparation

### `capture_templates_from_hand.py`
Capture card templates from your hand during gameplay.
- **When to use**: Collecting training data for card hand classifier
- **Output**: Images saved to `detection/card_templates/`
- **Usage**:
  ```bash
  python3 scripts/capture_templates_from_hand.py
  ```

### `add_new_images_to_dataset.py`
Add new card images to existing sorted dataset.
- **When to use**: Adding more training data to improve classifier
- **Usage**:
  ```bash
  python3 scripts/add_new_images_to_dataset.py
  ```

### `convert_templates_to_dataset.py`
Convert card templates into YOLOv8 training dataset format.
- **When to use**: Preparing data for training on Colab
- **Output**: `sorted_for_training/` directory
- **Usage**:
  ```bash
  python3 scripts/convert_templates_to_dataset.py
  ```

### `create_templates_from_screenshots.py`
Extract card templates from battle screenshots.
- **When to use**: Alternative method for collecting card images
- **Usage**:
  ```bash
  python3 scripts/create_templates_from_screenshots.py
  ```

### `sort_crops.py`
Manually sort cropped card images into folders.
- **When to use**: Organizing unlabeled card crops
- **Usage**:
  ```bash
  python3 scripts/sort_crops.py
  ```

### `count_sorted_images.py`
Count images per class in sorted dataset.
- **When to use**: Checking dataset balance before training
- **Output**: Prints class distribution
- **Usage**:
  ```bash
  python3 scripts/count_sorted_images.py
  ```

### `split_dataset.py`
Split dataset into train/val/test sets.
- **When to use**: Preparing dataset for training (if needed)
- **Usage**:
  ```bash
  python3 scripts/split_dataset.py
  ```

---

## Testing & Validation

### `test_card_detection.py`
Test YOLO card detection model on live gameplay.
- **When to use**: Verify card detector is working correctly
- **Usage**:
  ```bash
  python3 scripts/test_card_detection.py
  ```

### `test_classifier.py`
Test card hand classifier accuracy.
- **When to use**: Verify classifier model accuracy on test images
- **Output**: Accuracy metrics and predictions
- **Usage**:
  ```bash
  python3 scripts/test_classifier.py --classifier models/card_hand_classifier.pt --sorted detection/card_templates --samples 10
  ```

### `test_result_detection.py`
Test battle result detection (victory/defeat).
- **When to use**: Verify result screen detection works
- **Usage**:
  ```bash
  python3 scripts/test_result_detection.py
  ```

### `test_tower_hp.py`
Test tower HP detection.
- **When to use**: Verify tower HP tracking works correctly
- **Usage**:
  ```bash
  python3 scripts/test_tower_hp.py
  ```

### `test_state_manager.py`
Test game state management system.
- **When to use**: Verify state tracking for RL training
- **Usage**:
  ```bash
  python3 scripts/test_state_manager.py
  ```

---

## Model Evaluation

### `evaluate_card_classifier.py`
Comprehensive evaluation of card hand classifier.
- **When to use**: After training new classifier model
- **Output**: Confusion matrix, accuracy per class
- **Usage**:
  ```bash
  python3 scripts/evaluate_card_classifier.py
  ```

---

## Template Creation

### `create_result_templates.py`
Create templates for battle result screen detection.
- **When to use**: One-time setup for result detection
- **Output**: Templates saved to `detection/result_templates/`
- **Usage**:
  ```bash
  python3 scripts/create_result_templates.py
  ```

---

## Typical Workflows

### First Time Setup
1. `calibrate_arena.py` - Set up arena boundaries
2. `calibrate_tower_hp.py` - Set up tower HP detection
3. `create_result_templates.py` - Create result screen templates

### Collecting Training Data for Card Classifier
1. `capture_templates_from_hand.py` - Capture cards during gameplay
2. `count_sorted_images.py` - Check dataset balance
3. `convert_templates_to_dataset.py` - Prepare for training
4. Train on Colab using `TRAIN_HAND_CLASSIFIER_COLAB.md`
5. `test_classifier.py` - Verify trained model accuracy

### Testing Bot Components
1. `test_card_detection.py` - Verify YOLO detection
2. `test_classifier.py` - Verify card hand classification
3. `test_tower_hp.py` - Verify HP tracking
4. `test_result_detection.py` - Verify result screen detection

---

## Archived Scripts

Older/redundant scripts have been moved to `scripts/archive/`:
- Debug scripts (one-time use)
- Duplicate functionality scripts
- Roboflow integration scripts (if not using Roboflow)
- Obsolete extraction scripts

These can be deleted if you're sure you don't need them.
