# Train Card Hand Classifier - Google Colab

**File ready**: `card_hand_training_v2.zip` (66.8 MB, 1,939 images, 25 card types)

✅ Much better dataset! With this many images, expect 95%+ accuracy.

---

## Step 1: Upload to Google Drive

1. Upload `card_hand_training_v2.zip` to your Google Drive
2. Put it in the root folder or remember the path

---

## Step 2: Open Google Colab

Go to: https://colab.research.google.com/

Create a new notebook

---

## Step 3: Copy-Paste These Cells

### Cell 1: Setup
```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install Ultralytics
!pip install ultralytics
```

### Cell 2: Extract Training Data from Google Drive
```python
# IMPORTANT: Remove old extraction if it exists
!rm -rf /content/sorted_for_training

# Copy from Google Drive (ensure you uploaded card_hand_training_v2.zip there)
!cp /content/drive/MyDrive/card_hand_training_v2.zip /content/

# Verify file integrity before extraction
print("Verifying zip file integrity...")
!unzip -t /content/card_hand_training_v2.zip | tail -5

# Extract the uploaded file
!unzip -o /content/card_hand_training_v2.zip -d /content/sorted_for_training

# Verify extraction
print("\n✅ Extraction complete!")
!find /content/sorted_for_training -name "*.png" | wc -l
print("\nClasses found:")
!ls /content/sorted_for_training/

# Expected: 1,939 images across 25 classes
```

### Cell 3: Train the Model
```python
from ultralytics import YOLO

# Load pretrained YOLOv8 SMALL classification model (not nano!)
model = YOLO('yolov8s-cls.pt')

# Train on your data
results = model.train(
    data='/content/sorted_for_training',
    epochs=100,              # Number of training iterations
    imgsz=224,               # Image size (224 is standard)
    batch=32,                # Batch size (reduce to 16 if GPU memory issues)
    patience=15,             # Early stopping if no improvement
    save=True,               # Save best model
    project='card_hand_classifier',
    name='run1',
    verbose=True,

    # Data augmentation (helps with small datasets)
    hsv_h=0.015,             # Hue variation
    hsv_s=0.7,               # Saturation variation
    hsv_v=0.4,               # Brightness variation
    degrees=5,               # Rotation (±5 degrees)
    translate=0.1,           # Translation
    scale=0.2,               # Scaling
    fliplr=0.5,              # 50% horizontal flip
)

print("\n✅ Training complete!")
```

### Cell 4: Validate
```python
# Validate the trained model
results = model.val()
print("\n✅ Validation complete!")
```

### Cell 5: Test on Sample
```python
# Test on a random card
import random
import glob

# Get random test image
test_images = glob.glob('/content/sorted_for_training/*/*.png')
test_img = random.choice(test_images)

# Predict
results = model.predict(test_img, verbose=False)
probs = results[0].probs

print(f"\nTest image: {test_img}")
print(f"Predicted: {model.names[int(probs.top1)]}")
print(f"Confidence: {float(probs.top1conf):.2%}")
```

### Cell 6: Save to Google Drive
```python
# Copy best model to Google Drive
!cp /content/card_hand_classifier/run1/weights/best.pt /content/drive/MyDrive/card_hand_classifier.pt

# Also copy training plots
!cp /content/card_hand_classifier/run1/results.png /content/drive/MyDrive/training_results.png
!cp /content/card_hand_classifier/run1/confusion_matrix.png /content/drive/MyDrive/confusion_matrix.png

print("✅ Saved to Google Drive:")
print("  - card_hand_classifier.pt (the model)")
print("  - training_results.png (training curves)")
print("  - confusion_matrix.png (accuracy breakdown)")
```

---

## Step 4: Download and Install

1. Download `card_hand_classifier.pt` from Google Drive
2. Save it to: `models/card_hand_classifier.pt`
3. Test it:

```bash
python3 scripts/test_classifier.py --classifier models/card_hand_classifier.pt --sorted sorted_for_training --samples 3
```

---

## Expected Training Time

- **With Colab Free (T4 GPU)**: ~10-15 minutes
- **With Colab Pro (A100 GPU)**: ~5-8 minutes

---

## Current Dataset Stats

```
25 card types, 1,939 total images:
  archers: 181 images ✓
  barbarians: 127 images ✓
  battle_ram: 17 images
  bomb_tower: 15 images
  bomber: 61 images ✓
  cannon: 22 images
  electro_spirit: 3 images ⚠️ (very low, may need more)
  fire_spirit: 11 images
  furnace: 32 images
  giant: 74 images ✓
  goblin_brawler: 26 images
  goblin_cage: 8 images
  goblin_hut: 70 images ✓
  goblins: 183 images ✓
  inferno_tower: 8 images
  knight: 109 images ✓
  mega_minion: 36 images
  mini_pekka: 132 images ✓
  minions: 222 images ✓
  musketeer: 104 images ✓
  skeleton_dragons: 60 images ✓
  skeletons: 129 images ✓
  spear_goblins: 237 images ✓
  tombstone: 27 images
  valkyrie: 45 images
```

---

## To Improve Accuracy

If you have more card images in `sorted/` directory:

```bash
# Copy additional images to sorted_for_training/
cp -r sorted/* sorted_for_training/

# Re-zip
zip -r card_hand_training.zip sorted_for_training/

# Upload and retrain
```

Ideally you want **50+ images per card** for best results!
