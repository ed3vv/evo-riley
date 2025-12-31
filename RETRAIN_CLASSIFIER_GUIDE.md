# Retrain Card Classifier Guide

Complete workflow to retrain your card classifier with new arena cards from Roboflow.

## Prerequisites

- New detection model trained: `models/best.pt` ✅
- Roboflow dataset with new arena cards
- Roboflow API key

## Step-by-Step Workflow

### Step 1: Extract Card Crops from Roboflow Dataset

Use your NEW detection model to crop all cards from the Roboflow dataset:

```bash
python3 scripts/extract_cards_from_roboflow.py \
    --api-key YOUR_ROBOFLOW_API_KEY \
    --workspace YOUR_WORKSPACE \
    --project YOUR_PROJECT \
    --version 1 \
    --model models/best.pt \
    --output crops
```

**What this does:**
- Downloads your Roboflow dataset (includes new arena cards)
- Uses your NEW `best.pt` to detect all cards
- Crops each detection and saves to `crops/` folder
- Ready for manual sorting

**Example output:**
```
1. Downloading Roboflow dataset: workspace/project/v1
   Downloaded to: /path/to/dataset

2. Loading detection model: models/best.pt
   Found 500 images in dataset

3. Extracting cards from images...
   Processed 500/500 images, extracted 2500 cards...

✅ Extraction complete!
   Total cards extracted: 2500
   Saved to: crops/

📋 Next step: Sort these cards manually
   Run: python3 scripts/sort_crops.py --crops crops
```

---

### Step 2: Manually Sort Card Crops

Now you need to **manually classify** each crop by typing the card name:

```bash
python3 scripts/sort_crops.py --crops crops --sorted sorted
```

**What this does:**
- Shows each crop one-by-one in a window
- You type the card name (e.g., "knight", "archers", "giant")
- Script organizes crops into folders: `sorted/knight/`, `sorted/archers/`, etc.

**Interactive commands:**
- Type card name (e.g., `knight`) → Sorts to `sorted/knight/`
- `skip` or `s` → Skip this image (saves to `skipped/`)
- `delete` or `d` → Delete this image
- `undo` or `u` → Undo last action
- `quit` or `q` → Quit sorting

**Example session:**
```
Found 2500 crops to sort

=== COMMANDS ===
  Type card name (e.g., 'knight', 'musketeer', 'giant')
  'skip' or 's' = skip this image
  'delete' or 'd' = delete this image
  'undo' or 'u' = undo last action
  'quit' or 'q' = quit sorting
==================

[1/2500] Enter card name: knight
Sorted to 'knight': crop_00000.png (1 total)

[2/2500] Enter card name: archers
Sorted to 'archers': crop_00001.png (1 total)

[3/2500] Enter card name: giant
Sorted to 'giant': crop_00002.png (1 total)

... (continue sorting all 2500 cards)

=== SORTING SUMMARY ===
knight: 150
archers: 140
giant: 130
fireball: 120
... (all your cards)
skipped: 50
deleted: 10

Total sorted: 2450
Remaining: 50
```

**Tips:**
- **New arena cards**: When you see a new card, type its name carefully
  - Script will ask: "New card class: 'NEW_CARD' - Is this correct? (y/n)"
  - Confirm with `y` to create the new folder
- **Pause anytime**: Press `q` to quit, progress is saved
- **Resume later**: Just run the same command again, it will continue with remaining crops
- **Undo mistakes**: Type `u` to undo the last sort

---

### Step 3: Verify Sorted Dataset

Check that your sorted dataset looks good:

```bash
# Count images per class
find sorted -type f -name "*.png" | wc -l

# List all card classes
ls sorted/

# Count per class
for dir in sorted/*/; do echo "$(basename $dir): $(ls $dir/*.png 2>/dev/null | wc -l)"; done | sort
```

**Example output:**
```
archers: 140
arrows: 80
baby_dragon: 60
balloon: 55
barbarian_barrel: 45
barbarians: 90
bats: 70
... (all your cards)
zap: 100

Total: 2450 images across 30 classes
```

**What to check:**
- Each card has at least 20-30 images (more is better)
- New arena cards are included
- No weird/broken class names

---

### Step 4: Train Card Classifier

Now train the classifier on your sorted dataset:

```bash
yolo classify train \
    data=sorted \
    model=yolov8n-cls.pt \
    epochs=50 \
    imgsz=64 \
    batch=32 \
    patience=10 \
    name=card_classifier_v2
```

**Training parameters:**
- `data=sorted` - Your manually sorted dataset
- `model=yolov8n-cls.pt` - Start from YOLOv8 nano classification
- `epochs=50` - Train for 50 epochs (increase if needed)
- `imgsz=64` - 64x64 images (cards are small)
- `batch=32` - Batch size
- `patience=10` - Early stopping if no improvement
- `name=card_classifier_v2` - Save results to `runs/classify/card_classifier_v2/`

**Example output:**
```
Epoch    GPU_mem   train_loss   val_loss   top1_acc   top5_acc
  1/50      0.5G       2.450      2.123      0.350      0.750
  2/50      0.5G       1.890      1.687      0.520      0.850
  3/50      0.5G       1.456      1.312      0.680      0.920
 ...
 50/50      0.5G       0.234      0.189      0.945      0.990

✅ Training complete!
Best model saved to: runs/classify/card_classifier_v2/weights/best.pt
Results: runs/classify/card_classifier_v2/
```

---

### Step 5: Replace Old Classifier

Replace your old classifier with the new one:

```bash
# Backup old classifier
cp models/card_classifier.pt models/card_classifier_old.pt

# Copy new classifier
cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt

echo "✅ Classifier updated!"
```

---

### Step 6: Test New Classifier

Test that the new classifier works:

```bash
# Test on a few sorted images
yolo classify predict \
    model=models/card_classifier.pt \
    source=sorted/knight/ \
    imgsz=64 \
    save=False
```

Or use your existing evaluation script:

```bash
python3 scripts/evaluate_card_classifier.py
```

---

## Summary

**Complete workflow:**

1. **Extract crops from Roboflow** (using NEW detection model)
   ```bash
   python3 scripts/extract_cards_from_roboflow.py \
       --api-key YOUR_KEY \
       --workspace YOUR_WORKSPACE \
       --project YOUR_PROJECT \
       --version 1
   ```

2. **Sort crops manually** (this is the step you were asking about!)
   ```bash
   python3 scripts/sort_crops.py --crops crops --sorted sorted
   ```

3. **Train classifier** on sorted dataset
   ```bash
   yolo classify train data=sorted model=yolov8n-cls.pt epochs=50 imgsz=64 batch=32 name=card_classifier_v2
   ```

4. **Replace old model**
   ```bash
   cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt
   ```

---

## Alternative: Manual Download from Roboflow

If you prefer to manually download from Roboflow instead of using the script:

1. Go to Roboflow project → Download → YOLOv8 format
2. Extract the zip file to `datasets/roboflow_dataset/`
3. Use the existing script to crop:
   ```bash
   # Modify roboflow_crop.py to point to your dataset
   python3 scripts/roboflow_crop.py
   ```
4. Continue with Step 2 (sorting)

---

## Troubleshooting

### "No crops extracted"
- Check that `models/best.pt` exists and works
- Lower confidence threshold: `--conf 0.1`
- Verify dataset downloaded correctly

### "Too many crops to sort"
- You can quit anytime with `q` and resume later
- Sort in batches: move some crops to a different folder, sort in chunks

### "New card names unclear"
- Reference official Clash Royale card names
- Use underscores for multi-word cards: `mega_knight`, `electro_wizard`

### "Training accuracy low"
- Need more images per class (aim for 50+ each)
- Train longer: increase `epochs=100`
- Check that sorting was done correctly (no mislabeled images)

---

## Files Involved

- **scripts/extract_cards_from_roboflow.py** - Downloads Roboflow dataset, crops cards using detection model
- **scripts/sort_crops.py** - Manual sorting interface (already exists)
- **scripts/roboflow_crop.py** - Alternative cropping script (already exists)
- **models/best.pt** - Your NEW detection model (trained)
- **models/card_classifier.pt** - Card classifier (will be replaced)

---

Ready to start! Begin with Step 1 to extract crops from your Roboflow dataset.
