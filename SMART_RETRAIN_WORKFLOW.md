# Smart Retrain Workflow: Only Sort NEW Cards

Instead of re-sorting all 2500+ cards, this workflow **only sorts the cards that need it**.

## How It Works

1. **Extract crops** from Roboflow using your NEW detection model
2. **Classify each crop** using your EXISTING classifier
3. **Only save crops** that have low confidence (< 70%)
4. **Manually sort** only these uncertain cards
5. **Merge** with existing sorted dataset
6. **Retrain** classifier on combined dataset

This way you might only need to sort 50-200 cards instead of 2500+!

---

## Step-by-Step Workflow

### Step 1: Extract Only Unsorted/Low-Confidence Cards

Run the smart extraction script:

```bash
python3 scripts/extract_unsorted_cards.py \
    --api-key YOUR_ROBOFLOW_API_KEY \
    --workspace YOUR_WORKSPACE \
    --project YOUR_PROJECT \
    --version 1 \
    --detection-model models/best.pt \
    --classifier-model models/card_classifier.pt \
    --classifier-conf 0.7 \
    --output crops_unsorted
```

**What this does:**
- Downloads Roboflow dataset
- Uses NEW `best.pt` to detect cards
- Uses EXISTING `card_classifier.pt` to classify each crop
- Only saves crops with confidence < 0.7 (likely new cards or misclassifications)

**Parameters:**
- `--classifier-conf 0.7` - Confidence threshold
  - Lower (0.5) = more conservative, saves more crops to sort
  - Higher (0.9) = only saves very uncertain cards
  - Default (0.7) = good balance

**Example output:**
```
1. Downloading Roboflow dataset: workspace/project/v1
   Downloaded to: /path/to/dataset

2. Loading detection model: models/best.pt
   Loading classifier: models/card_classifier.pt
   Classifier knows 25 card types
   Found 500 images in dataset

3. Extracting unsorted/low-confidence cards...
   Confidence threshold: 0.7
   Processed 500/500 images | Total crops: 2500 | Need sorting: 150

✅ Extraction complete!

📊 Statistics:
   Total cards detected: 2500
   High confidence (>= 0.7): 2350 ✅
   Low confidence (< 0.7): 150 ⚠️

   Cards needing manual sorting: 150
   Saved to: crops_unsorted/
   Metadata: crops_unsorted/metadata/unsorted_info.txt

📋 Next step: Sort these 150 cards manually
   Run: python3 scripts/sort_crops.py --crops crops_unsorted
```

**Key insight:** Instead of sorting 2500 cards, you only sort 150!

---

### Step 2: Review Metadata (Optional)

Check what the classifier thinks these cards are:

```bash
cat crops_unsorted/SUMMARY.txt
cat crops_unsorted/metadata/unsorted_info.txt
```

**Example metadata:**
```
crop_00000.png:
  Source: battle_001.jpg
  Predicted: giant (confidence: 0.623)
  Reason: low_confidence_0.62
  Top-5:
    1. giant: 0.623
    2. royal_giant: 0.241
    3. mega_knight: 0.089
    4. golem: 0.032
    5. pekka: 0.015

crop_00001.png:
  Source: battle_042.jpg
  Predicted: phoenix (confidence: 0.453)
  Reason: low_confidence_0.45
  Top-5:
    1. phoenix: 0.453  ← NEW CARD from new arena!
    2. baby_dragon: 0.312
    3. inferno_dragon: 0.156
    4. lava_hound: 0.079
```

This helps you see:
- What the classifier is confused about
- What new cards appeared (classifier has never seen these)
- What confidence levels look like

---

### Step 3: Manually Sort Only These Cards

Now sort just the 150 cards:

```bash
python3 scripts/sort_crops.py \
    --crops crops_unsorted \
    --sorted sorted_new
```

**Interactive sorting:**
```
Found 150 crops to sort

[1/150] Enter card name: phoenix
New card class: 'phoenix' - Is this correct? (y/n): y
Sorted to 'phoenix': crop_00000.png (1 total)

[2/150] Enter card name: giant
Sorted to 'giant': crop_00001.png (1 total)

... (only 150 cards to sort, not 2500!)

=== SORTING SUMMARY ===
phoenix: 45       ← NEW card from new arena
mega_knight: 30
giant: 25
... (only uncertain cards)

Total sorted: 150
```

**Much faster!** You only sort the cards that need it.

---

### Step 4: Merge with Existing Sorted Dataset

You likely have an existing `sorted/` folder from before. Merge the new sorts:

```bash
python3 scripts/merge_sorted_datasets.py \
    --existing sorted \
    --new sorted_new \
    --output sorted_combined
```

Or manually:
```bash
# Create combined directory
mkdir -p sorted_combined

# Copy existing sorted cards
cp -r sorted/* sorted_combined/

# Merge new sorted cards
for dir in sorted_new/*/; do
    card_name=$(basename "$dir")
    mkdir -p "sorted_combined/$card_name"
    cp "$dir"/*.png "sorted_combined/$card_name/" 2>/dev/null || true
done

echo "✅ Merged datasets into sorted_combined/"
```

**Result:**
```
sorted_combined/
├── knight/          (150 images from before + 0 new)
├── giant/           (130 images from before + 25 new)
├── archers/         (140 images from before + 0 new)
├── phoenix/         (0 images from before + 45 new) ← NEW!
├── mega_knight/     (90 images from before + 30 new)
└── ...
```

---

### Step 5: Train Classifier on Combined Dataset

Train on the combined dataset (old + new):

```bash
yolo classify train \
    data=sorted_combined \
    model=yolov8n-cls.pt \
    epochs=50 \
    imgsz=64 \
    batch=32 \
    patience=10 \
    name=card_classifier_v2
```

**This trains on ALL your cards:**
- Old cards from before (already sorted)
- New arena cards (just sorted)
- Better overall accuracy

---

### Step 6: Replace Classifier

```bash
# Backup old classifier
cp models/card_classifier.pt models/card_classifier_old.pt

# Install new classifier
cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt

echo "✅ Classifier updated with new arena cards!"
```

---

## Quick Reference

### Full workflow (one command each):

```bash
# 1. Extract only unsorted cards
python3 scripts/extract_unsorted_cards.py \
    --api-key YOUR_KEY \
    --workspace YOUR_WORKSPACE \
    --project YOUR_PROJECT \
    --version 1 \
    --classifier-conf 0.7

# 2. Sort only these cards
python3 scripts/sort_crops.py --crops crops_unsorted --sorted sorted_new

# 3. Merge datasets (manual for now)
mkdir -p sorted_combined
cp -r sorted/* sorted_combined/
for dir in sorted_new/*/; do
    card_name=$(basename "$dir")
    mkdir -p "sorted_combined/$card_name"
    cp "$dir"/*.png "sorted_combined/$card_name/" 2>/dev/null || true
done

# 4. Train on combined dataset
yolo classify train \
    data=sorted_combined \
    model=yolov8n-cls.pt \
    epochs=50 \
    imgsz=64 \
    batch=32 \
    name=card_classifier_v2

# 5. Replace classifier
cp models/card_classifier.pt models/card_classifier_old.pt
cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt
```

---

## Adjusting Confidence Threshold

If you want to be more or less conservative:

**More conservative (sort more cards):**
```bash
--classifier-conf 0.5  # Saves cards with < 50% confidence
```
- Pros: Catches more potential misclassifications
- Cons: More cards to manually sort

**Less conservative (sort fewer cards):**
```bash
--classifier-conf 0.9  # Only saves cards with < 90% confidence
```
- Pros: Very few cards to sort
- Cons: Might miss some edge cases

**Recommended: 0.7** (70% confidence threshold)
- Good balance between catching new cards and minimizing manual work

---

## Comparison: Smart vs. Full Retrain

### Full Retrain (from RETRAIN_CLASSIFIER_GUIDE.md)
- Extract ALL cards: 2500+ crops
- Sort ALL cards: 2500+ manual labels
- Time: 3-5 hours of manual sorting
- Use when: Starting from scratch

### Smart Retrain (this guide)
- Extract only uncertain cards: 50-200 crops
- Sort only these: 50-200 manual labels
- Time: 10-30 minutes of manual sorting
- Use when: Adding new arena cards to existing classifier

**You should use this Smart Retrain workflow!** Much faster.

---

## Troubleshooting

### "No cards need sorting!"
This means your existing classifier already handles all cards well. Either:
- Lower the threshold: `--classifier-conf 0.5`
- Use `--save-all` to see all crops (debugging)

### "Still too many cards to sort (500+)"
Your classifier doesn't know many of these cards. Either:
- Increase threshold: `--classifier-conf 0.8`
- Or accept that you need to sort them (they're new)

### "How do I know which cards are truly new?"
Check the metadata file:
```bash
grep "confidence: 0\." crops_unsorted/metadata/unsorted_info.txt
```
Very low confidence (< 0.3) usually means completely new cards.

---

## Files Created

- **scripts/extract_unsorted_cards.py** - Smart extraction (only saves uncertain cards)
- **crops_unsorted/** - Directory with only cards needing sorting
- **crops_unsorted/metadata/unsorted_info.txt** - Detailed info about each unsorted card
- **crops_unsorted/SUMMARY.txt** - Summary statistics

---

Ready to start! Run Step 1 with your Roboflow credentials.
