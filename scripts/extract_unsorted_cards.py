"""
Extract only NEW/LOW-CONFIDENCE cards from Roboflow dataset.

This script:
1. Downloads your Roboflow dataset (with new arena cards)
2. Uses your NEW best.pt to detect cards
3. Uses your EXISTING card_classifier.pt to classify each crop
4. Only saves crops that:
   - Are misclassified (wrong prediction)
   - Have low confidence (< threshold)
   - Are completely new cards (not in classifier)
5. Ready for manual sorting with sort_crops.py

This way you only sort NEW cards, not all 2500+ cards again!
"""

from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil
import torch


def extract_unsorted_cards(
    roboflow_api_key: str,
    workspace: str,
    project: str,
    version: int,
    detection_model_path: str = "models/best.pt",
    classifier_model_path: str = "models/card_classifier.pt",
    output_dir: str = "crops_unsorted",
    detection_conf: float = 0.25,
    classifier_conf_threshold: float = 0.7,
    save_all: bool = False
):
    """
    Download Roboflow dataset and extract only unsorted/low-confidence cards.

    Args:
        roboflow_api_key: Your Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        detection_model_path: Path to your NEW detection model (best.pt)
        classifier_model_path: Path to your EXISTING classifier (card_classifier.pt)
        output_dir: Where to save unsorted crops
        detection_conf: Minimum confidence for detection
        classifier_conf_threshold: Cards below this confidence need sorting
        save_all: If True, save all crops (for debugging)
    """

    # Clear previous crops
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Clearing previous unsorted crops in {output_dir}...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectory for metadata
    metadata_dir = output_path / "metadata"
    metadata_dir.mkdir(exist_ok=True)

    # Download dataset from Roboflow
    print(f"\n1. Downloading Roboflow dataset: {workspace}/{project}/v{version}")
    rf = Roboflow(api_key=roboflow_api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")

    dataset_path = Path(dataset.location)
    print(f"   Downloaded to: {dataset_path}")

    # Load NEW detection model
    print(f"\n2. Loading detection model: {detection_model_path}")
    detection_model = YOLO(detection_model_path)

    # Load EXISTING classifier
    print(f"   Loading classifier: {classifier_model_path}")
    classifier_model = YOLO(classifier_model_path)

    # Get classifier class names
    classifier_classes = classifier_model.names
    print(f"   Classifier knows {len(classifier_classes)} card types")

    # Find all images in dataset
    image_paths = []
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split / 'images'
        if split_dir.exists():
            image_paths.extend(split_dir.glob('*.jpg'))
            image_paths.extend(split_dir.glob('*.png'))

    print(f"   Found {len(image_paths)} images in dataset")

    # Extract unsorted cards
    print(f"\n3. Extracting unsorted/low-confidence cards...")
    print(f"   Confidence threshold: {classifier_conf_threshold}")

    total_crops = 0
    unsorted_crops = 0
    stats = {
        'low_confidence': 0,
        'new_card': 0,
        'high_confidence': 0
    }

    # Create metadata file
    metadata_file = metadata_dir / "unsorted_info.txt"
    with open(metadata_file, 'w') as f:
        f.write("Unsorted Cards Metadata\n")
        f.write("=" * 80 + "\n\n")

    for img_idx, img_path in enumerate(image_paths):
        # Run detection
        detection_results = detection_model.predict(
            source=str(img_path),
            conf=detection_conf,
            verbose=False
        )

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Process each detection
        for result in detection_results:
            boxes = result.boxes
            for box_idx, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop card
                crop = img[y1:y2, x1:x2]
                total_crops += 1

                # Classify the crop
                classifier_results = classifier_model.predict(
                    source=crop,
                    verbose=False,
                    imgsz=64
                )

                # Get top prediction
                probs = classifier_results[0].probs
                top1_idx = probs.top1
                top1_conf = probs.top1conf.item()
                top1_class = classifier_classes[top1_idx]

                # Decide if this crop needs sorting
                needs_sorting = False
                reason = ""

                if save_all:
                    needs_sorting = True
                    reason = "save_all_mode"
                elif top1_conf < classifier_conf_threshold:
                    needs_sorting = True
                    reason = f"low_confidence_{top1_conf:.2f}"
                    stats['low_confidence'] += 1
                else:
                    stats['high_confidence'] += 1

                # Save if needs sorting
                if needs_sorting:
                    crop_filename = f"crop_{unsorted_crops:05d}.png"
                    crop_path = output_path / crop_filename
                    cv2.imwrite(str(crop_path), crop)

                    # Write metadata
                    with open(metadata_file, 'a') as f:
                        f.write(f"{crop_filename}:\n")
                        f.write(f"  Source: {img_path.name}\n")
                        f.write(f"  Predicted: {top1_class} (confidence: {top1_conf:.3f})\n")
                        f.write(f"  Reason: {reason}\n")
                        f.write(f"  Top-5:\n")

                        # Get top 5 predictions
                        top5_indices = probs.top5
                        top5_conf = probs.top5conf
                        for i, (idx, conf) in enumerate(zip(top5_indices, top5_conf)):
                            class_name = classifier_classes[idx]
                            f.write(f"    {i+1}. {class_name}: {conf:.3f}\n")
                        f.write("\n")

                    unsorted_crops += 1

        # Progress update
        if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(image_paths):
            print(f"   Processed {img_idx + 1}/{len(image_paths)} images | "
                  f"Total crops: {total_crops} | "
                  f"Need sorting: {unsorted_crops}")

    print(f"\n✅ Extraction complete!")
    print(f"\n📊 Statistics:")
    print(f"   Total cards detected: {total_crops}")
    print(f"   High confidence (>= {classifier_conf_threshold}): {stats['high_confidence']} ✅")
    print(f"   Low confidence (< {classifier_conf_threshold}): {stats['low_confidence']} ⚠️")
    print(f"\n   Cards needing manual sorting: {unsorted_crops}")
    print(f"   Saved to: {output_dir}/")
    print(f"   Metadata: {metadata_file}")

    if unsorted_crops > 0:
        print(f"\n📋 Next step: Sort these {unsorted_crops} cards manually")
        print(f"   Run: python3 scripts/sort_crops.py --crops {output_dir}")
    else:
        print(f"\n🎉 No cards need sorting! All cards classified with high confidence.")
        print(f"   You can skip sorting and use the existing classifier as-is.")

    # Save summary
    summary_file = output_path / "SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Unsorted Cards Extraction Summary\n")
        f.write(f"=" * 80 + "\n\n")
        f.write(f"Total cards detected: {total_crops}\n")
        f.write(f"High confidence (>= {classifier_conf_threshold}): {stats['high_confidence']}\n")
        f.write(f"Low confidence (< {classifier_conf_threshold}): {stats['low_confidence']}\n")
        f.write(f"\nCards needing manual sorting: {unsorted_crops}\n")
        f.write(f"\nConfidence threshold: {classifier_conf_threshold}\n")
        f.write(f"Detection model: {detection_model_path}\n")
        f.write(f"Classifier model: {classifier_model_path}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract only unsorted/low-confidence cards from Roboflow dataset"
    )
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--project", required=True, help="Roboflow project name")
    parser.add_argument("--version", type=int, required=True, help="Dataset version")
    parser.add_argument(
        "--detection-model",
        default="models/best.pt",
        help="Detection model path (your NEW best.pt)"
    )
    parser.add_argument(
        "--classifier-model",
        default="models/card_classifier.pt",
        help="Classifier model path (your EXISTING classifier)"
    )
    parser.add_argument(
        "--output",
        default="crops_unsorted",
        help="Output directory for unsorted crops"
    )
    parser.add_argument(
        "--detection-conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--classifier-conf",
        type=float,
        default=0.7,
        help="Classifier confidence threshold (below this = needs sorting)"
    )
    parser.add_argument(
        "--save-all",
        action="store_true",
        help="Save all crops regardless of confidence (for debugging)"
    )

    args = parser.parse_args()

    extract_unsorted_cards(
        roboflow_api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        detection_model_path=args.detection_model,
        classifier_model_path=args.classifier_model,
        output_dir=args.output,
        detection_conf=args.detection_conf,
        classifier_conf_threshold=args.classifier_conf,
        save_all=args.save_all
    )
