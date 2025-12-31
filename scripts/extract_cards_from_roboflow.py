"""
Extract card crops from Roboflow dataset using the NEW detection model.

This script:
1. Downloads your Roboflow dataset (with new arena cards)
2. Uses your NEW best.pt model to detect cards
3. Crops each detected card into the 'crops' folder
4. Ready for manual sorting with sort_crops.py
"""

from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import shutil


def extract_cards_from_roboflow(
    roboflow_api_key: str,
    workspace: str,
    project: str,
    version: int,
    detection_model_path: str = "models/best.pt",
    output_dir: str = "crops",
    confidence_threshold: float = 0.25
):
    """
    Download Roboflow dataset and extract card crops using detection model.

    Args:
        roboflow_api_key: Your Roboflow API key
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number
        detection_model_path: Path to your NEW detection model (best.pt)
        output_dir: Where to save cropped cards
        confidence_threshold: Minimum confidence for detections
    """

    # Clear previous crops
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"Clearing previous crops in {output_dir}...")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Download dataset from Roboflow
    print(f"\n1. Downloading Roboflow dataset: {workspace}/{project}/v{version}")
    rf = Roboflow(api_key=roboflow_api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset = project_obj.version(version).download("yolov8")

    dataset_path = Path(dataset.location)
    print(f"   Downloaded to: {dataset_path}")

    # Load NEW detection model
    print(f"\n2. Loading detection model: {detection_model_path}")
    model = YOLO(detection_model_path)

    # Find all images in dataset
    image_paths = []
    for split in ['train', 'valid', 'test']:
        split_dir = dataset_path / split / 'images'
        if split_dir.exists():
            image_paths.extend(split_dir.glob('*.jpg'))
            image_paths.extend(split_dir.glob('*.png'))

    print(f"   Found {len(image_paths)} images in dataset")

    # Extract cards from each image
    print(f"\n3. Extracting cards from images...")
    crop_count = 0

    for img_path in image_paths:
        # Run detection
        results = model.predict(source=str(img_path), conf=confidence_threshold, verbose=False)

        # Load image
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Crop each detection
        for result in results:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop card
                crop = img[y1:y2, x1:x2]

                # Save crop
                crop_filename = f"crop_{crop_count:05d}.png"
                crop_path = output_path / crop_filename
                cv2.imwrite(str(crop_path), crop)

                crop_count += 1

        if (len(image_paths) > 100 and crop_count % 100 == 0) or len(image_paths) <= 100:
            print(f"   Processed {len([p for p in image_paths if p <= img_path])}/{len(image_paths)} images, extracted {crop_count} cards...")

    print(f"\n✅ Extraction complete!")
    print(f"   Total cards extracted: {crop_count}")
    print(f"   Saved to: {output_dir}/")
    print(f"\n📋 Next step: Sort these cards manually")
    print(f"   Run: python3 scripts/sort_crops.py --crops {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract cards from Roboflow dataset")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")
    parser.add_argument("--workspace", required=True, help="Roboflow workspace name")
    parser.add_argument("--project", required=True, help="Roboflow project name")
    parser.add_argument("--version", type=int, required=True, help="Dataset version")
    parser.add_argument("--model", default="models/best.pt", help="Detection model path")
    parser.add_argument("--output", default="crops", help="Output directory for crops")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")

    args = parser.parse_args()

    extract_cards_from_roboflow(
        roboflow_api_key=args.api_key,
        workspace=args.workspace,
        project=args.project,
        version=args.version,
        detection_model_path=args.model,
        output_dir=args.output,
        confidence_threshold=args.conf
    )
