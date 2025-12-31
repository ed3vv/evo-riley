"""
Merge two sorted card datasets.

Takes an existing sorted dataset and merges it with newly sorted cards.
Useful for incrementally adding new cards without re-sorting everything.
"""

from pathlib import Path
import shutil
from collections import defaultdict


def merge_sorted_datasets(
    existing_dir: str = "sorted",
    new_dir: str = "sorted_new",
    output_dir: str = "sorted_combined",
    copy_mode: str = "copy"
):
    """
    Merge two sorted card datasets.

    Args:
        existing_dir: Directory with existing sorted cards
        new_dir: Directory with newly sorted cards
        output_dir: Output directory for combined dataset
        copy_mode: 'copy' or 'move' (move deletes source files)
    """

    existing_path = Path(existing_dir)
    new_path = Path(new_dir)
    output_path = Path(output_dir)

    # Validate inputs
    if not existing_path.exists():
        print(f"❌ Existing dataset not found: {existing_dir}")
        return

    if not new_path.exists():
        print(f"❌ New dataset not found: {new_dir}")
        return

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Merging sorted datasets:")
    print(f"  Existing: {existing_dir}")
    print(f"  New: {new_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Mode: {copy_mode}")
    print()

    stats = defaultdict(lambda: {'existing': 0, 'new': 0, 'total': 0})

    # Step 1: Copy existing sorted cards
    print("1. Copying existing sorted cards...")
    for class_dir in sorted(existing_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy all images in this class
        image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        for img_file in image_files:
            dest_file = output_class_dir / img_file.name

            if copy_mode == 'copy':
                shutil.copy2(img_file, dest_file)
            else:
                shutil.move(str(img_file), str(dest_file))

            stats[class_name]['existing'] += 1

        print(f"   {class_name}: {len(image_files)} images")

    # Step 2: Merge new sorted cards
    print("\n2. Merging newly sorted cards...")
    for class_dir in sorted(new_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        output_class_dir = output_path / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        # Copy all images in this class
        image_files = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
        for img_file in image_files:
            # Handle name conflicts
            dest_file = output_class_dir / img_file.name
            counter = 1
            while dest_file.exists():
                # Add suffix to avoid overwriting
                stem = img_file.stem
                suffix = img_file.suffix
                dest_file = output_class_dir / f"{stem}_new{counter}{suffix}"
                counter += 1

            if copy_mode == 'copy':
                shutil.copy2(img_file, dest_file)
            else:
                shutil.move(str(img_file), str(dest_file))

            stats[class_name]['new'] += 1

        if len(image_files) > 0:
            is_new_class = stats[class_name]['existing'] == 0
            marker = "← NEW CLASS!" if is_new_class else ""
            print(f"   {class_name}: +{len(image_files)} new images {marker}")

    # Calculate totals
    for class_name in stats:
        stats[class_name]['total'] = (
            stats[class_name]['existing'] + stats[class_name]['new']
        )

    # Print summary
    print("\n" + "=" * 80)
    print("MERGE SUMMARY")
    print("=" * 80)
    print(f"{'Class':<20} {'Existing':<12} {'New':<12} {'Total':<12}")
    print("-" * 80)

    total_existing = 0
    total_new = 0
    total_images = 0

    for class_name in sorted(stats.keys()):
        s = stats[class_name]
        is_new_class = s['existing'] == 0

        marker = " (NEW)" if is_new_class else ""
        print(f"{class_name:<20} {s['existing']:<12} {s['new']:<12} {s['total']:<12}{marker}")

        total_existing += s['existing']
        total_new += s['new']
        total_images += s['total']

    print("-" * 80)
    print(f"{'TOTAL':<20} {total_existing:<12} {total_new:<12} {total_images:<12}")
    print("=" * 80)

    print(f"\n✅ Merge complete!")
    print(f"   Total classes: {len(stats)}")
    print(f"   Total images: {total_images}")
    print(f"   Output: {output_dir}/")

    # Save summary to file
    summary_file = output_path / "MERGE_SUMMARY.txt"
    with open(summary_file, 'w') as f:
        f.write("Dataset Merge Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Existing dataset: {existing_dir}\n")
        f.write(f"New dataset: {new_dir}\n")
        f.write(f"Output dataset: {output_dir}\n")
        f.write(f"Mode: {copy_mode}\n\n")
        f.write(f"{'Class':<20} {'Existing':<12} {'New':<12} {'Total':<12}\n")
        f.write("-" * 80 + "\n")

        for class_name in sorted(stats.keys()):
            s = stats[class_name]
            is_new_class = s['existing'] == 0
            marker = " (NEW)" if is_new_class else ""
            f.write(f"{class_name:<20} {s['existing']:<12} {s['new']:<12} {s['total']:<12}{marker}\n")

        f.write("-" * 80 + "\n")
        f.write(f"{'TOTAL':<20} {total_existing:<12} {total_new:<12} {total_images:<12}\n")

    print(f"   Summary saved: {summary_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Merge two sorted card datasets"
    )
    parser.add_argument(
        "--existing",
        default="sorted",
        help="Directory with existing sorted cards"
    )
    parser.add_argument(
        "--new",
        default="sorted_new",
        help="Directory with newly sorted cards"
    )
    parser.add_argument(
        "--output",
        default="sorted_combined",
        help="Output directory for combined dataset"
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (deletes source files)"
    )

    args = parser.parse_args()

    merge_sorted_datasets(
        existing_dir=args.existing,
        new_dir=args.new,
        output_dir=args.output,
        copy_mode='move' if args.move else 'copy'
    )
