"""
Interactive card extraction from Roboflow.

Lists all your Roboflow datasets and lets you choose which one to extract from.
"""

from roboflow import Roboflow
import subprocess
import sys


def interactive_extract(api_key: str):
    """
    Interactive workflow to extract cards from Roboflow.

    Args:
        api_key: Your Roboflow API key
    """

    print("=" * 80)
    print("INTERACTIVE CARD EXTRACTION")
    print("=" * 80)

    # Step 1: List all datasets
    print("\n1. Fetching your Roboflow datasets...\n")

    try:
        # Import the list function
        sys.path.append('scripts')
        from list_roboflow_datasets import list_roboflow_datasets

        datasets = list_roboflow_datasets(api_key=api_key, verbose=False)

        if not datasets:
            print("❌ No datasets found!")
            return

    except Exception as e:
        print(f"Error listing datasets: {e}")
        return

    # Step 2: Let user choose dataset
    print("\n" + "=" * 80)
    print("SELECT DATASET")
    print("=" * 80)

    while True:
        choice = input(f"\nEnter dataset number (1-{len(datasets)}) or 'q' to quit: ").strip()

        if choice.lower() == 'q':
            print("Cancelled.")
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                selected = datasets[idx]
                break
            else:
                print(f"Invalid choice. Enter a number between 1 and {len(datasets)}.")
        except ValueError:
            print("Invalid input. Enter a number.")

    # Show selected dataset
    print("\n" + "=" * 80)
    print("SELECTED DATASET")
    print("=" * 80)
    print(f"Workspace: {selected['workspace']}")
    print(f"Project:   {selected['project']}")
    print(f"Version:   v{selected['version']}")
    print(f"Images:    {selected['images']}")

    # Step 3: Choose extraction mode
    print("\n" + "=" * 80)
    print("EXTRACTION MODE")
    print("=" * 80)
    print("1. Smart extraction (only unsorted/low-confidence cards) - RECOMMENDED")
    print("2. Full extraction (all cards from dataset)")

    while True:
        mode = input("\nEnter mode (1 or 2): ").strip()
        if mode in ['1', '2']:
            break
        print("Invalid choice. Enter 1 or 2.")

    # Step 4: Set confidence threshold (if smart mode)
    conf_threshold = 0.7
    if mode == '1':
        print("\n" + "=" * 80)
        print("CONFIDENCE THRESHOLD")
        print("=" * 80)
        print("Cards below this confidence will be extracted for manual sorting.")
        print("Recommended: 0.7 (70%)")
        print("  Lower (0.5) = extract more cards")
        print("  Higher (0.9) = extract fewer cards")

        conf_input = input("\nEnter threshold (0.0-1.0) or press Enter for 0.7: ").strip()
        if conf_input:
            try:
                conf_threshold = float(conf_input)
                conf_threshold = max(0.0, min(1.0, conf_threshold))
            except ValueError:
                print("Invalid input. Using default 0.7")
                conf_threshold = 0.7

    # Step 5: Confirm and run
    print("\n" + "=" * 80)
    print("READY TO EXTRACT")
    print("=" * 80)
    print(f"Dataset:   {selected['workspace']}/{selected['project']}/v{selected['version']}")
    print(f"Mode:      {'Smart (unsorted only)' if mode == '1' else 'Full (all cards)'}")
    if mode == '1':
        print(f"Threshold: {conf_threshold}")
    print()

    confirm = input("Proceed with extraction? (y/n): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("Cancelled.")
        return

    # Build command
    if mode == '1':
        # Smart extraction
        cmd = [
            'python3', 'scripts/extract_unsorted_cards.py',
            '--api-key', api_key,
            '--workspace', selected['workspace'],
            '--project', selected['project_id'],
            '--version', str(selected['version']),
            '--classifier-conf', str(conf_threshold)
        ]
        print(f"\n🚀 Running smart extraction...\n")
    else:
        # Full extraction
        cmd = [
            'python3', 'scripts/extract_cards_from_roboflow.py',
            '--api-key', api_key,
            '--workspace', selected['workspace'],
            '--project', selected['project_id'],
            '--version', str(selected['version'])
        ]
        print(f"\n🚀 Running full extraction...\n")

    # Run extraction
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Extraction failed: {e}")
        return
    except FileNotFoundError:
        print(f"\n❌ Extraction script not found. Make sure you're in the project root directory.")
        return

    # Next steps
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)

    if mode == '1':
        print("\n1. Sort the extracted cards:")
        print("   python3 scripts/sort_crops.py --crops crops_unsorted --sorted sorted_new")
        print("\n2. Merge with existing sorted dataset:")
        print("   python3 scripts/merge_sorted_datasets.py --existing sorted --new sorted_new --output sorted_combined")
        print("\n3. Train classifier:")
        print("   yolo classify train data=sorted_combined model=yolov8n-cls.pt epochs=50 imgsz=64 batch=32 name=card_classifier_v2")
        print("\n4. Replace classifier:")
        print("   cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt")
    else:
        print("\n1. Sort all extracted cards:")
        print("   python3 scripts/sort_crops.py --crops crops --sorted sorted")
        print("\n2. Train classifier:")
        print("   yolo classify train data=sorted model=yolov8n-cls.pt epochs=50 imgsz=64 batch=32 name=card_classifier_v2")
        print("\n3. Replace classifier:")
        print("   cp runs/classify/card_classifier_v2/weights/best.pt models/card_classifier.pt")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Interactive card extraction from Roboflow"
    )
    parser.add_argument(
        "--api-key",
        required=True,
        help="Roboflow API key"
    )

    args = parser.parse_args()

    interactive_extract(api_key=args.api_key)
