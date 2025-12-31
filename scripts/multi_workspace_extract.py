"""
Multi-workspace card extraction from Roboflow.

Handles multiple workspaces with different API keys.
You can list datasets from multiple workspaces and extract from any of them.
"""

from roboflow import Roboflow
import subprocess
import sys
import json
from pathlib import Path


def load_workspace_config(config_file: str = "roboflow_workspaces.json"):
    """
    Load workspace API keys from config file.

    Returns:
        List of workspace configs: [{'name': 'workspace1', 'api_key': 'key1'}, ...]
    """
    config_path = Path(config_file)

    if not config_path.exists():
        return []

    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return []


def save_workspace_config(workspaces: list, config_file: str = "roboflow_workspaces.json"):
    """Save workspace API keys to config file."""
    try:
        with open(config_file, 'w') as f:
            json.dump(workspaces, f, indent=2)
        print(f"✅ Workspace config saved to: {config_file}")
    except Exception as e:
        print(f"Error saving config: {e}")


def setup_workspaces():
    """
    Interactive setup for multiple workspaces with different API keys.
    """
    print("=" * 80)
    print("WORKSPACE SETUP")
    print("=" * 80)
    print("\nLet's set up your Roboflow workspaces with their API keys.")
    print("You can add multiple workspaces if you have different API keys.\n")

    workspaces = []

    while True:
        print("-" * 80)
        workspace_name = input("\nWorkspace name (or 'done' to finish): ").strip()

        if workspace_name.lower() == 'done':
            break

        api_key = input(f"API key for '{workspace_name}': ").strip()

        if api_key:
            workspaces.append({
                'name': workspace_name,
                'api_key': api_key
            })
            print(f"✅ Added workspace: {workspace_name}")
        else:
            print("❌ API key cannot be empty. Skipping.")

    if workspaces:
        save_workspace_config(workspaces)
        print(f"\n✅ Configured {len(workspaces)} workspace(s)")
    else:
        print("\n⚠️  No workspaces configured")

    return workspaces


def list_all_datasets(workspaces: list):
    """
    List datasets from all configured workspaces.

    Args:
        workspaces: List of workspace configs with API keys

    Returns:
        List of all datasets across all workspaces
    """
    all_datasets = []

    for workspace_config in workspaces:
        workspace_name = workspace_config['name']
        api_key = workspace_config['api_key']

        print(f"\n{'='*80}")
        print(f"Fetching datasets from workspace: {workspace_name}")
        print(f"{'='*80}")

        try:
            rf = Roboflow(api_key=api_key)
            workspace = rf.workspace()

            # Get all projects
            try:
                projects = workspace.project_list
                print(f"Found {len(projects)} project(s)\n")

                for project_data in projects:
                    # project_data is a dict with project info
                    if not isinstance(project_data, dict):
                        continue

                    project_name = project_data.get('name', project_data.get('id', 'unknown'))
                    full_project_id = project_data.get('id', project_name)

                    # Extract just the project part (remove workspace prefix)
                    # ID format: "workspace/project-name"
                    if '/' in full_project_id:
                        project_id = full_project_id.split('/')[-1]
                    else:
                        project_id = full_project_id

                    # Get version count and image info from project_data
                    version_count = project_data.get('versions', 0)
                    total_images = project_data.get('images', 0)
                    splits = project_data.get('splits', {})

                    print(f"  Project: {project_name}")
                    print(f"    ID: {project_id}")
                    print(f"    Images: {total_images}")
                    print(f"    Versions: {version_count}")

                    # For each version, we'll create a dataset entry
                    # Since we don't know individual version details from project_list,
                    # we'll need to fetch the project to get version details
                    try:
                        project = workspace.project(project_id)

                        # Get versions
                        if hasattr(project, 'versions'):
                            versions = project.versions()

                            for version_obj in versions:
                                if isinstance(version_obj, dict):
                                    version_num = version_obj.get('name', version_obj.get('id', 'unknown'))
                                    version_images = sum(version_obj.get('splits', {}).values()) if 'splits' in version_obj else total_images
                                else:
                                    version_num = version_obj.version if hasattr(version_obj, 'version') else "unknown"
                                    if hasattr(version_obj, 'splits'):
                                        version_images = sum(version_obj.splits.values())
                                    else:
                                        version_images = total_images

                                print(f"      - v{version_num}: {version_images} images")

                                # Store dataset info
                                all_datasets.append({
                                    'workspace': workspace_name,
                                    'project': project_name,
                                    'project_id': project_id,
                                    'version': version_num,
                                    'images': version_images,
                                    'api_key': api_key
                                })
                        else:
                            # Fallback: if we can't get versions, create entry with latest
                            print(f"      - (version info unavailable)")
                            all_datasets.append({
                                'workspace': workspace_name,
                                'project': project_name,
                                'project_id': project_id,
                                'version': 1,
                                'images': total_images,
                                'api_key': api_key
                            })

                    except Exception as e:
                        print(f"    Could not load versions: {e}")
                        # Still add an entry with what we know
                        all_datasets.append({
                            'workspace': workspace_name,
                            'project': project_name,
                            'project_id': project_id,
                            'version': 1,
                            'images': total_images,
                            'api_key': api_key
                        })

            except Exception as e:
                print(f"Could not list projects: {e}")

        except Exception as e:
            print(f"Error accessing workspace '{workspace_name}': {e}")

    return all_datasets


def interactive_extract():
    """
    Interactive workflow to extract cards from any workspace.
    """
    print("=" * 80)
    print("MULTI-WORKSPACE CARD EXTRACTION")
    print("=" * 80)

    # Step 1: Load or setup workspaces
    config_file = "roboflow_workspaces.json"
    workspaces = load_workspace_config(config_file)

    if not workspaces:
        print(f"\n⚠️  No workspace configuration found ({config_file})")
        print("\nOptions:")
        print("1. Set up workspaces now (recommended)")
        print("2. Use single API key for this session")
        print("3. Exit")

        choice = input("\nEnter choice (1-3): ").strip()

        if choice == '1':
            workspaces = setup_workspaces()
            if not workspaces:
                print("No workspaces configured. Exiting.")
                return
        elif choice == '2':
            api_key = input("Enter your Roboflow API key: ").strip()
            if not api_key:
                print("No API key provided. Exiting.")
                return
            workspace_name = input("Enter workspace name (optional, for display): ").strip() or "default"
            workspaces = [{'name': workspace_name, 'api_key': api_key}]
        else:
            print("Exiting.")
            return
    else:
        print(f"\n✅ Loaded {len(workspaces)} workspace(s) from {config_file}")
        for ws in workspaces:
            print(f"   - {ws['name']}")

    # Step 2: List all datasets
    print("\n" + "=" * 80)
    print("FETCHING DATASETS")
    print("=" * 80)

    datasets = list_all_datasets(workspaces)

    if not datasets:
        print("\n❌ No datasets found across all workspaces!")
        return

    # Step 3: Display summary
    print("\n" + "=" * 80)
    print("ALL DATASETS")
    print("=" * 80)
    print(f"{'#':<4} {'Workspace':<20} {'Project':<25} {'Ver':<6} {'Images':<10}")
    print("-" * 80)

    for idx, dataset in enumerate(datasets, 1):
        workspace = dataset['workspace'][:19]
        project = dataset['project'][:24]
        version = str(dataset['version'])
        images = str(dataset['images'])

        print(f"{idx:<4} {workspace:<20} {project:<25} v{version:<5} {images:<10}")

    print("=" * 80)

    # Step 4: Let user choose dataset
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
    print(f"API Key:   {'*' * 20}{selected['api_key'][-8:]}")  # Show last 8 chars

    # Step 5: Choose extraction mode
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

    # Step 6: Set confidence threshold (if smart mode)
    conf_threshold = 0.7
    if mode == '1':
        print("\n" + "=" * 80)
        print("CONFIDENCE THRESHOLD")
        print("=" * 80)
        print("Cards below this confidence will be extracted for manual sorting.")
        print("Recommended: 0.7 (70%)")

        conf_input = input("\nEnter threshold (0.0-1.0) or press Enter for 0.7: ").strip()
        if conf_input:
            try:
                conf_threshold = float(conf_input)
                conf_threshold = max(0.0, min(1.0, conf_threshold))
            except ValueError:
                print("Invalid input. Using default 0.7")
                conf_threshold = 0.7

    # Step 7: Confirm and run
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

    # Build command using the correct API key for this dataset
    if mode == '1':
        cmd = [
            'python3', 'scripts/extract_unsorted_cards.py',
            '--api-key', selected['api_key'],  # Use workspace-specific API key
            '--workspace', selected['workspace'],
            '--project', selected['project_id'],
            '--version', str(selected['version']),
            '--classifier-conf', str(conf_threshold)
        ]
        print(f"\n🚀 Running smart extraction...\n")
    else:
        cmd = [
            'python3', 'scripts/extract_cards_from_roboflow.py',
            '--api-key', selected['api_key'],  # Use workspace-specific API key
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
        print(f"\n❌ Extraction script not found.")
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
    else:
        print("\n1. Sort all extracted cards:")
        print("   python3 scripts/sort_crops.py --crops crops --sorted sorted")
        print("\n2. Train classifier:")
        print("   yolo classify train data=sorted model=yolov8n-cls.pt epochs=50 imgsz=64 batch=32 name=card_classifier_v2")

    print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-workspace card extraction from Roboflow"
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="Set up workspace API keys"
    )

    args = parser.parse_args()

    if args.setup:
        setup_workspaces()
    else:
        interactive_extract()
