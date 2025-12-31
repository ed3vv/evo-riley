"""
List all Roboflow workspaces, projects, and datasets.

Helps you find the right workspace/project/version for extracting cards.
"""

from roboflow import Roboflow
import json


def list_roboflow_datasets(api_key: str = None, verbose: bool = True):
    """
    List all accessible Roboflow workspaces, projects, and versions.

    Args:
        api_key: Your Roboflow API key (optional, uses default if not provided)
        verbose: If True, print detailed information
    """

    print("=" * 80)
    print("ROBOFLOW DATASETS")
    print("=" * 80)

    if api_key:
        rf = Roboflow(api_key=api_key)
    else:
        # Use default API key from environment or Roboflow config
        print("Note: Using default Roboflow credentials")
        print("For specific workspace, use: --api-key YOUR_WORKSPACE_API_KEY\n")
        rf = Roboflow()

    # Get all workspaces
    try:
        workspaces = rf.workspaces()
        print(f"\nFound {len(workspaces)} workspace(s)\n")
    except Exception as e:
        print(f"Error getting workspaces: {e}")
        print("\nTrying to get default workspace...")
        workspaces = [rf.workspace()]

    all_datasets = []

    for workspace_idx, workspace in enumerate(workspaces, 1):
        try:
            workspace_name = workspace.name if hasattr(workspace, 'name') else str(workspace)
            print(f"\n{'='*80}")
            print(f"WORKSPACE {workspace_idx}: {workspace_name}")
            print(f"{'='*80}")

            # Get all projects in this workspace
            try:
                projects = workspace.projects()
                print(f"Projects: {len(projects)}\n")
            except Exception as e:
                print(f"Could not list projects: {e}\n")
                continue

            for project_idx, project in enumerate(projects, 1):
                try:
                    project_name = project.name if hasattr(project, 'name') else project.id
                    project_id = project.id if hasattr(project, 'id') else project_name

                    print(f"\n  [{workspace_idx}.{project_idx}] Project: {project_name}")
                    print(f"      ID: {project_id}")

                    # Get all versions
                    try:
                        versions = project.versions()
                        print(f"      Versions: {len(versions)}")

                        for version in versions:
                            version_num = version.version if hasattr(version, 'version') else "unknown"

                            # Try to get version details
                            try:
                                # Get image count
                                if hasattr(version, 'splits'):
                                    splits = version.splits
                                    total_images = sum(splits.values()) if splits else 0
                                else:
                                    total_images = "unknown"

                                print(f"        - v{version_num}: {total_images} images")

                                # Store dataset info
                                all_datasets.append({
                                    'workspace': workspace_name,
                                    'project': project_name,
                                    'project_id': project_id,
                                    'version': version_num,
                                    'images': total_images,
                                    'api_key': api_key  # Store which API key accesses this
                                })

                            except Exception as e:
                                print(f"        - v{version_num}: (details unavailable)")
                                all_datasets.append({
                                    'workspace': workspace_name,
                                    'project': project_name,
                                    'project_id': project_id,
                                    'version': version_num,
                                    'images': 'unknown',
                                    'api_key': api_key
                                })

                    except Exception as e:
                        print(f"      Could not list versions: {e}")

                except Exception as e:
                    print(f"  Error processing project: {e}")

        except Exception as e:
            print(f"Error processing workspace: {e}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY: ALL DATASETS")
    print("=" * 80)
    print(f"{'#':<4} {'Workspace':<20} {'Project':<25} {'Ver':<6} {'Images':<10}")
    print("-" * 80)

    for idx, dataset in enumerate(all_datasets, 1):
        workspace = dataset['workspace'][:19]
        project = dataset['project'][:24]
        version = str(dataset['version'])
        images = str(dataset['images'])

        print(f"{idx:<4} {workspace:<20} {project:<25} v{version:<5} {images:<10}")

    print("=" * 80)
    print(f"\nTotal datasets: {len(all_datasets)}\n")

    # Print example commands
    if all_datasets:
        print("=" * 80)
        print("EXAMPLE COMMANDS")
        print("=" * 80)

        # Show first dataset as example
        example = all_datasets[0]
        workspace = example['workspace']
        project_id = example['project_id']
        version = example['version']

        print(f"\nTo extract cards from dataset #1:")
        print(f"\npython3 scripts/extract_unsorted_cards.py \\")
        print(f"    --api-key YOUR_API_KEY \\")
        print(f"    --workspace {workspace} \\")
        print(f"    --project {project_id} \\")
        print(f"    --version {version}")

        if len(all_datasets) > 1:
            print(f"\n(Replace workspace/project/version for other datasets)")

    # Save to JSON file
    output_file = "roboflow_datasets.json"
    with open(output_file, 'w') as f:
        json.dump(all_datasets, f, indent=2)

    print(f"\n📁 Dataset list saved to: {output_file}")

    return all_datasets


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="List all Roboflow workspaces, projects, and datasets"
    )
    parser.add_argument(
        "--api-key",
        help="Roboflow API key (optional, uses default if not provided)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed information"
    )

    args = parser.parse_args()

    list_roboflow_datasets(
        api_key=args.api_key,
        verbose=args.verbose
    )
