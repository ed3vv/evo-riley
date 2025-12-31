"""
Test Roboflow API to see what data structure it returns.
"""

from roboflow import Roboflow
import json


def test_roboflow_api(api_key: str):
    """
    Test what the Roboflow API returns for your workspace.
    """

    print("=" * 80)
    print("ROBOFLOW API TEST")
    print("=" * 80)

    try:
        rf = Roboflow(api_key=api_key)
        print("✅ Connected to Roboflow\n")

        # First, let's see what's available on rf object
        print("Roboflow object attributes:")
        print(f"   Type: {type(rf)}")

        # Check for workspace-related attributes
        if hasattr(rf, 'workspace'):
            print("   ✅ Has workspace() method")
        if hasattr(rf, 'workspaces'):
            print("   ✅ Has workspaces() method")

        # Try to get workspace URL from auth
        workspace_url = None
        if hasattr(rf, '__dict__'):
            print(f"\n   RF object dict keys: {list(rf.__dict__.keys())}")
            if 'api_url' in rf.__dict__:
                print(f"   API URL: {rf.__dict__['api_url']}")

        print("\nAttempting to get workspace...")

        # Try method 1: workspace() with no args
        try:
            workspace = rf.workspace()
            print(f"✅ Method 1 worked: rf.workspace()")
        except Exception as e1:
            print(f"❌ Method 1 failed: {e1}")

            # Try method 2: Get workspace from API directly
            try:
                # The API key is tied to a specific workspace
                # We need to extract workspace from the API response
                import requests

                # Try to get workspace info
                headers = {"Authorization": f"Bearer {api_key}"}
                response = requests.get("https://api.roboflow.com/", headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    print(f"\n✅ Got API response:")
                    print(f"   Data: {json.dumps(data, indent=2)}")

                    # Extract workspace from response
                    if 'workspace' in data:
                        workspace_id = data['workspace']
                        print(f"\n   Workspace ID from API: {workspace_id}")

                        # Now try to get workspace with ID
                        workspace = rf.workspace(workspace_id)
                        print(f"✅ Got workspace object: {type(workspace)}")
                    else:
                        print("❌ No workspace in API response")
                        return
                else:
                    print(f"❌ API request failed: {response.status_code}")
                    return

            except Exception as e2:
                print(f"❌ Method 2 failed: {e2}")
                import traceback
                traceback.print_exc()
                return

        print(f"✅ Workspace object: {type(workspace)}")
        print(f"   Workspace attributes: {dir(workspace)}\n")

        # Try to get workspace name
        if hasattr(workspace, 'name'):
            print(f"   Workspace name: {workspace.name}")
        if hasattr(workspace, 'url'):
            print(f"   Workspace URL: {workspace.url}")

        print("\n" + "-" * 80)
        print("Getting projects...")
        print("-" * 80 + "\n")

        # Try different ways to get projects
        projects = None

        if hasattr(workspace, 'project_list'):
            print("✅ Found workspace.project_list")
            projects = workspace.project_list
            print(f"   Type: {type(projects)}")
            print(f"   Count: {len(projects) if projects else 0}")

            if projects and len(projects) > 0:
                print(f"\n   First project structure:")
                first_project = projects[0]
                print(f"   Type: {type(first_project)}")

                if isinstance(first_project, dict):
                    print(f"   Keys: {first_project.keys()}")
                    print(f"   Data: {json.dumps(first_project, indent=2)}")
                else:
                    print(f"   Attributes: {dir(first_project)}")
                    if hasattr(first_project, 'name'):
                        print(f"   Name: {first_project.name}")
                    if hasattr(first_project, 'id'):
                        print(f"   ID: {first_project.id}")

        elif hasattr(workspace, 'projects'):
            print("✅ Found workspace.projects()")
            try:
                projects = workspace.projects()
                print(f"   Type: {type(projects)}")
                print(f"   Count: {len(projects) if projects else 0}")
            except Exception as e:
                print(f"   Error calling projects(): {e}")

        else:
            print("❌ No known method to get projects")
            print(f"   Available workspace methods:")
            for attr in dir(workspace):
                if not attr.startswith('_'):
                    print(f"     - {attr}")

        # If we have projects, try to access first one
        if projects and len(projects) > 0:
            print("\n" + "-" * 80)
            print("Testing first project...")
            print("-" * 80 + "\n")

            first_project_data = projects[0]

            # Get project ID
            if isinstance(first_project_data, dict):
                project_id = first_project_data.get('id', first_project_data.get('name'))
            else:
                project_id = first_project_data.id if hasattr(first_project_data, 'id') else None

            if project_id:
                print(f"Project ID: {project_id}")

                # Try to get the project object
                try:
                    print(f"\nTrying workspace.project('{project_id}')...")
                    project = workspace.project(project_id)
                    print(f"✅ Got project object: {type(project)}")
                    print(f"   Attributes: {dir(project)}")

                    # Try to get versions
                    print(f"\n   Getting versions...")

                    if hasattr(project, 'versions'):
                        print(f"   ✅ Found project.versions()")
                        try:
                            versions = project.versions()
                            print(f"      Type: {type(versions)}")
                            print(f"      Count: {len(versions) if versions else 0}")

                            if versions and len(versions) > 0:
                                first_version = versions[0]
                                print(f"\n      First version structure:")
                                print(f"      Type: {type(first_version)}")

                                if isinstance(first_version, dict):
                                    print(f"      Keys: {first_version.keys()}")
                                else:
                                    print(f"      Attributes: {dir(first_version)}")
                                    if hasattr(first_version, 'version'):
                                        print(f"      Version: {first_version.version}")

                        except Exception as e:
                            print(f"      Error calling versions(): {e}")
                    else:
                        print(f"   ❌ No versions() method")
                        print(f"   Available project methods:")
                        for attr in dir(project):
                            if not attr.startswith('_'):
                                print(f"     - {attr}")

                except Exception as e:
                    print(f"   Error getting project: {e}")
                    import traceback
                    traceback.print_exc()

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Roboflow API")
    parser.add_argument("--api-key", required=True, help="Roboflow API key")

    args = parser.parse_args()

    test_roboflow_api(args.api_key)
