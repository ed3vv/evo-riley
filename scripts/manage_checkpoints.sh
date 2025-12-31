#!/bin/bash
# Checkpoint Management Script

show_help() {
    echo "Checkpoint Management Tool"
    echo "=========================="
    echo ""
    echo "Usage: bash scripts/manage_checkpoints.sh [command]"
    echo ""
    echo "Commands:"
    echo "  list              List all checkpoints with details"
    echo "  info <file>       Show checkpoint information"
    echo "  clean <keep>      Delete old checkpoints, keep only last N"
    echo "  delete-all        Delete all checkpoints (requires confirmation)"
    echo "  backup <file>     Backup a checkpoint"
    echo ""
    echo "Examples:"
    echo "  bash scripts/manage_checkpoints.sh list"
    echo "  bash scripts/manage_checkpoints.sh info checkpoints/latest.pt"
    echo "  bash scripts/manage_checkpoints.sh clean 10"
    echo "  bash scripts/manage_checkpoints.sh backup checkpoints/checkpoint_game_100.pt"
    echo ""
}

list_checkpoints() {
    echo "Checkpoints in checkpoints/:"
    echo "============================"

    if [ ! -d "checkpoints" ]; then
        echo "No checkpoints directory found!"
        return
    fi

    # Count files
    count=$(ls checkpoints/*.pt 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "No checkpoints found!"
        return
    fi

    # List all checkpoints with size and date
    ls -lhtr checkpoints/*.pt

    echo ""
    echo "Total: $count checkpoint(s)"

    # Calculate total size
    total_size=$(du -sh checkpoints/ 2>/dev/null | cut -f1)
    echo "Total size: $total_size"
}

show_info() {
    checkpoint_file=$1

    if [ -z "$checkpoint_file" ]; then
        echo "Error: Please specify checkpoint file"
        echo "Usage: bash scripts/manage_checkpoints.sh info <file>"
        return 1
    fi

    if [ ! -f "$checkpoint_file" ]; then
        echo "Error: Checkpoint file not found: $checkpoint_file"
        return 1
    fi

    echo "Checkpoint Information"
    echo "====================="
    echo "File: $checkpoint_file"
    echo "Size: $(ls -lh "$checkpoint_file" | awk '{print $5}')"
    echo "Modified: $(ls -lh "$checkpoint_file" | awk '{print $6, $7, $8}')"
    echo ""

    # Extract info using Python
    python3 - <<END
import torch
try:
    checkpoint = torch.load('$checkpoint_file', map_location='cpu')
    print(f"Episodes:        {checkpoint['episodes']}")
    print(f"Steps:           {checkpoint['steps']}")
    print(f"Epsilon:         {checkpoint['epsilon']:.4f}")
    print(f"State size:      {checkpoint['state_size']}")
    print(f"Action size:     {checkpoint['action_size']}")
    print(f"Reward history:  {len(checkpoint['episode_rewards'])} episodes")

    if len(checkpoint['episode_rewards']) > 0:
        recent = checkpoint['episode_rewards'][-10:]
        avg_recent = sum(recent) / len(recent)
        print(f"Avg reward (last 10): {avg_recent:.2f}")
except Exception as e:
    print(f"Error reading checkpoint: {e}")
END
}

clean_checkpoints() {
    keep_count=$1

    if [ -z "$keep_count" ]; then
        echo "Error: Please specify how many checkpoints to keep"
        echo "Usage: bash scripts/manage_checkpoints.sh clean <count>"
        return 1
    fi

    if ! [[ "$keep_count" =~ ^[0-9]+$ ]]; then
        echo "Error: Keep count must be a number"
        return 1
    fi

    total=$(ls checkpoints/checkpoint_game_*.pt 2>/dev/null | wc -l)

    if [ "$total" -eq 0 ]; then
        echo "No game checkpoints found to clean!"
        return
    fi

    if [ "$total" -le "$keep_count" ]; then
        echo "Already have $total checkpoint(s), which is <= $keep_count. Nothing to delete."
        return
    fi

    to_delete=$((total - keep_count))

    echo "Found $total checkpoint(s)"
    echo "Keeping latest $keep_count checkpoint(s)"
    echo "Deleting $to_delete old checkpoint(s)"
    echo ""

    # Show which files will be deleted
    ls -t checkpoints/checkpoint_game_*.pt | tail -n +$((keep_count + 1))

    echo ""
    read -p "Delete these files? (y/N) " -n 1 -r
    echo

    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ls -t checkpoints/checkpoint_game_*.pt | tail -n +$((keep_count + 1)) | xargs rm
        echo "✅ Deleted $to_delete checkpoint(s)"
    else
        echo "Cancelled"
    fi
}

delete_all() {
    if [ ! -d "checkpoints" ]; then
        echo "No checkpoints directory found!"
        return
    fi

    count=$(ls checkpoints/*.pt 2>/dev/null | wc -l)

    if [ "$count" -eq 0 ]; then
        echo "No checkpoints found to delete!"
        return
    fi

    echo "⚠️  WARNING: This will delete ALL $count checkpoint(s)!"
    echo ""
    ls -lh checkpoints/*.pt
    echo ""
    read -p "Are you sure? (type 'yes' to confirm): " confirmation

    if [ "$confirmation" == "yes" ]; then
        rm checkpoints/*.pt
        echo "✅ Deleted all checkpoints"
    else
        echo "Cancelled"
    fi
}

backup_checkpoint() {
    checkpoint_file=$1

    if [ -z "$checkpoint_file" ]; then
        echo "Error: Please specify checkpoint file to backup"
        echo "Usage: bash scripts/manage_checkpoints.sh backup <file>"
        return 1
    fi

    if [ ! -f "$checkpoint_file" ]; then
        echo "Error: Checkpoint file not found: $checkpoint_file"
        return 1
    fi

    # Generate backup filename with timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    basename=$(basename "$checkpoint_file" .pt)
    backup_file="checkpoints/${basename}_backup_${timestamp}.pt"

    cp "$checkpoint_file" "$backup_file"
    echo "✅ Backup created: $backup_file"
    echo "Size: $(ls -lh "$backup_file" | awk '{print $5}')"
}

# Main script
case "$1" in
    list)
        list_checkpoints
        ;;
    info)
        show_info "$2"
        ;;
    clean)
        clean_checkpoints "$2"
        ;;
    delete-all)
        delete_all
        ;;
    backup)
        backup_checkpoint "$2"
        ;;
    *)
        show_help
        ;;
esac
