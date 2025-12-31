# Checkpoint Management Guide

## Overview

The RL agent automatically saves checkpoints during training to preserve your progress. Checkpoints contain the trained neural network weights, optimizer state, epsilon value, and training history.

## Checkpoint Structure

Each checkpoint file (`.pt`) contains:
- **Q-Network weights**: The main policy network
- **Target Network weights**: The stable target network
- **Optimizer state**: Adam optimizer parameters
- **Training progress**: Episodes, steps, epsilon value
- **Reward history**: Episode rewards for tracking performance

## Automatic Saving

Checkpoints are automatically saved during training:

- **Every 5 games**: `checkpoint_game_5.pt`, `checkpoint_game_10.pt`, etc.
- **Latest checkpoint**: `latest.pt` (always the most recent save)

Location: `checkpoints/` directory

## Managing Checkpoints

### 1. View Available Checkpoints

```bash
# List all checkpoints with sizes
ls -lh checkpoints/

# Count total checkpoints
ls checkpoints/*.pt | wc -l

# Show only game checkpoints (sorted)
ls checkpoints/checkpoint_game_*.pt | sort -V
```

### 2. Check Checkpoint Info

To see what's inside a checkpoint without loading it:

```python
import torch

checkpoint = torch.load('checkpoints/latest.pt', map_location='cpu')
print(f"Episodes: {checkpoint['episodes']}")
print(f"Steps: {checkpoint['steps']}")
print(f"Epsilon: {checkpoint['epsilon']:.4f}")
print(f"Reward history length: {len(checkpoint['episode_rewards'])}")
```

### 3. Load from Specific Checkpoint

**Start training from a checkpoint:**

```bash
# Continue from latest checkpoint
python3 main.py --rl --games 100 --checkpoint checkpoints/latest.pt

# Continue from specific game checkpoint
python3 main.py --rl --games 100 --checkpoint checkpoints/checkpoint_game_50.pt

# Start fresh (no checkpoint)
python3 main.py --rl --games 100
```

### 4. Delete Checkpoints

**Delete old checkpoints to save space:**

```bash
# Delete all checkpoints
rm -rf checkpoints/*.pt

# Keep only latest checkpoint
rm checkpoints/checkpoint_game_*.pt

# Keep only recent checkpoints (e.g., last 5)
ls -t checkpoints/checkpoint_game_*.pt | tail -n +6 | xargs rm

# Delete checkpoints older than specific game
rm checkpoints/checkpoint_game_{0..45}.pt  # Keep only game 50+
```

### 5. Restart Training from Scratch

```bash
# Option 1: Delete all checkpoints
rm -rf checkpoints/*.pt
python3 main.py --rl --games 100

# Option 2: Use a different checkpoint directory
mkdir checkpoints_v2
# (modify code to use checkpoints_v2/)

# Option 3: Just don't specify --checkpoint flag
python3 main.py --rl --games 100  # Starts fresh
```

## Checkpoint File Sizes

- Each checkpoint: ~16MB
- 100 checkpoints: ~1.6GB
- Recommend: Keep only last 10-20 checkpoints to save disk space

## Best Practices

1. **Backup important checkpoints** before long training sessions
   ```bash
   cp checkpoints/checkpoint_game_100.pt checkpoints/backup_game_100.pt
   ```

2. **Clean up old checkpoints regularly**
   ```bash
   # Keep only every 10th checkpoint
   ls checkpoints/checkpoint_game_*.pt | grep -v '[0]\.pt$' | xargs rm
   ```

3. **Monitor training progress**
   - Watch epsilon decay (should decrease over time)
   - Check average reward trends
   - Compare checkpoints from different stages

4. **Test checkpoints before deleting**
   - Run a few test games to verify performance
   - Keep checkpoints that show improvement

## Troubleshooting

### Checkpoint loading fails

```
Error: checkpoint['state_size'] != current state_size
```

**Solution**: The model architecture changed. You need to retrain from scratch or use a checkpoint from the same code version.

### Out of disk space

```bash
# Free up space by keeping only recent checkpoints
ls -t checkpoints/checkpoint_game_*.pt | tail -n +11 | xargs rm
```

### Want to compare different training runs

```bash
# Organize into versioned directories
mkdir -p checkpoints_runs/run1
mv checkpoints/*.pt checkpoints_runs/run1/

# Start new run
python3 main.py --rl --games 100
```

## Example Workflow

```bash
# 1. Train for 50 games
python3 main.py --rl --games 50

# 2. Check progress
ls -lh checkpoints/

# 3. Continue training from latest checkpoint
python3 main.py --rl --games 100 --checkpoint checkpoints/latest.pt

# 4. Clean up old checkpoints (keep every 5th)
ls checkpoints/checkpoint_game_{1..4}.pt | xargs rm
ls checkpoints/checkpoint_game_{6..9}.pt | xargs rm
# (repeat pattern)

# 5. Backup best checkpoint
cp checkpoints/checkpoint_game_100.pt checkpoints/best_model_v1.pt
```

## Quick Reference

| Task | Command |
|------|---------|
| Start fresh | `python3 main.py --rl --games 100` |
| Continue training | `python3 main.py --rl --games 100 --checkpoint checkpoints/latest.pt` |
| List checkpoints | `ls -lh checkpoints/` |
| Delete all | `rm checkpoints/*.pt` |
| Keep latest only | `rm checkpoints/checkpoint_game_*.pt` |
| Backup checkpoint | `cp checkpoints/latest.pt checkpoints/backup.pt` |
