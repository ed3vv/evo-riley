"""
Display utilities for cleaner terminal output
"""
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime


class GameDisplay:
    """
    Manages clean terminal display with in-place updates
    """

    def __init__(self, verbose: bool = False):
        """
        Initialize display manager

        Args:
            verbose: If True, show all debug info. If False, show compact dashboard.
        """
        self.verbose = verbose
        self.last_lines = 0

    def clear_lines(self, n: int):
        """Clear last n lines from terminal"""
        if not self.verbose:
            for _ in range(n):
                sys.stdout.write('\033[F')  # Move cursor up
                sys.stdout.write('\033[K')  # Clear line

    def clear_screen(self):
        """Clear entire screen"""
        if not self.verbose:
            os.system('clear' if os.name == 'posix' else 'cls')

    def print_dashboard(self, state: str, game_num: int,
                       hand: List[Optional[Dict]],
                       elixir: Optional[int],
                       recent_actions: List[str],
                       enemies: int = 0,
                       allies: int = 0):
        """
        Print compact game status dashboard

        Args:
            state: Current game state (MAIN_MENU, IN_BATTLE, etc.)
            game_num: Current game number
            hand: List of cards in hand
            elixir: Current elixir count
            recent_actions: List of recent actions taken
            enemies: Number of enemy units detected
            allies: Number of ally units detected
        """
        if self.verbose:
            # Verbose mode - don't use dashboard
            return

        # Clear previous dashboard
        self.clear_lines(self.last_lines)

        lines = []
        lines.append("=" * 70)
        lines.append(f" CLASH ROYALE RL AGENT | Game #{game_num} | {datetime.now().strftime('%H:%M:%S')}")
        lines.append("=" * 70)
        lines.append(f" State: {state:20s} | Elixir: {elixir if elixir is not None else '?':2} | Enemies: {enemies:2d} | Allies: {allies:2d}")
        lines.append("-" * 70)

        # Show hand
        lines.append(" Hand:")
        for slot_idx, card in enumerate(hand):
            if card:
                name = card['card_name'].replace('_', ' ').title()
                available = "✓" if card.get('available', True) else "✗"
                conf = card['confidence']
                lines.append(f"  [{slot_idx}] {available} {name:20s} (conf: {conf:.2f})")
            else:
                lines.append(f"  [{slot_idx}] - {'Empty':20s}")

        lines.append("-" * 70)

        # Show recent actions
        lines.append(" Recent Actions:")
        if recent_actions:
            for action in recent_actions[-5:]:  # Show last 5
                lines.append(f"  • {action}")
        else:
            lines.append("  (no actions yet)")

        lines.append("=" * 70)

        # Print all lines
        for line in lines:
            print(line)

        self.last_lines = len(lines)

    def log_action(self, message: str):
        """
        Log an action or event

        Args:
            message: Action description
        """
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
        # In non-verbose mode, actions are shown in dashboard

    def log_state_change(self, old_state: str, new_state: str):
        """Log state transition"""
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"STATE CHANGE: {old_state} → {new_state}")
            print(f"{'='*60}\n")

    def log_battle_result(self, result: str):
        """Log battle result prominently"""
        print(f"\n{'='*70}")
        print(f"                    BATTLE RESULT: {result.upper()}")
        print(f"{'='*70}\n")

    def log_error(self, message: str):
        """Log error message"""
        print(f"❌ ERROR: {message}")

    def log_detection(self, detections: List[Dict]):
        """
        Log battlefield detections (only in verbose mode)

        Args:
            detections: List of detected units
        """
        if not self.verbose:
            return

        if not detections:
            return

        allies = [d for d in detections if d['team'] == 'ally']
        enemies = [d for d in detections if d['team'] == 'enemy']

        if allies:
            print(f"\nAllies ({len(allies)}):")
            for d in allies:
                card_label = f"{d['team']} - {d['card_type']}"
                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {card_label:25s} [Grid: {grid_row:2d},{grid_col:2d}] | conf: {d['card_confidence']:.2f}")
                else:
                    print(f"  - {card_label:25s} at ({d['center'][0]:3d}, {d['center'][1]:3d}) | conf: {d['card_confidence']:.2f}")

        if enemies:
            print(f"\nEnemies ({len(enemies)}):")
            for d in enemies:
                card_label = f"{d['team']} - {d['card_type']}"
                if d['grid']:
                    grid_row, grid_col = d['grid']
                    print(f"  - {card_label:25s} [Grid: {grid_row:2d},{grid_col:2d}] | conf: {d['card_confidence']:.2f}")
                else:
                    print(f"  - {card_label:25s} at ({d['center'][0]:3d}, {d['center'][1]:3d}) | conf: {d['card_confidence']:.2f}")
