import time
import random
import argparse
import os
from datetime import datetime
import cv2
from controllers.game_controller import GameController
from detection.state_detector import StateDetector, GameState
from detection.card_detector import CardDetector
from detection.card_hand_detector import CardHandDetector
from detection.elixir_detector import ElixirDetector
from detection.battle_result_detector import BattleResultDetector
from vision.tower_hp_detector import TowerHPDetector
from rl.dqn_agent import DQNAgent
from rl.state_encoder import StateEncoder
from rl.reward_calculator import RewardCalculator
from utils.display import GameDisplay

class Agent:
    def __init__(self, instance_id=0, save_screenshots=False, use_model=False, use_rl=False, verbose=False):
        self.gc = GameController(instance_id)
        self.detector = StateDetector(self.gc.image_matcher)
        self.games_played = 0

        # Display system
        self.verbose = verbose
        self.display = GameDisplay(verbose=verbose)
        self.recent_actions = []
        self.last_state = None
        self.enemy_count = 0
        self.ally_count = 0

        # Screenshot saving setup
        self.save_screenshots = save_screenshots
        self.screenshot_interval = 5.0  # Fixed at 5 seconds
        self.last_save_time = 0
        self.screenshot_count = 0
        self.output_dir = None

        # Card hand detection
        self.hand_detector = CardHandDetector()
        self.current_hand = []

        # Elixir detection
        self.elixir_detector = ElixirDetector()
        self.current_elixir = None

        # Battle result detection
        self.result_detector = BattleResultDetector()
        self.battle_result = None  # 'victory', 'defeat', 'draw', or None

        # Battle state
        self.last_play_time = 0
        self.last_decision_elixir = None 

        # YOLO model setup
        self.use_model = use_model
        self.card_detector = None
        if self.use_model:
            print("Loading YOLO models...")
            self.card_detector = CardDetector(
                model_path="models/best.pt",  # Detection model (WHERE + ally/enemy)
                classifier_path="models/card_classifier.pt",  # Classification model (WHICH card)
                confidence_threshold=0.25,
                grid_system=self.gc.grid
            )
            print("Models loaded successfully!")

        # RL training setup
        self.use_rl = use_rl
        self.rl_agent = None
        self.state_encoder = None
        self.reward_calculator = None
        self.prev_state = None
        self.prev_action = None
        self.prev_state_vector = None

        # Tower HP detection (YOLOv8 digit detection)
        self.tower_hp_detector = None
        if self.use_rl or self.use_model:
            print("Loading Tower HP detector...")
            self.tower_hp_detector = TowerHPDetector(
                model_path="models/tower_hp.pt",
                confidence=0.3
            )
            print("Tower HP detector loaded!")

        self.cached_tower_hp = None
        self.last_tower_hp_update = 0
        self.TOWER_HP_UPDATE_INTERVAL = 0.5 

        # Tower HP history for detecting false positives (HP bar obstruction)
        self.tower_hp_history = {
            'enemy_left_princess': [],
            'enemy_king': [],
            'enemy_right_princess': [],
            'ally_left_princess': [],
            'ally_king': [],
            'ally_right_princess': []
        }
        self.TOWER_HP_HISTORY_LENGTH = 5  # Track last 5 readings

        # Initialize RL components for both --rl and --model modes
        if self.use_rl or self.use_model:
            mode_name = "RL Training System" if self.use_rl else "RL Inference System (--model)"
            print("\n" + "="*60)
            print(f"Initializing {mode_name}")
            print("="*60)

            # State encoder
            self.state_encoder = StateEncoder(grid_rows=32, grid_cols=18, max_cards=4)

            # Action space: 4 cards × 32 rows × 18 cols + 1 "do nothing" = 2305 actions

            self.action_size = 2305

            # RL agent
            if self.use_rl:
                # Training mode: enable exploration and learning
                self.rl_agent = DQNAgent(
                    state_size=self.state_encoder.state_size,
                    action_size=self.action_size,
                    learning_rate=0.0001,
                    gamma=0.99,
                    epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay=0.998,
                    buffer_size=10000,
                    batch_size=64
                )
            else:
                # Inference mode (--model): greedy only, no exploration
                self.rl_agent = DQNAgent(
                    state_size=self.state_encoder.state_size,
                    action_size=self.action_size,
                    learning_rate=0.0001,
                    gamma=0.99,
                    epsilon_start=0.0,  
                    epsilon_end=0.0,
                    epsilon_decay=1.0, 
                    buffer_size=10000,
                    batch_size=64
                )

            # Reward calculator (only used in --rl mode)
            if self.use_rl:
                self.reward_calculator = RewardCalculator(
                    tower_damage_reward=0.01,
                    tower_destroy_bonus=50.0,
                    elixir_advantage_reward=0.1,
                    win_reward=200.0,
                    loss_penalty=-200.0
                )

            # Try to load checkpoint
            checkpoint_path = "checkpoints/latest.pt"
            if os.path.exists(checkpoint_path):
                self.rl_agent.load_checkpoint(checkpoint_path)
                if self.use_model:
                    print("✓ Loaded RL weights for inference (greedy policy, no exploration)")
            else:
                if self.use_rl:
                    print("No checkpoint found - starting fresh")
                else:
                    print("⚠️  No checkpoint found - --model will use untrained network")

            print("="*60)
            if self.use_rl:
                print("RL Training Enabled!")
            else:
                print("RL Inference Enabled!")
            print(f"State size: {self.state_encoder.state_size}")
            print(f"Action size: {self.action_size}")
            print(f"Epsilon: {self.rl_agent.epsilon:.3f}")
            print("="*60 + "\n")

        if self.save_screenshots:
            self.output_dir = self._create_output_dir()

    def _create_output_dir(self):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = f"training_data/session_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        # Create subdirectory for unclassified card crops
        self.unclassified_cards_dir = os.path.join(output_dir, "unclassified_cards")
        os.makedirs(self.unclassified_cards_dir, exist_ok=True)

        print(f"Saving screenshots to: {output_dir}")
        print(f"Unclassified cards will be saved to: {self.unclassified_cards_dir}")
        # Initialize last_save_time to allow immediate first screenshot
        self.last_save_time = 0
        self.card_save_count = 0
        return output_dir

    def _save_screenshot(self, screenshot, state):
        self.screenshot_count += 1
        filename = f"screenshot_{self.screenshot_count:04d}_{state.name}.png"
        filepath = os.path.join(self.output_dir, filename)
        cv2.imwrite(filepath, screenshot)
        # print(f"Saved: {filename}")
        # ^ use this only for debugging

    def _save_unclassified_cards(self, screenshot, hand, confidence_threshold=0.7):
        """
        Save card crops that couldn't be classified with high confidence for manual labeling

        Args:
            screenshot: Full game screenshot
            hand: List of card detection results from CardHandDetector
            confidence_threshold: Save cards below this confidence (default: 0.7)
        """
        for slot_idx, card_info in enumerate(hand):
            # Save if card couldn't be detected or confidence is low
            if card_info is None or card_info['confidence'] < confidence_threshold:
                # Extract card crop from screenshot
                x1, y1, x2, y2 = self.hand_detector.CARD_SLOTS[slot_idx]
                card_crop = screenshot[y1:y2, x1:x2]

                # Generate filename
                self.card_save_count += 1
                card_name = card_info['card_name'] if card_info else 'unknown'
                conf = card_info['confidence'] if card_info else 0.0
                filename = f"card_{self.card_save_count:04d}_slot{slot_idx}_{card_name}_conf{conf:.2f}.png"
                filepath = os.path.join(self.unclassified_cards_dir, filename)

                # Save crop
                cv2.imwrite(filepath, card_crop)
                print(f"[CARD] Saved low-confidence card: {filename}")

    def play_games(self, num_games=1):
        if self.verbose:
            print(f"Agent is now playing {num_games} games...")
        else:
            self.display.clear_screen()

        while self.games_played < num_games:
            screenshot = self.gc.take_screenshot()

            if screenshot is None:
                continue

            state = self.detector.detect_state(screenshot, verbose=self.verbose)
            state_name = state.name

            # Track state changes
            if state_name != self.last_state and self.last_state is not None:
                # Print state change
                print(f"\n>>> STATE: {self.last_state} → {state_name}\n")
            self.last_state = state_name

            # Save screenshot if enabled and in battle
            if self.save_screenshots and state == GameState.IN_BATTLE:
                current_time = time.time()
                if current_time - self.last_save_time >= self.screenshot_interval:
                    self._save_screenshot(screenshot, state)
                    hand = self.hand_detector.get_hand(screenshot, verbose=False)
                    self._save_unclassified_cards(screenshot, hand, confidence_threshold=0.9)
                    self.last_save_time = current_time

            # Handle state
            if state == GameState.MAIN_MENU:
                self.handle_main_menu()
            elif state == GameState.IN_BATTLE:
                self.handle_battle(screenshot)
            elif state == GameState.BATTLE_END:
                self.handle_battle_end()
            elif state == GameState.QUEUEING:
                time.sleep(2)
            else:
                # UNKNOWN or LOADING state
                time.sleep(1)
    
    def _add_action(self, action: str):
        """Add action to recent actions list"""
        self.recent_actions.append(action)
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        # Don't use display.log_action - we print actions directly

    def handle_main_menu(self):
        print(">>> Clicking Battle button\n")
        self.gc.click_battle_button()
        time.sleep(5)
    
    def handle_battle(self, screenshot):
        # Check for battle result ONLY if elixir bar is NOT visible
        # This prevents false positives during active battle
        elixir_visible = self.detector._check_elixir_bar_visible(screenshot)

        if not elixir_visible:
            # Battle might be ending, check for result screen
            result = self.result_detector.detect_result(screenshot, threshold=0.92, verbose=self.verbose)
            if result is not None:
                self.battle_result = result
                self.display.log_battle_result(result)
                return

        # If RL mode OR model mode is enabled, use RL agent
        if self.use_rl or self.use_model:
            self._handle_battle_rl(screenshot)
            return

        # If model is not enabled, just wait (let user play manually)
        if not self.use_model:
            time.sleep(0.5)  # Just monitor, don't play
            return

        # Check cooldown before playing
        current_time = time.time()
        time_since_last_play = current_time - self.last_play_time
        if time_since_last_play < self.play_cooldown:
            return

        # Print action
        print(f"\n>>> ACTION: Play {card_name} (slot {card_slot}) at grid [{row},{col}]\n")

        action = f"Play {card_name} at [{row},{col}]"
        self._add_action(action)

        self.gc.play_card(card_slot, row, col)

        # Update last play time and decision elixir AFTER successfully playing
        self.last_play_time = current_time

        # Update last decision elixir to current elixir AFTER playing
        # This ensures we only update when we actually execute a play
        if self.current_elixir is not None:
            self.last_decision_elixir = self.current_elixir

        # Small delay after playing to prevent duplicate plays (card animation time)
        time.sleep(0.3)

    def _handle_battle_rl(self, screenshot):
        """
        RL-enabled battle handler with tower HP tracking and learning
        """
        # Check for battle result ONLY if elixir bar is NOT visible
        # This prevents false positives during active battle
        elixir_visible = self.detector._check_elixir_bar_visible(screenshot)

        if not elixir_visible:
            # Battle might be ending, check for result screen
            result = self.result_detector.detect_result(screenshot, threshold=0.92, verbose=self.verbose)
            if result is not None:
                self.battle_result = result
                self.display.log_battle_result(result)
                return

        # Detect all game state components
        current_time = time.time()

        # 1. Elixir (fast)
        elixir = self.elixir_detector.get_elixir(screenshot, verbose=False)
        if elixir is None:
            elixir = 5.0  # Default if detection fails

        # 2. Hand (fast)
        hand = self.hand_detector.get_hand(screenshot, verbose=False)

        # 3. Tower HP (throttled - YOLOv8 digit detection)
        if current_time - self.last_tower_hp_update > self.TOWER_HP_UPDATE_INTERVAL:
            if self.tower_hp_detector:
                self.cached_tower_hp = self.tower_hp_detector.detect_tower_hp(screenshot)
                self.last_tower_hp_update = current_time

        # Use cached tower HP, or set defaults for princess towers only
        # King towers should remain None until activated and detected
        if self.cached_tower_hp is not None:
            tower_hp = self.cached_tower_hp
        else:
            tower_hp = {
                'enemy_left_princess': 1768,
                'enemy_king': None,  # Don't assume - detect when activated
                'enemy_right_princess': 1768,
                'ally_left_princess': 1768,
                'ally_king': None,  # Don't assume - detect when activated
                'ally_right_princess': 1768,
            }

        # 4. Troop detections (if YOLO enabled)
        all_detections = []
        enemy_detections = []
        ally_detections = []

        if self.card_detector:
            all_detections = self.card_detector.detect(screenshot, verbose=False)
            enemy_detections = [d for d in all_detections if d['class_name'].startswith('enemy')]
            ally_detections = [d for d in all_detections if d['class_name'].startswith('ally')]

        # 5. Encode current state
        current_state_vector = self.state_encoder.encode_state(
            elixir=elixir,
            hand=hand,
            enemy_detections=enemy_detections,
            ally_detections=ally_detections,
            tower_hp=tower_hp
        )

        # Build state dict for reward calculation
        current_state_dict = {
            'tower_hp': tower_hp,
            'elixir': elixir,
            'enemy_troops': enemy_detections,
            'ally_troops': ally_detections,
            'battle_result': self.battle_result
        }

        # 5. Update tower HP history
        self._update_tower_hp_history(tower_hp)

        # 6. Calculate reward and train (only in --rl mode, skip in --model inference)
        if self.use_rl and self.prev_state is not None and self.prev_action is not None:
            reward = self.reward_calculator.calculate_step_reward(
                prev_state=self.prev_state,
                curr_state=current_state_dict,
                action_taken=self.prev_action
            )

            # Store transition in replay buffer
            done = self.battle_result is not None
            self.rl_agent.store_transition(
                state=self.prev_state_vector,
                action=self.prev_action,
                reward=reward,
                next_state=current_state_vector,
                done=done
            )

            # Train agent
            loss = self.rl_agent.train_step()

            # Print stats periodically
            if self.rl_agent.steps % 100 == 0:
                stats = self.rl_agent.get_stats()
                if self.verbose:
                    print(f"\n[RL STATS] Steps: {stats['steps']}, "
                          f"Epsilon: {stats['epsilon']:.3f}, "
                          f"Avg Reward: {stats['avg_reward_100']:.2f}, "
                          f"Avg Loss: {stats['avg_loss_100']:.4f}")

        # 7. Get valid actions
        valid_actions = self._get_valid_actions(hand, elixir)

        if len(valid_actions) == 0:
            # No valid actions - wait
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            time.sleep(0.1)
            return

        # 8. Select action using RL agent
        action = self.rl_agent.select_action(current_state_vector, valid_actions)

        # 9. Decode action to (card_slot, row, col)
        card_slot, row, col = self._decode_action(action)

        # 10. Check if action is "do nothing"
        if card_slot is None:
            # Do nothing action - just wait and observe
            # Still store this as prev_state for learning
            self.prev_state = current_state_dict
            self.prev_action = action
            self.prev_state_vector = current_state_vector
            self.current_hand = hand
            self.current_elixir = elixir
            return

        # Get card info and check elixir
        card_info = hand[card_slot] if card_slot < len(hand) else None
        if card_info:
            from detection.card_info import get_card_elixir
            card_name = card_info.get('card_name', 'unknown')

            # Don't play "empty" cards
            if card_name == 'empty':
                self.prev_state = None
                self.prev_action = None
                self.prev_state_vector = None
                return

            elixir_cost = get_card_elixir(card_name)

            # Check if we have enough elixir
            if elixir < elixir_cost:
                print(f"  ⏸️  Not enough elixir for {card_name} (need {elixir_cost}, have {elixir:.0f})")
                self.prev_state = None
                self.prev_action = None
                self.prev_state_vector = None
                return

        # Check cooldown before playing
        current_time = time.time()
        time_since_last_play = current_time - self.last_play_time
        if time_since_last_play < self.play_cooldown:
            remaining = self.play_cooldown - time_since_last_play
            print(f"  ⏸️  Card on cooldown - wait {remaining:.1f}s")
            # Don't execute action, but keep state for next step
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            return

        # 11. Execute action
        if card_info:
            mode = "Inference" if self.use_model else "RL"
            action_msg = f"{mode} Play {card_name} ({elixir:.0f}/{elixir_cost} elixir) at Grid[{row},{col}] (ε={self.rl_agent.epsilon:.2f})"
            self._add_action(action_msg)

        self.gc.play_card(card_slot, row, col)

        # Update last play time
        self.last_play_time = current_time

        # 11. Save state for next step
        self.prev_state = current_state_dict
        self.prev_state_vector = current_state_vector
        self.prev_action = action

        # Small delay to prevent duplicate plays
        time.sleep(0.5)

    def _get_valid_actions(self, hand, elixir):
        """
        Get list of valid action indices based on available cards and elixir
        """
        from detection.card_info import get_card_elixir

        valid_actions = []

        for card_slot in range(4):
            if card_slot >= len(hand) or hand[card_slot] is None:
                continue

            card = hand[card_slot]
            card_name = card.get('card_name', 'unknown')
            available = card.get('available', True)

            # Skip empty card slots
            if card_name == 'empty':
                continue

            if not available:
                continue

            elixir_cost = get_card_elixir(card_name)
            if elixir < elixir_cost:
                continue

            # Card is playable - add all positions as valid actions
            for row in range(16, 32):  # Only our territory
                for col in range(18):
                    action = self._encode_action(card_slot, row, col)
                    valid_actions.append(action)

        return valid_actions

    def _encode_action(self, card_slot, row, col):
        """Encode (card_slot, row, col) to single action index"""
        return card_slot * (32 * 18) + row * 18 + col

    def _decode_action(self, action):
        """
        Decode action index to (card_slot, row, col)

        Returns:
            (None, None, None) if action is "do nothing"
            (card_slot, row, col) otherwise
        """
        # Action 2304 = "do nothing"
        if action == 2304:
            return None, None, None

        # Actions 0-2303 = play card at position
        card_slot = action // (32 * 18)
        remainder = action % (32 * 18)
        row = remainder // 18
        col = remainder % 18
        return card_slot, row, col

    def _update_tower_hp_history(self, tower_hp: dict):
        """
        Update tower HP history for each tower

        Args:
            tower_hp: Dict of current tower HP values
        """
        for tower_name, hp_value in tower_hp.items():
            if tower_name in self.tower_hp_history:
                # Add current HP to history
                self.tower_hp_history[tower_name].append(hp_value)

                # Keep only last N readings
                if len(self.tower_hp_history[tower_name]) > self.TOWER_HP_HISTORY_LENGTH:
                    self.tower_hp_history[tower_name].pop(0)

    def _print_game_state(self, enemy_detections, ally_detections):
        """Print game state information relevant for RL model"""
        # Clear previous output (move cursor up and clear lines)
        print("\033[2J\033[H", end="")  # Clear screen and move to top

        # Elixir
        elixir = self.current_elixir if self.current_elixir is not None else 0

        # Hand cards (names only)
        hand = [card['card_name'] if card else 'empty' for card in self.current_hand]

        # Enemy troops (specific card name and grid position)
        enemies = []
        for d in enemy_detections:
            card_name = d.get('card_type', 'unknown')  # Get specific card like 'giant', 'wizard'
            grid = d.get('grid', (0, 0))
            enemies.append(f"{card_name}@{grid}")

        # Ally troops (specific card name and grid position)
        allies = []
        for d in ally_detections:
            card_name = d.get('card_type', 'unknown')  # Get specific card like 'giant', 'wizard'
            grid = d.get('grid', (0, 0))
            allies.append(f"{card_name}@{grid}")

        # Tower HP (if available)
        tower_hp = None
        if hasattr(self, 'tower_tracker') and self.tower_tracker:
            tower_hp = self.tower_tracker.get_tower_hp()

        # Print formatted state
        print("=" * 80)
        print("GAME STATE")
        print("=" * 80)
        print(f"Elixir: {elixir}")
        print(f"Hand:   [{', '.join(hand)}]")

        if tower_hp:
            print(f"Towers: Left={tower_hp['left_tower']}% | King={tower_hp['king_tower']}% | Right={tower_hp['right_tower']}%")

        print(f"\nEnemies ({len(enemies)}):")
        if enemies:
            for i, enemy in enumerate(enemies[:5]):  # Show first 5
                print(f"  {enemy}")
            if len(enemies) > 5:
                print(f"  ... and {len(enemies) - 5} more")
        else:
            print("  None")

        print(f"\nAllies ({len(allies)}):")
        if allies:
            for i, ally in enumerate(allies[:5]):  # Show first 5
                print(f"  {ally}")
            if len(allies) > 5:
                print(f"  ... and {len(allies) - 5} more")
        else:
            print("  None")

        print("=" * 80)

    def _select_best_card(self, enemy_detections):
        """
        Select a random playable card

        Returns:
            (card_slot, card_info) tuple, or (None, None) if no playable card
        """
        from detection.card_info import CARD_INFO

        # Build list of playable cards
        playable_cards = []

        for slot_idx, card in enumerate(self.current_hand):
            if card is None:
                continue

            card_name = card['card_name']
            available = card.get('available', True)

            # Skip if card is on cooldown
            if not available:
                continue

            # Get card stats from CARD_INFO
            card_stats = CARD_INFO.get(card_name, {})
            elixir_cost = card_stats.get('elixir_cost', 10)  # Default high cost if unknown

            # Skip if not enough elixir
            if self.current_elixir is not None and self.current_elixir < elixir_cost:
                continue

            # Add elixir cost to card info
            card_with_stats = card.copy()
            card_with_stats['elixir_cost'] = elixir_cost

            playable_cards.append((slot_idx, card_with_stats))

        if not playable_cards:
            return None, None

        # Pick a random playable card
        selected_slot, selected_card = random.choice(playable_cards)

        print(f"  ✅ Randomly selected: Slot {selected_slot} ({selected_card['card_name']}, {selected_card['elixir_cost']} elixir)")

        return selected_slot, selected_card

    def _get_smart_placement(self, card_info, enemy_detections):
        """
        Random placement in our territory for all cards

        Returns:
            (row, col) tuple
        """
        # Random placement in our territory
        row = random.randint(16, 31)  # Our territory (bottom half)
        col = random.randint(0, 17)   # Any column
        return row, col

    def handle_battle_end(self):
        # Check for battle result if not already detected
        if self.battle_result is None:
            screenshot = self.gc.adb.screenshot()
            if screenshot is not None:
                result = self.result_detector.detect_result(screenshot, threshold=0.92, verbose=self.verbose)
                if result is not None:
                    self.battle_result = result
                    self.display.log_battle_result(result)

        # Log battle result
        if self.battle_result is not None:
            self._add_action(f"Battle result: {self.battle_result.upper()}")

            # RL: Apply final reward for battle outcome
            if self.use_rl and self.rl_agent is not None:
                # Calculate outcome reward
                if self.battle_result == 'victory':
                    outcome_reward = 200.0
                    print(f"\n[BATTLE END] 🏆 VICTORY detected! Final reward: +{outcome_reward:.2f}")
                elif self.battle_result == 'defeat':
                    outcome_reward = -200.0
                    print(f"\n[BATTLE END] ☠️  DEFEAT detected! Final penalty: {outcome_reward:.2f}")
                elif self.battle_result == 'draw':
                    outcome_reward = 0.0
                    print(f"\n[BATTLE END] 🤝 DRAW detected! Final reward: {outcome_reward:.2f}")
                else:
                    outcome_reward = 0.0

                # Store final transition with outcome reward (if we have previous state)
                if outcome_reward != 0 and self.prev_state is not None:
                    # Create final state (battle ended)
                    final_state_vector = self.state_encoder.encode_state(
                        elixir=self.current_elixir if self.current_elixir else 0,
                        hand=self.current_hand if self.current_hand else [None, None, None, None],
                        enemy_detections=[],
                        ally_detections=[],
                        tower_hp=self.cached_tower_hp if self.cached_tower_hp else {}
                    )

                    self.rl_agent.store_transition(
                        state=self.prev_state_vector,
                        action=self.prev_action,
                        reward=outcome_reward,
                        next_state=final_state_vector,
                        done=True
                    )
                    print(f"[DEBUG] Victory/defeat reward stored in replay buffer (+{outcome_reward:.2f} to total_reward)")
                elif outcome_reward != 0:
                    # No previous state - just add to total reward directly
                    print(f"[DEBUG] No prev_state, adding outcome reward directly to total_reward")
                    self.rl_agent.total_reward += outcome_reward

        # RL: End episode and save checkpoint
        if self.use_rl and self.rl_agent is not None:
            # Mark episode as done
            self.rl_agent.end_episode()

            # Save checkpoint every 5 games
            if self.games_played % 1 == 0:
                checkpoint_path = f"checkpoints/checkpoint_game_{self.games_played}.pt"
                self.rl_agent.save_checkpoint(checkpoint_path)

                # Also save as "latest"
                self.rl_agent.save_checkpoint("checkpoints/latest.pt")

            # Print episode stats
            stats = self.rl_agent.get_stats()
            print(f"\n{'='*70}")
            print(f"RL EPISODE {stats['episodes']} END")
            print(f"{'='*70}")
            print(f"  This Game Reward:  {stats['last_episode_reward']:+.2f}")
            print(f"  Avg Reward (last 100 games): {stats['avg_reward_100']:+.2f}")
            print(f"  Epsilon: {stats['epsilon']:.3f} ({'exploring' if stats['epsilon'] > 0.5 else 'exploiting'})")
            print(f"  Buffer: {stats['buffer_size']}/{10000} ({stats['buffer_size']/100:.0f}%)")
            print(f"{'='*70}\n")

            # Reset state for next episode
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            self.cached_tower_hp = None

            # Reset tower HP detector for next battle
            if self.tower_hp_detector:
                self.tower_hp_detector.reset_tracking()

            # Reset tower HP history for next battle
            for tower_name in self.tower_hp_history:
                self.tower_hp_history[tower_name] = []

        # Reset state for --model mode (inference only, no training)
        if self.use_model and not self.use_rl:
            self.prev_state = None
            self.prev_action = None
            self.prev_state_vector = None
            self.cached_tower_hp = None

            # Reset tower HP detector for next battle
            if self.tower_hp_detector:
                self.tower_hp_detector.reset_tracking()

            # Reset tower HP history for next battle
            for tower_name in self.tower_hp_history:
                self.tower_hp_history[tower_name] = []

        # Reset decision tracking for next battle (--model mode)
        self.last_decision_elixir = None

        self._add_action("Clicking OK button to finish game")
        self.gc.click_ok_button()
        self.games_played += 1

        # Reset battle result for next game
        self.battle_result = None

        time.sleep(3)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clash Royale RL Agent")
    parser.add_argument("--instance", type=int, default=0, help="Bluestacks Instance ID")
    parser.add_argument("--games", type=int, default=1, help="Number of games to play")
    parser.add_argument("--screenshots", action="store_true", help="Save screenshots every 5 seconds for training")
    parser.add_argument("--model", action="store_true", help="Use YOLO model to detect troops and print detections")
    parser.add_argument("--rl", action="store_true", help="Enable RL training mode with DQN agent")
    parser.add_argument("--checkpoint", type=str, default=None, help="Load training from checkpoint file (e.g., checkpoints/latest.pt)")
    parser.add_argument("--verbose", action="store_true", help="Show all debug output (default: clean dashboard)")

    args = parser.parse_args()

    agent = Agent(
        instance_id=args.instance,
        save_screenshots=args.screenshots,
        use_model=args.model or args.rl,  # RL requires model for troop detection
        use_rl=args.rl,
        verbose=args.verbose
    )

    # Load checkpoint if specified
    if args.checkpoint and args.rl:
        if agent.rl_agent:
            agent.rl_agent.load_checkpoint(args.checkpoint)
        else:
            print(f"⚠️  Warning: --checkpoint specified but RL mode not enabled")

    agent.play_games(num_games=args.games)
