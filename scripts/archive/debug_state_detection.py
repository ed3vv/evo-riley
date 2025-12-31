#!/usr/bin/env python3
"""
Debug state detection issues.
Shows what the detector sees and why it makes decisions.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from controllers.adb_controller import ADBController
from detection.image_matcher import ImageMatcher
from detection.state_detector import StateDetector, GameState
import cv2


def debug_state_detection():
    """Debug state detection by showing all checks"""
    print("=" * 80)
    print("STATE DETECTION DEBUGGER")
    print("=" * 80)
    print("This will help identify why the model thinks battle has ended.\n")

    # Initialize
    adb = ADBController(adb_port=5555)
    matcher = ImageMatcher()
    detector = StateDetector(matcher)

    if not adb.test_connection():
        print("❌ ADB connection failed. Is BlueStacks running?")
        return

    print("✅ Connected to BlueStacks\n")
    input("Go into a battle, then press ENTER to debug...")

    # Capture screenshot
    screenshot = adb.screenshot()
    if screenshot is None:
        print("❌ Failed to capture screenshot")
        return

    h, w = screenshot.shape[:2]
    print(f"Screenshot size: {w}x{h}\n")

    # Check each detection step
    print("=" * 80)
    print("DETECTION CHECKS (in order)")
    print("=" * 80)

    # 1. Elixir bar check
    print("\n1. ELIXIR BAR CHECK:")
    print("-" * 40)
    test_positions = [
        (67, 1255),
        (67, 1260),
        (200, 1255),
        (200, 1260),
        (350, 1255),
    ]

    elixir_detected = False
    for x, y in test_positions:
        if y >= h or x >= w:
            print(f"  ({x:3d}, {y:4d}) - OUT OF BOUNDS (screenshot is {w}x{h})")
            continue

        pixel = screenshot[y, x]
        b, g, r = int(pixel[0]), int(pixel[1]), int(pixel[2])
        is_pink = (r > 150 and g < 150 and b > 150)

        status = "✓ PINK DETECTED" if is_pink else "✗ not pink"
        print(f"  ({x:3d}, {y:4d}) - RGB({r:3d}, {g:3d}, {b:3d}) - {status}")

        if is_pink:
            elixir_detected = True

    print(f"\n  Result: {'✅ IN_BATTLE (elixir visible)' if elixir_detected else '❌ Elixir not detected'}")

    if elixir_detected:
        print("\n  ⚠️  If elixir is detected, state should be IN_BATTLE!")
        print("     The detection should STOP HERE and not check OK button.")

    # 2. Main menu check
    print("\n2. MAIN MENU CHECK:")
    print("-" * 40)
    main_menu = detector._check_main_menu(screenshot, verbose=True)
    print(f"  Result: {'✅ MAIN_MENU detected' if main_menu else '❌ Not main menu'}")

    # 3. OK button check
    print("\n3. OK BUTTON CHECK:")
    print("-" * 40)
    ok_button = matcher.find_template(screenshot, 'ok_button', threshold=0.85)
    if ok_button:
        x, y, conf = ok_button
        print(f"  ✅ OK button found at ({x}, {y}) - confidence: {conf:.2%}")
        print(f"     Current threshold: 0.92 (requires {0.92:.2%})")

        if conf >= 0.92:
            print(f"     ⚠️  WOULD TRIGGER BATTLE_END (conf >= 0.92)")
        else:
            print(f"     ✓ Below threshold, won't trigger")
    else:
        print(f"  ❌ OK button not found (threshold: 0.85 for detection)")

    # Try lower threshold to see if there's a weak match
    ok_button_weak = matcher.find_template(screenshot, 'ok_button', threshold=0.7)
    if ok_button_weak and not ok_button:
        x, y, conf = ok_button_weak
        print(f"  ⚠️  Weak OK button match at ({x}, {y}) - confidence: {conf:.2%}")
        print(f"     (Below 0.85 detection threshold, but shows potential false positive)")

    # 4. Queueing check
    print("\n4. QUEUEING CHECK:")
    print("-" * 40)
    queueing = detector._check_queueing_state(screenshot)
    print(f"  Result: {'✅ QUEUEING detected' if queueing else '❌ Not queueing'}")

    # Final state
    print("\n" + "=" * 80)
    print("FINAL STATE DETECTION")
    print("=" * 80)
    state = detector.detect_state(screenshot, verbose=True)
    print(f"\nFinal State: {state.name}")

    # Save debug screenshot
    print("\n" + "=" * 80)
    save = input("\nSave screenshot for debugging? (y/n): ").strip().lower()
    if save == 'y':
        filename = f"debug_false_battle_end.png"
        cv2.imwrite(filename, screenshot)
        print(f"✅ Saved to {filename}")

        # Also save with annotations
        annotated = screenshot.copy()

        # Draw elixir check positions
        for x, y in test_positions:
            if y < h and x < w:
                cv2.circle(annotated, (x, y), 10, (0, 255, 255), 2)  # Yellow circles

        # Draw OK button location if found
        if ok_button:
            x, y, conf = ok_button
            cv2.rectangle(annotated, (x-50, y-50), (x+50, y+50), (0, 0, 255), 3)  # Red box
            cv2.putText(annotated, f"OK {conf:.2%}", (x-40, y-60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        annotated_file = f"debug_annotated.png"
        cv2.imwrite(annotated_file, annotated)
        print(f"✅ Saved annotated version to {annotated_file}")


if __name__ == "__main__":
    debug_state_detection()
