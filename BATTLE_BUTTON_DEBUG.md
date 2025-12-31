# Battle Button Detection - Debugging Guide

## How Battle Button Detection Works

The system uses a **two-stage approach**:

### Stage 1: Main Menu Detection (State Detection)
Location: [detection/state_detector.py](detection/state_detector.py#L98-L186)

The state detector first determines if you're on the main menu by checking for:

1. **Yellow/Orange Battle Button** pixels (samples 9 positions)
   - Y positions: `h-250`, `h-200`, `h-150`
   - X positions: Center, Center-60, Center+60
   - Color criteria (BGR):
     ```python
     r > 100 and g > 80 and b < 150
     r > b and g > b - 30
     ```
   - **Threshold**: 3+ yellow pixels = main menu detected

2. **Bottom Navigation Bar** (samples 9 positions)
   - Y positions: `h-30`, `h-50`, `h-70`
   - X positions: `w//5`, `w//2`, `4*w//5`
   - Brightness criteria: `< 100` or `> 350`
   - **Threshold**: 4+ nav elements = main menu detected

**Result**: If either check passes, state = `MAIN_MENU`

### Stage 2: Battle Button Click (Template Matching)
Location: [controllers/game_controller.py](controllers/game_controller.py#L231-L238)

Once in MAIN_MENU state, the system tries to click the battle button:

```python
def click_battle_button(self) -> bool:
    return self.click_button('battle_button.png', threshold=0.6)
```

This uses **template matching** with:
- **Templates**: 5 variants (battle_button_1.png through battle_button_5.png)
- **Method**: OpenCV `cv2.matchTemplate` with `TM_CCOEFF_NORMED`
- **Threshold**: 0.6 (60% confidence minimum)
- **Matching**: Grayscale comparison of screenshot vs template

The system tries all 5 variants and picks the best match above threshold.

## Common Failure Scenarios

### 1. State Detection Fails (Main Menu Not Detected)

**Symptoms:**
- Never sees ">>> Clicking Battle button"
- Stuck in other state (QUEUEING, BATTLE_END, etc.)

**Causes:**
- Screen layout changed (different device resolution)
- New UI update from Supercell
- Different page on main menu (Shop, Cards, etc.)
- Notification popups blocking the battle button

**Debug:**
```bash
# Run with verbose mode to see detection details
python3 main.py --verbose

# Look for lines like:
[MAIN_MENU] Yellow pixels: 2/9, Nav elements: 3/9  # Too low!
```

### 2. Template Matching Fails (Button Not Found)

**Symptoms:**
- Sees ">>> Clicking Battle button"
- Then sees "❌ battle_button.png not found"

**Causes:**
- Battle button looks different (animation, glow effects)
- Different screen resolution
- Templates were captured at different zoom level
- Button is partially obscured

**Debug:**
```bash
# Visualize what the matcher sees
python3 -c "
from controllers.game_controller import GameController
gc = GameController()
gc.visualize_button('battle_button.png', 'debug_battle.png')
"
# Check debug_battle.png to see if button is highlighted
```

## Debugging Tools

### 1. Check Current State Detection

```python
# Create a test script: test_state.py
from detection.state_detector import StateDetector
from detection.image_matcher import ImageMatcher
from controllers.adb_controller import ADBController

adb = ADBController()
matcher = ImageMatcher()
detector = StateDetector(matcher)

screenshot = adb.screenshot()
state = detector.detect_state(screenshot, verbose=True)
print(f"\nDetected state: {state}")
```

Run it:
```bash
python3 test_state.py
```

### 2. Visualize Template Matching

```python
# Create a test script: test_template.py
from detection.image_matcher import ImageMatcher
from controllers.adb_controller import ADBController

adb = ADBController()
matcher = ImageMatcher()

screenshot = adb.screenshot()
result = matcher.visualize_match(
    screenshot,
    'battle_button.png',
    threshold=0.6,
    output_path='battle_match.png'
)

if result:
    print("✓ Match found! Check battle_match.png")
else:
    print("✗ No match found")
    # Try lower threshold
    matcher.visualize_match(screenshot, 'battle_button.png', threshold=0.4, output_path='battle_match_low.png')
    print("Tried with threshold=0.4, check battle_match_low.png")
```

Run it:
```bash
python3 test_template.py
```

### 3. Manual Check of Detection Regions

```python
# test_regions.py
import cv2
from controllers.adb_controller import ADBController

adb = ADBController()
screenshot = adb.screenshot()
h, w = screenshot.shape[:2]

# Draw detection regions
viz = screenshot.copy()

# Battle button region (yellow check)
for y in [h-250, h-200, h-150]:
    for x in [w//2 - 60, w//2, w//2 + 60]:
        cv2.circle(viz, (x, y), 5, (0, 255, 0), -1)

# Nav bar region
for y in [h-30, h-50, h-70]:
    for x in [w//5, w//2, 4*w//5]:
        cv2.circle(viz, (x, y), 5, (255, 0, 0), -1)

cv2.imwrite('detection_regions.png', viz)
print("Saved detection_regions.png - green = battle button, blue = nav bar")
```

### 4. Check Template Files

```bash
# View available templates
ls -lh detection/images/battle_button*.png

# Expected output:
# battle_button_1.png (13K)
# battle_button_2.png (13K)
# battle_button_3.png (13K)
# battle_button_4.png (47K)
# battle_button_5.png (49K)
```

## Fixing Detection Issues

### Fix 1: Adjust State Detection Thresholds

If main menu detection is too strict, edit [detection/state_detector.py](detection/state_detector.py):

```python
# Line 178 - make it more lenient
if yellow_count >= 2 or nav_detected >= 3:  # Changed from 3 and 4
    return True
```

### Fix 2: Lower Template Matching Threshold

If button is found but confidence is low, edit [controllers/game_controller.py](controllers/game_controller.py):

```python
# Line 238 - lower threshold
return self.click_button('battle_button.png', threshold=0.5)  # Changed from 0.6
```

### Fix 3: Add New Template Variant

If button looks different:

1. Take a screenshot when on main menu
2. Crop just the battle button region
3. Save as `detection/images/battle_button_6.png`
4. The system will automatically try all variants

```bash
# Quick way to capture and crop:
python3 -c "
from controllers.adb_controller import ADBController
import cv2

adb = ADBController()
screenshot = adb.screenshot()
cv2.imwrite('main_menu_screenshot.png', screenshot)
print('Saved! Now crop the battle button and save as battle_button_6.png')
"
```

### Fix 4: Adjust Color Detection for Battle Button

If button color changed, edit [detection/state_detector.py](detection/state_detector.py#L134-L140):

```python
# More lenient yellow detection
is_yellow = (
    r > 80 and         # Lower red threshold
    g > 60 and         # Lower green threshold
    b < 180 and        # Higher blue tolerance
    r > b and
    g > b - 50         # More tolerance
)
```

## Real-Time Debugging During Run

Add debug logging to see what's happening:

```python
# In main.py, add to handle_main_menu():
def handle_main_menu(self):
    print(">>> Clicking Battle button\n")

    # DEBUG: Check if button is actually detectable
    result = self.gc.find_button('battle_button.png', threshold=0.6)
    if result:
        x, y, conf = result
        print(f"[DEBUG] Battle button found at ({x}, {y}) with confidence {conf:.2f}")
    else:
        print(f"[DEBUG] Battle button NOT found - trying lower threshold")
        result = self.gc.find_button('battle_button.png', threshold=0.4)
        if result:
            x, y, conf = result
            print(f"[DEBUG] Found at threshold 0.4: ({x}, {y}) confidence {conf:.2f}")

    success = self.gc.click_battle_button()
    print(f"[DEBUG] Click result: {success}")
    time.sleep(5)
```

## Quick Diagnostic Checklist

When battle button fails to click:

- [ ] Is state detection working? (Check for `[STATE] Detected: MAIN_MENU`)
- [ ] Are you actually on the main menu? (Not on a popup or different screen)
- [ ] Does `battle_button.png` template exist in `detection/images/`?
- [ ] Is the confidence threshold too high? (Try lowering from 0.6 to 0.5)
- [ ] Is the button partially obscured? (Notifications, season info, etc.)
- [ ] Has the UI changed? (Game update, different device)
- [ ] Run visualization to see what the matcher sees

## Summary

**Detection Flow:**
1. State detector checks yellow pixels + nav bar → `MAIN_MENU`
2. If `MAIN_MENU`, call `click_battle_button()`
3. Template matching finds button in screenshot
4. Click at center of matched region

**Key Files:**
- State detection: `detection/state_detector.py` (lines 98-186)
- Template matching: `detection/image_matcher.py` (lines 60-111)
- Button clicking: `controllers/game_controller.py` (lines 184-238)
- Templates: `detection/images/battle_button_*.png`

**Common Fixes:**
- Lower thresholds (state: 3→2, template: 0.6→0.5)
- Add new template variant
- Adjust color detection ranges
- Check for UI obstructions
