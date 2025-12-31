# clash-royale-rl

NOT meant to be used to push trophies

This is a multi model clash royale RL bot meant for me to practice using ML, CV, RL, etc, not to push trophies

Includes 7 models:

model 1: yolov8 object detector 
- 2 classes: ally, enemy

model 2: yolov8 object classifier
- up to 120+ classes, all the cards in the game (currently not supporting spells)

model 3: elixir template matcher
- uses template matching to find user's elixir count
- 11 classes: 0-10

model 4: yolov8 card hand classifier
- takes screenshots of the hand (4 cards playable) and classifies them
- up to 120+ classes

model 5: state tracker
- takes screenshots of the entire screen to detect state (main menu, queueing, in game, etc.)

model 6: yolov8 Tower HP tracker
- tracks tower hp by taking screenshots, finding individual digits, concatenating them to form numbers, and setting as tower HP
- 11 classes: digits 0-9, and level class for detecting when only the tower level is visible, signifying unactivated king tower

model 7: RL model
- reward system based on win/loss and tower damage suffered/dealt.
- Full model is unable to be completed without first reaching 10k
