#!/usr/bin/env python3
"""
Test that penalty rule is implemented correctly:
- Return top helping to grill face-up
- Turn the highest face-up helping on the grill face-down (unless returned helping is highest)
"""
import sys
sys.path.insert(0, '..')

from game_models import Helping, Player, GameState
from simulation import apply_turn_outcome, resolve_turn

def test_penalty_highest_flipped():
    # Setup grill with three helpings: 21, 22, 23 all face-up
    grill = [Helping(21,1), Helping(22,1), Helping(23,1)]
    for h in grill:
        h.face_up = True
    # Player 0 has a top helping 24 (not on grill)
    player0 = Player(0)
    top = Helping(24,2)
    player0.add_helping(top)
    # Player 1 empty
    player1 = Player(1)
    state = GameState(grill, [player0, player1], current_player=0)

    print("Initial state:")
    print("Grill face-up:", [h.number for h in grill if h.face_up])
    print("Player 0 top:", player0.top_helping.number if player0.top_helping else None)

    # Simulate failure: no worm in collection, failure_reward returns negative worm delta
    # We'll manually call apply_turn_outcome with failure parameters
    worm_delta = -player0.top_helping.worms  # -2
    helping_taken = None
    taken_from = -2
    apply_turn_outcome(state, worm_delta, helping_taken, taken_from)

    print("\nAfter failure:")
    print("Grill face-up:", [h.number for h in grill if h.face_up])
    print("Player 0 top:", player0.top_helping.number if player0.top_helping else None)
    print("Grill contents:", [(h.number, h.face_up) for h in grill])

    # Expected: returned helping 24 added to grill face-up
    # Highest face-up helping on grill before penalty? Let's think:
    # Initial grill: 21,22,23 face-up
    # After returning: 24 added face-up
    # Highest face-up is 24 (returned helping), so it should stay face-up (rule)
    # Thus no tile should be turned face-down.
    # All four tiles remain face-up.
    face_up_count = sum(1 for h in grill if h.face_up)
    print(f"Face-up count: {face_up_count}")
    if face_up_count != 4:
        print("BUG: Some tile turned face-down incorrectly")
        return False
    # Ensure tile 24 is in grill
    if not any(h.number == 24 for h in grill):
        print("BUG: Returned tile not in grill")
        return False
    print("Test passed: returned helping is highest, no flip needed")
    return True

def test_penalty_highest_not_returned():
    # Grill: 21,22,30 face-up
    grill = [Helping(21,1), Helping(22,1), Helping(30,3)]
    for h in grill:
        h.face_up = True
    # Player 0 has top helping 25 (lower than 30)
    player0 = Player(0)
    top = Helping(25,2)
    player0.add_helping(top)
    player1 = Player(1)
    state = GameState(grill, [player0, player1], current_player=0)

    print("\n=== Test 2: returned helping not highest ===")
    print("Initial grill face-up:", [h.number for h in grill if h.face_up])
    print("Player 0 top:", top.number)

    worm_delta = -top.worms
    apply_turn_outcome(state, worm_delta, None, -2)

    print("After failure:")
    for h in grill:
        print(f"  Tile {h.number}: face_up={h.face_up}")

    # Expected: tile 30 (highest) turned face-down, tile 25 added face-up
    # So face-up tiles: 21,22,25 (30 face-down)
    face_up_numbers = [h.number for h in grill if h.face_up]
    print("Face-up numbers:", face_up_numbers)
    if 30 in face_up_numbers:
        print("BUG: Highest tile 30 not turned face-down")
        return False
    if 25 not in face_up_numbers:
        print("BUG: Returned tile 25 not face-up")
        return False
    print("Test passed: highest tile turned face-down")
    return True

if __name__ == "__main__":
    success = True
    if not test_penalty_highest_flipped():
        success = False
    if not test_penalty_highest_not_returned():
        success = False
    if success:
        print("\nAll tests passed")
    else:
        print("\nSome tests failed - penalty rule not implemented correctly")
        sys.exit(1)