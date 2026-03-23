#!/usr/bin/env python3
"""
Test game end condition when no face-up helpings remain.
"""
import sys
sys.path.insert(0, '..')

from game_models import Helping, Player, GameState
from simulation import apply_turn_outcome, flip_highest_face_down

def test_game_end_condition():
    # Grill with one face-up tile
    grill = [Helping(21,1)]
    grill[0].face_up = True
    players = [Player(0), Player(1)]
    state = GameState(grill, players, current_player=0)

    # No top helping, failure with worm_delta = 0
    # apply_turn_outcome with worm_delta=0, helping_taken=None, taken_from=-2
    # This should flip the highest face-up tile (21) face-down
    apply_turn_outcome(state, 0, None, -2)

    # Now there should be zero face-up tiles
    face_up_count = sum(1 for h in grill if h.face_up)
    print(f"Face-up count after flipping last tile: {face_up_count}")
    if face_up_count != 0:
        print("BUG: Last face-up tile not turned face-down")
        return False

    # Check condition any(h.face_up for h in grill) should be False
    if any(h.face_up for h in grill):
        print("BUG: any(face_up) returned True")
        return False

    print("Test passed: game end condition works")
    return True

def test_flip_highest_face_down_no_face_up():
    # Grill with all face-down tiles
    grill = [Helping(21,1), Helping(22,1)]
    for h in grill:
        h.face_up = False
    # Should do nothing
    flip_highest_face_down(grill)
    # Still all face-down
    if any(h.face_up for h in grill):
        print("BUG: flip_highest_face_down changed face-up state when no face-up tiles")
        return False
    print("Test passed: flip_highest_face_down handles no face-up")
    return True

def test_flip_highest_face_down_excluded():
    # Grill with two face-up tiles
    grill = [Helping(30,3), Helping(25,2)]
    for h in grill:
        h.face_up = True
    # Exclude tile 30
    excluded = grill[0]
    flip_highest_face_down(grill, excluded_helping=excluded)
    # Highest is 30, excluded, so nothing should happen
    if not grill[0].face_up or not grill[1].face_up:
        print("BUG: flip_highest_face_down flipped when excluded helping is highest")
        return False
    # Now exclude tile 25 (lower), highest 30 should flip
    flip_highest_face_down(grill, excluded_helping=grill[1])
    if grill[0].face_up:
        print("BUG: highest tile not flipped when excluded is lower")
        return False
    if not grill[1].face_up:
        print("BUG: lower tile incorrectly flipped")
        return False
    print("Test passed: excluded helping works")
    return True

if __name__ == "__main__":
    success = True
    if not test_game_end_condition():
        success = False
    if not test_flip_highest_face_down_no_face_up():
        success = False
    if not test_flip_highest_face_down_excluded():
        success = False
    if success:
        print("\nAll game end tests passed")
    else:
        print("\nSome tests failed")
        sys.exit(1)