#!/usr/bin/env python3
"""Quick smoke test for MonteCarloStrategy."""

import sys
sys.path.insert(0, '..')

from game_models import GameState, Helping, Player
from strategies import MonteCarloStrategy
import random

def create_test_state(num_players=3):
    """Create a simple game state for testing."""
    # Create grill with all helpings face-up
    grill = [Helping(num, worms) for num, worms in [
        (21,1),(22,1),(23,1),(24,1),(25,2),(26,2),(27,2),(28,2),
        (29,3),(30,3),(31,3),(32,3),(33,4),(34,4),(35,4),(36,4)
    ]]
    # All tiles face-up
    for h in grill:
        h.face_up = True

    players = [Player(i) for i in range(num_players)]
    # Give each player a random helping from grill
    for p in players:
        if grill:
            h = grill.pop()
            p.add_helping(h)

    current_player = 0
    return GameState(grill, players, current_player)

def main():
    print("Creating test state...")
    state = create_test_state(num_players=3)
    print(f"Grill has {len(state.visible_grill())} face-up tiles")

    # Create MonteCarloStrategy with default parameters
    print("Creating MonteCarloStrategy...")
    strategy = MonteCarloStrategy(state, player_id=0, num_simulations=50, time_limit_ms=500)

    # Test should_stop with empty collection
    collection = (0,0,0,0,0,0)
    print("Testing should_stop with empty collection...")
    stop = strategy.should_stop(collection)
    print(f"Should stop: {stop}")

    # Test choose_symbol with a random roll
    roll_counts = (0,0,1,0,0,0)  # one '3' symbol
    symbol = strategy.choose_symbol(collection, roll_counts)
    print(f"Choose symbol from roll {roll_counts}: {symbol}")

    # Test with a non-empty collection
    collection2 = (0,0,1,0,0,0)  # one '3' taken
    stop2 = strategy.should_stop(collection2)
    print(f"Should stop with collection {collection2}: {stop2}")

    print("Smoke test completed.")

if __name__ == "__main__":
    main()