#!/usr/bin/env python3
import sys
sys.path.insert(0, '..')
import random
from simulation import simulate_game

def smoke_test():
    random.seed(42)
    try:
        # Run a short game with quiet output
        winner = simulate_game(num_players=2, verbose=False, max_turns=50)
        print(f"Smoke test passed. Winner: {winner}")
        return True
    except Exception as e:
        print(f"Smoke test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = smoke_test()
    sys.exit(0 if success else 1)