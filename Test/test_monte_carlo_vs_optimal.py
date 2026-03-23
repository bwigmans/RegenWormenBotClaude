#!/usr/bin/env python3
"""Test MonteCarloStrategy against OptimalExpectedValueStrategy."""

import sys
import time
import random
from typing import List, Tuple

sys.path.insert(0, '..')

from game_models import Helping, Player, GameState, EnhancedGameState
from strategies import MonteCarloStrategy, OptimalExpectedValueStrategy, create_strategy
from simulation import simulate_game

def run_single_game(player0_strategy: str, player0_params: dict,
                    player1_strategy: str, player1_params: dict,
                    seed: int = None,
                    verbose: bool = False) -> int:
    """Run a single game between two strategies.

    Returns:
        Winner player index (0 or 1)
    """
    if seed is not None:
        random.seed(seed)

    strategy_configs = [
        {"name": player0_strategy, "params": player0_params},
        {"name": player1_strategy, "params": player1_params}
    ]

    try:
        result = simulate_game(num_players=2, verbose=verbose,
                               max_turns=200, strategy_configs=strategy_configs)
        if isinstance(result, tuple):
            winner = result[0]
            if isinstance(winner, list):
                # Tie, pick random winner for simplicity
                return random.choice(winner)
            return winner
        else:
            return result
    except Exception as e:
        print(f"Error simulating game: {e}")
        import traceback
        traceback.print_exc()
        return -1

def run_match(num_games: int = 10,
              monte_carlo_params: dict = None,
              opponent_strategy: str = "optimal_expected",
              opponent_params: dict = None,
              seed: int = 42) -> Tuple[int, int]:
    """Run a match between MonteCarlo and another strategy.

    Returns:
        Tuple of (monte_carlo_wins, opponent_wins)
    """
    if monte_carlo_params is None:
        monte_carlo_params = {"num_simulations": 50, "time_limit_ms": 500}
    if opponent_params is None:
        opponent_params = {}

    random.seed(seed)
    seeds = [random.randint(0, 1000000) for _ in range(num_games)]

    mc_wins = 0
    opp_wins = 0

    for i, game_seed in enumerate(seeds):
        print(f"Game {i+1}/{num_games} (seed={game_seed})...")
        start_time = time.time()
        winner = run_single_game(
            "monte_carlo", monte_carlo_params,
            opponent_strategy, opponent_params,
            seed=game_seed, verbose=False
        )
        elapsed = time.time() - start_time
        print(f"  Winner: Player {winner}, time: {elapsed:.1f}s")

        if winner == 0:
            mc_wins += 1
        elif winner == 1:
            opp_wins += 1

    return mc_wins, opp_wins

def test_quick_decisions():
    """Test that MonteCarlo makes decisions within time limits."""
    print("\n=== Testing MonteCarlo decision times ===")

    # Create a simple game state
    from dice_utils import ROLL_OUTCOMES
    grill = [Helping(num, worms) for num, worms in [
        (21,1),(22,1),(23,1),(24,1),(25,2),(26,2),(27,2),(28,2),
        (29,3),(30,3),(31,3),(32,3),(33,4),(34,4),(35,4),(36,4)
    ]]
    for h in grill:
        h.face_up = True

    players = [Player(0), Player(1)]
    players[0].add_helping(grill.pop())
    players[1].add_helping(grill.pop())
    state = EnhancedGameState(grill, players, current_player=0)

    # Create MonteCarloStrategy with low simulation count for speed
    strategy = MonteCarloStrategy(state, player_id=0,
                                  num_simulations=20, time_limit_ms=1000)

    # Test should_stop with various collections
    test_collections = [
        (0,0,0,0,0,0),
        (0,0,1,0,0,0),
        (0,0,0,0,0,1),
        (1,1,1,1,1,1)
    ]

    for coll in test_collections:
        print(f"Testing collection {coll}...")
        start = time.time()
        stop = strategy.should_stop(coll)
        elapsed = (time.time() - start) * 1000
        print(f"  should_stop: {stop}, time: {elapsed:.1f}ms")

        # Test choose_symbol with a simple roll
        roll = (0,0,1,0,0,0)  # one '3'
        start = time.time()
        symbol = strategy.choose_symbol(coll, roll)
        elapsed = (time.time() - start) * 1000
        print(f"  choose_symbol: {symbol}, time: {elapsed:.1f}ms")

def main():
    print("=== MonteCarloStrategy Test Suite ===\n")

    # Test 1: Quick decision timing
    test_quick_decisions()

    # Test 2: Small match vs optimal_expected
    print("\n=== Match: MonteCarlo vs OptimalExpected (10 games) ===")
    print("MonteCarlo params: num_simulations=50, time_limit_ms=500")
    mc_wins, opt_wins = run_match(num_games=10)
    print(f"Results: MonteCarlo {mc_wins} - {opt_wins} OptimalExpected")

    # Test 3: Match vs aggressive (if user wants)
    print("\n=== Match: MonteCarlo vs Aggressive (5 games) ===")
    mc_wins, agg_wins = run_match(num_games=5, opponent_strategy="aggressive")
    print(f"Results: MonteCarlo {mc_wins} - {agg_wins} Aggressive")

    print("\n=== Test completed ===")

if __name__ == "__main__":
    main()