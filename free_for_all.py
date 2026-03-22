#!/usr/bin/env python3
"""
Free-for-all tournament with all available Regenwormen strategies.
Runs 100 games with 6 strategies competing simultaneously.
"""

import sys
import random
import time
from typing import List, Dict, Tuple, Any

from game_models import Helping, Player, EnhancedGameState
from dice_utils import collection_sum, has_worm
from strategies import create_strategy, STRATEGY_REGISTRY
from simulation import (
    initial_grill, simulate_turn, apply_turn_outcome,
    player_worms, random_roll
)

# Configuration
RANDOM_SEED = 42   # for reproducibility
MAX_TURNS = 1000   # max turns per game (safety)
NUM_GAMES = 100    # number of games to run

def run_game(strategy_configs: List[dict], seed: int = None) -> Tuple[Any, List[int]]:
    """
    Run a single game between multiple strategies.

    This is adapted from tournament.py's run_game function.

    Args:
        strategy_configs: List of strategy config dicts, one per player.
        seed: Random seed for reproducible game.

    Returns:
        Tuple (winner, scores):
        - winner: int (player index) or list of ints if tie
        - scores: list of worm counts for each player
    """
    if seed is not None:
        random.seed(seed)

    num_players = len(strategy_configs)
    grill = initial_grill()
    players = [Player(i) for i in range(num_players)]
    current_player = random.randrange(num_players)
    turn = 0

    # Game loop
    while grill and any(h.face_up for h in grill) and turn < MAX_TURNS:
        turn += 1

        # Create game state for this turn
        state = EnhancedGameState(grill, players, current_player)
        config = strategy_configs[current_player]
        policy = create_strategy(
            config.get("name", "optimal_expected"),
            state, current_player,
            **config.get("params", {})
        )

        # Simulate the turn
        worm_delta, helping_taken, taken_from = simulate_turn(
            state, policy, current_player, commentator=None
        )

        # Apply outcome
        apply_turn_outcome(state, worm_delta, helping_taken, taken_from)

        # Next player
        current_player = (current_player + 1) % num_players

    if turn >= MAX_TURNS:
        # Game exceeded turn limit - rare but handle gracefully
        print(f"Warning: Game exceeded {MAX_TURNS} turn limit")

    # Game over: compute scores and determine winner
    scores = [player_worms(p) for p in players]
    max_score = max(scores)
    winners = [i for i, s in enumerate(scores) if s == max_score]

    if len(winners) == 1:
        return winners[0], scores
    else:
        return winners, scores


def main():
    """Run free-for-all tournament with all strategies."""
    print("="*80)
    print("REGENWORMEN FREE-FOR-ALL TOURNAMENT")
    print("="*80)
    print(f"Running {NUM_GAMES} games with {len(STRATEGY_REGISTRY)} strategies")
    print()

    # Define all strategies with default parameters
    strategies = {
        "optimal_expected": {
            "name": "optimal_expected",
            "params": {}
        },
        "risk_adjusted": {
            "name": "risk_adjusted",
            "params": {"risk_aversion": 1.0}  # neutral risk
        },
        "conservative": {
            "name": "conservative",
            "params": {"stop_bias": 1.3}  # slightly conservative
        },
        "aggressive": {
            "name": "aggressive",
            "params": {"continue_bias": 1.3}  # slightly aggressive
        },
        "opponent_aware": {
            "name": "opponent_aware",
            "params": {"steal_preference": 1.2, "risk_modifier": 0.9}
        },
        "endgame_focused": {
            "name": "endgame_focused",
            "params": {}  # use defaults
        }
    }

    # Verify all strategies are in registry
    for name in list(strategies.keys()):
        if name not in STRATEGY_REGISTRY:
            print(f"Warning: Strategy '{name}' not in registry, skipping")
            del strategies[name]

    if not strategies:
        print("Error: No valid strategies found!")
        return 1

    strategy_names = list(strategies.keys())
    strategy_configs = [strategies[name] for name in strategy_names]

    print("Strategies participating:")
    for i, name in enumerate(strategy_names):
        print(f"  {i}: {name}")
    print()

    # Initialize statistics
    wins = [0] * len(strategies)
    ties = [0] * len(strategies)
    total_worms = [0] * len(strategies)
    total_games = 0

    start_time = time.time()

    # Run games
    for game_num in range(NUM_GAMES):
        # Set random seed for reproducibility but vary per game
        seed = RANDOM_SEED + game_num * 997  # use prime to avoid correlation

        # Run the game
        winner, scores = run_game(strategy_configs, seed=seed)

        # Update statistics
        if isinstance(winner, int):
            wins[winner] += 1
        else:
            # Tie - split points among winners
            for w in winner:
                wins[w] += 1 / len(winner)
                ties[w] += 1

        for i, score in enumerate(scores):
            total_worms[i] += score

        total_games += 1

        # Print progress every 10 games
        if (game_num + 1) % 10 == 0:
            elapsed = time.time() - start_time
            games_per_sec = (game_num + 1) / elapsed if elapsed > 0 else 0
            print(f"  Completed {game_num + 1}/{NUM_GAMES} games "
                  f"({games_per_sec:.1f} games/sec)")

    elapsed = time.time() - start_time

    print()
    print("="*80)
    print("TOURNAMENT RESULTS")
    print("="*80)
    print(f"Total games: {total_games}")
    print(f"Total time: {elapsed:.2f} seconds ({elapsed/total_games:.3f} sec/game)")
    print()

    # Sort strategies by wins (descending)
    results = []
    for i, name in enumerate(strategy_names):
        win_rate = wins[i] / total_games * 100
        avg_worms = total_worms[i] / total_games
        results.append({
            "name": name,
            "wins": wins[i],
            "win_rate": win_rate,
            "avg_worms": avg_worms,
            "total_worms": total_worms[i],
            "ties": ties[i]
        })

    results.sort(key=lambda x: x["wins"], reverse=True)

    # Print results table
    print(f"{'Strategy':<20} {'Wins':<8} {'Win %':<8} {'Avg Worms':<12} {'Total Worms':<12} {'Ties':<8}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:<20} {r['wins']:<8.1f} {r['win_rate']:<8.1f} "
              f"{r['avg_worms']:<12.2f} {r['total_worms']:<12.1f} {r['ties']:<8.0f}")

    print()

    # Determine champion
    champion = results[0]
    print(f"CHAMPION: {champion['name']} with {champion['wins']:.1f} wins "
          f"({champion['win_rate']:.1f}% win rate)")
    print(f"   Average worms per game: {champion['avg_worms']:.2f}")

    # Show runner-up
    if len(results) > 1:
        runner_up = results[1]
        print(f"RUNNER-UP: {runner_up['name']} with {runner_up['wins']:.1f} wins "
              f"({runner_up['win_rate']:.1f}% win rate)")

    # Show interesting stats
    print()
    print("INTERESTING STATISTICS:")
    print(f"  Total worm count across all games: {sum(total_worms):.0f}")
    print(f"  Average worms per game per strategy: {sum(total_worms)/len(strategies)/total_games:.2f}")

    # Check for ties
    total_ties = sum(ties)
    if total_ties > 0:
        tie_rate = total_ties / (total_games * len(strategies)) * 100
        print(f"  Tie games: {total_ties} total ({tie_rate:.2f}% of possible ties)")

    return 0


if __name__ == "__main__":
    sys.exit(main())