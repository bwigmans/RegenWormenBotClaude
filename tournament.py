#!/usr/bin/env python3
"""
Tournament bracket for Regenwormen strategies.
Runs 100-game matches between strategies in a single-elimination bracket.
"""

import sys
import random
import argparse
import time
from typing import List, Tuple, Union

from game_models import Helping, Player, EnhancedGameState
from dice_utils import collection_sum, has_worm
from strategies import create_strategy
from config_loader import load_config_file
from simulation import (
    initial_grill, simulate_turn, apply_turn_outcome,
    player_worms, random_roll
)

# Configuration
RANDOM_SEED = 42   # for reproducibility
MAX_TURNS = 1000   # max turns per game (safety)

def run_game(strategy_configs: List[dict], seed: int = None) -> Tuple[Union[int, List[int]], List[int]]:
    """
    Run a single game between strategies and return winner and all scores.

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
        pass

    # Game over: compute scores and determine winner
    scores = [player_worms(p) for p in players]
    max_score = max(scores)
    winners = [i for i, s in enumerate(scores) if s == max_score]

    if len(winners) == 1:
        return winners[0], scores
    else:
        return winners, scores


def run_match(strategy1: dict, strategy2: dict, match_name: str = "Match",
              verbose: bool = False, games_per_match: int = 100) -> Tuple[float, float, float, float]:
    """
    Run a match between two strategies (best of games_per_match).

    Args:
        strategy1: dict with "name" and optional "params"
        strategy2: same format
        match_name: identifier for logging
        verbose: print progress
        games_per_match: number of games per match

    Returns:
        Tuple (wins1, wins2, total_worms1, total_worms2)
    """
    wins1 = wins2 = 0.0
    worms1 = worms2 = 0.0

    if verbose:
        print(f"  {match_name}: {strategy1['name']} vs {strategy2['name']}")

    for game in range(games_per_match):
        # Set random seed for reproducibility but vary per game
        seed = RANDOM_SEED + game * 997  # use prime to avoid correlation

        # Run the game
        strategy_configs = [strategy1, strategy2]
        winner, scores = run_game(strategy_configs, seed=seed)

        # Track wins and worms
        if isinstance(winner, int):
            if winner == 0:
                wins1 += 1
                worms1 += scores[0]
            else:
                wins2 += 1
                worms2 += scores[1]
        else:
            # Tie - split points
            if 0 in winner:
                wins1 += 0.5
                worms1 += scores[0] / len(winner)
            if 1 in winner:
                wins2 += 0.5
                worms2 += scores[1] / len(winner)

        if verbose and (game + 1) % 20 == 0:
            print(f"    Game {game+1}/{games_per_match}: {wins1:.1f}-{wins2:.1f}")

    if verbose:
        print(f"    Final: {wins1:.1f}-{wins2:.1f}")
        if wins1 > wins2:
            print(f"    Winner: {strategy1['name']}")
        elif wins2 > wins1:
            print(f"    Winner: {strategy2['name']}")
        else:
            print(f"    Tie: {strategy1['name']} vs {strategy2['name']}")

    return wins1, wins2, worms1, worms2


def determine_match_winner(strategy1_name: str, wins1: float, worms1: float,
                          strategy2_name: str, wins2: float, worms2: float) -> Tuple[str, str]:
    """
    Determine match winner with tie-breaking.

    Returns:
        Tuple (winner_name, loser_name)
    """
    if wins1 > wins2:
        return strategy1_name, strategy2_name
    elif wins2 > wins1:
        return strategy2_name, strategy1_name
    else:
        # Tie in wins - use total worms as tiebreaker
        if worms1 > worms2:
            return strategy1_name, strategy2_name
        elif worms2 > worms1:
            return strategy2_name, strategy1_name
        else:
            # Still tied - randomly decide
            return (strategy1_name, strategy2_name) if random.random() < 0.5 \
                   else (strategy2_name, strategy1_name)


def run_bracket(strategies: dict, verbose: bool = False, games_per_match: int = 100) -> dict:
    """
    Run single-elimination bracket with given strategies.

    Args:
        strategies: dict mapping strategy_name -> strategy_config
        verbose: print progress
        games_per_match: number of games per match

    Returns:
        Dict with bracket results
    """
    # Create initial bracket (assuming 4 strategies)
    if len(strategies) != 4:
        raise ValueError("Bracket currently supports exactly 4 strategies")

    # Extract strategy names and configs
    strategy_names = list(strategies.keys())

    # Round 1 matches
    if verbose:
        print("\n" + "="*60)
        print("ROUND 1 - Quarterfinals")
        print("="*60)

    # Match 1: strategy_names[0] vs strategy_names[1]
    s1 = strategies[strategy_names[0]]
    s2 = strategies[strategy_names[1]]
    wins1, wins2, worms1, worms2 = run_match(s1, s2, "Quarterfinal 1", verbose, games_per_match)
    winner1, loser1 = determine_match_winner(
        strategy_names[0], wins1, worms1,
        strategy_names[1], wins2, worms2
    )

    # Match 2: strategy_names[2] vs strategy_names[3]
    s3 = strategies[strategy_names[2]]
    s4 = strategies[strategy_names[3]]
    wins3, wins4, worms3, worms4 = run_match(s3, s4, "Quarterfinal 2", verbose, games_per_match)
    winner2, loser2 = determine_match_winner(
        strategy_names[2], wins3, worms3,
        strategy_names[3], wins4, worms4
    )

    # Store round 1 results
    round1_results = {
        "match1": {
            "player1": strategy_names[0],
            "player2": strategy_names[1],
            "wins1": wins1,
            "wins2": wins2,
            "worms1": worms1,
            "worms2": worms2,
            "winner": winner1,
            "loser": loser1
        },
        "match2": {
            "player1": strategy_names[2],
            "player2": strategy_names[3],
            "wins1": wins3,
            "wins2": wins4,
            "worms1": worms3,
            "worms2": worms4,
            "winner": winner2,
            "loser": loser2
        }
    }

    if verbose:
        print("\n" + "="*60)
        print("ROUND 2 - Final")
        print("="*60)

    # Final match: winner1 vs winner2
    s_winner1 = strategies[winner1]
    s_winner2 = strategies[winner2]
    final_wins1, final_wins2, final_worms1, final_worms2 = run_match(
        s_winner1, s_winner2, "Final", verbose, games_per_match
    )
    champion, runner_up = determine_match_winner(
        winner1, final_wins1, final_worms1,
        winner2, final_wins2, final_worms2
    )

    final_results = {
        "player1": winner1,
        "player2": winner2,
        "wins1": final_wins1,
        "wins2": final_wins2,
        "worms1": final_worms1,
        "worms2": final_worms2,
        "champion": champion,
        "runner_up": runner_up
    }

    if verbose:
        print("\n" + "="*60)
        print("TOURNAMENT COMPLETE")
        print("="*60)

    return {
        "round1": round1_results,
        "final": final_results,
        "champion": champion,
        "runner_up": runner_up
    }


def display_results_table(results: dict, games_per_match: int = 100):
    """Display tournament results in a formatted table."""
    print("\n" + "="*80)
    print("REGENWORMEN STRATEGY TOURNAMENT RESULTS")
    print("="*80)

    # Quarterfinals
    print(f"\nQUARTERFINALS ({games_per_match} games each)")
    print("-" * 80)
    print(f"{'Match':<20} {'Strategy 1':<25} {'Score':<12} {'Strategy 2':<25}")
    print("-" * 80)

    m1 = results["round1"]["match1"]
    score1 = f"{m1['wins1']:.1f} - {m1['wins2']:.1f}"
    print(f"{'Quarterfinal 1':<20} {m1['player1']:<25} {score1:<12} {m1['player2']:<25}")

    m2 = results["round1"]["match2"]
    score2 = f"{m2['wins1']:.1f} - {m2['wins2']:.1f}"
    print(f"{'Quarterfinal 2':<20} {m2['player1']:<25} {score2:<12} {m2['player2']:<25}")

    # Final
    print(f"\nFINAL ({games_per_match} games)")
    print("-" * 80)
    final = results["final"]
    final_score = f"{final['wins1']:.1f} - {final['wins2']:.1f}"
    print(f"{'Final':<20} {final['player1']:<25} {final_score:<12} {final['player2']:<25}")

    # Champion
    print("\n" + "="*80)
    print(f"CHAMPION: {results['champion']}")
    print(f"RUNNER-UP: {results['runner_up']}")
    print("="*80)

    # Detailed statistics
    print("\nDETAILED STATISTICS")
    print("-" * 80)
    print(f"{'Match':<20} {'Winner':<25} {'Wins':<10} {'Total Worms':<15}")
    print("-" * 80)

    matches = [
        ("Quarterfinal 1", m1['winner'], max(m1['wins1'], m1['wins2']),
         m1['worms1'] if m1['wins1'] > m1['wins2'] else m1['worms2']),
        ("Quarterfinal 2", m2['winner'], max(m2['wins1'], m2['wins2']),
         m2['worms1'] if m2['wins1'] > m2['wins2'] else m2['worms2']),
        ("Final", final['champion'], max(final['wins1'], final['wins2']),
         final['worms1'] if final['wins1'] > final['wins2'] else final['worms2']),
    ]

    for match_name, winner, wins, worms in matches:
        print(f"{match_name:<20} {winner:<25} {wins:<10.1f} {worms:<15.1f}")


def main():
    """Main tournament runner."""
    parser = argparse.ArgumentParser(
        description="Run a Regenwormen strategy tournament bracket."
    )
    parser.add_argument(
        "-g", "--games", type=int, default=100,
        help="Number of games per match (default: 100)"
    )
    parser.add_argument(
        "-t", "--test", action="store_true",
        help="Test mode: run with 2 games per match"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Quiet mode: suppress progress output"
    )
    parser.add_argument(
        "-s", "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-c", "--config",
        help="Path to JSON configuration file (if not provided, uses default 4 strategies)"
    )
    args = parser.parse_args()

    games_per_match = 2 if args.test else args.games
    verbose = not args.quiet
    seed = args.seed

    # Load strategies from config file or use defaults
    if args.config:
        try:
            config = load_config_file(args.config)
            if len(config.players) != 4:
                print(f"Error: Tournament requires exactly 4 players, got {len(config.players)}", file=sys.stderr)
                sys.exit(1)
            # Convert PlayerConfig to tournament format
            strategies = {}
            for player_config in config.players:
                # Use strategy name with player ID as key to avoid collisions
                key = f"{player_config.strategy}_{player_config.player_id}"
                strategies[key] = {
                    "name": player_config.strategy,
                    "params": player_config.params
                }
            print(f"Loaded {len(strategies)} strategies from config file: {args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default 4 strategies for the bracket
        strategies = {
            "optimal_expected": {
                "name": "optimal_expected",
                "params": {}
            },
            "conservative": {
                "name": "conservative",
                "params": {"stop_bias": 1.3}
            },
            "aggressive": {
                "name": "aggressive",
                "params": {"continue_bias": 1.3}
            },
            "opponent_aware": {
                "name": "opponent_aware",
                "params": {"steal_preference": 1.2, "risk_modifier": 0.9}
            }
        }
        print("Using default tournament strategies")

    print("Starting Regenwormen Strategy Tournament")
    print(f"Each match: {games_per_match} games")
    print(f"Random seed: {seed}")
    print("\nParticipants:")
    for name in strategies:
        print(f"  - {name}")

    # Set random seed globally
    random.seed(seed)
    global RANDOM_SEED
    RANDOM_SEED = seed

    # Run the bracket
    results = run_bracket(strategies, verbose=verbose, games_per_match=games_per_match)

    # Display results
    display_results_table(results, games_per_match)


if __name__ == "__main__":
    main()