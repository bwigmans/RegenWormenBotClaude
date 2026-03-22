#!/usr/bin/env python3
"""Test free-for-all with 1 game."""

import sys
import random
from game_models import Helping, Player, EnhancedGameState
from dice_utils import collection_sum, has_worm
from strategies import create_strategy, STRATEGY_REGISTRY
from simulation import (
    initial_grill, simulate_turn, apply_turn_outcome,
    player_worms, random_roll
)

def run_game(strategy_configs, seed=None):
    if seed is not None:
        random.seed(seed)

    num_players = len(strategy_configs)
    grill = initial_grill()
    players = [Player(i) for i in range(num_players)]
    current_player = random.randrange(num_players)
    turn = 0
    MAX_TURNS = 1000

    while grill and any(h.face_up for h in grill) and turn < MAX_TURNS:
        turn += 1
        state = EnhancedGameState(grill, players, current_player)
        config = strategy_configs[current_player]
        policy = create_strategy(
            config.get("name", "optimal_expected"),
            state, current_player,
            **config.get("params", {})
        )
        worm_delta, helping_taken, taken_from = simulate_turn(
            state, policy, current_player, commentator=None
        )
        apply_turn_outcome(state, worm_delta, helping_taken, taken_from)
        current_player = (current_player + 1) % num_players

    scores = [player_worms(p) for p in players]
    max_score = max(scores)
    winners = [i for i, s in enumerate(scores) if s == max_score]
    if len(winners) == 1:
        return winners[0], scores
    else:
        return winners, scores

def main():
    print("Testing free-for-all with 1 game...")
    strategies = {
        "optimal_expected": {"name": "optimal_expected", "params": {}},
        "risk_adjusted": {"name": "risk_adjusted", "params": {"risk_aversion": 1.0}},
        "conservative": {"name": "conservative", "params": {"stop_bias": 1.3}},
        "aggressive": {"name": "aggressive", "params": {"continue_bias": 1.3}},
        "opponent_aware": {"name": "opponent_aware", "params": {"steal_preference": 1.2, "risk_modifier": 0.9}},
        "endgame_focused": {"name": "endgame_focused", "params": {}},
    }
    strategy_names = list(strategies.keys())
    strategy_configs = [strategies[name] for name in strategy_names]

    print(f"Running 1 game with {len(strategy_configs)} players...")
    try:
        winner, scores = run_game(strategy_configs, seed=42)
        print("Success!")
        print(f"Winner: {winner}")
        print(f"Scores: {scores}")
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())