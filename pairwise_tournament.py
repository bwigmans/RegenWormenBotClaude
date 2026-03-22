#!/usr/bin/env python3
"""
Round-robin 1v1 tournament between all strategies.
Confirms whether optimal_expected still dominates in head-to-head.
"""

import sys
import time
from typing import List, Dict, Tuple

from tournament import run_match, determine_match_winner

def main():
    print("="*80)
    print("REGENWORMEN PAIRWISE 1v1 TOURNAMENT")
    print("="*80)
    print("Each match: 50 games")
    print()

    # Same strategies as free-for-all
    strategies = {
        "optimal_expected": {"name": "optimal_expected", "params": {}},
        "risk_adjusted": {"name": "risk_adjusted", "params": {"risk_aversion": 1.0}},
        "conservative": {"name": "conservative", "params": {"stop_bias": 1.3}},
        "aggressive": {"name": "aggressive", "params": {"continue_bias": 1.3}},
        "opponent_aware": {"name": "opponent_aware", "params": {"steal_preference": 1.2, "risk_modifier": 0.9}},
        "endgame_focused": {"name": "endgame_focused", "params": {}},
    }

    strategy_names = list(strategies.keys())
    games_per_match = 50  # Keep it fast but statistically meaningful
    results = {}  # (strategy1, strategy2) -> (wins1, wins2, worms1, worms2)

    total_matches = len(strategy_names) * (len(strategy_names) - 1) // 2
    match_count = 0
    start_time = time.time()

    print(f"Running {total_matches} matches ({games_per_match} games each)...")
    print()

    # Round-robin: each pair once
    for i in range(len(strategy_names)):
        for j in range(i + 1, len(strategy_names)):
            match_count += 1
            name1 = strategy_names[i]
            name2 = strategy_names[j]
            s1 = strategies[name1]
            s2 = strategies[name2]

            print(f"Match {match_count}/{total_matches}: {name1} vs {name2}")
            wins1, wins2, worms1, worms2 = run_match(
                s1, s2, f"{name1} vs {name2}", verbose=False, games_per_match=games_per_match
            )

            results[(name1, name2)] = (wins1, wins2, worms1, worms2)
            winner, loser = determine_match_winner(name1, wins1, worms1, name2, wins2, worms2)

            win_rate1 = wins1 / games_per_match * 100
            win_rate2 = wins2 / games_per_match * 100
            print(f"  Result: {wins1:.1f}-{wins2:.1f} ({win_rate1:.1f}% - {win_rate2:.1f}%)")
            print(f"  Winner: {winner}")
            print()

    elapsed = time.time() - start_time
    print(f"Total time: {elapsed:.1f} seconds")
    print()

    # Aggregate wins
    total_wins = {name: 0.0 for name in strategy_names}
    match_wins = {name: 0 for name in strategy_names}  # matches won (not games)
    total_worms = {name: 0.0 for name in strategy_names}

    for (name1, name2), (wins1, wins2, worms1, worms2) in results.items():
        total_wins[name1] += wins1
        total_wins[name2] += wins2
        total_worms[name1] += worms1
        total_worms[name2] += worms2

        # Count match wins (best of games_per_match)
        if wins1 > wins2:
            match_wins[name1] += 1
        elif wins2 > wins1:
            match_wins[name2] += 1
        else:
            # Tie in wins - use worms tiebreaker
            if worms1 > worms2:
                match_wins[name1] += 1
            elif worms2 > worms1:
                match_wins[name2] += 1
            else:
                # Still tied - split
                match_wins[name1] += 0.5
                match_wins[name2] += 0.5

    # Sort by total wins
    sorted_strategies = sorted(strategy_names, key=lambda n: total_wins[n], reverse=True)

    print("="*80)
    print("FINAL STANDINGS (by total game wins)")
    print("="*80)
    print(f"{'Strategy':<20} {'Wins':<8} {'Matches Won':<12} {'Avg Win %':<12} {'Total Worms':<12}")
    print("-" * 80)

    for name in sorted_strategies:
        matches_played = len(strategy_names) - 1  # each plays N-1 opponents
        avg_win_rate = total_wins[name] / (matches_played * games_per_match) * 100
        print(f"{name:<20} {total_wins[name]:<8.1f} {match_wins[name]:<12.1f} "
              f"{avg_win_rate:<12.1f} {total_worms[name]:<12.1f}")

    print()

    # Head-to-head matrix
    print("HEAD-TO-HEAD MATRIX (win percentages)")
    print("-" * 80)
    print("Row = attacker, Column = defender")
    print(f"{'':<20}", end="")
    for name in strategy_names:
        print(f"{name[:8]:<8}", end="")
    print()

    for name1 in strategy_names:
        print(f"{name1:<20}", end="")
        for name2 in strategy_names:
            if name1 == name2:
                print(f"{'--':<8}", end="")
            elif (name1, name2) in results:
                wins1, wins2, _, _ = results[(name1, name2)]
                win_rate = wins1 / games_per_match * 100
                print(f"{win_rate:<8.1f}", end="")
            else:
                wins2, wins1, _, _ = results[(name2, name1)]
                win_rate = wins1 / games_per_match * 100
                print(f"{win_rate:<8.1f}", end="")
        print()

    print()

    # Champion
    champion = sorted_strategies[0]
    print(f"CHAMPION: {champion} with {total_wins[champion]:.1f} total wins "
          f"({total_wins[champion]/(len(strategy_names)-1)/games_per_match*100:.1f}% average win rate)")

    # Compare with free-for-all ranking
    free_for_all_rank = ["aggressive", "opponent_aware", "risk_adjusted",
                         "optimal_expected", "endgame_focused", "conservative"]

    print("\nCOMPARISON WITH FREE-FOR-ALL (6-player) RANKING:")
    print("  Free-for-all ranking (100 games):")
    for i, name in enumerate(free_for_all_rank, 1):
        print(f"    {i}. {name}")

    print("\n  Pairwise 1v1 ranking (this tournament):")
    for i, name in enumerate(sorted_strategies, 1):
        print(f"    {i}. {name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())