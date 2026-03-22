"""
Comprehensive benchmarking system for Regenwormen strategies.
"""
from typing import Dict, Tuple, List, Optional, Any
from collections import defaultdict
import random
import math
import statistics
import time
from dice_utils import collection_sum

class DecisionTracker:
    """Tracks decisions during simulation for later analysis."""

    def __init__(self):
        self.stop_counts: Dict[int, int] = defaultdict(int)  # collection_sum -> count
        self.continue_counts: Dict[int, int] = defaultdict(int)  # collection_sum -> count
        self.steal_attempts: int = 0
        self.successful_steals: int = 0
        self.total_stop_collection: int = 0
        self.total_stop_reward: float = 0.0
        self.total_continuation_value: float = 0.0
        self.stop_decisions: int = 0
        self.roll_counts: int = 0
        self.roll_symbol_counts: Dict[int, int] = defaultdict(int)  # symbol -> total count across all rolls
        self.symbol_choices: Dict[int, int] = defaultdict(int)  # symbol -> count

    def record_roll(self, collection: Tuple[int, ...],
                   roll_counts: Tuple[int, ...]) -> None:
        """Record a roll event."""
        self.roll_counts += 1
        for symbol, count in enumerate(roll_counts):
            if count > 0:
                self.roll_symbol_counts[symbol] += count

    def record_stop_decision(self, collection: Tuple[int, ...],
                           stop_reward: float,
                           continuation_value: float) -> None:
        """
        Record a stop decision with context.

        Args:
            collection: 6-tuple of dice taken so far
            stop_reward: Reward for stopping now
            continuation_value: Expected value of continuing
        """
        coll_sum = collection_sum(collection)
        self.stop_counts[coll_sum] += 1
        self.total_stop_collection += coll_sum
        self.total_stop_reward += stop_reward
        self.total_continuation_value += continuation_value
        self.stop_decisions += 1

    def record_continue_decision(self, collection: Tuple[int, ...],
                               chosen_symbol: int) -> None:
        """
        Record a continue decision.

        Args:
            collection: 6-tuple of dice taken so far
            chosen_symbol: Symbol chosen to take (0-5)
        """
        coll_sum = collection_sum(collection)
        self.continue_counts[coll_sum] += 1
        self.symbol_choices[chosen_symbol] += 1

    def record_steal_attempt(self, collection: Tuple[int, ...],
                           victim_player: int, success: bool) -> None:
        """
        Record a steal attempt.

        Args:
            collection: 6-tuple of dice taken so far
            victim_player: Player being stolen from
            success: Whether steal was successful
        """
        self.steal_attempts += 1
        if success:
            self.successful_steals += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get computed statistics from tracked decisions."""
        total_decisions = sum(self.stop_counts.values()) + sum(self.continue_counts.values())
        stop_frequency = 0.0
        if total_decisions > 0:
            stop_frequency = sum(self.stop_counts.values()) / total_decisions

        avg_stop_collection = 0.0
        if self.stop_decisions > 0:
            avg_stop_collection = self.total_stop_collection / self.stop_decisions

        steal_success_rate = 0.0
        if self.steal_attempts > 0:
            steal_success_rate = self.successful_steals / self.steal_attempts

        total_dice_rolled = sum(self.roll_symbol_counts.values())
        avg_stop_reward = self.total_stop_reward / self.stop_decisions if self.stop_decisions > 0 else 0.0
        avg_continuation_value = self.total_continuation_value / self.stop_decisions if self.stop_decisions > 0 else 0.0
        avg_value_difference = avg_stop_reward - avg_continuation_value
        return {
            "stop_frequency": stop_frequency,
            "avg_stop_collection": avg_stop_collection,
            "avg_stop_reward": avg_stop_reward,
            "avg_continuation_value": avg_continuation_value,
            "avg_value_difference": avg_value_difference,
            "steal_attempts": self.steal_attempts,
            "steal_success_rate": steal_success_rate,
            "rolls_per_turn": self.roll_counts / max(self.stop_decisions, 1),
            "total_decisions": total_decisions,
            "total_dice_rolled": total_dice_rolled,
            "avg_dice_per_roll": total_dice_rolled / self.roll_counts if self.roll_counts > 0 else 0.0
        }