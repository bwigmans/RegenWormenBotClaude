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


class BenchmarkMetrics:
    """Collects and aggregates benchmarking metrics across multiple games."""

    def __init__(self):
        self.wins: Dict[str, float] = defaultdict(float)
        self.total_worms: Dict[str, float] = defaultdict(float)
        self.worm_squares: Dict[str, float] = defaultdict(float)  # For variance calculation
        self.game_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        self.decision_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.timing_stats: Dict[str, List[float]] = defaultdict(list)

    def record_game_result(self, strategy1: str, worms1: float,
                          strategy2: str, worms2: float,
                          decision_stats1: Optional[Dict] = None,
                          decision_stats2: Optional[Dict] = None,
                          timing1: Optional[float] = None,
                          timing2: Optional[float] = None):
        """
        Record results from a single game.

        Args:
            strategy1: Name of first strategy
            worms1: Worm count for first strategy
            strategy2: Name of second strategy
            worms2: Worm count for second strategy
            decision_stats1: Decision statistics for strategy1 (optional)
            decision_stats2: Decision statistics for strategy2 (optional)
            timing1: Decision time for strategy1 in seconds (optional)
            timing2: Decision time for strategy2 in seconds (optional)
        """
        # Record worm counts
        self.total_worms[strategy1] += worms1
        self.total_worms[strategy2] += worms2

        # Record squares for variance calculation
        self.worm_squares[strategy1] += worms1 * worms1
        self.worm_squares[strategy2] += worms2 * worms2

        # Determine winner
        if worms1 > worms2:
            self.wins[strategy1] += 1
        elif worms2 > worms1:
            self.wins[strategy2] += 1
        else:
            # Tie - split win
            self.wins[strategy1] += 0.5
            self.wins[strategy2] += 0.5

        # Record game count for this matchup
        matchup = tuple(sorted([strategy1, strategy2]))
        self.game_counts[matchup] += 1

        # Record decision stats if provided
        if decision_stats1 is not None:
            self._merge_decision_stats(strategy1, decision_stats1)
        if decision_stats2 is not None:
            self._merge_decision_stats(strategy2, decision_stats2)

        # Record timing if provided
        if timing1 is not None:
            self.timing_stats[strategy1].append(timing1)
        if timing2 is not None:
            self.timing_stats[strategy2].append(timing2)

    def _merge_decision_stats(self, strategy: str, stats: Dict[str, Any]):
        """Merge decision statistics for a strategy."""
        if strategy not in self.decision_stats:
            self.decision_stats[strategy] = stats.copy()
        else:
            # Merge averages weighted by counts
            current = self.decision_stats[strategy]
            for key, value in stats.items():
                if isinstance(value, (int, float)):
                    if key in current:
                        # Simple average for now
                        current[key] = (current[key] + value) / 2
                    else:
                        current[key] = value
                else:
                    current[key] = value

    def get_strategy_metrics(self, strategy: str, total_games: int) -> Dict[str, Any]:
        """Get computed metrics for a specific strategy."""
        wins = self.wins.get(strategy, 0)
        total_worms = self.total_worms.get(strategy, 0)
        worm_squares = self.worm_squares.get(strategy, 0)

        win_rate = wins / total_games if total_games > 0 else 0.0
        avg_worms = total_worms / total_games if total_games > 0 else 0.0

        # Calculate variance: Var(X) = E[X^2] - (E[X])^2
        variance = 0.0
        if total_games > 0:
            mean_squares = worm_squares / total_games
            variance = mean_squares - (avg_worms * avg_worms)
            variance = max(variance, 0.0)  # Avoid floating point errors

        std_dev = variance ** 0.5 if variance > 0 else 0.0

        # Get decision stats
        decision_stats = self.decision_stats.get(strategy, {})

        # Get timing stats
        timing_data = self.timing_stats.get(strategy, [])
        avg_time = statistics.mean(timing_data) if timing_data else 0.0
        time_std = statistics.stdev(timing_data) if len(timing_data) > 1 else 0.0

        return {
            "win_rate": win_rate,
            "avg_worms": avg_worms,
            "worm_variance": variance,
            "worm_std_dev": std_dev,
            "total_wins": wins,
            "total_worms": total_worms,
            "decision_stats": decision_stats,
            "avg_decision_time": avg_time,
            "decision_time_std": time_std,
            "games_played": total_games
        }


class StrategyBenchmark:
    """Main benchmarking class for evaluating strategies."""

    def __init__(self, config):
        """
        Initialize benchmark with configuration.

        Args:
            config: Config object from config_loader
        """
        self.config = config
        self.metrics = BenchmarkMetrics()
        self.decision_trackers = {}  # strategy_name -> DecisionTracker
        self.rng = random.Random(config.random_seed)

    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run benchmark based on configuration.

        Returns:
            Dictionary with benchmark results
        """
        if len(self.config.players) == 2:
            return self._run_head_to_head_benchmark()
        else:
            return self._run_round_robin_benchmark()

    def _run_head_to_head_benchmark(self) -> Dict[str, Any]:
        """
        Run head-to-head benchmark for exactly two players.

        Returns:
            Dictionary with benchmark results
        """
        if len(self.config.players) != 2:
            raise ValueError("Head-to-head benchmark requires exactly 2 players")

        player1_config = self.config.players[0]
        player2_config = self.config.players[1]

        for game_num in range(self.config.num_games):
            # Generate deterministic game seed
            game_seed = self.config.random_seed + game_num * 997

            # Run single game
            worms1, worms2, stats1, stats2, time1, time2 = self._run_single_game(
                player1_config, player2_config, game_seed
            )

            # Record results
            strategy1_name = f"{player1_config.strategy}_{player1_config.player_id}"
            strategy2_name = f"{player2_config.strategy}_{player2_config.player_id}"

            self.metrics.record_game_result(
                strategy1_name, worms1,
                strategy2_name, worms2,
                decision_stats1=stats1,
                decision_stats2=stats2,
                timing1=time1,
                timing2=time2
            )

        # Compile and return results
        return self._compile_results([strategy1_name, strategy2_name])

    def _run_single_game(self, player1_config, player2_config, game_seed):
        """
        Run a single game between two players.

        Args:
            player1_config: PlayerConfig for player 1
            player2_config: PlayerConfig for player 2
            game_seed: Random seed for this game

        Returns:
            Tuple of (worms1, worms2, stats1, stats2, time1, time2)
        """
        # Set random seed for this game
        random.seed(game_seed)

        # Import here to avoid circular imports
        from simulation import initial_grill, simulate_turn, apply_turn_outcome, player_worms
        from game_models import Player, GameState, EnhancedGameState
        from strategies import create_strategy

        # Initialize game state
        grill = initial_grill()
        players = [Player(0), Player(1)]
        starting_player = random.randint(0, 1)
        global_state = GameState(grill, players, starting_player)

        # Initialize decision trackers if needed
        if self.config.collect_decision_stats:
            strategy1_name = f"{player1_config.strategy}_{player1_config.player_id}"
            strategy2_name = f"{player2_config.strategy}_{player2_config.player_id}"
            self.decision_trackers[strategy1_name] = DecisionTracker()
            self.decision_trackers[strategy2_name] = DecisionTracker()

        # Game loop
        turn = 0
        game_timing1 = []  # Store timing for player 1 in this game
        game_timing2 = []  # Store timing for player 2 in this game
        while any(h.face_up for h in grill) and turn < self.config.max_turns_per_game:
            current_player = global_state.current_player
            current_config = player1_config if current_player == 0 else player2_config

            # Create enhanced game state
            enhanced_state = EnhancedGameState(global_state)

            # Create strategy for current player
            strategy = create_strategy(
                current_config.strategy,
                enhanced_state,
                current_player,
                **current_config.params
            )

            # Get decision tracker for this player if collecting stats
            decision_tracker = None
            if self.config.collect_decision_stats:
                strategy_name = f"{current_config.strategy}_{current_config.player_id}"
                decision_tracker = self.decision_trackers.get(strategy_name)

            # Simulate turn with timing if needed
            start_time = time.perf_counter() if self.config.collect_timing else None

            worm_delta, helping_taken, taken_from = simulate_turn(
                global_state, strategy, player=current_player,
                decision_tracker=decision_tracker
            )

            turn_time = None
            if self.config.collect_timing and start_time is not None:
                turn_time = time.perf_counter() - start_time

            # Apply turn outcome
            apply_turn_outcome(global_state, worm_delta, helping_taken, taken_from)

            # Record timing for this game
            if turn_time is not None:
                if current_player == 0:
                    game_timing1.append(turn_time)
                else:
                    game_timing2.append(turn_time)

            turn += 1

        # Compute final worm counts
        worms1 = player_worms(players[0])
        worms2 = player_worms(players[1])

        # Get decision stats if collected
        stats1 = None
        stats2 = None
        if self.config.collect_decision_stats:
            strategy1_name = f"{player1_config.strategy}_{player1_config.player_id}"
            strategy2_name = f"{player2_config.strategy}_{player2_config.player_id}"
            tracker1 = self.decision_trackers.get(strategy1_name)
            tracker2 = self.decision_trackers.get(strategy2_name)
            if tracker1:
                stats1 = tracker1.get_stats()
            if tracker2:
                stats2 = tracker2.get_stats()

        # Get average timing if collected
        time1 = None
        time2 = None
        if self.config.collect_timing:
            if game_timing1:
                time1 = sum(game_timing1) / len(game_timing1)
            if game_timing2:
                time2 = sum(game_timing2) / len(game_timing2)

        return worms1, worms2, stats1, stats2, time1, time2

    def _run_round_robin_benchmark(self) -> Dict[str, Any]:
        """
        Run round-robin benchmark for multiple players.

        Returns:
            Dictionary with benchmark results
        """
        # TODO: Implement in Task 7A
        raise NotImplementedError("Round-robin benchmarking not yet implemented")

    def _compile_results(self, strategy_names: List[str]) -> Dict[str, Any]:
        """
        Compile benchmark results into a structured dictionary.

        Args:
            strategy_names: List of strategy names that participated

        Returns:
            Dictionary with compiled results
        """
        total_games = self.config.num_games
        num_players = len(strategy_names)

        # For round-robin, each strategy plays (num_players - 1) * total_games games
        games_per_strategy = (num_players - 1) * total_games if num_players > 1 else total_games

        results = {
            "config": {
                "num_games": total_games,
                "random_seed": self.config.random_seed,
                "players": [
                    {
                        "name": name,
                        "config": self.config.players[idx].__dict__
                    }
                    for idx, name in enumerate(strategy_names)
                ]
            },
            "metrics": {
                name: self.metrics.get_strategy_metrics(name, games_per_strategy)
                for name in strategy_names
            },
            "matchup_stats": {
                "games_played": total_games,
                "head_to_head_wins": {
                    name: self.metrics.wins.get(name, 0)
                    for name in strategy_names
                }
            }
        }

        return results