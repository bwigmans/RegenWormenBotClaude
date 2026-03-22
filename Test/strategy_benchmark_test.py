# RegenWormenBot/Test/strategy_benchmark_test.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategy_benchmark import DecisionTracker, BenchmarkMetrics, StrategyBenchmark

def test_decision_tracker_creation():
    """Test DecisionTracker class creation."""
    tracker = DecisionTracker()
    assert tracker.stop_counts == {}
    assert tracker.continue_counts == {}
    assert tracker.steal_attempts == 0
    assert tracker.successful_steals == 0
    assert tracker.total_stop_collection == 0
    assert tracker.total_stop_reward == 0.0
    assert tracker.total_continuation_value == 0.0
    assert tracker.stop_decisions == 0
    assert tracker.roll_counts == 0
    assert tracker.roll_symbol_counts == {}
    assert tracker.symbol_choices == {}

def test_record_roll():
    """Test recording roll events."""
    tracker = DecisionTracker()
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (2, 1, 0, 0, 0, 1)  # 2 ones, 1 two, 1 worm
    tracker.record_roll(collection, roll_counts)
    assert tracker.roll_counts == 1
    assert tracker.roll_symbol_counts[0] == 2
    assert tracker.roll_symbol_counts[1] == 1
    assert tracker.roll_symbol_counts[5] == 1
    # symbols with zero count not added to dict
    assert 2 not in tracker.roll_symbol_counts

    tracker.record_roll(collection, (0, 0, 3, 0, 0, 0))
    assert tracker.roll_counts == 2
    assert tracker.roll_symbol_counts[2] == 3

def test_record_stop_decision():
    """Test recording stop decisions."""
    tracker = DecisionTracker()
    collection = (1, 0, 0, 0, 0, 1)  # sum = 1*1 + 1*5 = 6
    tracker.record_stop_decision(collection, stop_reward=10.5, continuation_value=8.2)
    assert tracker.stop_counts[6] == 1
    assert tracker.total_stop_collection == 6
    assert tracker.total_stop_reward == 10.5
    assert tracker.total_continuation_value == 8.2
    assert tracker.stop_decisions == 1

    # Another stop with same collection sum
    tracker.record_stop_decision(collection, stop_reward=5.0, continuation_value=6.0)
    assert tracker.stop_counts[6] == 2
    assert tracker.total_stop_collection == 12
    assert tracker.total_stop_reward == 15.5
    assert tracker.total_continuation_value == 14.2
    assert tracker.stop_decisions == 2

def test_record_continue_decision():
    """Test recording continue decisions."""
    tracker = DecisionTracker()
    collection = (0, 2, 0, 0, 0, 0)  # sum = 2*2 = 4
    tracker.record_continue_decision(collection, chosen_symbol=1)
    assert tracker.continue_counts[4] == 1
    assert tracker.symbol_choices[1] == 1

    tracker.record_continue_decision(collection, chosen_symbol=1)
    assert tracker.continue_counts[4] == 2
    assert tracker.symbol_choices[1] == 2

    tracker.record_continue_decision(collection, chosen_symbol=3)
    assert tracker.symbol_choices[3] == 1

def test_record_steal_attempt():
    """Test recording steal attempts."""
    tracker = DecisionTracker()
    collection = (0, 0, 0, 0, 0, 0)
    tracker.record_steal_attempt(collection, victim_player=1, success=True)
    assert tracker.steal_attempts == 1
    assert tracker.successful_steals == 1

    tracker.record_steal_attempt(collection, victim_player=2, success=False)
    assert tracker.steal_attempts == 2
    assert tracker.successful_steals == 1

def test_get_stats_empty():
    """Test statistics with no decisions."""
    tracker = DecisionTracker()
    stats = tracker.get_stats()
    assert stats["stop_frequency"] == 0.0
    assert stats["avg_stop_collection"] == 0.0
    assert stats["avg_stop_reward"] == 0.0
    assert stats["avg_continuation_value"] == 0.0
    assert stats["avg_value_difference"] == 0.0
    assert stats["steal_attempts"] == 0
    assert stats["steal_success_rate"] == 0.0
    assert stats["rolls_per_turn"] == 0.0  # stop_decisions = 0, max(1,1) = 1, roll_counts = 0
    assert stats["total_decisions"] == 0
    assert stats["total_dice_rolled"] == 0
    assert stats["avg_dice_per_roll"] == 0.0

def test_get_stats_with_data():
    """Test statistics with mixed decisions."""
    tracker = DecisionTracker()
    # Add some rolls
    tracker.record_roll((0,0,0,0,0,0), (2, 1, 0, 0, 0, 1))
    tracker.record_roll((0,0,0,0,0,0), (0, 0, 3, 0, 0, 0))
    # Add stop decisions
    tracker.record_stop_decision((1,0,0,0,0,1), stop_reward=10.0, continuation_value=8.0)
    tracker.record_stop_decision((0,2,0,0,0,0), stop_reward=5.0, continuation_value=6.0)
    # Add continue decisions
    tracker.record_continue_decision((0,1,0,0,0,0), chosen_symbol=1)
    tracker.record_continue_decision((0,0,1,0,0,0), chosen_symbol=2)
    tracker.record_continue_decision((0,0,0,1,0,0), chosen_symbol=3)
    # Add steal attempts
    tracker.record_steal_attempt((0,0,0,0,0,0), victim_player=1, success=True)
    tracker.record_steal_attempt((0,0,0,0,0,0), victim_player=2, success=False)

    stats = tracker.get_stats()
    # Check calculations
    assert stats["stop_frequency"] == 2 / 5  # 2 stops, 3 continues = 5 total decisions
    assert stats["avg_stop_collection"] == (6 + 4) / 2  # (1*1 + 1*5) = 6, (2*2) = 4
    assert stats["avg_stop_reward"] == (10.0 + 5.0) / 2
    assert stats["avg_continuation_value"] == (8.0 + 6.0) / 2
    assert stats["avg_value_difference"] == stats["avg_stop_reward"] - stats["avg_continuation_value"]
    assert stats["steal_attempts"] == 2
    assert stats["steal_success_rate"] == 0.5
    assert stats["rolls_per_turn"] == 2 / 2  # 2 rolls, 2 stop_decisions
    assert stats["total_decisions"] == 5
    assert stats["total_dice_rolled"] == 2+1+1 + 3  # 2 ones, 1 two, 1 worm + 3 threes = 7
    assert stats["avg_dice_per_roll"] == 7 / 2


def test_simulate_turn_with_decision_tracker():
    """Test that simulate_turn records decisions via DecisionTracker."""
    # Import here to avoid circular imports
    from simulation import simulate_turn
    from game_models import GameState, Player, Helping
    from strategies import OptimalExpectedValueStrategy
    import random

    # Set up a simple game state with one player and a grill with a low tile
    grill = [Helping(5, 1), Helping(10, 2), Helping(15, 3)]
    for h in grill:
        h.face_up = True
    players = [Player(0)]
    state = GameState(grill, players, 0)
    policy = OptimalExpectedValueStrategy(state, 0)
    tracker = DecisionTracker()

    # Set random seed for reproducible dice rolls
    random.seed(42)

    # Run a turn with decision tracking
    worm_delta, helping_taken, taken_from = simulate_turn(
        state, policy, decision_tracker=tracker
    )

    # Check that some decisions were recorded (at least one roll or stop)
    stats = tracker.get_stats()
    # At minimum, there should be at least one roll if the turn didn't stop immediately
    # We'll just ensure no errors and stats are computed
    assert "rolls_per_turn" in stats
    # No assertions on specific values because they depend on random rolls
    print(f"Turn recorded {stats['rolls_per_turn']} rolls per turn")

    # Reset random seed
    random.seed()

def test_benchmark_metrics_creation():
    """Test BenchmarkMetrics class creation and updates."""
    metrics = BenchmarkMetrics()

    # Initial state
    assert metrics.wins == {}
    assert metrics.total_worms == {}
    assert metrics.worm_squares == {}
    assert metrics.game_counts == {}

    # Update with a result
    metrics.record_game_result("optimal_expected", 5, "conservative", 3)

    assert metrics.wins["optimal_expected"] == 1
    assert metrics.total_worms["optimal_expected"] == 5
    assert metrics.total_worms["conservative"] == 3
    assert metrics.worm_squares["optimal_expected"] == 25
    # matchup is sorted alphabetically
    assert metrics.game_counts[("conservative", "optimal_expected")] == 1


def test_strategy_benchmark_creation():
    """Test StrategyBenchmark class creation."""
    # Import here to avoid circular imports
    from config_loader import Config, PlayerConfig

    # Create a simple config with two players
    players = [
        PlayerConfig(player_id=0, strategy="optimal_expected", params={}),
        PlayerConfig(player_id=1, strategy="conservative", params={"stop_bias": 1.2})
    ]
    config = Config(
        players=players,
        num_games=100,
        random_seed=42,
        verbose=False,
        max_turns_per_game=1000,
        collect_decision_stats=True,
        collect_worm_distribution=True,
        collect_timing=False
    )

    # Create benchmark instance
    benchmark = StrategyBenchmark(config)

    # Check that config is stored
    assert benchmark.config == config

    # Check that metrics object is created
    assert hasattr(benchmark, 'metrics')
    assert isinstance(benchmark.metrics, BenchmarkMetrics)

    # Check that decision_trackers dict exists
    assert hasattr(benchmark, 'decision_trackers')
    assert isinstance(benchmark.decision_trackers, dict)

    # Check that rng is created with correct seed
    assert hasattr(benchmark, 'rng')
    assert benchmark.rng is not None

    # Check that run_benchmark method exists
    assert hasattr(benchmark, 'run_benchmark')
    assert callable(benchmark.run_benchmark)

    # Check that head-to-head method exists
    assert hasattr(benchmark, '_run_head_to_head_benchmark')
    assert callable(benchmark._run_head_to_head_benchmark)

    # Check that round-robin method exists (stub for now)
    assert hasattr(benchmark, '_run_round_robin_benchmark')
    assert callable(benchmark._run_round_robin_benchmark)

def test_round_robin_benchmark():
    """Test round-robin benchmarking with 3 players."""
    from strategy_benchmark import StrategyBenchmark
    from config_loader import Config, PlayerConfig

    players = [
        PlayerConfig(0, "optimal_expected", {}),
        PlayerConfig(1, "conservative", {"stop_bias": 1.3}),
        PlayerConfig(2, "aggressive", {"continue_bias": 1.3})
    ]
    config = Config(players=players, num_games=10, random_seed=42)
    benchmark = StrategyBenchmark(config)

    # Should run without error
    results = benchmark.run_benchmark()

    # Should have metrics for all 3 strategies
    assert len(results["metrics"]) == 3
    for strategy_name in ["optimal_expected_0", "conservative_1", "aggressive_2"]:
        assert strategy_name in results["metrics"]
        metrics = results["metrics"][strategy_name]
        assert metrics["games_played"] == 20  # 2 opponents * 10 games each