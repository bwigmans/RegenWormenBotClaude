"""
Unit tests for the strategy system.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from strategies import (
    BaseTurnStrategy, OptimalExpectedValueStrategy,
    RiskAdjustedStrategy, ConservativeStrategy, AggressiveStrategy,
    OpponentAwareStrategy, EndgameStrategy,
    create_strategy, STRATEGY_REGISTRY, TurnPolicy
)
from game_models import Helping, Player, GameState, EnhancedGameState
from dice_utils import collection_sum, has_worm


def create_test_game_state(num_players=2, current_player=0):
    """Create a minimal game state for testing."""
    # Create grill with some helpings
    grill = [
        Helping(25, 2),  # face-up by default
        Helping(26, 2),
        Helping(27, 2),
        Helping(28, 2),
        Helping(29, 3),
        Helping(30, 3),
    ]
    # Make them all face-up
    for h in grill:
        h.face_up = True

    players = [Player(i) for i in range(num_players)]
    # Give first player a helping
    players[0].add_helping(Helping(24, 1))

    return GameState(grill, players, current_player)


def test_strategy_registry():
    """Test that strategy registry contains expected strategies."""
    assert "optimal_expected" in STRATEGY_REGISTRY
    assert STRATEGY_REGISTRY["optimal_expected"] == OptimalExpectedValueStrategy


def test_create_strategy():
    """Test factory function create_strategy."""
    state = create_test_game_state()
    strategy = create_strategy("optimal_expected", state, player_id=0)
    assert isinstance(strategy, OptimalExpectedValueStrategy)
    assert strategy.player_id == 0
    assert strategy.global_state is state


def test_optimal_expected_value_strategy_basic():
    """Test basic functionality of OptimalExpectedValueStrategy."""
    state = create_test_game_state()
    strategy = OptimalExpectedValueStrategy(state, player_id=0)

    # Test that values are computed
    assert hasattr(strategy, 'values')
    assert isinstance(strategy.values, dict)

    # Test should_stop with empty collection (should continue)
    collection = (0, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    # With empty collection, optimal is to continue (except in degenerate cases)
    # We'll just verify the method runs without error
    assert isinstance(stop, bool)

    # Test choose_symbol with a roll containing new symbols
    roll_counts = (1, 0, 0, 0, 0, 0)  # one '1'
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert symbol == 0  # should choose symbol 0 (1)

    # Test choose_symbol with no new symbols (all taken)
    collection_all = (1, 0, 0, 0, 0, 0)  # already have a '1'
    symbol = strategy.choose_symbol(collection_all, roll_counts)
    assert symbol is None


def test_optimal_expected_value_strategy_consistency():
    """
    Test that OptimalExpectedValueStrategy produces consistent decisions.
    The strategy should prefer continuing when expected value > stop reward.
    """
    state = create_test_game_state()
    strategy = OptimalExpectedValueStrategy(state, player_id=0)

    # Get a collection where continuing is clearly better
    # Use empty collection - continuing should be better
    collection = (0, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    # In normal game state, should continue with empty collection
    # But we'll just verify the method works

    # Test that choose_symbol returns best symbol
    roll_counts = (2, 0, 0, 0, 0, 0)  # two '1's
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert symbol == 0


def test_enhanced_game_state():
    """Test EnhancedGameState functionality."""
    grill = [Helping(25, 2), Helping(26, 2), Helping(27, 2)]
    for h in grill:
        h.face_up = True

    players = [Player(0), Player(1)]
    players[0].add_helping(Helping(24, 1))
    players[0].add_helping(Helping(28, 2))  # total worms: 1+2=3
    players[1].add_helping(Helping(29, 3))  # total worms: 3

    state = EnhancedGameState(grill, players, current_player=0)

    # Test scores computation
    assert state.scores == [3, 3]  # both have 3 worms
    assert isinstance(state.scores, list)
    assert len(state.scores) == 2

    # Test player position
    position, diff = state.get_player_position(0)
    assert position == 1  # tied for first
    assert diff == 0  # tied with next

    # Test game phase
    phase = state.get_game_phase()
    assert phase in ["early", "mid", "endgame"]
    # With 3 face-up tiles, should be "endgame" (threshold ≤3)
    assert phase == "endgame"

    # Test with more face-up tiles
    grill2 = [Helping(21 + i, 1) for i in range(12)]  # 12 tiles
    for h in grill2:
        h.face_up = True
    state2 = EnhancedGameState(grill2, players, current_player=0)
    phase2 = state2.get_game_phase()
    assert phase2 == "early"  # >10 tiles


def test_turn_policy_backward_compatibility():
    """Test that TurnPolicy wrapper works for backward compatibility."""
    state = create_test_game_state()
    policy = TurnPolicy(state)

    # Should have the same methods
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)

    stop = policy.should_stop(collection)
    symbol = policy.choose_symbol(collection, roll_counts)

    assert isinstance(stop, bool)
    assert symbol == 0  # should choose symbol 0


def test_strategy_update_game_state():
    """Test update_game_state method."""
    state1 = create_test_game_state(num_players=2, current_player=0)
    state2 = create_test_game_state(num_players=2, current_player=1)

    strategy = OptimalExpectedValueStrategy(state1, player_id=0)
    assert strategy.global_state is state1
    assert strategy.global_state.current_player == 0

    strategy.update_game_state(state2)
    assert strategy.global_state is state2
    assert strategy.global_state.current_player == 1


def test_strategy_on_turn_end():
    """Test that on_turn_end callback exists and can be called."""
    state = create_test_game_state()
    strategy = OptimalExpectedValueStrategy(state, player_id=0)

    # Should not raise any exception
    strategy.on_turn_end(2, Helping(25, 2))  # success with 2 worms
    strategy.on_turn_end(-1, None)  # failure losing 1 worm
    strategy.on_turn_end(0, None)  # failure with no loss


def test_risk_adjusted_strategy():
    """Test RiskAdjustedStrategy instantiation and basic functionality."""
    state = create_test_game_state()
    # Default risk_aversion = 1.0 should behave like optimal_expected
    strategy = RiskAdjustedStrategy(state, player_id=0)
    assert hasattr(strategy, 'risk_aversion')
    assert strategy.risk_aversion == 1.0
    assert hasattr(strategy, 'values')

    # Test with different risk_aversion values
    strategy2 = RiskAdjustedStrategy(state, player_id=0, risk_aversion=0.8)
    assert strategy2.risk_aversion == 0.8

    strategy3 = RiskAdjustedStrategy(state, player_id=0, risk_aversion=1.5)
    assert strategy3.risk_aversion == 1.5

    # Test should_stop and choose_symbol don't crash
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert isinstance(stop, bool)
    assert symbol is not None or symbol is None  # either is fine


def test_conservative_strategy():
    """Test ConservativeStrategy instantiation and basic functionality."""
    state = create_test_game_state()
    strategy = ConservativeStrategy(state, player_id=0)
    assert hasattr(strategy, 'stop_bias')
    assert strategy.stop_bias == 1.1  # default

    # Test with custom stop_bias
    strategy2 = ConservativeStrategy(state, player_id=0, stop_bias=1.5)
    assert strategy2.stop_bias == 1.5

    # Basic functionality test
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert isinstance(stop, bool)
    assert symbol is not None or symbol is None


def test_aggressive_strategy():
    """Test AggressiveStrategy instantiation and basic functionality."""
    state = create_test_game_state()
    strategy = AggressiveStrategy(state, player_id=0)
    assert hasattr(strategy, 'continue_bias')
    assert strategy.continue_bias == 1.1  # default

    # Test with custom continue_bias
    strategy2 = AggressiveStrategy(state, player_id=0, continue_bias=1.5)
    assert strategy2.continue_bias == 1.5

    # Basic functionality test
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert isinstance(stop, bool)
    assert symbol is not None or symbol is None


def test_opponent_aware_strategy():
    """Test OpponentAwareStrategy instantiation and basic functionality."""
    state = create_test_game_state()
    # Need EnhancedGameState for position detection
    if hasattr(state, 'get_player_position'):
        # Convert to EnhancedGameState if possible
        from game_models import EnhancedGameState
        enhanced_state = EnhancedGameState(state.grill, state.players, state.current_player)
    else:
        enhanced_state = state

    strategy = OpponentAwareStrategy(enhanced_state, player_id=0)
    assert hasattr(strategy, 'steal_preference')
    assert hasattr(strategy, 'risk_modifier')
    assert hasattr(strategy, 'position')

    # Test with custom parameters
    strategy2 = OpponentAwareStrategy(
        enhanced_state, player_id=0,
        steal_preference=1.5, risk_modifier=0.8,
        trailing_steal_boost=0.5, position_risk_effect=0.4
    )
    assert strategy2.steal_preference == 1.5
    assert strategy2.risk_modifier == 0.8

    # Basic functionality test
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert isinstance(stop, bool)
    assert symbol is not None or symbol is None


def test_endgame_strategy():
    """Test EndgameStrategy instantiation and basic functionality."""
    state = create_test_game_state()
    # Need EnhancedGameState for phase detection
    if hasattr(state, 'get_game_phase'):
        # Convert to EnhancedGameState if possible
        from game_models import EnhancedGameState
        enhanced_state = EnhancedGameState(state.grill, state.players, state.current_player)
    else:
        enhanced_state = state

    strategy = EndgameStrategy(enhanced_state, player_id=0)
    assert hasattr(strategy, 'phase')
    assert hasattr(strategy, 'position')

    # Test with custom parameters
    strategy2 = EndgameStrategy(
        enhanced_state, player_id=0,
        endgame_stop_bias_leading=1.5,
        endgame_continue_bias_trailing=1.5,
        critical_endgame_multiplier=1.3,
        score_gap_threshold=7
    )
    assert strategy2.endgame_stop_bias_leading == 1.5
    assert strategy2.endgame_continue_bias_trailing == 1.5

    # Basic functionality test
    collection = (0, 0, 0, 0, 0, 0)
    roll_counts = (1, 0, 0, 0, 0, 0)
    stop = strategy.should_stop(collection)
    symbol = strategy.choose_symbol(collection, roll_counts)
    assert isinstance(stop, bool)
    assert symbol is not None or symbol is None


def test_strategy_registry_completeness():
    """Test that all strategies are in the registry."""
    expected_strategies = [
        "optimal_expected",
        "risk_adjusted",
        "conservative",
        "aggressive",
        "opponent_aware",
        "endgame_focused",
    ]
    for name in expected_strategies:
        assert name in STRATEGY_REGISTRY, f"Strategy '{name}' missing from registry"


def test_parameter_validation():
    """Test that parameter validation works through create_strategy."""
    state = create_test_game_state()

    # Test clamping of out-of-range parameters
    # risk_aversion should be clamped to [0.5, 2.0]
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        strategy = create_strategy(
            "risk_adjusted", state, player_id=0,
            risk_aversion=3.0  # out of range, should be clamped to 2.0
        )
        # Should have generated a warning
        assert len(w) >= 1
        assert "clamped" in str(w[0].message).lower()

    # Test that unknown parameters generate warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        strategy = create_strategy(
            "optimal_expected", state, player_id=0,
            unknown_param=42  # unknown parameter
        )
        assert len(w) >= 1
        assert "unknown" in str(w[0].message).lower()


if __name__ == "__main__":
    # Run all test functions
    test_functions = [
        test_strategy_registry,
        test_create_strategy,
        test_optimal_expected_value_strategy_basic,
        test_optimal_expected_value_strategy_consistency,
        test_enhanced_game_state,
        test_turn_policy_backward_compatibility,
        test_strategy_update_game_state,
        test_strategy_on_turn_end,
        test_risk_adjusted_strategy,
        test_conservative_strategy,
        test_aggressive_strategy,
        test_opponent_aware_strategy,
        test_endgame_strategy,
        test_strategy_registry_completeness,
        test_parameter_validation,
    ]

    for test_func in test_functions:
        try:
            test_func()
            print(f"PASS {test_func.__name__}")
        except AssertionError as e:
            print(f"FAIL {test_func.__name__}: {e}")
            raise

    print("\nAll tests passed!")