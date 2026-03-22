# strategies.py
# Comprehensive strategy system for Regenwormen AI
#
# Base abstract class and strategy implementations for different AI behaviors.

from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod
import warnings
from game_models import GameState, Helping
from dice_utils import collection_sum, has_worm
from decision_engine import compute_turn_values, stop_reward, failure_reward, compute_turn_utilities, compute_turn_values_with_continue_bias, utility_transform


class BaseTurnStrategy(ABC):
    """
    Abstract base class for all turn strategies.

    Strategies decide when to stop and which symbols to take based on
    game state, risk preferences, opponent modeling, and long-term planning.
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        """
        Initialize strategy with current game state.

        Args:
            global_state: Current game state (grill, players, current player)
            player_id: ID of player using this strategy (0-based index)
            **params: Strategy-specific parameters
        """
        self.global_state = global_state
        self.player_id = player_id
        self.params = params
        self.setup()

    def setup(self):
        """Optional initialization after constructor."""
        pass

    @abstractmethod
    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """
        Determine whether to stop with current collection.

        Args:
            collection: 6-tuple (c1,c2,c3,c4,c5,c_worm) of dice taken so far

        Returns:
            True if stopping now is optimal, False if continuing is better
        """
        raise NotImplementedError

    @abstractmethod
    def choose_symbol(self, collection: Tuple[int, ...],
                     roll_counts: Tuple[int, ...]) -> Optional[int]:
        """
        Choose best symbol from roll, or None if none available.

        Args:
            collection: 6-tuple of dice taken so far
            roll_counts: 6-tuple of dice counts in the new roll

        Returns:
            Integer 0..5 representing the best symbol to take,
            or None if no new symbol is available
        """
        raise NotImplementedError

    def update_game_state(self, new_state: GameState):
        """
        Update strategy with new global state (for multi-turn planning).

        Args:
            new_state: Updated game state
        """
        self.global_state = new_state

    def on_turn_end(self, result_worms: int, helping_taken: Optional[Helping]):
        """
        Callback for end of turn (for learning/adaptation).

        Args:
            result_worms: Net worm change from the turn (positive if success,
                         negative if failure)
            helping_taken: Helping object that was taken, or None if failure
        """
        pass


class OptimalExpectedValueStrategy(BaseTurnStrategy):
    """
    Mathematically optimal strategy that maximizes expected worms
    for the current turn under perfect information.

    This is a refactored version of the original TurnPolicy class.
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        super().__init__(global_state, player_id, **params)
        # Compute expected values for all collections
        self.values = compute_turn_values(global_state)

    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """
        Determine whether stopping now is optimal.

        Because self.values[coll] = max(stop_val, continuation_value),
        equality (within tolerance) means stop is optimal.
        """
        coll = tuple(collection)
        stop_val = stop_reward(coll, self.global_state)
        # Tolerance for floating point comparison
        return abs(self.values[coll] - stop_val) < 1e-9

    def choose_symbol(self, collection: Tuple[int, ...],
                     roll_counts: Tuple[int, ...]) -> Optional[int]:
        """
        Choose the symbol that maximizes expected value.
        """
        coll = tuple(collection)
        # Symbols that are not yet taken and appear in the roll
        available = [s for s in range(6) if coll[s] == 0 and roll_counts[s] > 0]
        if not available:
            return None

        best_s = None
        best_val = -float('inf')
        for s in available:
            new_coll = list(coll)
            new_coll[s] += roll_counts[s]
            new_coll = tuple(new_coll)
            val = self.values[new_coll]
            if val > best_val:
                best_val = val
                best_s = s
        return best_s


class RiskAdjustedStrategy(OptimalExpectedValueStrategy):
    """
    Strategy that incorporates risk preferences using a power utility function.

    Computes expected utilities directly using the Bellman equation with
    utility-transformed terminal rewards: V(coll) = max( U(stop_reward), E[V(next_coll)] ).

    Parameters:
        risk_aversion (float): α ∈ [0.5, 2.0], default 1.0 (neutral)
            - α < 1.0: risk-seeking (concave utility)
            - α = 1.0: risk-neutral (linear, same as optimal_expected)
            - α > 1.0: risk-averse (convex utility)
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        # Set default risk_aversion
        self.risk_aversion = params.get("risk_aversion", 1.0)
        # Clamp to reasonable range
        self.risk_aversion = max(0.5, min(2.0, self.risk_aversion))

        # Compute expected utilities directly (not utility of expected worms)
        self.values = compute_turn_utilities(global_state, self.risk_aversion)
        # Call parent to set global_state, player_id, params (skip value computation)
        # We'll set attributes manually to avoid redundant computation
        self.global_state = global_state
        self.player_id = player_id
        self.params = params
        self.setup()

    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """
        Determine whether to stop with current collection.

        Compares utility of stopping with expected continuation utility.
        """
        coll = tuple(collection)
        stop_val = stop_reward(coll, self.global_state)
        stop_utility = utility_transform(stop_val, self.risk_aversion)
        # Tolerance for floating point comparison
        return abs(self.values[coll] - stop_utility) < 1e-9


class ConservativeStrategy(OptimalExpectedValueStrategy):
    """
    Strategy biased toward stopping early.

    Multiplies stop reward by bias factor, making stopping more attractive.

    Parameters:
        stop_bias (float): Factor to multiply stop reward by, default 1.1
            - stop_bias > 1.0: more conservative (stop earlier)
            - stop_bias = 1.0: neutral (same as optimal_expected)
            - stop_bias < 1.0: less conservative (stop later)
            Valid range: [0.5, 3.0]
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        self.stop_bias = params.get("stop_bias", 1.1)
        # Clamp to reasonable range
        self.stop_bias = max(0.5, min(3.0, self.stop_bias))
        super().__init__(global_state, player_id, **params)

    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """Stop if biased stop reward exceeds continuation value."""
        coll = tuple(collection)
        stop_val = stop_reward(coll, self.global_state) * self.stop_bias
        # Compare with original expected value (not biased)
        return abs(self.values[coll] - stop_val) < 1e-9 or stop_val > self.values[coll]


class AggressiveStrategy(OptimalExpectedValueStrategy):
    """
    Strategy biased toward continuing.

    Applies continue bias during dynamic programming comparison:
    values[coll] = max(stop_val, cont_val * continue_bias)

    Parameters:
        continue_bias (float): Factor to multiply continuation value by, default 1.1
            - continue_bias > 1.0: more aggressive (continue more)
            - continue_bias = 1.0: neutral (same as optimal_expected)
            - continue_bias < 1.0: less aggressive (stop earlier)
            Valid range: [0.5, 3.0]
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        # Set default continue_bias
        self.continue_bias = params.get("continue_bias", 1.1)
        # Clamp to reasonable range
        self.continue_bias = max(0.5, min(3.0, self.continue_bias))

        # Compute values with continue bias applied during DP
        self.values = compute_turn_values_with_continue_bias(global_state, self.continue_bias)
        # Call parent to set global_state, player_id, params (skip value computation)
        self.global_state = global_state
        self.player_id = player_id
        self.params = params
        self.setup()


class OpponentAwareStrategy(OptimalExpectedValueStrategy):
    """
    Strategy that adjusts decisions based on opponent positions and scores.

    Applies position-dependent adjustments:
    1. Steal preference enhancement: Multiply steal rewards based on position
    2. Position-based risk adjustment: Modify decision thresholds based on
       relative position (leading vs trailing)

    Parameters:
        steal_preference (float): Base multiplier for steal rewards,
            default 1.0, range [0.5, 2.0]
        risk_modifier (float): Base risk adjustment factor,
            default 1.0, range [0.5, 2.0]
            >1.0: more conservative (higher stop bias)
            <1.0: more aggressive (lower stop bias)
        trailing_steal_boost (float): Additional steal multiplier when trailing,
            default 0.3, range [0.0, 1.0]
        position_risk_effect (float): Magnitude of position-based risk adjustment,
            default 0.3, range [0.0, 0.5]
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        # Set parameters with defaults and clamping
        self.steal_preference = max(0.5, min(2.0, params.get("steal_preference", 1.0)))
        self.risk_modifier = max(0.5, min(2.0, params.get("risk_modifier", 1.0)))
        self.trailing_steal_boost = max(0.0, min(1.0, params.get("trailing_steal_boost", 0.3)))
        self.position_risk_effect = max(0.0, min(0.5, params.get("position_risk_effect", 0.3)))

        super().__init__(global_state, player_id, **params)

        # Get player position and score difference
        if hasattr(global_state, 'get_player_position'):
            self.position, self.score_diff = global_state.get_player_position(player_id)
        else:
            # Fallback if not EnhancedGameState
            self.position = 1
            self.score_diff = 0

        self._adjusted_stop_bias = self._compute_adjusted_stop_bias()

    def _compute_adjusted_stop_bias(self):
        """Compute position-adjusted stop bias using multiplicative risk_modifier."""
        position_effect = 0.0
        if self.position == 1:  # Leading
            position_effect = -self.position_risk_effect  # Less conservative
        elif self.position > 1:  # Trailing
            position_effect = self.position_risk_effect   # More conservative

        return self.risk_modifier * (1.0 + position_effect)

    def _adjusted_stop_reward(self, collection):
        """Compute stop reward with steal preference adjustments."""
        # Use extended stop_reward that returns (reward, source)
        from decision_engine import stop_reward
        base_reward, source = stop_reward(collection, self.global_state, return_source=True)
        if source == 'steal':
            # Apply steal multiplier based on position and score difference
            position_factor = 1.0
            if self.position > 1:  # Trailing
                trailing_bonus = self.trailing_steal_boost * min(0.5, abs(self.score_diff) / 10.0)
                position_factor = 1.0 + trailing_bonus
            steal_multiplier = self.steal_preference * position_factor
            return base_reward * steal_multiplier
        return base_reward

    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """
        Determine whether to stop with current collection.

        Uses adjusted stop reward (with steal preference) and position-adjusted
        bias compared to continuation value.
        """
        coll = tuple(collection)
        stop_val = self._adjusted_stop_reward(coll) * self._adjusted_stop_bias
        # Compare with original expected value (not biased)
        return abs(self.values[coll] - stop_val) < 1e-9 or stop_val > self.values[coll]


class EndgameStrategy(OptimalExpectedValueStrategy):
    """
    Strategy that adjusts behavior during endgame phase with position-dependent
    parameter switching.

    Uses enhanced phase detection considering tile count, score gaps, and
    remaining high-value tiles.

    Parameters:
        endgame_stop_bias_leading (float): Stop bias multiplier when leading
            in endgame, default 1.3, range [0.5, 3.0]
        endgame_continue_bias_trailing (float): Continue bias multiplier when
            trailing in endgame, default 1.3, range [0.5, 3.0]
        critical_endgame_multiplier (float): Additional multiplier for critical
            endgame, default 1.2, range [1.0, 2.0]
        score_gap_threshold (int): Score difference for "large lead" detection,
            default 5, range [0, 10]
    """

    def __init__(self, global_state: GameState, player_id: int, **params):
        # Set parameters with defaults and clamping
        self.endgame_stop_bias_leading = max(0.5, min(3.0, params.get("endgame_stop_bias_leading", 1.3)))
        self.endgame_continue_bias_trailing = max(0.5, min(3.0, params.get("endgame_continue_bias_trailing", 1.3)))
        self.critical_multiplier = max(1.0, min(2.0, params.get("critical_endgame_multiplier", 1.2)))
        self.score_gap_threshold = max(0, min(10, params.get("score_gap_threshold", 5)))

        super().__init__(global_state, player_id, **params)

        # Get player position and score difference
        if hasattr(global_state, 'get_player_position'):
            self.position, self.score_diff = global_state.get_player_position(player_id)
        else:
            self.position = 1
            self.score_diff = 0

        # Detect game phase
        self.phase = self._detect_phase()
        self._apply_phase_adjustments()

    def _detect_phase(self):
        """Determine game phase using enhanced detection."""
        if hasattr(self.global_state, 'get_enhanced_game_phase'):
            return self.global_state.get_enhanced_game_phase(score_gap_threshold=self.score_gap_threshold)
        else:
            # Fallback to basic phase detection
            if hasattr(self.global_state, 'get_game_phase'):
                return self.global_state.get_game_phase()
            else:
                return "mid"  # Default if no phase detection available

    def _apply_phase_adjustments(self):
        """Apply parameter adjustments based on phase and position."""
        self.stop_bias = 1.0  # Default no bias
        self.continue_bias = 1.0  # Default no bias

        if self.phase in ["endgame", "critical_endgame"]:
            multiplier = self.critical_multiplier if self.phase == "critical_endgame" else 1.0
            if self.position == 1:  # Leading
                # More conservative: increase stop bias
                self.stop_bias = self.endgame_stop_bias_leading * multiplier
            else:  # Trailing
                # More aggressive: increase continue bias
                self.continue_bias = self.endgame_continue_bias_trailing * multiplier
        # Early/mid game: keep default biases (1.0)

    def should_stop(self, collection: Tuple[int, ...]) -> bool:
        """
        Determine whether to stop with current collection.

        Applies phase- and position-adjusted biases.
        """
        coll = tuple(collection)
        stop_val = stop_reward(coll, self.global_state) * self.stop_bias

        # For trailing aggressive endgame, we need to compare with biased continuation value
        if self.continue_bias != 1.0 and self.phase in ["endgame", "critical_endgame"] and self.position > 1:
            # Recompute values with continue bias (similar to AggressiveStrategy)
            from decision_engine import compute_turn_values_with_continue_bias
            biased_values = compute_turn_values_with_continue_bias(self.global_state, self.continue_bias)
            cont_val = biased_values[coll]
        else:
            cont_val = self.values[coll]

        return abs(cont_val - stop_val) < 1e-9 or stop_val > cont_val


# Strategy registry and factory
STRATEGY_REGISTRY: Dict[str, type] = {
    "optimal_expected": OptimalExpectedValueStrategy,
    "risk_adjusted": RiskAdjustedStrategy,
    "conservative": ConservativeStrategy,
    "aggressive": AggressiveStrategy,
    "opponent_aware": OpponentAwareStrategy,
    "endgame_focused": EndgameStrategy,
}

# Parameter validation ranges for each strategy
PARAMETER_RANGES: Dict[str, Dict[str, tuple]] = {
    "risk_adjusted": {
        "risk_aversion": (0.5, 2.0),
    },
    "conservative": {
        "stop_bias": (0.5, 3.0),
    },
    "aggressive": {
        "continue_bias": (0.5, 3.0),
    },
    "opponent_aware": {
        "steal_preference": (0.5, 2.0),
        "risk_modifier": (0.5, 2.0),
        "trailing_steal_boost": (0.0, 1.0),
        "position_risk_effect": (0.0, 0.5),
    },
    "endgame_focused": {
        "endgame_stop_bias_leading": (0.5, 3.0),
        "endgame_continue_bias_trailing": (0.5, 3.0),
        "critical_endgame_multiplier": (1.0, 2.0),
        "score_gap_threshold": (0, 10),
    },
}

def validate_parameters(params: Dict[str, Any], strategy_name: str) -> Dict[str, Any]:
    """
    Validate and normalize strategy parameters.

    Args:
        params: Dictionary of parameter names to values
        strategy_name: Name of strategy for which parameters are intended

    Returns:
        Validated parameter dictionary with clamped values and warnings for
        out-of-range or unknown parameters.
    """
    validated = {}
    for key, value in params.items():
        if strategy_name in PARAMETER_RANGES and key in PARAMETER_RANGES[strategy_name]:
            min_val, max_val = PARAMETER_RANGES[strategy_name][key]
            if value < min_val or value > max_val:
                warnings.warn(
                    f"Parameter {key}={value} for strategy '{strategy_name}' "
                    f"clamped to [{min_val}, {max_val}]"
                )
                validated[key] = max(min_val, min(max_val, value))
            else:
                validated[key] = value
        else:
            warnings.warn(
                f"Unknown parameter '{key}' for strategy '{strategy_name}'. "
                "This parameter will be ignored."
            )
    return validated


def create_strategy(strategy_name: str, global_state: GameState,
                   player_id: int, **params) -> BaseTurnStrategy:
    """
    Factory function to create strategies by name.

    Args:
        strategy_name: Name of strategy (must be in STRATEGY_REGISTRY)
        global_state: Current game state
        player_id: ID of player using this strategy
        **params: Strategy-specific parameters

    Returns:
        Instance of the requested strategy

    Raises:
        KeyError: If strategy_name is not in registry
    """
    cls = STRATEGY_REGISTRY[strategy_name]
    validated_params = validate_parameters(params, strategy_name)
    return cls(global_state, player_id, **validated_params)


# Backward compatibility wrapper
class TurnPolicy:
    """
    Legacy wrapper for backward compatibility.

    Deprecated: Use OptimalExpectedValueStrategy instead.
    """

    def __init__(self, global_state: GameState):
        import warnings
        warnings.warn(
            "TurnPolicy is deprecated. Use OptimalExpectedValueStrategy instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self._strategy = OptimalExpectedValueStrategy(global_state, global_state.current_player)

    def should_stop(self, collection):
        return self._strategy.should_stop(collection)

    def choose_symbol(self, collection, roll_counts):
        return self._strategy.choose_symbol(collection, roll_counts)