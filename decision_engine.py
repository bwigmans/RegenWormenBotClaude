# decision_engine.py
# Reward functions for stopping and failing a turn

import math
from game_models import SYM_VALUE, SYM_TO_ID, ID_TO_SYM
from dice_utils import collection_sum, has_worm

def utility_transform(worms, alpha):
    """
    Power utility function: U(w) = sign(w) * |w|^α

    Args:
        worms: Worm count (can be negative for failure penalty)
        alpha: Risk aversion parameter (0.5 ≤ α ≤ 2.0)

    Returns:
        Utility-transformed value
    """
    if worms >= 0:
        return worms ** alpha
    else:
        return -((-worms) ** alpha)


# ------------------------------------------------------------
# Failure reward
# ------------------------------------------------------------
def failure_reward(collection, global_state):
    """
    Reward when the turn fails (no worm taken or no helping obtained).
    The player loses their top helping (if any) and it is returned to the grill.
    Returns negative of the top helping's worms, or 0 if the player has no stack.
    """
    player = global_state.players[global_state.current_player]
    if player.top_helping:
        return -player.top_helping.worms
    return 0

# ------------------------------------------------------------
# Stop reward
# ------------------------------------------------------------
def stop_reward(collection, global_state, return_source=False):
    """
    Net worm change if the player stops now with the given dice collection.

    Args:
        collection: 6-tuple of dice counts
        global_state: Current game state
        return_source: If True, return (reward, source) where source is
            'grill', 'steal', or 'failure'. Default False returns reward only.

    Returns:
        Reward (worm count) or (reward, source) tuple
    """
    # No worm -> automatic failure
    if not has_worm(collection):
        reward = failure_reward(collection, global_state)
        return (reward, 'failure') if return_source else reward

    s = collection_sum(collection)

    # 1. Exact match on a face‑up helping on the grill
    for h in global_state.visible_grill():
        if h.number == s:
            return (h.worms, 'grill') if return_source else h.worms

    # 2. Exact match on an opponent's top helping
    for _, h in global_state.other_top_helpings():
        if h.number == s:
            return (h.worms, 'steal') if return_source else h.worms

    # 3. Next lower face‑up helping on the grill
    visible = sorted(global_state.visible_grill(), key=lambda h: h.number)
    lower = [h for h in visible if h.number < s]
    if lower:
        h = lower[-1]
        return (h.worms, 'grill') if return_source else h.worms

    # 4. No lower available → failure
    reward = failure_reward(collection, global_state)
    return (reward, 'failure') if return_source else reward

# decision_engine.py (continued)
# Dynamic programming for expected values

from dice_utils import ROLL_OUTCOMES, ALL_COLLECTIONS, COLLECTION_TO_IDX
from game_models import NUM_DICE

def compute_turn_values(global_state):
    """
    Returns a dict {collection: expected worms from this collection onward}
    given the current global state (grill, opponents' tops).
    """
    num_coll = len(ALL_COLLECTIONS)
    values = [0.0] * num_coll

    # Group collection indices by remaining dice (NUM_DICE - total dice taken)
    by_rem = [[] for _ in range(NUM_DICE + 1)]
    for idx, coll in enumerate(ALL_COLLECTIONS):
        taken = sum(coll)
        rem = NUM_DICE - taken
        by_rem[rem].append(idx)

    # Process from 0 remaining dice upward
    for rem in range(NUM_DICE + 1):
        for idx in by_rem[rem]:
            coll = ALL_COLLECTIONS[idx]
            stop_val = stop_reward(coll, global_state)

            if rem == 0:
                values[idx] = stop_val
                continue

            # Expected value if we continue
            cont_val = 0.0
            for outcome_counts, prob in ROLL_OUTCOMES[rem]:
                # Available symbols: those not yet taken and present in outcome
                available = [
                    s for s in range(6)
                    if coll[s] == 0 and outcome_counts[s] > 0
                ]

                if not available:
                    # No new symbol can be taken → forced failure
                    cont_val += prob * failure_reward(coll, global_state)
                else:
                    # Choose the best symbol to take
                    best_for_outcome = -float('inf')
                    for s in available:
                        new_coll = list(coll)
                        new_coll[s] += outcome_counts[s]
                        new_coll = tuple(new_coll)
                        new_val = values[COLLECTION_TO_IDX[new_coll]]
                        if new_val > best_for_outcome:
                            best_for_outcome = new_val
                    cont_val += prob * best_for_outcome

            # Optimal action: stop now or continue
            values[idx] = max(stop_val, cont_val)

    # Return dictionary mapping each collection to its value
    return {coll: values[i] for i, coll in enumerate(ALL_COLLECTIONS)}


def compute_turn_utilities(global_state, risk_aversion=1.0):
    """
    Compute expected utilities with risk-adjusted preferences.

    Returns dict {collection: expected_utility} where utility is
    U(worms) = sign(worms) * |worms|^risk_aversion.

    Args:
        global_state: Current game state
        risk_aversion: α ∈ [0.5, 2.0], default 1.0 (risk-neutral)

    Returns:
        Dictionary mapping collections to expected utilities
    """
    num_coll = len(ALL_COLLECTIONS)
    values = [0.0] * num_coll

    # Group collection indices by remaining dice (NUM_DICE - total dice taken)
    by_rem = [[] for _ in range(NUM_DICE + 1)]
    for idx, coll in enumerate(ALL_COLLECTIONS):
        taken = sum(coll)
        rem = NUM_DICE - taken
        by_rem[rem].append(idx)

    # Process from 0 remaining dice upward
    for rem in range(NUM_DICE + 1):
        for idx in by_rem[rem]:
            coll = ALL_COLLECTIONS[idx]
            stop_val = utility_transform(stop_reward(coll, global_state), risk_aversion)

            if rem == 0:
                values[idx] = stop_val
                continue

            # Expected utility if we continue
            cont_val = 0.0
            for outcome_counts, prob in ROLL_OUTCOMES[rem]:
                # Available symbols: those not yet taken and present in outcome
                available = [
                    s for s in range(6)
                    if coll[s] == 0 and outcome_counts[s] > 0
                ]

                if not available:
                    # No new symbol can be taken → forced failure
                    cont_val += prob * utility_transform(failure_reward(coll, global_state), risk_aversion)
                else:
                    # Choose the best symbol to take
                    best_for_outcome = -float('inf')
                    for s in available:
                        new_coll = list(coll)
                        new_coll[s] += outcome_counts[s]
                        new_coll = tuple(new_coll)
                        new_val = values[COLLECTION_TO_IDX[new_coll]]
                        if new_val > best_for_outcome:
                            best_for_outcome = new_val
                    cont_val += prob * best_for_outcome

            # Optimal action: stop now or continue
            values[idx] = max(stop_val, cont_val)

    # Return dictionary mapping each collection to its utility value
    return {coll: values[i] for i, coll in enumerate(ALL_COLLECTIONS)}


def compute_turn_values_with_continue_bias(global_state, continue_bias=1.0):
    """
    Compute expected values with continue bias in decision comparison.

    Args:
        global_state: Current game state
        continue_bias: Factor to multiply continuation value by, default 1.0
            - continue_bias > 1.0: continuation relatively more attractive
            - continue_bias = 1.0: no bias (same as compute_turn_values)
            - continue_bias < 1.0: continuation relatively less attractive

    Returns:
        Dictionary mapping collections to expected values with bias applied
    """
    num_coll = len(ALL_COLLECTIONS)
    values = [0.0] * num_coll

    # Group collection indices by remaining dice (NUM_DICE - total dice taken)
    by_rem = [[] for _ in range(NUM_DICE + 1)]
    for idx, coll in enumerate(ALL_COLLECTIONS):
        taken = sum(coll)
        rem = NUM_DICE - taken
        by_rem[rem].append(idx)

    # Process from 0 remaining dice upward
    for rem in range(NUM_DICE + 1):
        for idx in by_rem[rem]:
            coll = ALL_COLLECTIONS[idx]
            stop_val = stop_reward(coll, global_state)

            if rem == 0:
                values[idx] = stop_val
                continue

            # Expected value if we continue
            cont_val = 0.0
            for outcome_counts, prob in ROLL_OUTCOMES[rem]:
                # Available symbols: those not yet taken and present in outcome
                available = [
                    s for s in range(6)
                    if coll[s] == 0 and outcome_counts[s] > 0
                ]

                if not available:
                    # No new symbol can be taken → forced failure
                    cont_val += prob * failure_reward(coll, global_state)
                else:
                    # Choose the best symbol to take
                    best_for_outcome = -float('inf')
                    for s in available:
                        new_coll = list(coll)
                        new_coll[s] += outcome_counts[s]
                        new_coll = tuple(new_coll)
                        new_val = values[COLLECTION_TO_IDX[new_coll]]
                        if new_val > best_for_outcome:
                            best_for_outcome = new_val
                    cont_val += prob * best_for_outcome

            # Optimal action: stop now or continue (with bias applied to continuation)
            values[idx] = max(stop_val, cont_val * continue_bias)

    # Return dictionary mapping each collection to its value
    return {coll: values[i] for i, coll in enumerate(ALL_COLLECTIONS)}


# decision_engine.py (continued)
# TurnPolicy class for making optimal decisions

class TurnPolicy:
    """
    Given a global game state, computes expected values for all dice collections
    and provides methods to decide whether to stop and which symbol to take.
    """
    def __init__(self, global_state):
        """
        Args:
            global_state: GameState object representing the current visible state.
        """
        self.global_state = global_state
        self.values = compute_turn_values(global_state)  # dict {collection: expected_value}

    def should_stop(self, collection):
        """
        Determine whether stopping now is optimal.

        Args:
            collection: list or tuple of 6 integers (counts of symbols 1..5, worm).

        Returns:
            True if stopping yields expected value equal to the optimal value,
            False if continuing is better.
        """
        coll = tuple(collection)
        stop_val = stop_reward(coll, self.global_state)
        # Because self.values[coll] = max(stop_val, continuation_value),
        # equality (within tolerance) means stop is optimal.
        return abs(self.values[coll] - stop_val) < 1e-9

    def choose_symbol(self, collection, roll_counts):
        """
        Given the current collection and the outcome of a new roll,
        return the best symbol to take.

        Args:
            collection: list or tuple of 6 integers (current counts).
            roll_counts: list of 6 integers (counts of each symbol in the new roll).

        Returns:
            Integer 0..5 representing the best symbol to take,
            or None if no new symbol is available.
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