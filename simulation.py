"""
Simulation of a Regenwormen game between two optimal players.
Generates a play‑by‑play commentary with humorous remarks.
"""
import random
from typing import Tuple, Optional, List
from game_models import Helping, Player, GameState, EnhancedGameState, SYM_TO_ID, ID_TO_SYM, NUM_DICE, HELPINGS
from dice_utils import collection_sum, has_worm
from strategies import BaseTurnStrategy, create_strategy, OptimalExpectedValueStrategy, TurnPolicy
from decision_engine import stop_reward, compute_turn_values
from strategy_benchmark import DecisionTracker

class Commentator:
    """Generates funny commentary for game events."""

    def __init__(self, seed=None):
        if seed is not None:
            random.seed(seed)

    def turn_start(self, player: int, visible_grill: List[int]):
        phrases = [
            f"Player {player} steps up to the dice, eyeing grill {visible_grill}.",
            f"Player {player} licks their lips, ready to roll.",
            f"Player {player} cracks their knuckles. Let's roll!",
            f"The crowd holds its breath as Player {player} prepares to roll.",
        ]
        return random.choice(phrases)

    def roll_result(self, player: int, counts: Tuple[int, ...]):
        # Convert counts to readable dice faces
        faces = []
        for i, c in enumerate(counts):
            if c == 0:
                continue
            sym = ID_TO_SYM[i]
            faces.extend([str(sym)] * c)
        desc = ", ".join(faces)
        phrases = [
            f"Player {player} rolls: {desc}. Interesting...",
            f"The dice clatter: {desc}. Player {player} nods.",
            f"Roll result: {desc}. Player {player} ponders.",
            f"Ka-chunk! {desc}. Player {player} scratches their head.",
        ]
        return random.choice(phrases)

    def choose_symbol(self, player: int, symbol: int):
        sym_name = ID_TO_SYM[symbol]
        phrases = [
            f"Player {player} decides to keep the {sym_name}s. Wise choice?",
            f"Player {player} grabs the {sym_name}s. Greedy!",
            f"{sym_name}s it is! Player {player} smirks.",
            f"Player {player} selects the {sym_name}s. The audience gasps.",
        ]
        return random.choice(phrases)

    def stop_success_grill(self, player: int, helping: Helping):
        phrases = [
            f"Player {player} snatches tile {helping.number} ({helping.worms} worms) from the grill! Get in!",
            f"Tile {helping.number} ({helping.worms} worms) is mine! says Player {player} triumphantly.",
            f"Player {player} proudly takes tile {helping.number} ({helping.worms} worms). That's a nice worm.",
            f"Grill tile {helping.number} ({helping.worms} worms) disappears into Player {player}' stack. Yum!",
        ]
        return random.choice(phrases)

    def stop_success_steal(self, player: int, helping: Helping, victim: int):
        phrases = [
            f"Player {player} STEALS tile {helping.number} ({helping.worms} worms) from Player {victim}! Ouch!",
            f"Player {player} yanks tile {helping.number} ({helping.worms} worms) right from under Player {victim}'s nose! Savage!",
            f"Highway robbery! Player {player} takes tile {helping.number} ({helping.worms} worms) from Player {victim}.",
            f"Player {victim} watches helplessly as Player {player} swipes tile {helping.number} ({helping.worms} worms).",
        ]
        return random.choice(phrases)

    def failure_loss(self, player: int, lost_worms: int):
        phrases = [
            f"Player {player} fails and loses {lost_worms} worm(s) from their stack! Womp womp.",
            f"Disaster! Player {player} loses {lost_worms} worm(s). The crowd groans.",
            f"Player {player} crashes and burns, sacrificing {lost_worms} worm(s). Oof.",
            f"Epic fail! Player {player} forfeits {lost_worms} worm(s). Better luck next turn.",
        ]
        return random.choice(phrases)

    def failure_no_loss(self, player: int):
        phrases = [
            f"Player {player} fails but has no tile to lose. Tough luck.",
            f"Player {player} comes up empty-handed. At least they have nothing to lose!",
            f"Player {player} fails miserably, but their stack is safe. Silver lining?",
            f"Player {player} whiffs. Zero worms lost, zero worms gained. Exciting!",
        ]
        return random.choice(phrases)

    def turn_end(self, player: int, collection_sum: int):
        phrases = [
            f"Player {player} ends their turn with a sum of {collection_sum}.",
            f"Turn over. Player {player} managed to rack up {collection_sum} points.",
            f"Player {player} steps back with {collection_sum} points.",
            f"And that's a wrap for Player {player}. Total: {collection_sum}.",
        ]
        return random.choice(phrases)

    def game_over(self, winners, max_worms):
        if isinstance(winners, int):
            phrases = [
                f"Player {winners} wins with {max_worms} worms! All hail the Worm King!",
                f"Player {winners} triumphs with {max_worms} worms. Victory dance!",
                f"After an epic battle, Player {winners} claims victory with {max_worms} worms.",
                f"The winner is Player {winners} with {max_worms} worms. Congratulations!",
            ]
        else:
            phrases = [
                f"It's a tie between players {winners} with {max_worms} worms each! Share the glory.",
                f"Players {winners} tie with {max_worms} worms. Everyone's a winner!",
                f"Stalemate! Players {winners} each have {max_worms} worms.",
                f"No clear winner—players {winners} tie with {max_worms} worms.",
            ]
        return random.choice(phrases)

def random_roll(num_dice: int) -> Tuple[int, ...]:
    """
    Simulate rolling `num_dice` fair dice with faces 1..5 and worm.
    Returns a 6‑tuple of counts (c1, c2, c3, c4, c5, c_worm).
    """
    # Map faces to indices: 1->0, 2->1, 3->2, 4->3, 5->4, worm->5
    faces = list(range(6))  # indices
    results = random.choices(faces, k=num_dice)
    counts = [0] * 6
    for idx in results:
        counts[idx] += 1
    return tuple(counts)

def resolve_turn(collection: Tuple[int, ...], global_state: GameState) -> Tuple[int, Optional[Helping], int]:
    """
    Determine the outcome of stopping with the given collection.
    Returns:
        worm_delta: net change in worms for the current player
            (positive if a helping is taken, negative if failed).
        helping_taken: the Helping object that is taken (or None if failure).
        taken_from: index of player from whom the helping is taken,
            or -1 if taken from the grill, or -2 if failure.
    """
    # If no worm in collection, automatic failure
    if not has_worm(collection):
        return failure_reward(collection, global_state), None, -2

    s = collection_sum(collection)

    # 1. Exact match on a face‑up helping on the grill
    for h in global_state.visible_grill():
        if h.number == s:
            return h.worms, h, -1

    # 2. Exact match on an opponent's top helping
    for i, h in global_state.other_top_helpings():
        if h.number == s:
            return h.worms, h, i

    # 3. Next lower face‑up helping on the grill
    visible = sorted(global_state.visible_grill(), key=lambda h: h.number)
    lower = [h for h in visible if h.number < s]
    if lower:
        h = lower[-1]
        return h.worms, h, -1

    # 4. No lower available → failure
    return failure_reward(collection, global_state), None, -2

def failure_reward(collection, global_state):
    """Copy of decision_engine.failure_reward for local use."""
    player = global_state.players[global_state.current_player]
    if player.top_helping:
        return -player.top_helping.worms
    return 0

def apply_turn_outcome(global_state: GameState, worm_delta: int, helping_taken: Optional[Helping], taken_from: int):
    """
    Update the game state after a turn.
    - If helping_taken is not None, remove it from grill or opponent's stack.
    - If taken from grill, flip the next face‑down helping (if any).
    - If worm_delta is negative (failure), remove top helping from current player
      and return it to the grill (face‑up).
    """
    current = global_state.current_player
    players = global_state.players
    grill = global_state.grill

    if worm_delta >= 0 and helping_taken is not None:
        # Successfully took a helping
        if taken_from == -1:
            # Taken from grill: remove it
            grill.remove(helping_taken)
        else:
            # Taken from opponent's stack
            opponent = players[taken_from]
            removed = opponent.remove_top_helping()
            assert removed is helping_taken
        # Add the helping to the current player's stack
        players[current].add_helping(helping_taken)
    else:
        # Failure: worm_delta is negative (or zero if no top helping)
        # Remove top helping from current player (if any) and return to grill face‑up
        player = players[current]
        lost = None
        if player.top_helping:
            lost = player.remove_top_helping()
            # Return to grill face‑up
            lost.face_up = True
            grill.append(lost)
            # Keep grill sorted? Not necessary for logic.
        # Turn the highest face‑up helping on the grill face‑down
        # (if a helping was returned and is highest, leave it face‑up)
        flip_highest_face_down(grill, excluded_helping=lost)
    # Note: worm count is implicit in the helpings, no separate tracking needed.

def simulate_turn(global_state: GameState, policy: BaseTurnStrategy, player: int = None, commentator=None, decision_tracker: Optional[DecisionTracker] = None) -> Tuple[int, Optional[Helping], int]:
    """
    Simulate a single turn for the current player using the given policy.
    Returns the same as resolve_turn.
    If commentator is provided, call its methods for play‑by‑play commentary.
    If decision_tracker is provided, record decision events for later analysis.
    """
    if player is None:
        player = global_state.current_player
    collection = (0, 0, 0, 0, 0, 0)
    remaining = NUM_DICE - sum(collection)
    # Cache for continuation values (computed lazily)
    continuation_values = None

    while True:
        # Decide whether to stop now
        if policy.should_stop(collection):
            if decision_tracker is not None:
                if continuation_values is None:
                    continuation_values = compute_turn_values(global_state)
                stop_val = stop_reward(collection, global_state)
                cont_val = continuation_values.get(collection, 0.0)
                decision_tracker.record_stop_decision(collection, stop_val, cont_val)
            if commentator:
                print(commentator.turn_end(player, collection_sum(collection)))
            break

        # Roll dice
        roll = random_roll(remaining)
        if decision_tracker is not None:
            decision_tracker.record_roll(collection, roll)
        if commentator:
            print(commentator.roll_result(player, roll))
        # Choose symbol to take
        symbol = policy.choose_symbol(collection, roll)
        if decision_tracker is not None and symbol is not None:
            decision_tracker.record_continue_decision(collection, symbol)
        if symbol is None:
            # No new symbol can be taken → forced failure
            # According to the rules, the turn ends immediately with failure.
            # We'll break and let resolve_turn handle failure (no worm).
            if commentator:
                print(f"Player {player} cannot take any new symbol. Disaster!")
            break

        if commentator:
            print(commentator.choose_symbol(player, symbol))
        # Update collection
        new_coll = list(collection)
        new_coll[symbol] += roll[symbol]
        collection = tuple(new_coll)
        remaining = NUM_DICE - sum(collection)

    # Resolve outcome
    worm_delta, helping_taken, taken_from = resolve_turn(collection, global_state)
    if decision_tracker is not None and taken_from >= 0:
        # Successful steal attempt
        decision_tracker.record_steal_attempt(collection, taken_from, success=True)
    return worm_delta, helping_taken, taken_from

def initial_grill() -> List[Helping]:
    """Create the initial grill with all helpings face‑up."""
    grill = []
    for number, worms in HELPINGS:
        h = Helping(number, worms)
        h.face_up = True
        grill.append(h)
    return grill

def flip_next_face_down(grill: List[Helping]):
    """Flip the next face‑down helping (lowest number) face‑up."""
    face_down = [h for h in grill if not h.face_up]
    if not face_down:
        return
    lowest = min(face_down, key=lambda h: h.number)
    lowest.face_up = True

def flip_highest_face_down(grill: List[Helping], excluded_helping=None):
    """Turn the highest face‑up helping on the grill face‑down.
    If excluded_helping is provided and is the highest, do nothing."""
    face_up = [h for h in grill if h.face_up]
    if not face_up:
        return
    highest = max(face_up, key=lambda h: h.number)
    if excluded_helping is not None and highest is excluded_helping:
        return
    highest.face_up = False

def player_worms(player: Player) -> int:
    """Total worms in player's stack."""
    return sum(h.worms for h in player.stack)

def simulate_game(num_players=2, verbose=False, max_turns=1000, strategy_configs=None):
    """Simulate a full game between players using configurable strategies.

    Args:
        num_players: Number of players (default: 2)
        verbose: Whether to print commentary (default: False)
        max_turns: Maximum turns before forcing game end (default: 1000)
        strategy_configs: List of strategy configurations, one per player.
            Each config is a dict with keys:
            - "name": strategy name (default: "optimal_expected")
            - "params": dict of strategy-specific parameters (optional)
            If None, all players use "optimal_expected" strategy.
    """
    # Initialize grill
    grill = initial_grill()

    players = [Player(i) for i in range(num_players)]
    current_player = random.randrange(num_players)

    # Set up strategies
    if strategy_configs is None:
        strategy_configs = [{"name": "optimal_expected"}] * num_players
    elif len(strategy_configs) != num_players:
        raise ValueError(f"strategy_configs length ({len(strategy_configs)}) must match num_players ({num_players})")

    turn = 0
    commentator = Commentator() if verbose else None
    if verbose:
        print(f"=== REGENWORMEN SHOWDOWN ===\n{num_players} players enter, one leaves with the most worms!\n")
        print(f"Player {current_player} starts.\n")

    # Game loop
    while grill and any(h.face_up for h in grill) and turn < max_turns:  # while there are still face‑up helpings on the grill
        turn += 1
        if verbose:
            print(f"--- Turn {turn} | Player {current_player} ---")
            visible = [h.number for h in grill if h.face_up]
            print(f"Visible grill: {sorted(visible)}")
            for i, p in enumerate(players):
                worms = player_worms(p)
                top = p.top_helping.number if p.top_helping else None
                print(f"  Player {i}: {worms} worms, top tile {top}")
            print(commentator.turn_start(current_player, visible))

        # Create game state for this turn
        state = EnhancedGameState(grill, players, current_player)
        config = strategy_configs[current_player]
        policy = create_strategy(config.get("name", "optimal_expected"),
                                 state, current_player,
                                 **config.get("params", {}))
        worm_delta, helping_taken, taken_from = simulate_turn(state, policy, current_player, commentator)

        # Apply outcome
        apply_turn_outcome(state, worm_delta, helping_taken, taken_from)

        # Commentary
        if verbose:
            if worm_delta >= 0 and helping_taken:
                if taken_from == -1:
                    print(commentator.stop_success_grill(current_player, helping_taken))
                else:
                    print(commentator.stop_success_steal(current_player, helping_taken, taken_from))
            else:
                if worm_delta < 0:
                    lost = -worm_delta
                    print(commentator.failure_loss(current_player, lost))
                else:
                    print(commentator.failure_no_loss(current_player))
            print()

        # Next player
        current_player = (current_player + 1) % num_players

    if turn >= max_turns:
        if verbose:
            print(f"\nReached turn limit {max_turns}. Stopping.")
    # Game over: determine winner by total worms
    if verbose:
        print("\n=== GAME OVER ===")
    max_worms = -1
    winners = []
    for i, p in enumerate(players):
        w = player_worms(p)
        if verbose:
            print(f"Player {i}: {w} worms")
        if w > max_worms:
            max_worms = w
            winners = [i]
        elif w == max_worms:
            winners.append(i)
    if len(winners) == 1:
        if verbose:
            print(commentator.game_over(winners[0], max_worms))
        return winners[0], max_worms
    else:
        if verbose:
            print(commentator.game_over(winners, max_worms))
        return winners, max_worms

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Simulate a Regenwormen game between optimal solvers with funny commentary.")
    parser.add_argument("-n", "--players", type=int, default=2, help="Number of players (default: 2)")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible games")
    parser.add_argument("--max-turns", type=int, default=1000, help="Maximum turns before forcing game end (default: 1000)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress commentary, only output winner")
    parser.add_argument("-o", "--output", help="Save output to file (default: stdout)")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    if args.quiet:
        verbose = False
    else:
        verbose = True

    if args.output:
        import sys
        sys.stdout = open(args.output, 'w', encoding='utf-8')

    simulate_game(num_players=args.players, verbose=verbose, max_turns=args.max_turns)