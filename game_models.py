# game_models.py
# Constants and data structures for Pikomino/Regenwormen optimal play

# ------------------------------------------------------------
# Game constants
# ------------------------------------------------------------
SYMBOLS = [1, 2, 3, 4, 5, 'worm']
SYM_TO_ID = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 'worm': 5}
ID_TO_SYM = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 'worm'}
SYM_VALUE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 'worm': 5}
NUM_DICE = 8

# Full helpings list (number, worms)
HELPINGS = [(21, 1), (22, 1), (23, 1), (24, 1), (25, 2), (26, 2), (27, 2), (28, 2),
            (29, 3), (30, 3), (31, 3), (32, 3), (33, 4), (34, 4), (35, 4), (36, 4)]

# ------------------------------------------------------------
# Game state classes
# ------------------------------------------------------------
class Helping:
    """A single helping tile with a number and worm count."""
    def __init__(self, number, worms):
        self.number = number
        self.worms = worms
        self.face_up = True   # whether it's face‑up on the grill

    def __repr__(self):
        return f"H({self.number},{self.worms})"

class Player:
    """A player with a stack of collected helpings."""
    def __init__(self, pid):
        self.pid = pid
        self.stack = []       # list of Helping objects, oldest first

    @property
    def top_helping(self):
        """The topmost (most recently acquired) helping, or None if stack is empty."""
        return self.stack[-1] if self.stack else None

    def add_helping(self, h):
        """Add a helping to the top of the stack."""
        h.face_up = True      # when taken, it's face up (not needed for this module but kept)
        self.stack.append(h)

    def remove_top_helping(self):
        """Remove and return the top helping (when it is stolen)."""
        return self.stack.pop()

class GameState:
    """Complete visible state of the game at the start of a turn."""
    def __init__(self, grill, players, current_player, turn_dice=None):
        self.grill = grill                # list of all Helping objects (some face‑down)
        self.players = players            # list of Player objects
        self.current_player = current_player  # index of player whose turn it is
        self.turn_dice = turn_dice         # not used for decision, kept for completeness

    def visible_grill(self):
        """Return list of helpings that are currently face‑up on the grill."""
        return [h for h in self.grill if h.face_up]

    def other_top_helpings(self):
        """Return list of (player_index, helping) for opponents' top helpings."""
        result = []
        for i, p in enumerate(self.players):
            if i != self.current_player and p.top_helping:
                result.append((i, p.top_helping))
        return result


class EnhancedGameState(GameState):
    """
    Extended game state with additional information for strategy support.

    Adds game phase detection, player position tracking, and score computation.
    """

    def __init__(self, grill, players, current_player, turn_dice=None):
        super().__init__(grill, players, current_player, turn_dice)
        self.turn_number = 0
        self.scores = self._compute_scores()

    def _compute_scores(self):
        """Compute worm scores for all players."""
        return [sum(h.worms for h in p.stack) for p in self.players]

    def get_player_position(self, player_id):
        """
        Return position (1st, 2nd, etc.) and score difference.

        Args:
            player_id: Player index (0-based)

        Returns:
            Tuple of (position, score_difference) where:
            - position: 1 for first place, 2 for second, etc.
            - score_difference: Positive if leading next player (score above
              player immediately behind), negative if trailing (score below
              player immediately ahead), 0 if tied or single player.
        """
        scores = self.scores
        player_score = scores[player_id]
        sorted_scores = sorted(scores, reverse=True)
        position = sorted_scores.index(player_score) + 1

        if position == 1:
            # Leading: compare to next lower score
            next_lower = sorted_scores[1] if len(sorted_scores) > 1 else 0
            score_diff = player_score - next_lower
        else:
            # Trailing: compare to next higher score
            next_higher = sorted_scores[position - 2]  # position-2 is index of player ahead
            score_diff = player_score - next_higher

        return position, score_diff

    def get_game_phase(self):
        """
        Return game phase: early, mid, endgame.

        Based on number of face-up tiles on grill:
        - early: >10 face-up tiles
        - mid: 4-10 face-up tiles
        - endgame: ≤3 face-up tiles
        """
        face_up_tiles = len([h for h in self.grill if h.face_up])
        if face_up_tiles > 10:
            return "early"
        elif face_up_tiles > 3:
            return "mid"
        else:
            return "endgame"

    def get_enhanced_game_phase(self, score_gap_threshold=5):
        """
        Enhanced phase detection considering multiple factors.

        Returns one of four phases: "early", "mid", "endgame", "critical_endgame".

        Args:
            score_gap_threshold: Score difference for "large lead" detection,
                default 5 worms.

        Detection logic:
        1. Early: >10 face-up tiles
        2. Mid: 4-10 face-up tiles
        3. Endgame: ≤3 face-up tiles
        4. Critical endgame: Endgame AND (no high-value tiles remain OR
           large score gap exists)

        High-value tiles: worms ≥ 3
        Large score gap: max_score - min_score ≥ score_gap_threshold
        """
        face_up = len([h for h in self.grill if h.face_up])
        if face_up > 10:
            return "early"
        elif face_up > 3:
            return "mid"
        else:
            # Endgame by tile count
            # Check if endgame characteristics are strong
            high_value_remaining = any(h.worms >= 3 for h in self.grill if h.face_up)
            scores = self.scores
            max_score = max(scores) if scores else 0
            min_score = min(scores) if scores else 0
            score_gap = max_score - min_score

            # Critical endgame if either no high-value tiles remain OR large score gap
            if not high_value_remaining or score_gap >= score_gap_threshold:
                return "critical_endgame"
            else:
                return "endgame"