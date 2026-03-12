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