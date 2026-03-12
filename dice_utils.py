# dice_utils.py
# Precomputation of all dice roll outcomes and dice collections
import math
import itertools
from game_models import SYM_VALUE, SYM_TO_ID, NUM_DICE, ID_TO_SYM

# ------------------------------------------------------------
# Multinomial probability
# ------------------------------------------------------------
def multinomial_prob(counts, n):
    """
    Probability of a specific dice outcome given by counts (list of 6 ints)
    when rolling n fair dice.
    """
    numerator = math.factorial(n)
    denominator = 6 ** n
    for c in counts:
        numerator //= math.factorial(c)
    return numerator / denominator

# ------------------------------------------------------------
# Generate all possible roll outcomes for each number of dice
# ------------------------------------------------------------
def generate_all_rolls(max_dice):
    """
    Returns a list outcomes_by_dice[d] for d = 0..max_dice.
    Each entry is a list of (counts_tuple, probability) for all possible rolls
    with exactly d dice.
    """
    outcomes_by_dice = [[] for _ in range(max_dice + 1)]

    def gen_counts(remaining, idx, current):
        if idx == 5:  # last symbol (worm)
            current.append(remaining)
            counts = tuple(current)
            prob = multinomial_prob(counts, remaining + sum(current[:-1]))
            # Actually total dice = remaining + sum of previous = sum(current)
            # But we already know total = sum(current). Let's compute properly.
            # We'll compute probability after we have full counts.
            # Let's instead compute probability after constructing full tuple.
            # We'll just use the final counts.
            outcomes_by_dice[sum(counts)].append((counts, multinomial_prob(counts, sum(counts))))
            current.pop()
            return
        for v in range(remaining + 1):
            current.append(v)
            gen_counts(remaining - v, idx + 1, current)
            current.pop()

    # Start with total dice = d, but we need to generate for all d.
    # We'll generate for each total d by setting remaining = d.
    for d in range(max_dice + 1):
        gen_counts(d, 0, [])
    return outcomes_by_dice

# ------------------------------------------------------------
# Generate all possible dice collections (states within a turn)
# ------------------------------------------------------------
def generate_all_collections(max_dice):
    """
    Returns a list of all 6‑tuples (c1,c2,c3,c4,c5,c_worm) such that
    the total number of dice taken is between 0 and max_dice inclusive.
    """
    collections = []

    def recurse(idx, current, current_sum):
        if idx == 5:  # last symbol (worm)
            # worm can be 0 .. max_dice - current_sum
            for v in range(max_dice - current_sum + 1):
                current.append(v)
                collections.append(tuple(current))
                current.pop()
            return
        for v in range(max_dice - current_sum + 1):
            current.append(v)
            recurse(idx + 1, current, current_sum + v)
            current.pop()

    recurse(0, [], 0)
    # Sort by total number of dice taken (sum of tuple)
    collections.sort(key=lambda c: sum(c))
    return collections

# ------------------------------------------------------------
# Helper functions for collections
# ------------------------------------------------------------
def collection_sum(c):
    """Total point value of a collection (tuple of 6 counts)."""
    total = 0
    for i, count in enumerate(c):
        sym = ID_TO_SYM[i]   # need ID_TO_SYM; we'll import at top
        total += SYM_VALUE[sym] * count
    return total

def has_worm(c):
    """Return True if the collection contains at least one worm."""
    return c[SYM_TO_ID['worm']] > 0

# ------------------------------------------------------------
# Precomputed global data (computed once at module load)
# ------------------------------------------------------------
ROLL_OUTCOMES = generate_all_rolls(NUM_DICE)
ALL_COLLECTIONS = generate_all_collections(NUM_DICE)
COLLECTION_TO_IDX = {c: i for i, c in enumerate(ALL_COLLECTIONS)}