

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dice_utils import (
    multinomial_prob, generate_all_rolls, generate_all_collections,
    collection_sum, has_worm, ROLL_OUTCOMES, ALL_COLLECTIONS, COLLECTION_TO_IDX
)
from game_models import SYM_TO_ID, NUM_DICE

def test_multinomial_prob():
    # With 2 dice, probability of two worms: (1/6)^2 = 1/36
    counts = [0,0,0,0,0,2]
    assert abs(multinomial_prob(counts, 2) - 1/36) < 1e-9

def test_generate_all_rolls():
    outcomes = generate_all_rolls(2)
    # For 2 dice, total probability should sum to 1
    total_prob = sum(p for _, p in outcomes[2])
    assert abs(total_prob - 1.0) < 1e-9
    # There should be 6^2 = 36 outcomes, but grouped by counts, number of distinct count tuples = 28? Actually combos with repetition: C(6+2-1,2)=21. Let's check length.
    assert len(outcomes[2]) == 21  # number of combinations with repetition

def test_generate_all_collections():
    colls = generate_all_collections(8)
    # Number of collections: stars and bars: C(6+8,8)=C(14,8)=3003? Wait, that's for all nonnegative solutions to sum <=8, but we have sum from 0 to 8. Actually number of solutions to sum = k for each k: C(6+k-1,k). Sum over k=0..8 gives sum_{k=0..8} C(k+5,k). This is known to be C(8+6,8)=C(14,8)=3003? Let's compute: C(14,8)=3003. So total collections = 3003. But we previously thought 1287. Which is correct? Let's derive: For each k, number of ways to allocate k indistinguishable dice to 6 symbols = C(k+5,5). Sum k=0..8 C(k+5,5). This sum is C(8+6,6) = C(14,6) = 3003? Actually combinatorial identity: sum_{k=0}^{n} C(k+r, r) = C(n+r+1, r+1). Here r=5, n=8 => sum = C(8+5+1,5+1) = C(14,6)=3003. So yes, 3003 collections. But the code uses generate_all_collections which recursively generates all possibilities. We should check that length is 3003.
    assert len(colls) == 3003
    # Check that they are sorted by sum
    sums = [sum(c) for c in colls]
    assert sums == sorted(sums)

def test_collection_sum():
    # Collection (1,0,0,0,0,0) should sum to 1
    assert collection_sum((1,0,0,0,0,0)) == 1
    # (0,0,0,0,0,1) should sum to 5
    assert collection_sum((0,0,0,0,0,1)) == 5
    # (1,1,1,1,1,1) sum = 1+2+3+4+5+5 = 20
    assert collection_sum((1,1,1,1,1,1)) == 20

def test_has_worm():
    assert has_worm((0,0,0,0,0,1)) == True
    assert has_worm((0,0,0,0,0,0)) == False