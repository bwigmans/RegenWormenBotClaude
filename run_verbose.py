#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')
from simulation import simulate_game
import random

random.seed(42)
simulate_game(num_players=5, verbose=True, max_turns=5)