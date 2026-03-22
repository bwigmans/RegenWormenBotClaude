"""
Unit tests for configuration loader.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import tempfile
import os
from config_loader import Config, PlayerConfig


def test_config_class_creation():
    """Test basic Config class creation."""
    players = [PlayerConfig(0, "optimal_expected", {})]
    config = Config(players, num_games=100, random_seed=42)
    assert len(config.players) == 1
    assert config.players[0].strategy == "optimal_expected"
    assert config.num_games == 100
    assert config.random_seed == 42