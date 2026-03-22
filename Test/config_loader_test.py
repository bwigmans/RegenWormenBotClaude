"""Tests for config_loader module."""

import json
import tempfile
import os
import pytest
from config_loader import Config, PlayerConfig, load_config_file


def test_config_class_creation():
    """Test basic Config class creation."""
    players = [PlayerConfig(0, "optimal_expected", {})]
    config = Config(players, num_games=100, random_seed=42)
    assert len(config.players) == 1
    assert config.players[0].strategy == "optimal_expected"
    assert config.num_games == 100
    assert config.random_seed == 42


def test_player_config_class_creation():
    """Test basic PlayerConfig class creation."""
    player = PlayerConfig(
        player_id=1,
        strategy="risk_adjusted",
        params={"risk_aversion": 0.5, "utility_power": 1.5}
    )
    assert player.player_id == 1
    assert player.strategy == "risk_adjusted"
    assert player.params == {"risk_aversion": 0.5, "utility_power": 1.5}


def test_config_default_values():
    """Test that Config uses default values correctly."""
    players = [PlayerConfig(0, "optimal_expected", {})]
    config = Config(players)
    assert config.num_games == 1000
    assert config.random_seed == 42
    assert config.verbose is False
    assert config.max_turns_per_game == 1000
    assert config.collect_decision_stats is True
    assert config.collect_worm_distribution is True
    assert config.collect_timing is False


def test_config_custom_values():
    """Test Config with custom values."""
    players = [
        PlayerConfig(0, "optimal_expected", {}),
        PlayerConfig(1, "conservative", {"stop_bias": 1.2})
    ]
    config = Config(
        players=players,
        num_games=500,
        random_seed=123,
        verbose=True,
        max_turns_per_game=500,
        collect_decision_stats=False,
        collect_worm_distribution=False,
        collect_timing=True
    )
    assert len(config.players) == 2
    assert config.players[0].strategy == "optimal_expected"
    assert config.players[1].strategy == "conservative"
    assert config.players[1].params == {"stop_bias": 1.2}
    assert config.num_games == 500
    assert config.random_seed == 123
    assert config.verbose is True
    assert config.max_turns_per_game == 500
    assert config.collect_decision_stats is False
    assert config.collect_worm_distribution is False
    assert config.collect_timing is True


def test_load_config_file_valid():
    """Test loading a valid configuration file."""
    config_data = {
        "players": [
            {"id": 0, "strategy": "optimal_expected", "params": {}},
            {"id": 1, "strategy": "conservative", "params": {"stop_bias": 1.3}}
        ],
        "game_settings": {
            "num_games": 500,
            "random_seed": 123
        }
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config_data, f)
        fname = f.name

    try:
        config = load_config_file(fname)
        assert len(config.players) == 2
        assert config.players[0].strategy == "optimal_expected"
        assert config.players[1].strategy == "conservative"
        assert config.players[1].params["stop_bias"] == 1.3
        assert config.num_games == 500
        assert config.random_seed == 123
    finally:
        os.unlink(fname)


def test_load_config_file_not_found():
    """Test loading a non-existent configuration file."""
    with pytest.raises(FileNotFoundError, match="Configuration file not found:"):
        load_config_file("/nonexistent/path/to/file.json")


def test_load_config_file_invalid_json():
    """Test loading a configuration file with invalid JSON."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        f.write("{ invalid json }")
        fname = f.name

    try:
        with pytest.raises(json.JSONDecodeError, match="Invalid JSON in config file:"):
            load_config_file(fname)
    finally:
        os.unlink(fname)