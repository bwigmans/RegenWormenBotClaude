"""Configuration loader for Regenwormen AI strategy benchmarking."""

import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class PlayerConfig:
    """Configuration for a single player in a benchmark."""

    player_id: int
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    """Main configuration for benchmarking runs."""

    players: List[PlayerConfig]
    num_games: int = 1000
    random_seed: int = 42
    verbose: bool = False
    max_turns_per_game: int = 1000
    collect_decision_stats: bool = True
    collect_worm_distribution: bool = True
    collect_timing: bool = False


def load_config_file(filepath: str) -> Config:
    """
    Load and validate configuration from JSON file.

    Args:
        filepath: Path to JSON configuration file

    Returns:
        Config object

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If invalid JSON
        ValueError: If configuration validation fails
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in config file: {e.msg}", e.doc, e.pos) from e
    except OSError as e:
        raise OSError(f"Error reading config file {filepath}: {e}") from e

    return _create_config_from_dict(data)


def _create_config_from_dict(data: dict) -> Config:
    """Create Config from dictionary with basic validation."""
    # Extract players
    players_data = data.get("players", [])
    players = []
    for p_data in players_data:
        player = PlayerConfig(
            player_id=p_data.get("id"),
            strategy=p_data.get("strategy"),
            params=p_data.get("params", {})
        )
        players.append(player)

    # Extract game settings with defaults
    game_settings = data.get("game_settings", {})
    num_games = game_settings.get("num_games", 1000)
    random_seed = game_settings.get("random_seed", 42)
    verbose = game_settings.get("verbose", False)
    max_turns_per_game = game_settings.get("max_turns_per_game", 1000)

    # Extract benchmark metrics settings
    benchmark_metrics = data.get("benchmark_metrics", {})
    collect_decision_stats = benchmark_metrics.get("collect_decision_stats", True)
    collect_worm_distribution = benchmark_metrics.get("collect_worm_distribution", True)
    collect_timing = benchmark_metrics.get("collect_timing", False)

    return Config(
        players=players,
        num_games=num_games,
        random_seed=random_seed,
        verbose=verbose,
        max_turns_per_game=max_turns_per_game,
        collect_decision_stats=collect_decision_stats,
        collect_worm_distribution=collect_worm_distribution,
        collect_timing=collect_timing
    )


