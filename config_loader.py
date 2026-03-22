"""Configuration loader for Regenwormen AI strategy benchmarking."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class PlayerConfig:
    """Configuration for a single player in a benchmark."""

    player_id: int
    strategy: str
    params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the player configuration after initialization."""
        if not isinstance(self.player_id, int):
            raise TypeError(f"player_id must be an integer, got {type(self.player_id)}")
        if not isinstance(self.strategy, str):
            raise TypeError(f"strategy must be a string, got {type(self.strategy)}")
        if not isinstance(self.params, dict):
            raise TypeError(f"params must be a dictionary, got {type(self.params)}")


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

    def __post_init__(self):
        """Validate the configuration after initialization."""
        if not isinstance(self.players, list):
            raise TypeError(f"players must be a list, got {type(self.players)}")
        if len(self.players) == 0:
            raise ValueError("players list cannot be empty")
        if not all(isinstance(p, PlayerConfig) for p in self.players):
            raise TypeError("All players must be PlayerConfig instances")

        if not isinstance(self.num_games, int):
            raise TypeError(f"num_games must be an integer, got {type(self.num_games)}")
        if self.num_games <= 0:
            raise ValueError(f"num_games must be positive, got {self.num_games}")

        if not isinstance(self.random_seed, int):
            raise TypeError(f"random_seed must be an integer, got {type(self.random_seed)}")

        if not isinstance(self.verbose, bool):
            raise TypeError(f"verbose must be a boolean, got {type(self.verbose)}")

        if not isinstance(self.max_turns_per_game, int):
            raise TypeError(f"max_turns_per_game must be an integer, got {type(self.max_turns_per_game)}")
        if self.max_turns_per_game <= 0:
            raise ValueError(f"max_turns_per_game must be positive, got {self.max_turns_per_game}")

        if not isinstance(self.collect_decision_stats, bool):
            raise TypeError(f"collect_decision_stats must be a boolean, got {type(self.collect_decision_stats)}")

        if not isinstance(self.collect_worm_distribution, bool):
            raise TypeError(f"collect_worm_distribution must be a boolean, got {type(self.collect_worm_distribution)}")

        if not isinstance(self.collect_timing, bool):
            raise TypeError(f"collect_timing must be a boolean, got {type(self.collect_timing)}")


def load_config_file(filepath: str) -> Config:
    """Load configuration from a JSON file.

    Args:
        filepath: Path to the JSON configuration file

    Returns:
        Config object populated from the file

    Raises:
        FileNotFoundError: If the file doesn't exist
        JSONDecodeError: If the file contains invalid JSON
        ValueError: If the configuration is invalid
    """
    # This will be implemented in Task 2
    raise NotImplementedError("load_config_file will be implemented in Task 2")