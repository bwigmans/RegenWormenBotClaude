"""Configuration loader for Regenwormen AI strategy benchmarking."""

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


