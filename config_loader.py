"""
Configuration loader for Regenwormen strategy benchmarking.
"""
import json
from typing import List, Dict, Any, Tuple, Optional
import warnings
from dataclasses import dataclass

@dataclass
class PlayerConfig:
    """Configuration for a single player."""
    player_id: int
    strategy: str
    params: Dict[str, Any]

@dataclass
class Config:
    """Complete benchmark configuration."""
    players: List[PlayerConfig]
    num_games: int = 1000
    random_seed: int = 42
    verbose: bool = False
    max_turns_per_game: int = 1000
    collect_decision_stats: bool = True
    collect_worm_distribution: bool = True
    collect_timing: bool = False