"""
Secure configuration management for the thesis evaluation pipeline.
"""

import os
from dataclasses import dataclass, field
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_required_env_var(var_name: str) -> str:
    """Get required environment variable or raise clear error."""
    value = os.getenv(var_name)
    if not value:
        raise ValueError(
            f"Missing {var_name}. Please add it to your .env file.\n"
            f"See .env.example for template."
        )
    return value

@dataclass
class Config:
    """Main configuration class with secure API key loading."""
    
    # API Keys - Loaded securely from environment variables
    openai_api_key: str = field(default_factory=lambda: get_required_env_var("OPENAI_API_KEY"))
    anthropic_api_key: str = field(default_factory=lambda: get_required_env_var("ANTHROPIC_API_KEY"))
    gemini_api_key: str = field(default_factory=lambda: get_required_env_var("GEMINI_API_KEY"))
    
    # Providers and Strategies
    providers: List[str] = None
    strategies: List[str] = None
    
    # Domains and Scenarios  
    domains: List[str] = None
    scenarios_per_domain: int = 90
    
    # Execution Settings
    workers: int = 8
    batch_size: int = 300
    max_retries: int = 5
    log_level: str = "INFO"
    
    # Output Settings
    output_dir: str = "results_full"
    preserve_raw_results: bool = True  # Never auto-delete raw evaluation results
    
    # Determinism
    random_seed: int = 42
    
    def __post_init__(self):
        if self.providers is None:
            self.providers = ["gpt4", "claude3", "gemini"]
        
        if self.strategies is None:
            self.strategies = ["srlp", "cot", "tot", "react"]
        
        if self.domains is None:
            self.domains = [
                "travel_planning",
                "software_project", 
                "event_organization",
                "research_study",
                "business_launch"
            ]
    
    @property
    def total_scenarios(self) -> int:
        """Calculate total scenarios."""
        return len(self.domains) * self.scenarios_per_domain
    
    @property
    def total_experiments(self) -> int:
        """Calculate total experiments."""
        return len(self.providers) * len(self.strategies) * self.total_scenarios

def load_config() -> Config:
    """Load configuration."""
    return Config()
