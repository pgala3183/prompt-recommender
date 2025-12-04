"""
Model pricing configuration and cost calculation.
"""
from typing import Dict
from ..utils.logging import get_logger

logger = get_logger(__name__)

# Pricing in USD per 1K tokens
PRICING = {
    "gpt-4": {
        "input": 0.03,
        "output": 0.06
    },
    "gpt-3.5-turbo": {
        "input": 0.0015,
        "output": 0.002
    },
    "claude-3-opus": {
        "input": 0.015,
        "output": 0.075
    },
    "claude-3-sonnet": {
        "input": 0.003,
        "output": 0.015
    },
    "gemini-1.5-pro": {
        "input": 0.00125,
        "output": 0.005
    },
}


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int
) -> float:
    """
    Calculate total cost for a model run.
    
    Args:
        model: Model name
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        
    Returns:
        Total cost in USD
    """
    # Normalize model name
    model_key = model.lower()
    pricing = None
    
    for key in PRICING.keys():
        if key in model_key:
            pricing = PRICING[key]
            break
    
    if pricing is None:
        logger.warning(f"Unknown model {model}, using gpt-4 pricing")
        pricing = PRICING["gpt-4"]
    
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    total_cost = input_cost + output_cost
    
    logger.debug(
        "Cost calculated",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_cost=total_cost
    )
    
    return total_cost


def get_pricing_info(model: str) -> Dict[str, float]:
    """
    Get pricing information for a model.
    
    Args:
        model: Model name
        
    Returns:
        Dictionary with input and output pricing
    """
    model_key = model.lower()
    
    for key in PRICING.keys():
        if key in model_key:
            return PRICING[key].copy()
    
    logger.warning(f"Unknown model {model}, returning gpt-4 pricing")
    return PRICING["gpt-4"].copy()
