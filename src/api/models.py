"""
Pydantic models for API requests and responses.
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any


class RecommendationRequest(BaseModel):
    """Request model for recommendation endpoint."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    task_description: str = Field(..., description="Description of the task to recommend templates for")
    domain: Optional[str] = Field(None, description="Domain category (e.g., customer_support, code_generation)")
    max_cost_usd: Optional[float] = Field(1.0, description="Maximum acceptable cost in USD")
    min_safety_score: Optional[float] = Field(70.0, description="Minimum safety score (0-100)")
    num_recommendations: Optional[int] = Field(5, description="Number of recommendations to return")
    model_preference: Optional[str] = Field("gpt-4", description="Preferred model for cost estimation")


class ItemizedCost(BaseModel):
    """Itemized cost breakdown."""
    
    input_tokens: int
    output_tokens: int
    cost_usd: float


class TemplateRecommendation(BaseModel):
    """Single template recommendation."""
    
    model_config = ConfigDict(protected_namespaces=())
    
    template_id: str
    template_text: str
    domain: Optional[str]
    description: Optional[str]
    predicted_quality: float
    estimated_cost_usd: float
    safety_score: float
    combined_score: float
    model_recommendation: str
    itemized_cost: ItemizedCost


class RecommendationResponse(BaseModel):
    """Response model for recommendation endpoint."""
    
    recommendations: List[TemplateRecommendation]
    query: str
    total_candidates_retrieved: int
    filters_applied: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    models_loaded: Dict[str, bool]
