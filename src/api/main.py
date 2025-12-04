"""
FastAPI application for prompt recommendation API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from typing import Dict
import yaml

from .models import (
    RecommendationRequest,
    RecommendationResponse,
    TemplateRecommendation,
    ItemizedCost,
    HealthResponse
)
from ..retrieval import EmbeddingRetriever, LightGBMReranker, TemplateCandidate
from ..safety import SafetyRewardModel, SafetyClassifier
from ..safety.ollama_grader import OllamaGrader
from ..utils.logging import configure_logging, get_logger

# Configure logging
configure_logging("INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Prompt Recommendation API",
    description="Two-stage prompt recommendation system with cost optimization and safety scoring",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for models (loaded on startup)
retriever: EmbeddingRetriever = None
reranker: LightGBMReranker = None
safety_model: SafetyRewardModel = None
safety_classifier: SafetyClassifier = None
ollama_grader: OllamaGrader = None
config: Dict = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup."""
    global retriever, reranker, safety_model, safety_classifier, config
    
    logger.info("Starting up Prompt Recommendation API...")
    
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../../config/config.yaml")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {}
        logger.warning("Config file not found, using defaults")
    
    # Initialize retriever and load templates
    try:
        from ..data.ingestion import DataIngestionManager
        from ..data.schema import Template
        
        # Get templates from database
        logger.info("Loading templates from database...")
        ingestion_manager = DataIngestionManager()
        session = ingestion_manager.get_session()
        
        templates_db = session.query(Template).all()
        logger.info(f"Found {len(templates_db)} templates in database")
        
        if len(templates_db) > 0:
            # Convert to TemplateCandidate objects
            from ..retrieval import TemplateCandidate
            candidates = [
                TemplateCandidate(
                    template_id=t.template_id,
                    template_text=t.template_text,
                    domain=t.domain,
                    description=t.description
                )
                for t in templates_db
            ]
            
            # Initialize retriever
            retriever = EmbeddingRetriever(
                model_name=config.get('models', {}).get('embedding', {}).get('name', 'sentence-transformers/all-MiniLM-L6-v2')
            )
            
            # Build index
            logger.info("Building FAISS index...")
            retriever.build_index(candidates)
            logger.info("Embedding retriever initialized with index")
        else:
            logger.warning("No templates found in database. Run: python scripts/generate_synthetic_data.py")
            retriever = None
            
        session.close()
        
    except Exception as e:
        logger.error(f"Failed to initialize retriever: {e}")
        retriever = None
    
    # Initialize reranker
    try:
        reranking_config = config.get('reranking', {})
        reranker = LightGBMReranker(
            quality_weight=reranking_config.get('quality_weight', 0.5),
            cost_weight=reranking_config.get('cost_weight', 0.3),
            safety_weight=reranking_config.get('safety_weight', 0.2)
        )
        logger.info("LightGBM reranker initialized")
    except Exception as e:
        logger.error(f"Failed to initialize reranker: {e}")
        reranker = None
    
    # Initialize safety models
    try:
        safety_classifier = SafetyClassifier()
        safety_model = SafetyRewardModel()
        logger.info("Safety models initialized")
    except Exception as e:
        logger.error(f"Failed to initialize safety models: {e}")
        safety_classifier = None
        safety_model = None
    
    # Initialize Ollama grader
    try:
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        use_ollama = os.getenv("USE_OLLAMA_GRADER", "true").lower() == "true"
        
        if use_ollama:
            ollama_grader = OllamaGrader(
                model_name=ollama_model,
                base_url=ollama_url
            )
            logger.info(f"Ollama grader initialized with {ollama_model}")
        else:
            logger.info("Ollama grader disabled (USE_OLLAMA_GRADER=false)")
            ollama_grader = None
    except Exception as e:
        logger.warning(f"Ollama grader not available: {e}")
        ollama_grader = None
    
    logger.info("API startup complete")
    
    # Log warnings if models not loaded
    if retriever is None:
        logger.warning("⚠️  Retriever not loaded - make sure templates exist in database")
    if reranker is None:
        logger.warning("⚠️  Reranker not loaded")
    if safety_model is None or safety_classifier is None:
        logger.warning("⚠️  Safety models not loaded")
    if ollama_grader is None:
        logger.warning("⚠️  Ollama grader not loaded - using rule-based safety only")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        models_loaded={
            "retriever": retriever is not None,
            "reranker": reranker is not None,
            "safety_model": safety_model is not None,
            "safety_classifier": safety_classifier is not None,
            "ollama_grader": ollama_grader is not None,
        }
    )


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_templates(request: RecommendationRequest):
    """
    Recommend prompt templates based on task description.
    
    This endpoint performs two-stage retrieval:
    1. Embedding-based retrieval to get top-K candidates
    2. Re-ranking with quality, cost, and safety scores
    """
    if retriever is None or reranker is None:
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please check server logs."
        )
    
    logger.info(f"Recommendation request received", task=request.task_description[:50])
    
    try:
        # Stage 1: Retrieve candidates
        top_k = config.get('retrieval', {}).get('top_k_candidates', 50)
        candidates = retriever.retrieve_candidates(
            query=request.task_description,
            k=top_k
        )
        
        if not candidates:
            logger.warning("No candidates retrieved")
            return RecommendationResponse(
                recommendations=[],
                query=request.task_description,
                total_candidates_retrieved=0,
                filters_applied={
                    "max_cost_usd": request.max_cost_usd,
                    "min_safety_score": request.min_safety_score
                }
            )
        
        # Compute safety scores
        safety_scores = {}
        
        # Use Ollama grader if available for AI-powered safety scoring
        if ollama_grader:
            logger.info("Using Ollama for AI-powered safety grading...")
            for candidate in candidates:
                try:
                    flags = ollama_grader.grade(candidate.template_text, use_cache=True)
                    # Convert SafetyFlags to 0-100 score
                    # Safe templates get high scores, unsafe get low scores
                    if flags.is_safe():
                        score = 90.0
                    elif flags.contains_disallowed_content:
                        score = 30.0
                    elif flags.over_refuses_safe_request:
                        score = 60.0
                    else:
                        score = 75.0
                    
                    safety_scores[candidate.template_id] = score
                    logger.debug(f"Ollama safety score: {score}", template_id=candidate.template_id[:8])
                except Exception as e:
                    logger.warning(f"Ollama grading failed, using default: {e}")
                    safety_scores[candidate.template_id] = 80.0
        
        # Fallback: use safety model if trained, otherwise rule-based
        elif safety_model and safety_model.is_trained:
            logger.info("Using trained safety model...")
            for candidate in candidates:
                score = safety_model.compute_safety_score(candidate.template_text)
                safety_scores[candidate.template_id] = score
        
        # Final fallback: use rule-based classifier
        else:
            logger.info("Using rule-based safety classifier...")
            for candidate in candidates:
                flags = safety_classifier.classify(candidate.template_text) if safety_classifier else None
                if flags and flags.is_safe():
                    safety_scores[candidate.template_id] = 85.0
                elif flags and flags.contains_disallowed_content:
                    safety_scores[candidate.template_id] = 40.0
                else:
                    safety_scores[candidate.template_id] = 80.0
        
        # Stage 2: Re-rank candidates
        scored_templates = reranker.rerank(
            candidates=candidates,
            query=request.task_description,
            safety_scores=safety_scores,
            model_name=request.model_preference
        )
        
        # Apply filters
        filtered_templates = [
            t for t in scored_templates
            if t.estimated_cost_usd <= request.max_cost_usd
            and t.safety_score >= request.min_safety_score
        ]
        
        # Take top N
        top_n = request.num_recommendations
        recommendations = filtered_templates[:top_n]
        
        # Convert to response format
        recommendation_list = [
            TemplateRecommendation(
                template_id=t.template_id,
                template_text=t.template_text,
                domain=t.domain,
                description=t.description,
                predicted_quality=t.predicted_quality,
                estimated_cost_usd=t.estimated_cost_usd,
                safety_score=t.safety_score,
                combined_score=t.combined_score,
                model_recommendation=t.model_recommendation,
                itemized_cost=ItemizedCost(**t.itemized_cost)
            )
            for t in recommendations
        ]
        
        response = RecommendationResponse(
            recommendations=recommendation_list,
            query=request.task_description,
            total_candidates_retrieved=len(candidates),
            filters_applied={
                "max_cost_usd": request.max_cost_usd,
                "min_safety_score": request.min_safety_score,
                "domain": request.domain
            }
        )
        
        logger.info(f"Returning {len(recommendation_list)} recommendations")
        return response
        
    except Exception as e:
        logger.error(f"Error processing recommendation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Prompt Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/recommend": "Get template recommendations (POST)",
            "/docs": "API documentation"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
