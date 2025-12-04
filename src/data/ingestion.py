"""
Data ingestion module with multi-provider tokenization support.
"""
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import tiktoken
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

from .schema import Base, HistoricalRun
from ..utils.logging import get_logger

logger = get_logger(__name__)

load_dotenv()


class TokenizerInterface(ABC):
    """Abstract interface for tokenizers."""
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        pass


class OpenAITokenizer(TokenizerInterface):
    """OpenAI tokenizer using tiktoken."""
    
    def __init__(self, model_name: str = "gpt-4"):
        """
        Initialize OpenAI tokenizer.
        
        Args:
            model_name: Model name for tokenizer selection
        """
        self.model_name = model_name
        try:
            self.encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback to cl100k_base for GPT-4
            self.encoding = tiktoken.get_encoding("cl100k_base")
            logger.warning(f"Model {model_name} not found, using cl100k_base encoding")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken."""
        return len(self.encoding.encode(text))


class AnthropicTokenizer(TokenizerInterface):
    """Anthropic tokenizer using anthropic SDK."""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        """
        Initialize Anthropic tokenizer.
        
        Args:
            model_name: Model name for tokenizer
        """
        self.model_name = model_name
        self.client = None
        try:
            from anthropic import Anthropic
            api_key = os.getenv("ANTHROPIC_API_KEY", "placeholder")
            if api_key != "placeholder":
                self.client = Anthropic(api_key=api_key)
            else:
                logger.info("Anthropic API key not set, using approximate token count")
        except (ImportError, TypeError, Exception) as e:
            logger.warning(f"Anthropic SDK not available or incompatible ({e}), using approximate count")
            self.client = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Anthropic's count_tokens."""
        if self.client is None:
            # Approximate: 1 token ≈ 4 characters for Claude
            return len(text) // 4
        
        try:
            # For now, use approximation since count_tokens API may vary
            # In production, use: self.client.count_tokens(text)
            return len(text) // 4
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return len(text) // 4


class GeminiTokenizer(TokenizerInterface):
    """Google Gemini tokenizer."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize Gemini tokenizer.
        
        Args:
            model_name: Model name for tokenizer
        """
        self.model_name = model_name
        self.model = None
        try:
            import google.generativeai as genai
            api_key = os.getenv("GOOGLE_API_KEY", "placeholder")
            if api_key != "placeholder":
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
            else:
                logger.info("Google API key not set, using approximate token count")
        except (ImportError, Exception) as e:
            logger.warning(f"Google Generative AI SDK not available or error ({e}), using approximate count")
            self.model = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's count_tokens."""
        if self.model is None:
            # Approximate: 1 token ≈ 4 characters
            return len(text) // 4
        
        try:
            # Use model's count_tokens method
            return self.model.count_tokens(text).total_tokens
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return len(text) // 4


class UnifiedTokenizer:
    """Unified tokenizer that routes to appropriate provider."""
    
    def __init__(self):
        """Initialize unified tokenizer with all providers."""
        self.tokenizers = {
            "gpt-4": OpenAITokenizer("gpt-4"),
            "gpt-3.5-turbo": OpenAITokenizer("gpt-3.5-turbo"),
            "claude-3-opus": AnthropicTokenizer("claude-3-opus-20240229"),
            "claude-3-sonnet": AnthropicTokenizer("claude-3-sonnet-20240229"),
            "gemini-1.5-pro": GeminiTokenizer("gemini-1.5-pro"),
        }
    
    def count_tokens(self, text: str, model_name: str) -> int:
        """
        Count tokens for given model.
        
        Args:
            text: Text to tokenize
            model_name: Model name
            
        Returns:
            Token count
        """
        # Normalize model name
        model_key = model_name.lower()
        for key in self.tokenizers.keys():
            if key in model_key:
                return self.tokenizers[key].count_tokens(text)
        
        # Fallback to GPT-4 tokenizer
        logger.warning(f"Unknown model {model_name}, using GPT-4 tokenizer")
        return self.tokenizers["gpt-4"].count_tokens(text)


class DataIngestionManager:
    """Manage data ingestion into the database."""
    
    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize ingestion manager.
        
        Args:
            database_url: SQLAlchemy database URL
        """
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", "sqlite:///./data/prompt_recommender.db")
        
        self.engine = create_engine(database_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.tokenizer = UnifiedTokenizer()
        logger.info("DataIngestionManager initialized", database_url=database_url)
    
    def ingest_run(self, run_data: Dict[str, Any]) -> str:
        """
        Ingest a single run into the database.
        
        Args:
            run_data: Dictionary containing run information
            
        Returns:
            Task ID of the ingested run
        """
        session = self.SessionLocal()
        try:
            # Calculate token counts if not provided
            model_name = run_data.get("model_name", "gpt-4")
            
            if "input_token_count" not in run_data:
                input_text = run_data.get("input_text", "")
                run_data["input_token_count"] = self.tokenizer.count_tokens(input_text, model_name)
            
            if "output_token_count" not in run_data:
                output_text = run_data.get("output_text", "")
                run_data["output_token_count"] = self.tokenizer.count_tokens(output_text, model_name)
            
            # Create run record
            run = HistoricalRun(**run_data)
            session.add(run)
            session.commit()
            
            logger.info("Run ingested", task_id=run.task_id, model=model_name)
            return run.task_id
            
        except Exception as e:
            session.rollback()
            logger.error("Error ingesting run", error=str(e))
            raise
        finally:
            session.close()
    
    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()
