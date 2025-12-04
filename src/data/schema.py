"""
SQLAlchemy database models for the prompt recommender system.
"""
from datetime import datetime
from typing import Dict, Any
from sqlalchemy import Column, String, Text, Float, Integer, DateTime, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()


class HistoricalRun(Base):
    """Store historical prompt execution data."""
    
    __tablename__ = "historical_runs"
    
    task_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    task_description = Column(Text, nullable=False)
    template_text = Column(Text, nullable=False)
    model_name = Column(String(100), nullable=False)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    quality_score = Column(Float, nullable=False)
    safety_flags = Column(JSON, nullable=True)
    input_token_count = Column(Integer, nullable=False)
    output_token_count = Column(Integer, nullable=False)
    total_cost_usd = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Indexes for common queries
    __table_args__ = (
        Index('idx_task_description', 'task_description'),
        Index('idx_model_name', 'model_name'),
        Index('idx_timestamp', 'timestamp'),
        Index('idx_quality_score', 'quality_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'task_id': self.task_id,
            'task_description': self.task_description,
            'template_text': self.template_text,
            'model_name': self.model_name,
            'input_text': self.input_text,
            'output_text': self.output_text,
            'quality_score': self.quality_score,
            'safety_flags': self.safety_flags,
            'input_token_count': self.input_token_count,
            'output_token_count': self.output_token_count,
            'total_cost_usd': self.total_cost_usd,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class Experiment(Base):
    """Store A/B testing experiment data."""
    
    __tablename__ = "experiments"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    exp_id = Column(String(100), nullable=False, index=True)
    variant = Column(String(50), nullable=False)
    user_context = Column(JSON, nullable=True)
    template_id = Column(String(36), nullable=True)
    outcome = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    __table_args__ = (
        Index('idx_exp_id', 'exp_id'),
        Index('idx_variant', 'variant'),
        Index('idx_timestamp_exp', 'timestamp'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'id': self.id,
            'exp_id': self.exp_id,
            'variant': self.variant,
            'user_context': self.user_context,
            'template_id': self.template_id,
            'outcome': self.outcome,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class Template(Base):
    """Store prompt templates with metadata."""
    
    __tablename__ = "templates"
    
    template_id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    template_text = Column(Text, nullable=False)
    domain = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    avg_quality_score = Column(Float, nullable=True)
    avg_cost_usd = Column(Float, nullable=True)
    avg_safety_score = Column(Float, nullable=True)
    usage_count = Column(Integer, default=0)
    
    __table_args__ = (
        Index('idx_domain', 'domain'),
        Index('idx_avg_quality', 'avg_quality_score'),
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'template_id': self.template_id,
            'template_text': self.template_text,
            'domain': self.domain,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'avg_quality_score': self.avg_quality_score,
            'avg_cost_usd': self.avg_cost_usd,
            'avg_safety_score': self.avg_safety_score,
            'usage_count': self.usage_count,
        }
