"""
Embedding-based retrieval using sentence transformers and FAISS.
"""
from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import pickle
import os

try:
    from sentence_transformers import SentenceTransformer
    import faiss
except ImportError:
    SentenceTransformer = None
    faiss = None

from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateCandidate:
    """Container for template retrieval candidates."""
    
    template_id: str
    template_text: str
    domain: Optional[str]
    description: Optional[str]
    embedding: Optional[np.ndarray] = None
    similarity_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'template_id': self.template_id,
            'template_text': self.template_text,
            'domain': self.domain,
            'description': self.description,
            'similarity_score': self.similarity_score,
        }


class EmbeddingRetriever:
    """
    Embedding-based retrieval using sentence transformers.
    
    Uses FAISS for efficient similarity search to retrieve top-K candidates.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu"
    ):
        """
        Initialize embedding retriever.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cpu' or 'cuda')
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers not installed")
        
        self.model_name = model_name
        self.device = device
        self.model = SentenceTransformer(model_name)
        self.model.to(device)
        
        self.index: Optional[faiss.Index] = None
        self.templates: List[TemplateCandidate] = []
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(
            "EmbeddingRetriever initialized",
            model=model_name,
            device=device,
            embedding_dim=self.embedding_dim
        )
    
    def _create_search_text(self, template: TemplateCandidate) -> str:
        """
        Create text for embedding from template metadata.
        
        Args:
            template: Template candidate
            
        Returns:
            Combined text for embedding
        """
        parts = []
        
        if template.domain:
            parts.append(f"Domain: {template.domain}")
        
        if template.description:
            parts.append(f"Description: {template.description}")
        
        # Include first 200 chars of template
        template_preview = template.template_text[:200]
        parts.append(f"Template: {template_preview}")
        
        return " ".join(parts)
    
    def build_index(self, templates: List[TemplateCandidate]) -> None:
        """
        Build FAISS index from templates.
        
        Args:
            templates: List of template candidates
        """
        if faiss is None:
            raise ImportError("faiss-cpu not installed")
        
        if not templates:
            logger.warning("No templates provided, index not built")
            return
        
        self.templates = templates
        
        # Create embeddings
        logger.info(f"Encoding {len(templates)} templates...")
        texts = [self._create_search_text(t) for t in templates]
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Store embeddings in candidates
        for template, embedding in zip(templates, embeddings):
            template.embedding = embedding
        
        # Build FAISS index (IndexFlatIP for inner product = cosine similarity after normalization)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(embeddings)
        
        logger.info(
            "FAISS index built",
            num_templates=len(templates),
            index_size=self.index.ntotal
        )
    
    def retrieve_candidates(
        self,
        query: str,
        k: int = 50
    ) -> List[TemplateCandidate]:
        """
        Retrieve top-K candidates for query.
        
        Args:
            query: Query text (task description)
            k: Number of candidates to retrieve
            
        Returns:
            List of top-K templates with similarity scores
        """
        if self.index is None:
            logger.error("Index not built, call build_index() first")
            return []
        
        # Encode query
        query_embedding = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)
        
        # Search
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, k)
        
        # Create result candidates
        candidates = []
        for idx, score in zip(indices[0], scores[0]):
            template = self.templates[idx]
            # Create a copy with updated score
            candidate = TemplateCandidate(
                template_id=template.template_id,
                template_text=template.template_text,
                domain=template.domain,
                description=template.description,
                similarity_score=float(score)
            )
            candidates.append(candidate)
        
        logger.info(f"Retrieved {len(candidates)} candidates", query_length=len(query))
        return candidates
    
    def save_index(self, path: str) -> None:
        """
        Save index and templates to disk.
        
        Args:
            path: Directory path to save to
        """
        if self.index is None:
            logger.warning("No index to save")
            return
        
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(path, "faiss.index")
        faiss.write_index(self.index, index_path)
        
        # Save templates
        templates_path = os.path.join(path, "templates.pkl")
        with open(templates_path, 'wb') as f:
            pickle.dump(self.templates, f)
        
        logger.info("Index saved", path=path)
    
    def load_index(self, path: str) -> None:
        """
        Load index and templates from disk.
        
        Args:
            path: Directory path to load from
        """
        # Load FAISS index
        index_path = os.path.join(path, "faiss.index")
        self.index = faiss.read_index(index_path)
        
        # Load templates
        templates_path = os.path.join(path, "templates.pkl")
        with open(templates_path, 'rb') as f:
            self.templates = pickle.load(f)
        
        logger.info("Index loaded", path=path, num_templates=len(self.templates))
