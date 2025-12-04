"""
Feature extraction for prompt templates.
"""
from dataclasses import dataclass
from typing import Optional, List
import re
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TemplateFeatures:
    """Container for template features."""
    
    template_length_tokens: int
    few_shot_count: int
    has_cot_directive: bool
    has_safety_clause: bool
    domain_category: str
    avg_input_size: Optional[float] = None
    
    def to_dict(self):
        """Convert to dictionary."""
        return {
            'template_length_tokens': self.template_length_tokens,
            'few_shot_count': self.few_shot_count,
            'has_cot_directive': self.has_cot_directive,
            'has_safety_clause': self.has_safety_clause,
            'domain_category': self.domain_category,
            'avg_input_size': self.avg_input_size,
        }


class FeatureExtractor:
    """Extract features from prompt templates."""
    
    # Chain-of-thought directive patterns
    COT_PATTERNS = [
        r'\bstep\s+by\s+step\b',
        r'\bthink\s+(carefully|through)\b',
        r'\breasoning\b',
        r'\blet\'s\s+think\b',
        r'\bexplain\s+your\s+(thinking|reasoning)\b',
    ]
    
    # Safety clause patterns
    SAFETY_PATTERNS = [
        r'\bsafe(ty)?\b',
        r'\bharmful\b',
        r'\boffensive\b',
        r'\brespectful\b',
        r'\bethical\b',
        r'\bdo\s+not\s+(generate|create|produce)\b',
        r'\bavoid\b',
    ]
    
    # Domain keywords
    DOMAIN_KEYWORDS = {
        'customer_support': ['customer', 'support', 'help', 'service', 'refund', 'issue', 'complaint'],
        'code_generation': ['code', 'function', 'class', 'implement', 'python', 'javascript', 'programming'],
        'creative_writing': ['story', 'creative', 'write', 'narrative', 'character', 'plot', 'novel'],
        'data_analysis': ['analyze', 'data', 'statistics', 'chart', 'visualization', 'insights'],
        'summarization': ['summarize', 'summary', 'brief', 'key points', 'tldr', 'overview'],
        'question_answering': ['question', 'answer', 'explain', 'what is', 'how to', 'why'],
    }
    
    def __init__(self, tokenizer=None):
        """
        Initialize feature extractor.
        
        Args:
            tokenizer: Optional tokenizer for counting tokens
        """
        self.tokenizer = tokenizer
    
    def _count_few_shot_examples(self, template: str) -> int:
        """
        Count few-shot examples in template.
        
        Looks for patterns like:
        - Example 1:, Example 2:, etc.
        - Input: ... Output: ...
        - Q: ... A: ...
        
        Args:
            template: Template text
            
        Returns:
            Number of examples found
        """
        patterns = [
            r'example\s+\d+\s*:',
            r'input\s*:.*?output\s*:',
            r'q\s*:.*?a\s*:',
        ]
        
        total_count = 0
        for pattern in patterns:
            matches = re.findall(pattern, template.lower())
            total_count += len(matches)
        
        return total_count
    
    def _has_cot_directive(self, template: str) -> bool:
        """
        Check if template has chain-of-thought directive.
        
        Args:
            template: Template text
            
        Returns:
            True if CoT directive found
        """
        template_lower = template.lower()
        for pattern in self.COT_PATTERNS:
            if re.search(pattern, template_lower):
                return True
        return False
    
    def _has_safety_clause(self, template: str) -> bool:
        """
        Check if template has safety-related clauses.
        
        Args:
            template: Template text
            
        Returns:
            True if safety clause found
        """
        template_lower = template.lower()
        for pattern in self.SAFETY_PATTERNS:
            if re.search(pattern, template_lower):
                return True
        return False
    
    def _classify_domain(self, template: str, task_description: str = "") -> str:
        """
        Classify template domain based on keywords.
        
        Args:
            template: Template text
            task_description: Optional task description
            
        Returns:
            Domain category (or 'general' if no match)
        """
        combined_text = (template + " " + task_description).lower()
        
        domain_scores = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in combined_text)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores, key=domain_scores.get)
        
        return 'general'
    
    def extract(
        self,
        template: str,
        task_description: str = "",
        avg_input_size: Optional[float] = None
    ) -> TemplateFeatures:
        """
        Extract all features from template.
        
        Args:
            template: Template text
            task_description: Optional task description for context
            avg_input_size: Optional average input size from historical data
            
        Returns:
            TemplateFeatures object
        """
        # Count tokens
        if self.tokenizer:
            template_length = self.tokenizer.count_tokens(template, "gpt-4")
        else:
            # Approximate: 1 token ≈ 4 characters
            template_length = len(template) // 4
        
        features = TemplateFeatures(
            template_length_tokens=template_length,
            few_shot_count=self._count_few_shot_examples(template),
            has_cot_directive=self._has_cot_directive(template),
            has_safety_clause=self._has_safety_clause(template),
            domain_category=self._classify_domain(template, task_description),
            avg_input_size=avg_input_size,
        )
        
        logger.debug("Features extracted", features=features.to_dict())
        return features
