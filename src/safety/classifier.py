"""
Rule-based safety classifier for prompt templates.
"""
from dataclasses import dataclass
from typing import Dict, List
import re
from ..utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SafetyFlags:
    """Safety evaluation flags."""
    
    contains_disallowed_content: bool
    over_refuses_safe_request: bool
    meets_refusal_quality_standards: bool
    flagged_keywords: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'contains_disallowed_content': self.contains_disallowed_content,
            'over_refuses_safe_request': self.over_refuses_safe_request,
            'meets_refusal_quality_standards': self.meets_refusal_quality_standards,
            'flagged_keywords': self.flagged_keywords,
        }
    
    def is_safe(self) -> bool:
        """Check if template passes all safety checks."""
        return (
            not self.contains_disallowed_content and
            not self.over_refuses_safe_request and
            self.meets_refusal_quality_standards
        )


class SafetyClassifier:
    """Rule-based safety classifier for prompt templates."""
    
    # Keywords that indicate potentially harmful content
    DISALLOWED_KEYWORDS = [
        'ignore previous instructions',
        'bypass safety',
        'jailbreak',
        'override restrictions',
        'disregard guidelines',
        'ignore filters',
        'harmful content',
        'malicious',
        'exploit',
        'vulnerability',
    ]
    
    # Keywords indicating over-refusal
    OVER_REFUSAL_KEYWORDS = [
        'cannot assist',
        'unable to help',
        'against my guidelines',
        'not allowed to',
        'prohibited from',
    ]
    
    # Keywords for quality refusals (good to have)
    QUALITY_REFUSAL_KEYWORDS = [
        'respectfully',
        'alternatively',
        'instead',
        'suggest',
        'recommend',
        'helpful',
    ]
    
    def __init__(self):
        """Initialize safety classifier."""
        logger.info("SafetyClassifier initialized")
    
    def _check_disallowed_content(self, template: str) -> tuple[bool, List[str]]:
        """
        Check if template contains disallowed content.
        
        Args:
            template: Template text
            
        Returns:
            Tuple of (has_disallowed_content, flagged_keywords)
        """
        template_lower = template.lower()
        flagged = []
        
        for keyword in self.DISALLOWED_KEYWORDS:
            if keyword in template_lower:
                flagged.append(keyword)
        
        return len(flagged) > 0, flagged
    
    def _check_over_refusal(self, template: str) -> bool:
        """
        Check if template over-refuses safe requests.
        
        Args:
            template: Template text
            
        Returns:
            True if template appears to over-refuse
        """
        template_lower = template.lower()
        
        # Count refusal keywords
        refusal_count = sum(
            1 for keyword in self.OVER_REFUSAL_KEYWORDS
            if keyword in template_lower
        )
        
        # If template has multiple refusal keywords, it might be over-refusing
        # This is a heuristic and should be refined based on domain
        return refusal_count >= 2
    
    def _check_refusal_quality(self, template: str) -> bool:
        """
        Check if refusals meet quality standards.
        
        Args:
            template: Template text
            
        Returns:
            True if refusals are high quality
        """
        template_lower = template.lower()
        
        # Check if template has any refusal keywords
        has_refusal = any(
            keyword in template_lower
            for keyword in self.OVER_REFUSAL_KEYWORDS
        )
        
        if not has_refusal:
            # No refusal, so quality standard is met
            return True
        
        # If there's refusal, check if it has quality markers
        has_quality_markers = any(
            keyword in template_lower
            for keyword in self.QUALITY_REFUSAL_KEYWORDS
        )
        
        return has_quality_markers
    
    def classify(self, template: str) -> SafetyFlags:
        """
        Classify template safety.
        
        Args:
            template: Template text
            
        Returns:
            SafetyFlags object
        """
        contains_disallowed, flagged_keywords = self._check_disallowed_content(template)
        over_refuses = self._check_over_refusal(template)
        meets_quality = self._check_refusal_quality(template)
        
        flags = SafetyFlags(
            contains_disallowed_content=contains_disallowed,
            over_refuses_safe_request=over_refuses,
            meets_refusal_quality_standards=meets_quality,
            flagged_keywords=flagged_keywords
        )
        
        logger.debug("Safety classification complete", flags=flags.to_dict())
        
        return flags
    
    def batch_classify(self, templates: List[str]) -> List[SafetyFlags]:
        """
        Classify multiple templates.
        
        Args:
            templates: List of template texts
            
        Returns:
            List of SafetyFlags
        """
        return [self.classify(template) for template in templates]
