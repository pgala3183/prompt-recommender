"""
MinHash and LSH-based deduplication for prompt templates.
"""
from typing import List, Tuple, Set
from datasketch import MinHash, MinHashLSH
from ..utils.logging import get_logger

logger = get_logger(__name__)


class MinHashDeduplicator:
    """
    Deduplicate templates using MinHash and LSH.
    
    Uses MinHash for efficient similarity estimation and LSH for fast nearest
    neighbor search to find templates with >90% Jaccard similarity.
    """
    
    def __init__(
        self,
        num_perm: int = 128,
        threshold: float = 0.9,
        shingle_size: int = 3
    ):
        """
        Initialize deduplicator.
        
        Args:
            num_perm: Number of permutations for MinHash
            threshold: Jaccard similarity threshold for duplicates
            shingle_size: Size of character shingles
        """
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        logger.info(
            "MinHashDeduplicator initialized",
            num_perm=num_perm,
            threshold=threshold,
            shingle_size=shingle_size
        )
    
    def _create_shingles(self, text: str) -> Set[str]:
        """
        Create character shingles from text.
        
        Args:
            text: Input text
            
        Returns:
            Set of shingles
        """
        text = text.lower().strip()
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            shingles.add(shingle)
        return shingles
    
    def _compute_minhash(self, text: str) -> MinHash:
        """
        Compute MinHash signature for text.
        
        Args:
            text: Input text
            
        Returns:
            MinHash object
        """
        mh = MinHash(num_perm=self.num_perm)
        shingles = self._create_shingles(text)
        for shingle in shingles:
            mh.update(shingle.encode('utf-8'))
        return mh
    
    def find_duplicates(self, templates: List[str]) -> List[Tuple[int, int]]:
        """
        Find duplicate pairs in template list.
        
        Args:
            templates: List of template strings
            
        Returns:
            List of (index1, index2) tuples representing duplicate pairs
        """
        duplicates = []
        lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)
        
        # Build LSH index
        for idx, template in enumerate(templates):
            mh = self._compute_minhash(template)
            # Query for duplicates before inserting
            matches = lsh.query(mh)
            for match_idx in matches:
                duplicates.append((match_idx, idx))
            # Insert current template
            lsh.insert(idx, mh)
        
        logger.info(f"Found {len(duplicates)} duplicate pairs")
        return duplicates
    
    def filter_duplicates(self, templates: List[str]) -> List[str]:
        """
        Filter out duplicate templates, keeping only unique ones.
        
        Args:
            templates: List of template strings
            
        Returns:
            List of unique templates
        """
        if not templates:
            return []
        
        duplicate_pairs = self.find_duplicates(templates)
        
        # Build set of indices to remove (keep the first occurrence)
        to_remove = set()
        for idx1, idx2 in duplicate_pairs:
            to_remove.add(idx2)  # Remove the second occurrence
        
        # Filter templates
        unique_templates = [
            template for idx, template in enumerate(templates)
            if idx not in to_remove
        ]
        
        logger.info(
            "Deduplication complete",
            original_count=len(templates),
            unique_count=len(unique_templates),
            removed_count=len(to_remove)
        )
        
        return unique_templates
    
    def estimate_similarity(self, text1: str, text2: str) -> float:
        """
        Estimate Jaccard similarity between two texts.
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Estimated Jaccard similarity (0.0 to 1.0)
        """
        mh1 = self._compute_minhash(text1)
        mh2 = self._compute_minhash(text2)
        return mh1.jaccard(mh2)
