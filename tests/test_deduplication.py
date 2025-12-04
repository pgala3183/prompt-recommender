"""Test deduplication functionality."""
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.deduplication import MinHashDeduplicator


class TestMinHashDeduplicator:
    """Test deduplication system."""
    
    def test_identical_templates(self):
        """Test that identical templates are detected as duplicates."""
        template = "You are a helpful assistant. Please help with the following task."
        templates = [template, template]
        
        dedup = MinHashDeduplicator(threshold=0.9)
        duplicates = dedup.find_duplicates(templates)
        
        assert len(duplicates) > 0
    
    def test_similar_templates(self):
        """Test that very similar templates are detected."""
        template1 = "You are a helpful assistant. Please help with the following task."
        template2 = "You are a helpful assistant. Please assist with the following task."
        templates = [template1, template2]
        
        dedup = MinHashDeduplicator(threshold=0.9)
        duplicates = dedup.find_duplicates(templates)
        
        # These should be similar enough to be duplicates
        assert len(duplicates) > 0
    
    def test_different_templates(self):
        """Test that different templates are not marked as duplicates."""
        template1 = "You are a helpful customer service assistant."
        template2 = "Generate Python code for the following task."
        templates = [template1, template2]
        
        dedup = MinHashDeduplicator(threshold=0.9)
        duplicates = dedup.find_duplicates(templates)
        
        assert len(duplicates) == 0
    
    def test_filter_duplicates(self):
        """Test filtering keeps only unique templates."""
        templates = [
            "Template A",
            "Template A",
            "Template B",
            "Template B",
            "Template C"
        ]
        
        dedup = MinHashDeduplicator(threshold=0.9)
        unique = dedup.filter_duplicates(templates)
        
        # Should have fewer templates after dedup
        assert len(unique) < len(templates)
        assert len(unique) >= 3  # At least A, B, C
    
    def test_similarity_estimation(self):
        """Test Jaccard similarity estimation."""
        template1 = "hello world"
        template2 = "hello world"
        template3 = "goodbye world"
        
        dedup = MinHashDeduplicator()
        
        # Identical templates should have similarity ~1.0
        sim_identical = dedup.estimate_similarity(template1, template2)
        assert sim_identical > 0.99
        
        # Different templates should have lower similarity
        sim_different = dedup.estimate_similarity(template1, template3)
        assert sim_different < sim_identical


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
