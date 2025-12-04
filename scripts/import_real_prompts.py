"""
Import Stanford Alpaca dataset (52K prompts).
Large-scale, high-quality instruction-following dataset.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
import json
import uuid
from src.data.schema import Template
from src.data.ingestion import DataIngestionManager
from src.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def download_alpaca_dataset():
    """Download Stanford Alpaca dataset (52K instructions)."""
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    
    logger.info("Downloading Stanford Alpaca dataset (52K prompts)...")
    logger.info("This may take a moment...")
    
    response = requests.get(url, timeout=60)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download dataset: {response.status_code}")
    
    data = json.loads(response.text)
    logger.info(f"Downloaded {len(data)} instruction examples")
    
    return data


def categorize_instruction(instruction: str, input_text: str = "") -> str:
    """Categorize instruction into domain."""
    text = (instruction + " " + input_text).lower()
    
    # Domain classification with keywords
    if any(word in text for word in ['write', 'story', 'poem', 'creative', 'describe', 'imagine', 'narrative', 'character', 'plot']):
        return 'creative_writing'
    elif any(word in text for word in ['code', 'program', 'function', 'algorithm', 'debug', 'implement', 'python', 'javascript', 'sql', 'api']):
        return 'code_generation'
    elif any(word in text for word in ['analyze', 'data', 'calculate', 'statistics', 'compare', 'evaluate', 'measure', 'chart', 'graph', 'trend']):
        return 'data_analysis'
    elif any(word in text for word in ['explain', 'help', 'assist', 'guide', 'support', 'answer', 'solve', 'advise', 'recommend']):
        return 'customer_support'
    else:
        return 'general'


def import_alpaca_to_db(data, max_templates=5000):
    """
    Import Alpaca dataset to database.
    
    Args:
        data: Alpaca dataset
        max_templates: Maximum templates to import (default 5000 for performance)
    """
    manager = DataIngestionManager()
    session = manager.get_session()
    
    # Clear existing templates
    logger.info("Clearing existing templates...")
    session.query(Template).delete()
    session.commit()
    
    added = 0
    domain_counts = {}
    
    for item in data[:max_templates]:  # Limit for performance
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if not instruction:
            continue
        
        # Categorize
        domain = categorize_instruction(instruction, input_text)
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Format template
        if input_text:
            template_text = f"{instruction}\n\nContext: {input_text}\n\nUser query: {{input}}"
            description = f"{instruction[:100]}... (with context)"
        else:
            template_text = f"{instruction}\n\nUser query: {{input}}"
            description = instruction[:150]
        
        # Estimate quality based on output length (longer = more detailed)
        quality = min(0.9, 0.5 + len(output) / 2000)
        
        template = Template(
            template_id=str(uuid.uuid4()),
            template_text=template_text,
            domain=domain,
            description=description,
            avg_quality_score=quality,
            avg_cost_usd=0.15,
            avg_safety_score=88.0,
            usage_count=0
        )
        session.add(template)
        added += 1
        
        if added % 500 == 0:
            logger.info(f"Imported {added}/{max_templates} templates...")
            session.commit()
    
    session.commit()
    session.close()
    
    logger.info(f"✅ Successfully imported {added} Alpaca prompt templates!")
    logger.info(f"Domain distribution: {domain_counts}")
    
    return added, domain_counts


def main():
    """Main function."""
    try:
        print("\n" + "="*70)
        print("Stanford Alpaca Dataset Importer")
        print("="*70)
        print("\nDataset: 52,000 instruction-following examples")
        print("Source: Stanford Alpaca Project")
        print("URL: https://github.com/tatsu-lab/stanford_alpaca")
        print("\nImporting 5,000 templates (configurable)...")
        print("="*70 + "\n")
        
        # Download dataset
        data = download_alpaca_dataset()
        
        # Import to database
        count, domains = import_alpaca_to_db(data, max_templates=52000)
        
        print(f"\n{'='*70}")
        print(f"✅ SUCCESS! Imported {count} real-world prompts")
        print(f"{'='*70}")
        
        print("\nDomain Distribution:")
        for domain, cnt in sorted(domains.items(), key=lambda x: x[1], reverse=True):
            print(f"  • {domain}: {cnt} templates")
        
        print("\n" + "="*70)
        print("Next steps:")
        print("1. Restart API: uvicorn src.api.main:app --reload")
        print("2. Test dashboard: streamlit run src\\dashboard\\app.py")
        print("3. Try diverse queries to see variety!")
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to import Alpaca dataset: {e}")
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("- Check internet connection")
        print("- GitHub may be rate-limiting (wait a few minutes)")
        print("- Try reducing max_templates in the script")


if __name__ == "__main__":
    main()
