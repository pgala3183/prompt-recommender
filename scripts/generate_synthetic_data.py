"""
Generate synthetic historical data for testing.
"""
import random
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.ingestion import DataIngestionManager
from src.data.schema import Template
from src.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


# Sample templates for different domains
TEMPLATES = {
    "customer_support": [
        "You are a helpful customer service assistant. Handle the customer's request professionally and empathetically. Customer request: {input}",
        "Please assist the customer with their inquiry in a respectful and timely manner. Be sure to address their specific concerns. Request: {input}",
        "Customer Service Protocol: Listen to the customer, acknowledge their issue, and provide a clear solution. Always maintain a positive tone. Input: {input}",
        "Step by step, address the customer's concern: 1) Understand the issue 2) Provide options 3) Confirm resolution. Customer message: {input}",
    ],
    "code_generation": [
        "Generate a Python function that accomplishes the following task. Include docstrings and type hints. Task: {input}",
        "Write clean, well-documented code for this requirement:\n\n{input}\n\nProvide the complete implementation with error handling.",
        "Code Generation Task:\n{input}\n\nRequirements:\n- Follow PEP 8 style guidelines\n- Include unit tests\n- Add comments for complex logic",
        "Implement the following function step by step. Think through the logic carefully before coding.\n\nTask: {input}",
    ],
    "creative_writing": [
        "Write a creative story based on the following prompt. Make it engaging and vivid. Prompt: {input}",
        "Create a narrative that captivates the reader. Include rich descriptions and character development. Story prompt: {input}",
        "Creative Writing Exercise:\n{input}\n\nFocus on:\n- Strong opening hook\n- Sensory details\n- Emotional resonance",
        "Develop a compelling story with a clear beginning, middle, and end. Prompt: {input}",
    ],
    "data_analysis": [
        "Analyze the following data and provide insights. Include visualizations and key findings. Data: {input}",
        "Perform a comprehensive data analysis:\n{input}\n\nProvide:\n- Summary statistics\n- Trends and patterns\n- Actionable recommendations",
        "Data Analysis Task:\n{input}\n\nUse statistical methods to identify significant patterns and outliers.",
        "Step-by-step data analysis: 1) Clean data 2) Explore patterns 3) Generate insights 4) Visualize findings. Dataset: {input}",
    ],
}

# Sample task descriptions
TASK_DESCRIPTIONS = {
    "customer_support": [
        "Handle customer refund request for defective product",
        "Assist customer with account login issues",
        "Resolve billing dispute",
        "Answer product availability question",
        "Process return request",
    ],
    "code_generation": [
        "Create function to parse CSV files",
        "Implement binary search algorithm",
        "Write API client for REST service",
        "Build data validation utility",
        "Generate report from database query",
    ],
    "creative_writing": [
        "Write short story about time travel",
        "Create character description for fantasy novel",
        "Develop plot outline for mystery",
        "Write dialogue between two characters",
        "Compose opening paragraph for thriller",
    ],
    "data_analysis": [
        "Analyze sales trends over last quarter",
        "Identify customer churn patterns",
        "Evaluate marketing campaign effectiveness",
        "Compare product performance metrics",
        "Forecast revenue for next month",
    ],
}


def generate_synthetic_run(domain: str, template: str, task_description: str) -> dict:
    """Generate a single synthetic run."""
    
    # Generate input and output
    input_text = f"Task: {task_description}\\n\\nPlease proceed."
    output_text = f"[Generated response for: {task_description}]" + " Lorem ipsum " * random.randint(20, 100)
    
    # Quality score: normally distributed around 0.7
    quality_score = max(0.0, min(1.0, random.gauss(0.7, 0.15)))
    
    # Token counts based on text length
    input_token_count = len(template.split()) + len(input_text.split())
    output_token_count = len(output_text.split())
    
    # Model selection
    models = ["gpt-4", "claude-3-opus", "gemini-1.5-pro"]
    model_name = random.choice(models)
    
    # Calculate cost
    pricing = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    }
    model_pricing = pricing[model_name]
    total_cost_usd = (input_token_count / 1000) * model_pricing["input"] + (output_token_count / 1000) * model_pricing["output"]
    
    # Safety flags
    safety_flags = {
        "contains_disallowed_content": False,
        "over_refuses_safe_request": False,
        "meets_refusal_quality_standards": True,
    }
    
    # Randomly add some safety issues
    if random.random() < 0.05:
        safety_flags["contains_disallowed_content"] = True
        quality_score *= 0.5
    
    # Timestamp within last 30 days
    timestamp = datetime.utcnow() - timedelta(days=random.randint(0, 30))
    
    return {
        "task_description": task_description,
        "template_text": template,
        "model_name": model_name,
        "input_text": input_text,
        "output_text": output_text,
        "quality_score": quality_score,
        "safety_flags": safety_flags,
        "input_token_count": input_token_count,
        "output_token_count": output_token_count,
        "total_cost_usd": total_cost_usd,
        "timestamp": timestamp,
    }


def main(num_runs: int = 500):
    """Generate synthetic data."""
    logger.info(f"Generating {num_runs} synthetic runs...")
    
    # Initialize ingestion manager
    ingestion_manager = DataIngestionManager()
    
    # Get database session
    session = ingestion_manager.get_session()
    
    # Generate runs
    runs_generated = 0
    templates_added = set()
    
    for i in range(num_runs):
        # Select random domain
        domain = random.choice(list(TEMPLATES.keys()))
        
        # Select random template and task
        template = random.choice(TEMPLATES[domain])
        task_description = random.choice(TASK_DESCRIPTIONS[domain])
        
        # Generate run
        run_data = generate_synthetic_run(domain, template, task_description)
        
        # Ingest run
        try:
            task_id = ingestion_manager.ingest_run(run_data)
            runs_generated += 1
            
            # Add template to templates table if not already added
            template_key = (template, domain)
            if template_key not in templates_added:
                template_obj = Template(
                    template_text=template,
                    domain=domain,
                    description=f"Template for {domain}",
                    avg_quality_score=run_data["quality_score"],
                    avg_cost_usd=run_data["total_cost_usd"],
                    avg_safety_score=85.0,
                    usage_count=1
                )
                session.add(template_obj)
                templates_added.add(template_key)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{num_runs} runs")
                session.commit()
                
        except Exception as e:
            logger.error(f"Error generating run {i}: {e}")
    
    # Final commit
    session.commit()
    session.close()
    
    logger.info(f"Successfully generated {runs_generated} runs and {len(templates_added)} unique templates")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic historical data")
    parser.add_argument("--num-runs", type=int, default=500, help="Number of runs to generate")
    args = parser.parse_args()
    
    main(args.num_runs)
