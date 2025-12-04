"""
Add more diverse templates to the database for better recommendations.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.schema import Template
from src.data.ingestion import DataIngestionManager
import uuid

# Diverse templates for better differentiation
NEW_TEMPLATES = [
    # Customer Support - Specific scenarios
    {
        "domain": "customer_support",
        "template": "You are a billing specialist. Focus on resolving payment and invoice issues. Always verify account information before proceeding. Input: {input}",
        "description": "Specialized billing support template"
    },
    {
        "domain": "customer_support",
        "template": "Technical support agent. Guide users through troubleshooting steps methodically. Ask clarifying questions. Escalate if needed. Issue: {input}",
        "description": "Technical troubleshooting template"
    },
    {
        "domain": "customer_support",
        "template": "You specialize in product returns and exchanges. Be empathetic about dissatisfaction. Explain policies clearly. Offer alternatives. Request: {input}",
        "description": "Returns and refunds specialist"
    },
    
    # Code Generation - Different languages/frameworks
    {
        "domain": "code_generation",
        "template": "Generate TypeScript code with proper typing. Include JSDoc comments. Follow async/await best practices. Task: {input}",
        "description": "TypeScript code generation"
    },
    {
        "domain": "code_generation",
        "template": "Create a React component with TypeScript. Include props interface, error handling, and accessibility. Component: {input}",
        "description": "React/TypeScript component builder"
    },
    {
        "domain": "code_generation",
        "template": "Write SQL query with proper indexing considerations. Include comments explaining the logic. Optimize for performance. Query for: {input}",
        "description": "SQL query generator"
    },
    {
        "domain": "code_generation",
        "template": "Generate REST API endpoint using FastAPI. Include Pydantic models, error handling, and OpenAPI documentation. Endpoint: {input}",
        "description": "FastAPI endpoint generator"
    },
    
    # Data Analysis - Different focuses
    {
        "domain": "data_analysis",
        "template": "Perform time-series analysis on the data. Identify trends, seasonality, and anomalies. Provide statistical confidence intervals. Data: {input}",
        "description": "Time-series analysis specialist"
    },
    {
        "domain": "data_analysis",
        "template": "Conduct A/B test analysis. Calculate statistical significance, effect size, and confidence intervals. Provide actionable recommendations. Test results: {input}",
        "description": "A/B testing analyzer"
    },
    {
        "domain": "data_analysis",
        "template": "Create customer segmentation analysis. Use clustering techniques. Identify key characteristics of each segment. Dataset: {input}",
        "description": "Customer segmentation expert"
    },
    {
        "domain": "data_analysis",
        "template": "Build predictive model for forecasting. Compare multiple algorithms. Report accuracy metrics and feature importance. Target: {input}",
        "description": "Predictive modeling specialist"
    },
    
    # Creative Writing - Different styles
    {
        "domain": "creative_writing",
        "template": "Write in a noir detective style. Use first-person narrative. Create atmospheric tension with vivid noir imagery. Story prompt: {input}",
        "description": "Noir detective style"
    },
    {
        "domain": "creative_writing",
        "template": "Compose in the style of magical realism. Blend fantastical elements with everyday reality. Focus on sensory details. Prompt: {input}",
        "description": "Magical realism style"
    },
    {
        "domain": "creative_writing",
        "template": "Create a dialogue-driven scene. Show character through conversation. Use subtext and tension. Avoid exposition dumps. Scene: {input}",
        "description": "Dialogue-focused writing"
    },
    {
        "domain": "creative_writing",
        "template": "Write flash fiction (under 500 words). Every word must earn its place. Strong opening hook. Surprising ending. Theme: {input}",
        "description": "Flash fiction specialist"
    },
]

def add_diverse_templates():
    """Add diverse templates to database."""
    manager = DataIngestionManager()
    session = manager.get_session()
    
    added = 0
    for item in NEW_TEMPLATES:
        template = Template(
            template_id=str(uuid.uuid4()),
            template_text=item["template"],
            domain=item["domain"],
            description=item["description"],
            avg_quality_score=0.75,  # Higher quality for specialized templates
            avg_cost_usd=0.25,
            avg_safety_score=90.0,
            usage_count=0
        )
        session.add(template)
        added += 1
    
    session.commit()
    session.close()
    
    print(f"✅ Added {added} diverse templates to database!")
    print("Restart API for changes to take effect.")

if __name__ == "__main__":
    add_diverse_templates()
