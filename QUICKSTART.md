# Quickstart Guide

Get the LLM Prompt Recommender System up and running in 5 minutes!

## 1. Installation (2 minutes)

```bash
# Navigate to project directory
cd c:\Users\User\Desktop\ESP\prompt-recommender

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 2. Generate Sample Data (1 minute)

```bash
# Generate 500 synthetic historical runs
python scripts\generate_synthetic_data.py --num-runs 500
```

This creates:
- SQLite database at `data/prompt_recommender.db`
- Sample templates across 4 domains
- Realistic quality scores, costs, and safety flags

## 3. Start the API (1 minute)

```bash
# Start FastAPI server
cd src
python -m api.main
```

The API will be available at: http://localhost:8000

Test it:
```bash
curl http://localhost:8000/health
```

## 4. Use the Dashboard (1 minute)

In a new terminal:

```bash
# Make sure venv is activated
venv\Scripts\activate

# Launch Streamlit
streamlit run src\dashboard\app.py
```

Navigate to: http://localhost:8501

## 5. Make Your First Recommendation

### Option A: Via Dashboard

1. Open http://localhost:8501
2. Enter task: `"Handle customer refund request for defective product"`
3. Set filters (Max Cost: $0.50, Min Safety: 70)
4. Click "Get Recommendations"
5. View ranked templates with quality, cost, and safety scores!

### Option B: Via API

```bash
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d "{\"task_description\": \"customer support refund request\", \"max_cost_usd\": 0.5, \"min_safety_score\": 70}"
```

### Option C: Via Python

```python
import requests

response = requests.post(
    "http://localhost:8000/recommend",
    json={
        "task_description": "Generate Python function to parse JSON",
        "domain": "code_generation",
        "max_cost_usd": 0.5,
        "min_safety_score": 70,
        "num_recommendations": 3
    }
)

recommendations = response.json()["recommendations"]
for rec in recommendations:
    print(f"Quality: {rec['predicted_quality']:.2f}")
    print(f"Cost: ${rec['estimated_cost_usd']:.4f}")
    print(f"Safety: {rec['safety_score']:.0f}/100")
    print(f"Template: {rec['template_text'][:100]}...")
    print("-" * 80)
```

## What You Get

✅ **50 candidate templates** retrieved via embedding similarity  
✅ **Re-ranked by quality, cost, and safety** using LightGBM  
✅ **Filtered by your constraints** (max cost, min safety)  
✅ **Top-N recommendations** with detailed breakdowns  

## Next Steps

### Add Real Templates

Edit `scripts/generate_synthetic_data.py` to include your actual templates, or insert directly:

```python
from src.data.schema import Template
from src.data.ingestion import DataIngestionManager

ingestion = DataIngestionManager()
session = ingestion.get_session()

template = Template(
    template_text="Your custom template here: {input}",
    domain="your_domain",
    description="Template for X task",
    avg_quality_score=0.8,
    avg_cost_usd=0.15,
    avg_safety_score=90.0
)
session.add(template)
session.commit()
```

### Train Models on Real Data

Once you have production logs:

```bash
python scripts/train_models.py --data-path data/prompt_recommender.db
```

This will:
1. Build FAISS index from templates
2. Train LightGBM quality/cost models
3. Train safety reward model
4. Save artifacts to `models/` directory

### Run Tests

```bash
pytest tests/ -v --cov=src
```

### Set API Keys (Optional)

For production accuracy:

```bash
# Copy example env
copy .env.example .env

# Edit .env and add:
# OPENAI_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-...
# GOOGLE_API_KEY=...
```

This enables:
- Accurate tokenization per model
- LLM-based safety grading
- Better cost estimates

## Common Issues

**Issue**: `ModuleNotFoundError: No module named 'sentence_transformers'`  
**Fix**: `pip install sentence-transformers`

**Issue**: API returns empty recommendations  
**Fix**: 
1. Check templates exist: Open `data/prompt_recommender.db` in SQLite browser
2. Relax filters: Increase `max_cost_usd` or lower `min_safety_score`
3. Re-run data generation: `python scripts/generate_synthetic_data.py`

**Issue**: Dashboard shows "Cannot connect to API"  
**Fix**: Make sure API is running at http://localhost:8000

**Issue**: Slow initial startup  
**Fix**: First load downloads sentence-transformers model (~100MB). Subsequent starts are faster.

## Architecture Quick Reference

```
Request → Embedding Retrieval (FAISS) → Top-50 Candidates
       ↓
    LightGBM Re-ranking (Quality/Cost/Safety)
       ↓
    Filter (Cost & Safety Thresholds)
       ↓
    Top-N Results
```

## Sample Output

```json
{
  "recommendations": [
    {
      "template_id": "abc-123",
      "template_text": "You are a helpful customer service...",
      "predicted_quality": 0.87,
      "estimated_cost_usd": 0.24,
      "safety_score": 92.0,
      "itemized_cost": {
        "input_tokens": 120,
        "output_tokens": 80,
        "cost_usd": 0.24
      }
    }
  ]
}
```

## Performance Expectations

- **Retrieval**: ~50ms (10K templates)
- **Re-ranking**: ~100ms (50 candidates)
- **Total**: ~200ms end-to-end
- **Throughput**: ~100 req/sec (single instance)

## Support

Having issues? Check:
1. Full README.md for detailed documentation
2. API docs at http://localhost:8000/docs
3. Logs in terminal output

---

**Congratulations!** 🎉 You now have a working prompt recommendation system with cost optimization and safety scoring!
