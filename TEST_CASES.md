# Test Cases for Prompt Recommender System
## With 52K+ Stanford Alpaca Dataset

### Prerequisites
✅ Alpaca dataset imported (52,000 templates)  
✅ API running at http://localhost:8000  
✅ Dashboard running at http://localhost:8501  
✅ Ollama grader enabled (optional)

---

## Test Suite 1: Diversity & Specificity Tests

These tests verify that different queries return genuinely different recommendations.

### Test 1.1: Python Code Generation
**Query:** `"Write a Python function to parse JSON files"`

**Expected:**
- Top results should be **Python-specific** templates
- Keywords: "python", "function", "json", "parse"
- Domain: `code_generation`
- Different from JavaScript/SQL queries

**Validation:**
```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"Write a Python function to parse JSON files\", \"num_recommendations\": 3}"
```

✅ **Pass if:** Template text contains "python" or "function"

---

### Test 1.2: SQL Query Generation
**Query:** `"Generate SQL query for customer analytics"`

**Expected:**
- Top results mention **SQL, query, database**
- Domain: `code_generation` or `data_analysis`
- **Different templates** than Test 1.1

**Validation:**
Check that top recommendation text differs from Python test.

---

### Test 1.3: Creative Story Writing
**Query:** `"Write a short story about time travel"`

**Expected:**
- Domain: `creative_writing`
- Keywords: "story", "write", "creative", "narrative"
- **Completely different** from code generation tests

---

### Test 1.4: Data Analysis
**Query:** `"Analyze sales trends and create visualization"`

**Expected:**
- Domain: `data_analysis`
- Keywords: "analyze", "data", "trends", "chart"
- Mentions: statistics, visualization, insights

---

### Test 1.5: Customer Support
**Query:** `"Help customer troubleshoot login issues"`

**Expected:**
- Domain: `customer_support`
- Keywords: "help", "assist", "support", "troubleshoot"
- Professional, empathetic tone

---

## Test Suite 2: Edge Cases & Filters

### Test 2.1: Very Low Cost Filter
**Query:** `"Any task"`  
**Filters:** `max_cost_usd=0.01`

**Expected:**
- Returns 0-2 recommendations (very restrictive)
- Only cheapest templates pass filter

**Command:**
```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"test\", \"max_cost_usd\": 0.01}"
```

---

### Test 2.2: High Safety Requirement
**Query:** `"Generate content"`  
**Filters:** `min_safety_score=95`

**Expected:**
- Only safest templates returned
- Fewer results than default

---

### Test 2.3: Maximum Results
**Query:** `"Code generation"`  
**Filters:** `num_recommendations=10`

**Expected:**
- Returns exactly 10 recommendations
- All ranked by combined score
- Scores should be descending

---

## Test Suite 3: Specificity vs Generality

### Test 3.1: Specific Framework Query
**Query:** `"Create React component with hooks"`

**Expected:**
- Top result specifically mentions React/components
- Better match than generic "create web app"

---

### Test 3.2: Generic Query
**Query:** `"Help me with a task"`

**Expected:**
- Returns general-purpose templates
- Higher variety in top results

---

### Test 3.3: Domain-Specific Language
**Query:** `"Implement binary search tree in C++"`

**Expected:**
- Top results mention: algorithm, data structure, C++
- Not Python or JavaScript templates

---

## Test Suite 4: Quality vs Cost Trade-offs

### Test 4.1: Quality Priority
**Query:** `"Complex data analysis"`  
**Filters:** `max_cost_usd=2.0, min_safety_score=70`

**Expected:**
- Returns high-quality templates
- May have higher costs
- Quality scores: 0.7-0.9

---

### Test 4.2: Cost Priority
**Query:** `"Simple greeting message"`  
**Filters:** `max_cost_usd=0.10`

**Expected:**
- Returns shorter, simpler templates
- Lower costs
- Still relevant to task

---

## Test Suite 5: Dashboard UI Tests

### Test 5.1: Visual Variety Check
**Steps:**
1. Open dashboard: http://localhost:8501
2. Query: `"Python programming"`
3. Note top 3 template IDs
4. Query: `"Story writing"`
5. Compare template IDs

**Expected:**
- ✅ **0% overlap** in template IDs
- Different domains shown
- Different quality/cost/safety scores

---

### Test 5.2: Safety Badge Colors
**Query:** Multiple queries to find different safety scores

**Expected:**
- 🟢 Green badge (90-100): "Excellent"
- 🟡 Yellow badge (70-89): "Good"
- 🔴 Red badge (<70): "Review Required"

---

### Test 5.3: Chart Rendering
**Query:** Any query with 5 results

**Expected:**
- ✅ "Quality vs Safety Scores" bar chart displays
- ✅ "Cost Comparison" bar chart displays
- ✅ Charts have different values per template

---

## Test Suite 6: Performance Tests

### Test 6.1: Cold Start Latency
**Test:**
```bash
time curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"first query\"}"
```

**Expected:**
- With Ollama: 10-30 seconds (grades 50 candidates)
- Without Ollama: 1-3 seconds
- Response format: Valid JSON

---

### Test 6.2: Warm Request Latency
**Test:** Same query as 6.1, run again

**Expected:**
- With Ollama: <2 seconds (cached grading)
- Without Ollama: <1 second
- Much faster than cold start

---

### Test 6.3: Concurrent Requests
**Test:** Run 10 simultaneous requests

**Expected:**
- All return 200 status
- No crashes
- Response times: <5 seconds each

---

## Test Suite 7: Retrieval Quality Tests

### Test 7.1: Semantic Similarity
**Query:** `"Explain machine learning concepts"`  
**vs**  
**Query:** `"Teach me about AI"`

**Expected:**
- Should return **similar** top recommendations
- Both understand ML/AI are related

---

### Test 7.2: Different Intent Recognition
**Query:** `"Debug Python code"`  
**vs**  
**Query:** `"Write Python code"`

**Expected:**
- Different top recommendations
- Debug = troubleshooting focus
- Write = creation focus

---

## Test Suite 8: Comprehensive Integration Test

### Test 8.1: End-to-End Workflow
**Steps:**
1. Health check: `curl http://localhost:8000/health`
   - ✅ All models loaded: true
   
2. Query: `"Create REST API endpoint"`
   - ✅ Returns 5 recommendations
   - ✅ Domain: code_generation
   - ✅ Contains API/endpoint keywords
   
3. Apply strict filters: `max_cost=0.20, min_safety=85`
   - ✅ Returns 1-3 results
   - ✅ All pass filters
   
4. Request 10 results
   - ✅ Returns full 10
   - ✅ Ranked by combined_score

---

## Test Suite 9: Ollama AI Safety Tests

### Test 9.1: Safe Template Detection
**Query:** `"Professional business email"`

**Expected (with Ollama):**
- Safety score: 85-95
- No disallowed content flags
- High quality template

---

### Test 9.2: Potentially Problematic Content
**Query:** `"Aggressive marketing tactics"`

**Expected (with Ollama):**
- Ollama analyzes context
- May flag if unsafe
- Safety score: 60-80 range

---

## Quick Validation Script

Run all core tests quickly:

```bash
# Test 1: Diversity
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"Python code\"}" | jq '.recommendations[0].domain'

curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"Write story\"}" | jq '.recommendations[0].domain'

# Test 2: Filters
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"test\", \"max_cost_usd\": 0.05}" | jq '.recommendations | length'

# Test 3: Performance
time curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" -d "{\"task_description\": \"test\"}" > /dev/null
```

---

## Success Criteria

### ✅ System is working correctly if:

1. **Diversity**: Different queries return different top recommendations
2. **Specificity**: Specific queries match specific templates better
3. **Filters**: Cost and safety filters correctly exclude templates
4. **Performance**: Responses within acceptable time (<5s with Ollama, <1s without)
5. **Quality**: Recommendations are genuinely relevant to query
6. **Safety**: Safety scores vary based on template content
7. **Scale**: Works with 52K templates in database
8. **UI**: Dashboard displays results, charts, and badges correctly

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Same recommendations for everything | Restart API to rebuild FAISS index with new data |
| Very slow responses | First request with Ollama takes longer (model loading) |
| Empty recommendations | Check filters aren't too restrictive |
| API 503 error | Models not loaded - check startup logs |
| Low diversity | Verify 52K templates imported: check database |

---

## Expected Results with 52K Dataset

**Before** (48 synthetic templates):
- Limited variety
- Same ~5 templates for similar queries
- Generic matching

**After** (52,000 Alpaca templates):
- ✅ Massive variety
- ✅ Highly specific matching
- ✅ Genuinely different results per query
- ✅ Better semantic understanding
- ✅ Real-world, tested prompts

---

**Your system should now demonstrate production-grade recommendation quality!** 🚀
