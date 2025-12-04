# Troubleshooting Guide - Prompt Recommender System

## Issue: PowerShell Script Execution Disabled

### Problem
When trying to activate the virtual environment, you see:
```
.ps1 cannot be loaded because running scripts is disabled on this system
```

### Solutions

#### **Option 1: Change PowerShell Execution Policy (Recommended)**

Run PowerShell as Administrator and execute:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then try activating the virtual environment again:
```powershell
.\venv\Scripts\activate
```

#### **Option 2: Use Command Prompt Instead**

Instead of PowerShell, use Command Prompt (cmd.exe):
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
venv\Scripts\activate.bat
```

#### **Option 3: Bypass Policy for Single Command**

```powershell
powershell -ExecutionPolicy Bypass -File .\venv\Scripts\Activate.ps1
```

#### **Option 4: Run Python Directly Without Activation**

You can run commands without activating the virtual environment:
```cmd
venv\Scripts\python.exe scripts\generate_synthetic_data.py --num-runs 500
venv\Scripts\python.exe -m src.api.main
```

---

## Complete Startup Guide (Updated for Windows)

### Step 1: Navigate to Project Directory
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
```

### Step 2: Activate Virtual Environment

**Using Command Prompt (easiest):**
```cmd
venv\Scripts\activate.bat
```

**Using PowerShell (if execution policy fixed):**
```powershell
.\venv\Scripts\activate
```

### Step 3: Verify Installation
```cmd
python --version
pip list
```

### Step 4: Generate Sample Data (if not already done)
```cmd
python scripts\generate_synthetic_data.py --num-runs 500
```

Expected output:
- Creates `data/prompt_recommender.db`
- Generates sample templates and historical runs

### Step 5: Start the API Server
```cmd
cd src
python -m api.main
```

Or using uvicorn directly:
```cmd
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Expected output:
```
INFO:     Started server process [PID]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 6: Test the API (in a new terminal)
```cmd
curl http://localhost:8000/health
```

Or open in browser: http://localhost:8000/docs

### Step 7: Start the Dashboard (Optional)

In a new Command Prompt window:
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
venv\Scripts\activate.bat
streamlit run src\dashboard\app.py
```

Navigate to: http://localhost:8501

---

## Common Issues and Fixes

### Issue: `ModuleNotFoundError`

**Cause:** Dependencies not installed or virtual environment not activated

**Fix:**
```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt
```

### Issue: API Server Won't Start

**Possible causes and fixes:**

1. **Port 8000 already in use:**
   ```cmd
   # Find process using port 8000
   netstat -ano | findstr :8000
   
   # Kill the process (replace PID with actual process ID)
   taskkill /PID <PID> /F
   
   # Or use a different port
   uvicorn src.api.main:app --port 8001
   ```

2. **Missing database:**
   ```cmd
   python scripts\generate_synthetic_data.py --num-runs 500
   ```

3. **Import errors:**
   ```cmd
   # Test imports
   python test_imports.py
   ```

### Issue: Empty Recommendations

**Cause:** Database empty or filters too restrictive

**Fix:**
1. Check database exists:
   ```cmd
   dir data\prompt_recommender.db
   ```

2. Regenerate data:
   ```cmd
   python scripts\generate_synthetic_data.py --num-runs 500
   ```

3. Relax filters in your request:
   ```json
   {
     "task_description": "your task",
     "max_cost_usd": 10.0,
     "min_safety_score": 0
   }
   ```

### Issue: Slow Startup

**Cause:** First run downloads sentence-transformers model (~100MB)

**Fix:** Wait for download to complete. Subsequent starts will be faster.

### Issue: Dashboard Shows "Cannot Connect to API"

**Fix:**
1. Ensure API server is running at http://localhost:8000
2. Test with: `curl http://localhost:8000/health`
3. Check firewall settings

---

## Quick Reference Commands

### Starting Everything (3 Terminal Windows Needed)

**Terminal 1 - API Server:**
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
venv\Scripts\activate.bat
cd src
python -m api.main
```

**Terminal 2 - Dashboard:**
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
venv\Scripts\activate.bat
streamlit run src\dashboard\app.py
```

**Terminal 3 - Testing:**
```cmd
cd c:\Users\User\Desktop\recommender System\prompt-recommender
venv\Scripts\activate.bat
curl http://localhost:8000/health
```

---

## Checking System Status

### Verify Virtual Environment
```cmd
where python
# Should show: c:\Users\User\Desktop\recommender System\prompt-recommender\venv\Scripts\python.exe
```

### Verify Dependencies
```cmd
pip list | findstr sentence-transformers
pip list | findstr fastapi
pip list | findstr streamlit
```

### Verify Database
```cmd
# Check if database exists
dir data\prompt_recommender.db

# Check database size (should be > 0 bytes)
```

### Test API Connection
```cmd
curl http://localhost:8000/health
# Or open in browser: http://localhost:8000/docs
```

---

## Environment Variables (Optional)

For production use, set API keys in `.env`:

```cmd
copy .env.example .env
notepad .env
```

Add your keys:
```
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-your-key-here
GOOGLE_API_KEY=your-key-here
```

**Note:** System works without API keys using approximations.

---

## Reinstalling from Scratch

If all else fails:

```cmd
# Navigate to project
cd c:\Users\User\Desktop\recommender System\prompt-recommender

# Remove old virtual environment
rmdir /s /q venv

# Remove old database
del data\prompt_recommender.db

# Create fresh virtual environment
python -m venv venv

# Activate (using cmd)
venv\Scripts\activate.bat

# Upgrade pip
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Generate data
python scripts\generate_synthetic_data.py --num-runs 500

# Start API
cd src
python -m api.main
```

---

## Need More Help?

1. Check full README.md for detailed documentation
2. Review QUICKSTART.md for basic setup
3. Check API docs at http://localhost:8000/docs
4. Review error messages in terminal output
