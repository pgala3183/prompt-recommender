-- Initial database schema
-- SQLite version

-- Historical runs table
CREATE TABLE IF NOT EXISTS historical_runs (
    task_id TEXT PRIMARY KEY,
    task_description TEXT NOT NULL,
    template_text TEXT NOT NULL,
    model_name TEXT NOT NULL,
    input_text TEXT NOT NULL,
    output_text TEXT NOT NULL,
    quality_score REAL NOT NULL,
    safety_flags TEXT,  -- JSON
    input_token_count INTEGER NOT NULL,
    output_token_count INTEGER NOT NULL,
    total_cost_usd REAL NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for historical_runs
CREATE INDEX IF NOT EXISTS idx_task_description ON historical_runs(task_description);
CREATE INDEX IF NOT EXISTS idx_model_name ON historical_runs(model_name);
CREATE INDEX IF NOT EXISTS idx_timestamp ON historical_runs(timestamp);
CREATE INDEX IF NOT EXISTS idx_quality_score ON historical_runs(quality_score);

-- Templates table
CREATE TABLE IF NOT EXISTS templates (
    template_id TEXT PRIMARY KEY,
    template_text TEXT NOT NULL,
    domain TEXT,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    avg_quality_score REAL,
    avg_cost_usd REAL,
    avg_safety_score REAL,
    usage_count INTEGER DEFAULT 0
);

-- Indexes for templates
CREATE INDEX IF NOT EXISTS idx_domain ON templates(domain);
CREATE INDEX IF NOT EXISTS idx_avg_quality ON templates(avg_quality_score);

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    exp_id TEXT NOT NULL,
    variant TEXT NOT NULL,
    user_context TEXT,  -- JSON  
    template_id TEXT,
    outcome TEXT,  -- JSON
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for experiments
CREATE INDEX IF NOT EXISTS idx_exp_id ON experiments(exp_id);
CREATE INDEX IF NOT EXISTS idx_variant ON experiments(variant);
CREATE INDEX IF NOT EXISTS idx_timestamp_exp ON experiments(timestamp);
