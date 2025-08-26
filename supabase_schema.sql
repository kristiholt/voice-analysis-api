-- Create tables for X Voice API

-- API Keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    label TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    rate_limit INTEGER DEFAULT 1000,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Requests table (logs all API calls)
CREATE TABLE IF NOT EXISTS requests (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    api_key_id UUID REFERENCES api_keys(id),
    filename TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    cache_hit BOOLEAN DEFAULT false,
    error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Results table (stores analysis results)
CREATE TABLE IF NOT EXISTS results (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    request_id TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    scores JSONB NOT NULL,
    normalization JSONB NOT NULL,
    meta JSONB NOT NULL,
    processing_ms INTEGER NOT NULL,
    audio_ms INTEGER NOT NULL,
    warnings TEXT[] DEFAULT '{}',
    version TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Normalization baselines table
CREATE TABLE IF NOT EXISTS normalization_baselines (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    scheme TEXT NOT NULL,
    window_days INTEGER NOT NULL,
    sample_count INTEGER NOT NULL,
    emotion_stats JSONB NOT NULL,
    trait_stats JSONB NOT NULL,
    computed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(scheme)
);

-- Audio blobs table (optional, for storing audio files)
CREATE TABLE IF NOT EXISTS audio_blobs (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    storage_path TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    bucket TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_requests_content_hash ON requests(content_hash);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_results_content_hash ON results(content_hash);
CREATE INDEX IF NOT EXISTS idx_results_created_at ON results(created_at);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX IF NOT EXISTS idx_normalization_scheme ON normalization_baselines(scheme);

-- Enable Row Level Security (RLS) 
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE results ENABLE ROW LEVEL SECURITY;
ALTER TABLE normalization_baselines ENABLE ROW LEVEL SECURITY;
ALTER TABLE audio_blobs ENABLE ROW LEVEL SECURITY;

-- Create policies to allow service role access
CREATE POLICY "Service role can do everything on api_keys" ON api_keys
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on requests" ON requests
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on results" ON results
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on normalization_baselines" ON normalization_baselines
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Service role can do everything on audio_blobs" ON audio_blobs
    FOR ALL USING (auth.role() = 'service_role');