-- Run this in your PRODUCTION Supabase SQL editor:

-- Generate your first production API key
SELECT * FROM issue_api_key(
  (SELECT id FROM projects WHERE name = 'Voice Analysis API' LIMIT 1),
  'Production Client API Key',
  1000  -- 1000 requests per minute
);

-- The result will show:
-- id: [uuid] (store this for key management)
-- raw_token: [64-char hex token] (give this to your client)

-- View all API keys
SELECT 
  ak.id,
  ak.label,
  ak.rate_limit_per_min,
  ak.is_active,
  ak.created_at,
  p.name as project_name
FROM api_keys ak
JOIN projects p ON ak.project_id = p.id
ORDER BY ak.created_at DESC;
