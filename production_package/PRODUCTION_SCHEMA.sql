-- COPY THIS COMPLETE SCHEMA TO PRODUCTION SUPABASE:

-- Required extensions
create extension if not exists "pgcrypto";
create extension if not exists "uuid-ossp";

-- Tenancy tables
create table if not exists tenants (
  id uuid primary key default gen_random_uuid(),
  name text not null unique,
  status text not null default 'active',
  billing_plan text not null default 'pro',
  created_at timestamptz not null default now()
);

create table if not exists projects (
  id uuid primary key default gen_random_uuid(),
  tenant_id uuid not null references tenants(id) on delete cascade,
  name text not null,
  is_active boolean not null default true,
  created_at timestamptz not null default now(),
  unique (tenant_id, name)
);

-- Enhanced API keys
create table if not exists api_keys (
  id uuid primary key default gen_random_uuid(),
  project_id uuid references projects(id) on delete cascade,
  key_hash text not null unique,
  label text,
  is_active boolean not null default true,
  rate_limit_per_min int not null default 60,
  created_at timestamptz not null default now()
);

-- Request/result tables
create table if not exists requests (
  id uuid primary key default gen_random_uuid(),
  request_id text,
  project_id uuid,
  key_hash text,
  filename text,
  content_hash text,
  file_size int,
  cache_hit boolean,
  status_code int,
  received_at timestamptz default now(),
  error text
);

create table if not exists results (
  request_id text primary key,
  content_hash text,
  scores jsonb,
  normalization jsonb,
  meta jsonb,
  processing_ms int,
  audio_ms int,
  warnings text[],
  version text,
  created_at timestamptz default now()
);

-- Usage analytics
create table if not exists usage_events (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references projects(id) on delete cascade,
  key_hash text not null,
  ts timestamptz not null default now(),
  status_code int not null,
  processing_ms int not null,
  bytes_in int not null default 0,
  bytes_out int not null default 0
);

create table if not exists usage_hourly (
  project_id uuid not null references projects(id) on delete cascade,
  key_hash text not null,
  hour timestamptz not null,
  calls int not null default 0,
  ms_total bigint not null default 0,
  ms_max int not null default 0,
  bytes_in_total bigint not null default 0,
  bytes_out_total bigint not null default 0,
  primary key (project_id, key_hash, hour)
);

-- Webhooks and audit
create table if not exists webhooks (
  id uuid primary key default gen_random_uuid(),
  project_id uuid not null references projects(id) on delete cascade,
  url text not null,
  secret text not null,
  is_active boolean not null default true,
  created_at timestamptz not null default now()
);

create table if not exists audit_log (
  id uuid primary key default gen_random_uuid(),
  ts timestamptz not null default now(),
  actor text not null,
  action text not null,
  target_type text not null,
  target_id uuid,
  details jsonb
);

-- Secure key management functions
create or replace function issue_api_key(
  p_project_id uuid,
  p_label text,
  p_rate_limit int
) returns table(id uuid, raw_token text) language plpgsql as $$
declare
  tok text := encode(gen_random_bytes(32), 'hex');
  sha text := encode(digest(tok, 'sha256'), 'hex');
  new_id uuid;
begin
  insert into api_keys (id, project_id, key_hash, label, is_active, rate_limit_per_min)
  values (gen_random_uuid(), p_project_id, sha, p_label, true, p_rate_limit)
  returning api_keys.id into new_id;

  insert into audit_log(actor, action, target_type, target_id, details)
  values ('system', 'ISSUE_KEY', 'api_key', new_id, jsonb_build_object('label', p_label, 'rate_limit', p_rate_limit));

  return query select new_id, tok;
end $$;

create or replace function revoke_api_key(p_api_key_id uuid) returns void language plpgsql as $$
begin
  update api_keys set is_active=false where id=p_api_key_id;
  insert into audit_log(actor, action, target_type, target_id) values ('system','REVOKE_KEY','api_key', p_api_key_id);
end $$;

-- Record usage function
create or replace function record_usage(
  p_project_id uuid,
  p_key_hash text,
  p_status int,
  p_ms int,
  p_bytes_in int,
  p_bytes_out int
) returns void language plpgsql as $$
declare
  h timestamptz := date_trunc('hour', now());
begin
  insert into usage_events (project_id, key_hash, ts, status_code, processing_ms, bytes_in, bytes_out)
  values (p_project_id, p_key_hash, now(), p_status, p_ms, p_bytes_in, p_bytes_out);

  insert into usage_hourly (project_id, key_hash, hour, calls, ms_total, ms_max, bytes_in_total, bytes_out_total)
  values (p_project_id, p_key_hash, h, 1, p_ms, p_ms, p_bytes_in, p_bytes_out)
  on conflict (project_id, key_hash, hour) do update
    set calls = usage_hourly.calls + 1,
        ms_total = usage_hourly.ms_total + excluded.ms_total,
        ms_max = greatest(usage_hourly.ms_max, excluded.ms_max),
        bytes_in_total = usage_hourly.bytes_in_total + excluded.bytes_in_total,
        bytes_out_total = usage_hourly.bytes_out_total + excluded.bytes_out_total;
end $$;

-- Indexes for performance
create index if not exists idx_api_keys_project on api_keys(project_id);
create index if not exists idx_requests_project on requests(project_id);
create index if not exists idx_requests_keyhash on requests(key_hash);
create index if not exists idx_requests_received_at on requests(received_at);
create index if not exists idx_usage_events_proj_ts on usage_events(project_id, ts);

-- RLS (Row Level Security)
alter table tenants enable row level security;
alter table projects enable row level security;
alter table usage_events enable row level security;
alter table usage_hourly enable row level security;
alter table webhooks enable row level security;
alter table audit_log enable row level security;

-- Initial data
insert into tenants (name, billing_plan) values ('Voxcentia', 'enterprise') on conflict (name) do nothing;
insert into projects (tenant_id, name) 
select t.id, 'Voice Analysis API' 
from tenants t 
where t.name = 'Voxcentia'
on conflict (tenant_id, name) do nothing;
