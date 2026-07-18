// Shapes mirrored from the daemon API. No session IDs anywhere.

export interface Entry {
  id: number
  ts_ms: number
  kind: 'user' | 'assistant' | 'memory' | 'evict' | 'system' | string
  text: string
  meta?: Inspector
  ephemeral?: boolean
}

export interface TurnAction {
  type: 'web' | 'tool'
  query?: string
  name?: string
  results?: number
  error?: string
}

export interface Inspector {
  actions?: TurnAction[]
  images?: string[]
  turn_id?: number
  namespace?: string
  loaded?: number
  budget?: number
  retrieval_ms?: number
  generation_ms?: number
  faulted?: boolean
  fault_topic?: string
  retried?: boolean
  provider?: string
  writes?: { kind: string; content: string; branch: string }[]
  cancelled?: boolean
  privacy_mode?: string
  kind?: string
  branch?: string
  n?: number
}

export interface Settings {
  provider: string
  model: string
  base_url: string
  local_model: string
  memory_model: string
  embed_model: string
  num_ctx: number
  max_response_tokens: number
  temperature: number
  max_retrieved: number
  window_budget: number
  privacy_mode: 'persistent' | 'incognito' | 'paused' | string
  social_enabled: boolean
  web_enabled: boolean
  keys_present?: string[]
}

export interface Version {
  value: string
  timestamp: number
  source: string
}

export interface VersionedValue {
  current: string
  last_updated: number
  versions: Version[]
}

export interface BranchView {
  name: string
  created_at: number
  summary: VersionedValue
  details: VersionedValue[]
  archive: number
  tags: string[]
}

export interface BrowseResult {
  identity: VersionedValue
  branches: BranchView[]
}

export interface SearchTurn {
  idx: number
  speaker: string
  text: string
  timestamp: string
}

export interface SearchMemory {
  branch: string
  score: number
  summary: string
  details: { index: number; value: string; last_updated: number; versions: Version[] }[]
}

export interface SearchResult {
  query: string
  turns: SearchTurn[]
  memories: SearchMemory[]
  search_ms: number
}

export interface Status {
  provider: string
  model: string
  local_model: string
  privacy_mode: string
  ollama_up: boolean
  kv: { mounted: boolean; restored_tokens: number }
  pressure: { used: number; budget: number; level: string; evictions: number }
  counters: {
    turns_served: number
    journal_entries: number
    store: { branches: number; details: number; archive_entries: number; total_versions: number }
  }
}

// SSE events out of POST /v1/turn.
export type TurnEvent =
  | { t: 'turn'; id: number; user_entry: number }
  | { t: 'route'; loaded: number; namespace: string; budget: number; retrieval_ms: number }
  | { t: 'tok'; v: string }
  | { t: 'fault'; topic: string }
  | { t: 'web'; query: string }
  | { t: 'tool'; name: string }
  | { t: 'mem'; kind: string; content: string; branch: string }
  | { t: 'evict'; n: number }
  | { t: 'err'; message: string }
  | { t: 'done'; turn_id: number; entry: number; reply: string; inspector: Inspector; error?: boolean }
