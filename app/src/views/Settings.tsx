// Settings, in a drawer. The common moves (model switching) live in the
// header; this is the quieter machinery. Keys live in ~/.aios/keys and
// never pass through this UI.

import { useEffect, useState } from 'react'
import { getJSON, sendJSON } from '../api'
import type { Settings as S } from '../types'

export function Settings({ onSaved }: { onSaved: () => void }) {
  const [s, setS] = useState<S | null>(null)
  const [saved, setSaved] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON<S>('/v1/settings').then(setS).catch(() => {})
  }, [])

  if (!s) return <div className="drawer-body dim">loading…</div>

  const set = (patch: Partial<S>) => {
    setS({ ...s, ...patch })
    setSaved(false)
  }

  const save = async () => {
    setError('')
    try {
      const next = await sendJSON<S>('PUT', '/v1/settings', s)
      setS(next)
      setSaved(true)
      onSaved()
    } catch (e) {
      setError(String(e))
    }
  }

  return (
    <div className="drawer-body settings">
      <h2>Privacy</h2>
      <div className="segmented">
        {(['persistent', 'incognito', 'paused'] as const).map((mode) => (
          <button
            key={mode}
            className={s.privacy_mode === mode ? 'active' : ''}
            onClick={() => set({ privacy_mode: mode })}
          >
            {mode}
          </button>
        ))}
      </div>
      <p className="hint">
        {s.privacy_mode === 'persistent' && 'Remembers everything.'}
        {s.privacy_mode === 'incognito' && 'Talks, remembers nothing, forgets this stretch on exit.'}
        {s.privacy_mode === 'paused' && 'Recalls freely, writes nothing.'}
      </p>

      <h2>Memory</h2>
      <label>
        Memory formed by
        <div className="segmented">
          <button className={s.memory_model === 'local' ? 'active' : ''} onClick={() => set({ memory_model: 'local' })}>
            local model
          </button>
          <button className={s.memory_model === 'answer' ? 'active' : ''} onClick={() => set({ memory_model: 'answer' })}>
            answer model
          </button>
        </div>
      </label>
      <p className="hint">
        {s.memory_model === 'local'
          ? 'Private: what you say is classified into memory on your machine, whichever model answers.'
          : 'Sharper: the active model also forms the memories. With a hosted model, the exchange leaves your machine for classification too.'}
      </p>
      <label>
        Local model
        <input value={s.local_model} onChange={(e) => set({ local_model: e.target.value })} />
      </label>
      <label>
        Max memories per reply
        <input
          type="number"
          min={1}
          max={100}
          value={s.max_retrieved}
          onChange={(e) => set({ max_retrieved: Number(e.target.value) })}
        />
      </label>
      <label>
        Session budget (tokens)
        <input
          type="number"
          min={200}
          step={100}
          value={s.window_budget}
          onChange={(e) => set({ window_budget: Number(e.target.value) })}
        />
      </label>

      <h2>Live actions</h2>
      <label className="rowlabel">
        <input type="checkbox" checked={s.web_enabled} onChange={(e) => set({ web_enabled: e.target.checked })} />
        Web search
      </label>
      <p className="hint">
        The assistant can search when a question needs current information. Keyless by
        default; add {'{"brave": "..."}'} to ~/.aios/keys for a proper search API. Tools
        come from MCP servers declared in ~/.aios/mcp.json (restart the daemon to pick
        them up).
      </p>

      <h2>Generation</h2>
      <label>
        Reply length (tokens)
        <input
          type="number"
          min={64}
          step={64}
          value={s.max_response_tokens}
          onChange={(e) => set({ max_response_tokens: Number(e.target.value) })}
        />
      </label>
      <label>
        Temperature
        <input
          type="number"
          min={0}
          max={2}
          step={0.1}
          value={s.temperature}
          onChange={(e) => set({ temperature: Number(e.target.value) })}
        />
      </label>

      <h2>Provider</h2>
      <label>
        Provider
        <select value={s.provider} onChange={(e) => set({ provider: e.target.value })}>
          <option value="ollama">Ollama (local)</option>
          <option value="claude">Claude</option>
          <option value="openai_compat">OpenAI-compatible</option>
          <option value="llamaserver">llama-server</option>
        </select>
      </label>
      <label>
        Model
        <input value={s.model} onChange={(e) => set({ model: e.target.value })} />
      </label>
      {s.provider === 'openai_compat' && (
        <label>
          Base URL
          <input value={s.base_url} onChange={(e) => set({ base_url: e.target.value })} />
        </label>
      )}
      <p className="hint">
        keys on file: {s.keys_present?.length ? s.keys_present.join(', ') : 'none'} — add
        them as JSON in ~/.aios/keys (chmod 600), e.g. {'{"anthropic": "sk-..."}'}
      </p>

      <div className="savebar">
        <button className="primary" onClick={save}>
          Save
        </button>
        {saved && <span className="ok">saved</span>}
        {error && <span className="error">{error}</span>}
      </div>
    </div>
  )
}
