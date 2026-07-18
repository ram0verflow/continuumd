// The per-response memory inspector: what was recalled, from where, how
// fast, whether the model faulted, and what got written back. All of it
// comes straight off the turn's QueryResult, carried in the journal meta.

import type { Inspector } from '../types'

export function InspectorPanel({ meta }: { meta: Inspector }) {
  if (meta.loaded === undefined && meta.provider === undefined) return null
  return (
    <details className="inspector">
      <summary>
        {meta.loaded ?? 0} memories · {Math.round(meta.retrieval_ms ?? 0)}ms recall ·{' '}
        {((meta.generation_ms ?? 0) / 1000).toFixed(1)}s
        {meta.faulted ? ' · fault' : ''}
        {meta.cancelled ? ' · stopped' : ''}
      </summary>
      <div className="grid">
        <span>namespace</span>
        <b>{meta.namespace || '—'}</b>
        <span>memories loaded</span>
        <b>{meta.loaded ?? 0}</b>
        <span>memory budget</span>
        <b>{meta.budget ?? 0} tokens</b>
        <span>recall</span>
        <b>{Math.round(meta.retrieval_ms ?? 0)} ms</b>
        <span>generation</span>
        <b>{Math.round(meta.generation_ms ?? 0)} ms</b>
        <span>page fault</span>
        <b>{meta.faulted ? `yes — “${meta.fault_topic}”${meta.retried ? ', retried' : ', unresolved'}` : 'no'}</b>
        <span>model</span>
        <b>{meta.provider || '—'}</b>
        {meta.privacy_mode && meta.privacy_mode !== 'persistent' && (
          <>
            <span>privacy</span>
            <b>{meta.privacy_mode}</b>
          </>
        )}
      </div>
      {meta.actions && meta.actions.length > 0 && (
        <div className="acts">
          {meta.actions.map((a, i) => (
            <div key={i}>
              {a.type === 'web'
                ? `⌕ web — “${a.query}”${a.error ? ` (failed: ${a.error})` : ` · ${a.results} results`}`
                : `⚙ ${a.name}${a.error ? ` (failed: ${a.error})` : ''}`}
            </div>
          ))}
        </div>
      )}
      {meta.writes && meta.writes.length > 0 && (
        <div className="writes">
          {meta.writes.map((w, i) => (
            <div key={i}>
              ◆ {w.kind.toLowerCase().replace('_', ' ')}
              {w.branch ? ` · ${w.branch}` : ''} — {w.content}
            </div>
          ))}
        </div>
      )}
    </details>
  )
}
