// The memory browser, drawn as what it is: a hierarchy. Identity on top,
// topics as blocks, and inside each topic a spine of typed facts with their
// copy-on-write history. Search searches memory, not chats.

import { useEffect, useState } from 'react'
import { getJSON, sendJSON } from '../api'
import type { BrowseResult, BranchView, SearchResult, Version, VersionedValue } from '../types'

/** "[BRANCH_UPDATE] JSONL for the journal" -> typed fact. */
function parseFact(raw: string): { kind: string; text: string } {
  const m = /^\[([A-Z_]+)\]\s*(.*)$/.exec(raw)
  if (!m) return { kind: '', text: raw }
  const kind = m[1].toLowerCase().replace('branch_update', 'update').replace('preference_change', 'preference').replace('identity_update', 'identity')
  return { kind, text: m[2] }
}

export function Memory({ onJump }: { onJump: (entryId: number) => void }) {
  const [browse, setBrowse] = useState<BrowseResult | null>(null)
  const [open, setOpen] = useState<string | null>(null)
  const [q, setQ] = useState('')
  const [result, setResult] = useState<SearchResult | null>(null)
  const [searching, setSearching] = useState(false)

  const refresh = () => getJSON<BrowseResult>('/v1/memory/browse').then(setBrowse).catch(() => {})
  useEffect(() => {
    refresh()
  }, [])

  const search = async () => {
    if (!q.trim()) {
      setResult(null)
      return
    }
    setSearching(true)
    try {
      setResult(await getJSON<SearchResult>(`/v1/memory/search?q=${encodeURIComponent(q)}`))
    } finally {
      setSearching(false)
    }
  }

  const totalFacts = browse?.branches.reduce((n, b) => n + b.details.length, 0) ?? 0

  return (
    <div className="memoryview">
      <div className="searchbar">
        <input
          value={q}
          placeholder="Ask your memory — “when did I decide to switch to Postgres?”"
          onChange={(e) => {
            setQ(e.target.value)
            if (!e.target.value.trim()) setResult(null)
          }}
          onKeyDown={(e) => e.key === 'Enter' && search()}
        />
        <button onClick={search} disabled={searching}>
          {searching ? '…' : 'Search'}
        </button>
      </div>

      {result ? (
        <SearchResults result={result} onJump={onJump} />
      ) : browse ? (
        <>
          <IdentityBlock identity={browse.identity} onChanged={refresh} onJump={onJump} />

          <div className="mem-meta">
            <span>
              {browse.branches.length} topics · {totalFacts} facts
            </span>
            <span className="dim">every value keeps its history — nothing is overwritten</span>
          </div>

          {browse.branches.length === 0 && (
            <div className="empty">
              No memories yet. Go talk — the first conversation is where it learns who you are.
            </div>
          )}

          <div className="topic-grid">
            {browse.branches.map((b) => (
              <TopicBlock
                key={b.name}
                branch={b}
                open={open === b.name}
                onToggle={() => setOpen(open === b.name ? null : b.name)}
                onChanged={refresh}
                onJump={onJump}
              />
            ))}
          </div>
        </>
      ) : null}
    </div>
  )
}

function IdentityBlock({
  identity,
  onChanged,
  onJump,
}: {
  identity: VersionedValue
  onChanged: () => void
  onJump: (id: number) => void
}) {
  return (
    <div className="id-block">
      <div className="id-label">
        <span className="node-dot identity" /> identity · always loaded
        <span className="id-ops">
          <Correct
            initial={identity.current}
            onSave={async (value) => {
              await sendJSON('POST', '/v1/memory/correct', { target: 'identity', value })
              onChanged()
            }}
          />
        </span>
      </div>
      <p className="id-value">{identity.current || 'Not established yet — introduce yourself.'}</p>
      <VersionTrail versions={identity.versions} onJump={onJump} />
    </div>
  )
}

function TopicBlock({
  branch,
  open,
  onToggle,
  onChanged,
  onJump,
}: {
  branch: BranchView
  open: boolean
  onToggle: () => void
  onChanged: () => void
  onJump: (id: number) => void
}) {
  const del = async (target: 'branch' | 'detail', index?: number) => {
    const what = target === 'branch' ? `the whole “${branch.name}” topic` : 'this memory'
    if (!window.confirm(`Delete ${what}? This is the one true delete — history and all.`)) return
    await sendJSON('POST', '/v1/memory/delete', { branch: branch.name, target, index, confirm: true })
    onChanged()
  }

  return (
    <div className={`topic${open ? ' open' : ''}`}>
      <button className="topic-head" onClick={onToggle}>
        <span className="node-dot" />
        <span className="topic-name">{branch.name}</span>
        <span className="topic-counts">
          {branch.details.length > 0 && <i className="count-pip">{branch.details.length} facts</i>}
          {branch.archive > 0 && <i className="count-pip dim">{branch.archive} archived</i>}
        </span>
        <span className="topic-chev">{open ? '▾' : '▸'}</span>
      </button>
      {!open && branch.summary.current && <p className="topic-preview">{branch.summary.current}</p>}

      {open && (
        <div className="tree">
          {branch.summary.current && (
            <div className="tree-node summary">
              <div className="tree-line" />
              <div className="tree-content">
                <span className="fact-tag summary">summary</span>
                {branch.summary.current}
                <span className="factops">
                  <Correct
                    initial={branch.summary.current}
                    onSave={async (value) => {
                      await sendJSON('POST', '/v1/memory/correct', { target: 'summary', branch: branch.name, value })
                      onChanged()
                    }}
                  />
                </span>
                <VersionTrail versions={branch.summary.versions} onJump={onJump} />
              </div>
            </div>
          )}
          {branch.details.map((d, i) => {
            const f = parseFact(d.current)
            return (
              <div className="tree-node" key={i}>
                <div className="tree-line" />
                <div className="tree-content">
                  {f.kind && <span className={`fact-tag ${f.kind}`}>{f.kind}</span>}
                  {f.text}
                  <span className="factops">
                    <Correct
                      initial={f.text}
                      onSave={async (value) => {
                        await sendJSON('POST', '/v1/memory/correct', { target: 'detail', branch: branch.name, index: i, value })
                        onChanged()
                      }}
                    />
                    <button className="danger" onClick={() => del('detail', i)}>
                      ×
                    </button>
                  </span>
                  <VersionTrail versions={d.versions} onJump={onJump} />
                </div>
              </div>
            )
          })}
          <div className="tree-foot">
            <span className="dim">
              {branch.archive} raw exchanges archived beneath this topic
            </span>
            <button className="danger" onClick={() => del('branch')}>
              delete topic
            </button>
          </div>
        </div>
      )}
    </div>
  )
}

function SearchResults({ result, onJump }: { result: SearchResult; onJump: (id: number) => void }) {
  return (
    <div className="results">
      <div className="sectionlabel">
        memories · {result.memories.length} — {Math.round(result.search_ms)}ms
      </div>
      {result.memories.map((m) => (
        <div className="card" key={m.branch}>
          <div className="cardhead">
            <b>{m.branch}</b>
          </div>
          {m.summary && <p>{m.summary}</p>}
          {m.details.map((d) => {
            const f = parseFact(d.value)
            return (
              <div className="fact" key={d.index}>
                {f.kind && <span className={`fact-tag ${f.kind}`}>{f.kind}</span>}
                {f.text}
                <VersionTrail versions={d.versions} onJump={onJump} />
              </div>
            )
          })}
        </div>
      ))}
      <div className="sectionlabel">source turns · {result.turns.length}</div>
      {result.turns.map((t) => (
        <div className="card turn" key={t.idx}>
          <span className="who">{t.speaker}</span>
          <span className="when">{t.timestamp}</span>
          <p>{t.text}</p>
        </div>
      ))}
      {result.memories.length === 0 && result.turns.length === 0 && (
        <div className="empty">nothing recalled for “{result.query}”</div>
      )}
    </div>
  )
}

/** Copy-on-write history: every value keeps its versions; show them. */
function VersionTrail({ versions, onJump }: { versions: Version[]; onJump: (id: number) => void }) {
  if (versions.length <= 1) return null
  return (
    <details className="versions">
      <summary>{versions.length} versions</summary>
      {versions
        .slice()
        .reverse()
        .map((v, i) => {
          const m = /^journal:(\d+)$/.exec(v.source)
          return (
            <div className="version" key={i}>
              <span className="when">{fmtTs(v.timestamp)}</span>
              <span>{parseFact(v.value).text}</span>
              {m ? (
                <button className="link" onClick={() => onJump(Number(m[1]))}>
                  source turn →
                </button>
              ) : (
                <span className="dim">{v.source}</span>
              )}
            </div>
          )
        })}
    </details>
  )
}

function Correct({ initial, onSave }: { initial: string; onSave: (v: string) => Promise<void> }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(initial)
  if (!editing) {
    return (
      <button
        className="link"
        onClick={() => {
          setValue(initial)
          setEditing(true)
        }}
      >
        correct
      </button>
    )
  }
  return (
    <span className="correct">
      <input value={value} onChange={(e) => setValue(e.target.value)} autoFocus />
      <button
        onClick={async () => {
          if (value.trim()) await onSave(value.trim())
          setEditing(false)
        }}
      >
        save
      </button>
      <button onClick={() => setEditing(false)}>cancel</button>
    </span>
  )
}

function fmtTs(ts: number): string {
  if (!ts) return ''
  if (ts > 1_000_000_000) {
    return new Date(ts * 1000).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  }
  return `t=${ts}`
}
