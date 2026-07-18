// The app shell. Opens straight into the one timeline — no chat picker, no
// titles, no session IDs anywhere.

import { useCallback, useEffect, useState } from 'react'
import { getJSON } from './api'
import type { Status } from './types'
import { Timeline } from './views/Timeline'
import { Memory } from './views/Memory'
import { Settings } from './views/Settings'
import { ContextGauge } from './components/ContextGauge'
import { ModelSwitcher } from './components/ModelSwitcher'
import { Link } from './router'

type View = 'timeline' | 'memory'

export default function App() {
  // ?view=memory deep-links straight into a view.
  const initialView = (): View =>
    new URLSearchParams(window.location.search).get('view') === 'memory' ? 'memory' : 'timeline'
  const [view, setView] = useState<View>(initialView)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [status, setStatus] = useState<Status | null>(null)
  const [jumpTo, setJumpTo] = useState<number | null>(null)

  const pollStatus = useCallback(() => {
    getJSON<Status>('/v1/status').then(setStatus).catch(() => setStatus(null))
  }, [])

  useEffect(() => {
    pollStatus()
    const t = setInterval(pollStatus, 5000)
    return () => clearInterval(t)
  }, [pollStatus])

  const jump = (entryId: number) => {
    setJumpTo(entryId)
    setView('timeline')
  }

  const mode = status?.privacy_mode ?? 'persistent'

  return (
    <div className="app">
      <header className="app-head">
        <Link to="/" className="wordmark app-mark">
          AIOS
        </Link>
        <nav className="app-nav">
          {(
            [
              ['timeline', 'Timeline'],
              ['memory', 'Memory'],
            ] as [View, string][]
          ).map(([v, label]) => (
            <button key={v} className={view === v ? 'active' : ''} onClick={() => setView(v)}>
              {label}
            </button>
          ))}
        </nav>
        <div className="head-right">
          {status ? (
            <>
              <ModelSwitcher
                current={status.model}
                onSwitched={pollStatus}
                onOpenSettings={() => setSettingsOpen(true)}
              />
              <ContextGauge status={status} />
            </>
          ) : (
            <span className="offline">
              <span className="dot bad" /> daemon offline
            </span>
          )}
          <button className="gear" title="Settings" aria-label="Settings" onClick={() => setSettingsOpen(true)}>
            <svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" strokeWidth="1.7">
              <circle cx="12" cy="12" r="3.2" />
              <path d="M19.4 15a1.7 1.7 0 0 0 .34 1.87l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.7 1.7 0 0 0-1.87-.34 1.7 1.7 0 0 0-1.03 1.56V21a2 2 0 1 1-4 0v-.09a1.7 1.7 0 0 0-1.11-1.56 1.7 1.7 0 0 0-1.87.34l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.7 1.7 0 0 0 .34-1.87 1.7 1.7 0 0 0-1.56-1.03H3a2 2 0 1 1 0-4h.09a1.7 1.7 0 0 0 1.56-1.11 1.7 1.7 0 0 0-.34-1.87l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.7 1.7 0 0 0 1.87.34h.09a1.7 1.7 0 0 0 1.03-1.56V3a2 2 0 1 1 4 0v.09a1.7 1.7 0 0 0 1.03 1.56 1.7 1.7 0 0 0 1.87-.34l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.7 1.7 0 0 0-.34 1.87v.09a1.7 1.7 0 0 0 1.56 1.03H21a2 2 0 1 1 0 4h-.09a1.7 1.7 0 0 0-1.51 1.03z" />
            </svg>
          </button>
        </div>
      </header>

      {mode !== 'persistent' && (
        <div className={`banner ${mode}`}>
          {mode === 'incognito'
            ? 'Incognito — this stretch will be forgotten when you leave it.'
            : 'Memory paused — recalling freely, writing nothing.'}
        </div>
      )}

      <main>
        {view === 'timeline' && (
          <Timeline jumpTo={jumpTo} onJumped={() => setJumpTo(null)} onTurnDone={pollStatus} />
        )}
        {view === 'memory' && <Memory onJump={jump} />}
      </main>

      {settingsOpen && (
        <>
          <div className="backdrop" onClick={() => setSettingsOpen(false)} />
          <aside className="drawer">
            <div className="drawer-head">
              <b>Settings</b>
              <button className="drawer-close" onClick={() => setSettingsOpen(false)} aria-label="Close">
                ✕
              </button>
            </div>
            <Settings onSaved={pollStatus} />
          </aside>
        </>
      )}
    </div>
  )
}
