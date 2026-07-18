// Working-memory gauge: how full the session's attention is, live. Click
// for the details. Eviction here isn't loss — it's demotion into long-term
// memory, and the copy says so.

import { useEffect, useRef, useState } from 'react'
import type { Status } from '../types'

export function ContextGauge({ status }: { status: Status | null }) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', close)
    return () => window.removeEventListener('mousedown', close)
  }, [open])

  const p = status?.pressure
  const used = p?.used ?? 0
  const budget = Math.max(1, p?.budget ?? 1)
  const ratio = Math.min(1, used / budget)
  const R = 8
  const C = 2 * Math.PI * R
  const level = p?.level ?? 'OK'
  const tone = level === 'OK' ? 'var(--ok)' : level === 'WARNING' ? 'var(--warm)' : 'var(--hot)'

  return (
    <div className="gauge-wrap" ref={ref}>
      <button
        className="gauge"
        title="Working memory"
        aria-label={`Working memory ${Math.round(ratio * 100)}% full`}
        onClick={() => setOpen(!open)}
      >
        <svg viewBox="0 0 22 22" width="22" height="22">
          <circle cx="11" cy="11" r={R} fill="none" stroke="var(--border)" strokeWidth="2.5" />
          <circle
            cx="11"
            cy="11"
            r={R}
            fill="none"
            stroke={tone}
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeDasharray={C}
            strokeDashoffset={C * (1 - ratio)}
            transform="rotate(-90 11 11)"
            className="gauge-arc"
          />
        </svg>
        <span className="gauge-pct">{Math.round(ratio * 100)}%</span>
      </button>
      {open && (
        <div className="popover gauge-pop">
          <div className="pop-title">Working memory</div>
          <div className="gauge-bar">
            <div className="gauge-fill" style={{ width: `${ratio * 100}%`, background: tone }} />
          </div>
          <div className="pop-row">
            <span>in use</span>
            <b>
              {used} / {budget} tokens
            </b>
          </div>
          <div className="pop-row">
            <span>pressure</span>
            <b style={{ color: tone }}>{level.toLowerCase()}</b>
          </div>
          <div className="pop-row">
            <span>demoted so far</span>
            <b>{p?.evictions ?? 0} messages</b>
          </div>
          <p className="pop-note">
            When this fills, older messages aren't lost — they're demoted into long-term
            memory and recalled when relevant.
          </p>
        </div>
      )}
    </div>
  )
}
