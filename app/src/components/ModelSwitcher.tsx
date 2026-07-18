// The model switcher: hosted or self-hosted, swapped mid-relationship.
// Continuity is the daemon's job; this is just a preference.

import { useEffect, useRef, useState } from 'react'
import { getJSON, sendJSON } from '../api'

interface ModelEntry {
  provider: string
  model: string
  label: string
  note?: string
  available: boolean
  needs_key?: string
  soon?: boolean
  custom?: boolean
}

interface ModelsResponse {
  current: { provider: string; model: string }
  hosted: ModelEntry[]
  self_hosted: ModelEntry[]
}

export function ModelSwitcher({
  current,
  onSwitched,
  onOpenSettings,
}: {
  current: string
  onSwitched: () => void
  onOpenSettings: () => void
}) {
  const [open, setOpen] = useState(false)
  const [models, setModels] = useState<ModelsResponse | null>(null)
  const [switching, setSwitching] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    getJSON<ModelsResponse>('/v1/models').then(setModels).catch(() => {})
    const close = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    window.addEventListener('mousedown', close)
    return () => window.removeEventListener('mousedown', close)
  }, [open])

  const pick = async (m: ModelEntry) => {
    if (!m.available || m.soon) return
    if (m.custom) {
      setOpen(false)
      onOpenSettings()
      return
    }
    setSwitching(m.model)
    try {
      await sendJSON('PUT', '/v1/settings', { provider: m.provider, model: m.model })
      onSwitched()
      setOpen(false)
    } finally {
      setSwitching('')
    }
  }

  const row = (m: ModelEntry, i: number) => {
    const active = models?.current.model === m.model && models?.current.provider === m.provider
    return (
      <button
        key={`${m.provider}-${m.model}-${i}`}
        className={`model-row${active ? ' active' : ''}${!m.available || m.soon ? ' off' : ''}`}
        onClick={() => pick(m)}
      >
        <span className="model-label">
          {m.label}
          {m.soon && <em className="soon">soon</em>}
        </span>
        <span className="model-note">
          {switching === m.model
            ? 'switching…'
            : m.needs_key
              ? `add ${m.needs_key} key in ~/.aios/keys`
              : m.note}
        </span>
        {active && <span className="model-check">✓</span>}
      </button>
    )
  }

  return (
    <div className="switcher-wrap" ref={ref}>
      <button className="switcher" onClick={() => setOpen(!open)}>
        <span className="switcher-dot" />
        {current}
        <span className="chev">▾</span>
      </button>
      {open && (
        <div className="popover model-pop">
          <div className="pop-title">
            Model <span className="pop-hint">memory survives the swap</span>
          </div>
          <div className="model-group">Hosted</div>
          {models?.hosted.map(row)}
          <div className="model-group">Self-hosted</div>
          {models?.self_hosted.map(row)}
          {!models && <div className="model-note pad">loading…</div>}
        </div>
      )}
    </div>
  )
}
