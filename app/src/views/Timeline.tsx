// The one timeline. Not a chat list — there is nothing to pick, no "new
// conversation". Grouped by day, infinite scroll backwards through the
// journal, composer at the bottom, streaming replies with a per-response
// inspector disclosure.

import { useCallback, useEffect, useRef, useState } from 'react'
import { cancelTurn, getJSON, streamTurn } from '../api'
import type { Entry, Inspector } from '../types'
import { InspectorPanel } from '../components/Inspector'

interface Stream {
  turnId: number | null
  text: string
  route?: { loaded: number; namespace: string; retrieval_ms: number }
  fault?: string
  web?: string
  tool?: string
  error?: string
}

let tempId = -1

// Voice in: browser speech recognition where available (Chrome/Safari).
type Recognition = { start: () => void; stop: () => void } | null
function makeRecognition(onText: (t: string) => void, onEnd: () => void): Recognition {
  const w = window as unknown as { webkitSpeechRecognition?: new () => any; SpeechRecognition?: new () => any }
  const Ctor = w.SpeechRecognition ?? w.webkitSpeechRecognition
  if (!Ctor) return null
  const rec = new Ctor()
  rec.continuous = true
  rec.interimResults = true
  rec.onresult = (e: any) => {
    let text = ''
    for (const res of e.results) text += res[0].transcript
    onText(text)
  }
  rec.onend = onEnd
  rec.onerror = onEnd
  return rec
}

// Voice out: system TTS, no cloud.
function speak(text: string) {
  const clean = text.replace(/https?:\/\/\S+/g, 'link').slice(0, 1200)
  if (!clean) return
  window.speechSynthesis.cancel()
  window.speechSynthesis.speak(new SpeechSynthesisUtterance(clean))
}

export function Timeline({
  jumpTo,
  onJumped,
  onTurnDone,
}: {
  jumpTo: number | null
  onJumped: () => void
  onTurnDone?: () => void
}) {
  const [entries, setEntries] = useState<Entry[]>([])
  const [hasMore, setHasMore] = useState(true)
  const [digest, setDigest] = useState<string | null>(null)
  const [stream, setStream] = useState<Stream | null>(null)
  const [draft, setDraft] = useState('')
  const [attached, setAttached] = useState<string[]>([])
  const [recording, setRecording] = useState(false)
  const [voiceOut, setVoiceOut] = useState(() => localStorage.getItem('aios-voice') === 'on')
  const listRef = useRef<HTMLDivElement>(null)
  const stickBottom = useRef(true)
  const recRef = useRef<Recognition>(null)
  const fileRef = useRef<HTMLInputElement>(null)
  const streaming = stream !== null

  const toggleVoiceOut = () => {
    const next = !voiceOut
    setVoiceOut(next)
    localStorage.setItem('aios-voice', next ? 'on' : 'off')
    if (!next) window.speechSynthesis.cancel()
  }

  const toggleMic = () => {
    if (recording) {
      recRef.current?.stop()
      return
    }
    const rec = makeRecognition(
      (text) => setDraft(text),
      () => setRecording(false),
    )
    if (!rec) {
      alert('Voice input needs Chrome or Safari (SpeechRecognition).')
      return
    }
    recRef.current = rec
    setRecording(true)
    rec.start()
  }

  const attach = (files: FileList | null) => {
    if (!files) return
    for (const f of Array.from(files).slice(0, 4)) {
      if (!f.type.startsWith('image/')) continue
      const reader = new FileReader()
      reader.onload = () => setAttached((cur) => [...cur, String(reader.result)])
      reader.readAsDataURL(f)
    }
  }

  const loadOlder = useCallback(async (before: number): Promise<Entry[]> => {
    const res = await getJSON<{ entries: Entry[] }>(
      `/v1/timeline?limit=100${before ? `&before=${before}` : ''}`,
    )
    const older = res.entries.slice().reverse() // newest-first -> ascending
    if (res.entries.length < 100) setHasMore(false)
    return older
  }, [])

  // First render: latest page plus the daemon-composed digest.
  useEffect(() => {
    loadOlder(0).then(setEntries).catch(() => {})
    getJSON<{ text: string }>('/v1/digest')
      .then((d) => setDigest(d.text))
      .catch(() => {})
  }, [loadOlder])

  // Keep pinned to the bottom while streaming, unless the user scrolled up.
  useEffect(() => {
    const el = listRef.current
    if (el && stickBottom.current) el.scrollTop = el.scrollHeight
  }, [entries, stream, digest])

  // Jump from memory search: page back until the entry is loaded, then flash it.
  useEffect(() => {
    if (jumpTo == null) return
    let alive = true
    ;(async () => {
      let current = entries
      let more = hasMore
      while (alive && !current.some((e) => e.id === jumpTo) && more && current.length > 0) {
        const older = await loadOlder(current[0].id)
        if (older.length === 0) more = false
        current = [...older, ...current]
      }
      if (!alive) return
      setEntries(current)
      setHasMore(more)
      requestAnimationFrame(() => {
        const el = document.querySelector(`[data-eid="${jumpTo}"]`)
        if (el) {
          stickBottom.current = false
          el.scrollIntoView({ block: 'center' })
          el.classList.add('flash')
          setTimeout(() => el.classList.remove('flash'), 2200)
        }
        onJumped()
      })
    })()
    return () => {
      alive = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jumpTo])

  const onScroll = async () => {
    const el = listRef.current
    if (!el) return
    stickBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80
    if (el.scrollTop < 60 && hasMore && entries.length > 0 && !streaming) {
      const prevHeight = el.scrollHeight
      const older = await loadOlder(entries[0].id)
      if (older.length > 0) {
        setEntries((cur) => [...older, ...cur])
        requestAnimationFrame(() => {
          el.scrollTop = el.scrollHeight - prevHeight
        })
      }
    }
  }

  const send = async () => {
    const text = draft.trim()
    const images = attached
    if ((!text && images.length === 0) || streaming) return
    if (recording) recRef.current?.stop()
    setDraft('')
    setAttached([])
    setDigest(null) // the greeting made way for the conversation
    stickBottom.current = true
    const userTemp = tempId--
    setEntries((cur) => [
      ...cur,
      { id: userTemp, ts_ms: Date.now(), kind: 'user', text: text || '(shared an image)', meta: { images } },
    ])
    setStream({ turnId: null, text: '' })
    try {
      await streamTurn(text, images, (ev) => {
        switch (ev.t) {
          case 'turn':
            setStream((s) => s && { ...s, turnId: ev.id })
            setEntries((cur) => cur.map((e) => (e.id === userTemp ? { ...e, id: ev.user_entry } : e)))
            break
          case 'route':
            setStream((s) => s && { ...s, route: ev })
            break
          case 'tok':
            setStream((s) => s && { ...s, text: s.text + ev.v })
            break
          case 'fault':
            setStream((s) => s && { ...s, fault: ev.topic, text: '' })
            break
          case 'web':
            setStream((s) => s && { ...s, web: ev.query, tool: undefined, text: '' })
            setEntries((cur) => [
              ...cur,
              { id: tempId--, ts_ms: Date.now(), kind: 'web', text: ev.query },
            ])
            break
          case 'tool':
            setStream((s) => s && { ...s, tool: ev.name, web: undefined, text: '' })
            setEntries((cur) => [
              ...cur,
              { id: tempId--, ts_ms: Date.now(), kind: 'tool', text: ev.name },
            ])
            break
          case 'mem':
            setEntries((cur) => [
              ...cur,
              { id: tempId--, ts_ms: Date.now(), kind: 'memory', text: ev.content, meta: { kind: ev.kind, branch: ev.branch } },
            ])
            break
          case 'evict':
            setEntries((cur) => [
              ...cur,
              { id: tempId--, ts_ms: Date.now(), kind: 'evict', text: `${ev.n} messages demoted`, meta: { n: ev.n } },
            ])
            break
          case 'err':
            setStream((s) => s && { ...s, error: ev.message })
            break
          case 'done':
            setEntries((cur) => [
              ...cur,
              { id: ev.entry ?? tempId--, ts_ms: Date.now(), kind: 'assistant', text: ev.reply ?? '', meta: ev.inspector },
            ])
            setStream(null)
            if (voiceOut && ev.reply) speak(ev.reply)
            onTurnDone?.()
            break
        }
      })
    } catch (e) {
      setStream(null)
      setEntries((cur) => [
        ...cur,
        { id: tempId--, ts_ms: Date.now(), kind: 'assistant', text: `[connection error: ${String(e)}]` },
      ])
    }
    setStream((s) => (s && s.turnId === null ? null : s)) // stream ended without done
  }

  const stop = () => {
    if (stream?.turnId != null) cancelTurn(stream.turnId).catch(() => {})
  }

  return (
    <div className="timeline">
      <div className="entries" ref={listRef} onScroll={onScroll}>
        {!hasMore && <div className="edge">the beginning</div>}
        {renderGrouped(entries)}
        {digest && entries.length >= 0 && (
          <div className="row assistant">
            <div className="bubble digest">{digest}</div>
          </div>
        )}
        {stream && (
          <div className="row assistant live">
            <div className="bubble">
              {stream.route && (
                <div className="recall-pill">
                  ◆ recalled {stream.route.loaded} memories · {Math.round(stream.route.retrieval_ms)}ms
                </div>
              )}
              {stream.fault && !stream.text && (
                <div className="faultnote">reaching deeper — “{stream.fault}”…</div>
              )}
              {stream.web && !stream.text && (
                <div className="actnote">searching the web — “{stream.web}”…</div>
              )}
              {stream.tool && !stream.text && (
                <div className="actnote">using {stream.tool}…</div>
              )}
              {stream.text ? (
                <>
                  {stream.text}
                  <span className="caret" />
                </>
              ) : (
                !stream.fault && !stream.web && !stream.tool && <span className="thinking">thinking</span>
              )}
              {stream.error && <div className="error">{stream.error}</div>}
            </div>
          </div>
        )}
      </div>
      <div className="composer-zone">
        {attached.length > 0 && (
          <div className="attach-strip">
            {attached.map((src, i) => (
              <span className="attach-thumb" key={i}>
                <img src={src} alt="" />
                <button onClick={() => setAttached((cur) => cur.filter((_, j) => j !== i))}>✕</button>
              </span>
            ))}
          </div>
        )}
        <div className="composer">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            multiple
            hidden
            onChange={(e) => {
              attach(e.target.files)
              e.target.value = ''
            }}
          />
          <button className="ghostbtn" title="Attach an image" onClick={() => fileRef.current?.click()}>
            ＋
          </button>
          <textarea
            value={draft}
            placeholder={recording ? 'listening…' : 'Say something — it remembers.'}
            rows={Math.min(6, draft.split('\n').length)}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
          />
          <button
            className={`ghostbtn mic${recording ? ' rec' : ''}`}
            title={recording ? 'Stop listening' : 'Dictate'}
            onClick={toggleMic}
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
              <rect x="9" y="3" width="6" height="11" rx="3" />
              <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
            </svg>
          </button>
          <button
            className={`ghostbtn${voiceOut ? ' on' : ''}`}
            title={voiceOut ? 'Voice replies on' : 'Voice replies off'}
            onClick={toggleVoiceOut}
          >
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M11 5 6 9H3v6h3l5 4V5z" />
              {voiceOut && <path d="M15.5 8.5a5 5 0 0 1 0 7M18 6a8.5 8.5 0 0 1 0 12" />}
            </svg>
          </button>
          {streaming ? (
            <button className="stop" onClick={stop} title="Stop">
              ◼
            </button>
          ) : (
            <button className="send" onClick={send} disabled={!draft.trim() && attached.length === 0} title="Send">
              ↑
            </button>
          )}
        </div>
      </div>
    </div>
  )
}

function renderGrouped(entries: Entry[]) {
  const out: JSX.Element[] = []
  let lastDay = ''
  for (const e of entries) {
    const day = dayLabel(e.ts_ms)
    if (day !== lastDay) {
      lastDay = day
      out.push(
        <div key={`day-${e.id}`} className="day">
          {day}
        </div>,
      )
    }
    out.push(<EntryRow key={e.id} entry={e} />)
  }
  return out
}

function EntryRow({ entry }: { entry: Entry }) {
  switch (entry.kind) {
    case 'user':
      return (
        <div className="row user" data-eid={entry.id}>
          <div className="bubble">
            {(entry.meta?.images ?? []).length > 0 && (
              <div className="bubble-imgs">
                {entry.meta!.images!.map((im, i) => (
                  <img key={i} src={im.startsWith('data:') ? im : `/v1/media/${im}`} alt="" />
                ))}
              </div>
            )}
            {entry.text}
          </div>
        </div>
      )
    case 'web':
      return (
        <div className="chip act" data-eid={entry.id}>
          ⌕ searched the web — “{entry.text}”
        </div>
      )
    case 'tool':
      return (
        <div className="chip act" data-eid={entry.id}>
          ⚙ used {entry.text}
        </div>
      )
    case 'assistant':
      return (
        <div className="row assistant" data-eid={entry.id}>
          <div className="bubble">
            {entry.text}
            {entry.meta && <InspectorPanel meta={entry.meta as Inspector} />}
          </div>
        </div>
      )
    case 'memory':
      return (
        <div className="memchip" data-eid={entry.id} title={String(entry.meta?.kind ?? '')}>
          <span className="memchip-icon">◆</span>
          <span className="memchip-body">
            <span className="memchip-label">
              remembered{entry.meta?.branch ? ` · ${entry.meta.branch}` : ''}
            </span>
            {entry.text}
          </span>
        </div>
      )
    case 'evict':
      return (
        <div className="chip evict" data-eid={entry.id}>
          · {entry.text} to long-term memory
        </div>
      )
    default:
      return (
        <div className="chip" data-eid={entry.id}>
          {entry.text}
        </div>
      )
  }
}

function dayLabel(ts: number): string {
  const d = new Date(ts)
  const today = new Date()
  const yesterday = new Date(today.getTime() - 86400000)
  const same = (a: Date, b: Date) =>
    a.getFullYear() === b.getFullYear() && a.getMonth() === b.getMonth() && a.getDate() === b.getDate()
  if (same(d, today)) return 'Today'
  if (same(d, yesterday)) return 'Yesterday'
  const opts: Intl.DateTimeFormatOptions =
    d.getFullYear() === today.getFullYear()
      ? { weekday: 'long', month: 'long', day: 'numeric' }
      : { year: 'numeric', month: 'long', day: 'numeric' }
  return d.toLocaleDateString(undefined, opts)
}
