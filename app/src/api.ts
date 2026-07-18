// Thin client of the daemon. Same-origin in production (the daemon serves
// the build); the Vite dev server proxies /v1.

import type { TurnEvent } from './types'

export async function getJSON<T>(path: string): Promise<T> {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export async function sendJSON<T>(method: string, path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}))
    throw new Error((detail as { error?: string }).error ?? `${path}: ${r.status}`)
  }
  return r.json()
}

/** Stream one turn; calls onEvent for every SSE event until done. */
export async function streamTurn(
  text: string,
  images: string[],
  onEvent: (e: TurnEvent) => void,
): Promise<void> {
  const r = await fetch('/v1/turn', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, images }),
  })
  if (!r.ok || !r.body) throw new Error(`turn failed: ${r.status}`)
  const reader = r.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      for (const line of frame.split('\n')) {
        if (line.startsWith('data: ')) {
          try {
            onEvent(JSON.parse(line.slice(6)) as TurnEvent)
          } catch {
            // partial/garbled frame: skip
          }
        }
      }
    }
  }
}

export function cancelTurn(turnId: number): Promise<unknown> {
  return sendJSON('POST', '/v1/turn/cancel', { turn_id: turnId })
}
