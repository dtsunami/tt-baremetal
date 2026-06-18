import { writable } from 'svelte/store'
import { getJSON } from './api.js'

// Static chip model (built once on the backend) — tiles + card-image registration.
export const floorplan = writable(null)
getJSON('/api/floorplan').then((fp) => floorplan.set(fp)).catch((e) => console.error(e))

// Latest live telemetry frame. Polled over plain HTTP at ~2 Hz (the backend poll loop samples
// the device at `hz` and caches the frame, so this just reads the cache). Polling instead of a
// WebSocket: no persistent socket, no reconnect/half-open states, no "reconnecting…" — a missed
// poll just shows disconnected for one tick and self-heals on the next.
export const frame = writable(null)
export const connected = writable(false)

const POLL_MS = 500   // ~2 Hz, matches the server sample rate
let inflight = false

async function poll() {
  if (inflight) return                       // never stack requests if one is slow
  inflight = true
  try {
    const r = await fetch('/api/telemetry')
    if (!r.ok) throw new Error(r.status)
    frame.set(await r.json())
    connected.set(true)
  } catch {
    connected.set(false)
  } finally {
    inflight = false
  }
}
poll()
setInterval(poll, POLL_MS)
