import { writable } from 'svelte/store'
import { getJSON } from './api.js'

// Static chip model (built once on the backend) — tiles + card-image registration.
export const floorplan = writable(null)
getJSON('/api/floorplan').then((fp) => floorplan.set(fp)).catch((e) => console.error(e))

// Latest live telemetry frame, pushed over WebSocket at the poll rate (~2 Hz).
export const frame = writable(null)
export const connected = writable(false)

function connectWS() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws'
  const ws = new WebSocket(`${proto}://${location.host}/ws/telemetry`)
  ws.onopen = () => connected.set(true)
  ws.onmessage = (e) => frame.set(JSON.parse(e.data))
  ws.onclose = () => {
    connected.set(false)
    setTimeout(connectWS, 1500) // auto-reconnect
  }
  ws.onerror = () => ws.close()
}
connectWS()
