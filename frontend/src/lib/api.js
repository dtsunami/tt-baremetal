export async function getJSON(path) {
  const r = await fetch(path)
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

export async function postJSON(path, body) {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  const data = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(data.detail || `${path}: ${r.status}`)
  return data
}

export function fmtBW(b) {
  if (!b) return '0'
  if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB/s'
  if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB/s'
  if (b >= 1e3) return (b / 1e3).toFixed(1) + ' kB/s'
  return b.toFixed(0) + ' B/s'
}

export const tileKey = (noc0) => `${noc0[0]},${noc0[1]}`
