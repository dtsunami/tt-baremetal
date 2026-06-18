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

// ---- per-kernel meta-params: client mirror of web/kernmeta.coerce + route ----
function coerceOne(t, v) {
  if (t === 'int') return parseInt(v, 10)
  if (t === 'hex') return typeof v === 'number' ? v : parseInt(String(v))   // parseInt auto-detects 0x
  if (t === 'bool') return v === true || String(v).toLowerCase() === 'true' || v === 1 || v === '1'
  return v   // enum | str (kept as label/string)
}

export function coerceParam(p, v) {
  if (v === undefined || v === null) v = p.default
  if (p.multi) {
    const seq = Array.isArray(v) ? v : v === '' || v == null ? [] : [v]
    return seq.map((x) => coerceOne(p.type, x))
  }
  return coerceOne(p.type, v)
}

function arg0Of(p, v) {
  const c = coerceParam(p, v)
  if (p.type === 'enum') { const i = (p.choices || []).indexOf(c); return p.vals ? Number(p.vals[i]) : i }
  if (p.type === 'bool') return c ? 1 : 0
  return Number(c)
}

// Split {name: value} into the three application buckets (mirrors kernmeta.route):
//   {defines:{NAME:int}, deploy:{name:value}, mailbox:[{name,op,arg0}]}
export function routeParams(params, values) {
  const out = { defines: {}, deploy: {}, mailbox: [] }
  for (const p of params || []) {
    const raw = values && p.name in values ? values[p.name] : p.default
    if (p.kind === 'define') out.defines[p.name] = coerceParam({ ...p, type: 'hex' }, raw)
    else if (p.kind === 'deploy') out.deploy[p.name] = coerceParam(p, raw)
    else if (p.kind === 'mailbox') out.mailbox.push({ name: p.name, op: p.op, arg0: arg0Of(p, raw) })
    // rtarg / ctarg are tt-metal host-owned (documentation-only) — not routed (mirrors kernmeta.route)
  }
  return out
}

// Poll an async-job `*/last` endpoint ({running, result}) until it finishes, then call
// onDone(d). Replaces the per-lab setTimeout-recursion pollers. Returns a cancel fn.
export function pollJob(url, onDone, interval = 2000) {
  let cancelled = false
  const tick = async () => {
    if (cancelled) return
    let d
    try { d = await getJSON(url) } catch (e) { if (!cancelled) onDone({ error: String(e) }); return }
    if (cancelled) return
    if (d.running) setTimeout(tick, interval)
    else onDone(d)
  }
  setTimeout(tick, interval)
  return () => { cancelled = true }
}
