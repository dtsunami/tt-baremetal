<script>
  import { floorplan, frame } from '../lib/stores.js'
  import { fmtBW, tileKey, getJSON, postJSON } from '../lib/api.js'
  import TilePane from '../lib/TilePane.svelte'
  import TileCartoon from '../lib/TileCartoon.svelte'

  // NoC rail colours are reactive — driven by the style config (cfg) below

  // tile-type glyphs (item 6) — mirrors floorplan.KINDS; DRAM shows its controller (D0…D7)
  const GLYPH = { tensix: 'T', dram: 'D', eth: 'E', arc: 'A', pcie: 'P', l2cpu: 'C', security: 'S', empty: '·' }
  const glyphOf = (t) => t.kind === 'dram' && t.dram_ctrl != null ? `D${t.dram_ctrl}` : (GLYPH[t.kind] || '?')
  const KIND_NAME = { tensix: 'Tensix', dram: 'DRAM', eth: 'Ethernet', l2cpu: 'L2CPU x280', arc: 'ARC', pcie: 'PCIe', security: 'Security', empty: 'router·NIU' }
  const KIND_ORDER = ['tensix', 'l2cpu', 'dram', 'eth', 'pcie', 'arc', 'security', 'empty']

  let svgEl
  let scale = 1, tx = 0, ty = 0
  let hovered = null, hx = 0, hy = 0
  let dragging = false, lastX = 0, lastY = 0, moved = false

  let align = false
  let box = null // [x0,y0,x1,y1] live calibration of the package footprint
  let selected = null            // tile shown in the right-hand pane (item 3)
  let dramEdit = false           // drag GDDR6 chip badges to place them on the photo (item 7)
  let dramPos = null             // [[x,y],…] image-coord centres, one per controller
  let chipDrag = null
  let hidden = new Set(JSON.parse(localStorage.getItem('bhtop_hidden') || '[]'))  // tile kinds to hide
  let dramSize = JSON.parse(localStorage.getItem('bhtop_dram_size3') || 'null') || [150, 54]  // badge [w,h] — wide + short
  $: if (dramSize) localStorage.setItem('bhtop_dram_size3', JSON.stringify(dramSize))
  // ---- per-element style: colours, opacity, per-NoC wrap returns (H+V), NIU markers — each
  //      tuned via a ⚙ on its legend chip; persisted per-browser + savable to the repo (git) ----
  const hexToRgb = (h) => { const n = parseInt(h.slice(1), 16); return [(n >> 16) & 255, (n >> 8) & 255, n & 255] }
  const dmerge = (b, o) => { const r = Array.isArray(b) ? [...b] : { ...b }; for (const k in (o || {})) r[k] = (o[k] && typeof o[k] === 'object' && !Array.isArray(o[k])) ? dmerge(b[k] || {}, o[k]) : o[k]; return r }
  const RET = () => ({ s: 26, w: 14, a: 0 })                 // wrap-return hook: stretch / width / angle°
  const ELNAME = { noc0: 'NoC0', noc1: 'NoC1', niu0: 'NIU0', niu1: 'NIU1', tensix: 'Tensix', dram: 'DRAM', eth: 'Ethernet', l2cpu: 'L2CPU', pcie: 'PCIe', arc: 'ARC', security: 'Security' }
  const DEF_CFG = {
    noc0: { color: '#cf83ff', op: 1, show: true, h: RET(), v: RET() }, noc1: { color: '#36ecff', op: 1, show: false, h: RET(), v: RET() },
    niu0: { color: '#cf83ff', op: 0.85, show: false }, niu1: { color: '#36ecff', op: 0.85, show: false },
    tensix: { color: '#ff9f38', op: 1 }, eth: { color: '#38d2f0', op: 1 }, l2cpu: { color: '#c884ff', op: 1 },
    pcie: { color: '#6398ff', op: 1 }, arc: { color: '#ff6060', op: 1 }, security: { color: '#f6d344', op: 1 },
    dram: { op: 1, ch: ['#34e082', '#38dcf0', '#6398ff', '#b482ff', '#f078dc', '#ff6e6e', '#ffae3c', '#dede5c'] },
  }
  let cfg = dmerge(DEF_CFG, JSON.parse(localStorage.getItem('bhtop_cfg') || '{}'))
  $: localStorage.setItem('bhtop_cfg', JSON.stringify(cfg))
  let styleEl = null                                          // which element's ⚙ popover is open
  let saveMsg = ''
  $: NOC0 = cfg.noc0.color
  $: NOC1 = cfg.noc1.color

  // save the chip-view style/placement to a tracked repo file (git), or pull it back
  async function saveDefaults() {
    try { await postJSON('/api/uiconfig', { cfg, box, dramPos, dramSize }); saveMsg = 'saved → ui-defaults.json (git) ✓' }
    catch (e) { saveMsg = 'save failed: ' + e }
    setTimeout(() => (saveMsg = ''), 4000)
  }
  async function loadDefaults() {
    try {
      const d = await getJSON('/api/uiconfig')
      if (d.cfg) cfg = dmerge(DEF_CFG, d.cfg)
      if (d.box) box = d.box
      if (d.dramPos) dramPos = d.dramPos
      if (d.dramSize) dramSize = d.dramSize
      saveMsg = 'loaded server defaults'
    } catch (e) { saveMsg = 'load failed: ' + e }
    setTimeout(() => (saveMsg = ''), 4000)
  }

  $: fp = $floorplan
  $: img = fp?.image
  $: ftiles = $frame?.tiles ?? {}
  $: maxBW = Math.max(1, ...Object.values(ftiles).map((t) => (t.noc0 || 0) + (t.noc1 || 0)))

  $: if (fp && box === null) box = JSON.parse(localStorage.getItem('bhtop_cal') || 'null') || [...img.package]
  $: if (box) localStorage.setItem('bhtop_cal', JSON.stringify(box))

  $: lod = scale >= 4.5 ? 'high' : scale >= 2 ? 'mid' : 'low'
  $: sw = 1 / scale
  $: view = img ? { w: img.w, h: img.h } : { w: 1, h: 1 }

  // remap a server rect (computed for the default box) to the live-adjusted box — pure affine
  function remap(r) {
    const D = img.package, B = box
    const sx = (B[2] - B[0]) / (D[2] - D[0]), sy = (B[3] - B[1]) / (D[3] - D[1])
    return { x: B[0] + (r.x - D[0]) * sx, y: B[1] + (r.y - D[1]) * sy, w: r.w * sx, h: r.h * sy }
  }
  const pos = (t) => remap(t.rect)

  // accurate routing = noc0 grid adjacency with torus wrap (NOT physical neighbours)
  let adj = []
  $: if (fp) adj = buildAdj(fp)
  function buildAdj(fp) {
    const [c, r] = fp.noc0_dims
    const m = new Map(fp.tiles.map((t) => [t.noc0.join(','), t]))
    const out = []
    for (const t of fp.tiles) {
      const [x, y] = t.noc0
      for (const [dx, dy] of [[1, 0], [0, 1]]) {
        const nb = m.get(`${(x + dx) % c},${(y + dy) % r}`)
        if (nb) out.push({ a: t, b: nb, wrap: x + dx >= c || y + dy >= r })
      }
    }
    return out
  }
  // a torus WRAP renders as a compact 180° hook at the edge: it exits in the flow direction,
  // turns, and ends pointing the OPPOSITE way (so a full-width/height wrap doesn't balloon
  // off-screen the way a single connecting arc did). fx,fy = forward (flow) unit vector.
  function hook(px, py, fx, fy, reach, sep, angDeg) {
    const a = (angDeg || 0) * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a)
    const rfx = fx * ca - fy * sa, rfy = fx * sa + fy * ca, nx = -rfy, ny = rfx   // rotated forward + perp
    return `M${px},${py} C${px + rfx * reach},${py + rfy * reach} ${px + rfx * reach + nx * sep},${py + rfy * reach + ny * sep} ${px + nx * sep},${py + ny * sep}`
  }
  // rail geometry (recomputes on calibration change, not per frame)
  $: rails = box
    ? adj.map((l) => {
        const ca = pos(l.a), cb = pos(l.b)
        const ax = ca.x + ca.w / 2, ay = ca.y + ca.h / 2, bx = cb.x + cb.w / 2, by = cb.y + cb.h / 2
        if (l.wrap) {
          const horiz = l.a.noc0[1] === l.b.noc0[1]        // horizontal wrap → east; vertical → south
          const fx = horiz ? 1 : 0, fy = horiz ? 0 : 1
          const r0 = horiz ? cfg.noc0.h : cfg.noc0.v, r1 = horiz ? cfg.noc1.h : cfg.noc1.v
          return { l, d0: hook(ax, ay, fx, fy, r0.s, r0.w, r0.a), d1: hook(bx, by, -fx, -fy, r1.s, r1.w, r1.a) }
        }
        const ux = bx - ax, uy = by - ay, L = Math.hypot(ux, uy) || 1
        const px = -uy / L, py = ux / L, o = Math.min(ca.w, ca.h) * 0.18
        return {
          l,
          d0: `M${ax + px * o},${ay + py * o} L${bx + px * o},${by + py * o}`, // NoC0 a→b
          d1: `M${bx - px * o},${by - py * o} L${ax - px * o},${ay - py * o}`, // NoC1 b→a
        }
      })
    : []

  function bw(t, n) { const f = ftiles[tileKey(t.noc0)]; return f ? (n === 0 ? f.noc0 : f.noc1) || 0 : 0 }
  function linkAct(r, n) { return Math.max(bw(r.l.a, n), bw(r.l.b, n)) / maxBW }
  function safe(t) { return fp.safe_kinds.includes(t.kind) }
  function selBW(t) { const s0 = cfg.noc0.show, s1 = cfg.noc1.show; return s0 && !s1 ? bw(t, 0) : s1 && !s0 ? bw(t, 1) : bw(t, 0) + bw(t, 1) }
  // NIU "stop" number per tile, in noc0 raster order (N1, N2, …)
  $: niuIdx = (() => {
    const m = {}
    if (fp) fp.tiles.filter((t) => t.kind !== 'empty').sort((a, b) => a.noc0[1] - b.noc0[1] || a.noc0[0] - b.noc0[0]).forEach((t, i) => m[tileKey(t.noc0)] = i + 1)
    return m
  })()
  // DRAM channel colour (per controller) + per-element opacity, both from cfg
  const dramColor = (ctrl, c) => c.dram.ch[ctrl % 8]
  const kindOp = (t, c) => (t.kind === 'dram' ? c.dram.op : (c[t.kind]?.op ?? 1))
  const kindSwatch = (k, c) => k === 'dram' ? c.dram.ch[0] : (c[k]?.color || `rgb(${(fp.kind_rgb[k] || [120, 120, 140]).join(',')})`)
  // tile label colour = its configured kind colour (DRAM by channel), going white-hot when busy
  function labelFill(t, c) {
    const hex = t.kind === 'dram' && t.dram_ctrl != null ? c.dram.ch[t.dram_ctrl % 8] : (c[t.kind]?.color || '#e2e6ee')
    const rgb = hexToRgb(hex)
    if (!safe(t)) return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`
    const a = Math.min(1, (selBW(t) / maxBW) * 1.5)
    const m = (i) => Math.min(255, rgb[i] + (255 - rgb[i]) * a) | 0
    return `rgb(${m(0)},${m(1)},${m(2)})`
  }
  // labels are the resting identity; they fade out as you ZOOM in (reveal the photo + rails)
  $: labelOp = scale <= 1.6 ? 1 : Math.max(0, 1 - (scale - 1.6) / 3.2)
  // …and past ~6× the detailed per-tile cartoon fades IN (only the tiles currently in view)
  $: cartoonOp = scale < 6 ? 0 : Math.min(1, (scale - 6) / 3)
  $: visibleTiles = (cartoonOp > 0 && fp && box) ? fp.tiles.filter((t) => {
    if (hidden.has(t.kind) || t.kind === 'empty') return false
    const p = pos(t)
    return p.x + p.w > -tx / scale && p.x < (view.w - tx) / scale && p.y + p.h > -ty / scale && p.y < (view.h - ty) / scale
  }) : []
  // per-tile short id: tensix = 00,01,…; others = glyph + index (D0/E0/C0/A0/…)
  $: tileLabel = (() => {
    const m = {}, ctr = {}
    if (fp) for (const t of fp.tiles) {
      if (t.kind === 'empty') { m[tileKey(t.noc0)] = ''; continue }
      const i = (ctr[t.kind] = (ctr[t.kind] ?? -1) + 1)
      m[tileKey(t.noc0)] = t.kind === 'tensix' ? String(i).padStart(2, '0')
        : t.kind === 'dram' ? 'd' + t.dram_ctrl          // name by controller, matching the chip badges
        : (GLYPH[t.kind] || '?') + i
    }
    return m
  })()
  $: kinds = fp ? KIND_ORDER.filter((k) => fp.tiles.some((t) => t.kind === k)) : []
  $: kindCount = fp ? fp.tiles.reduce((m, t) => ((m[t.kind] = (m[t.kind] || 0) + 1), m), {}) : {}
  function toggleKind(k) { hidden.has(k) ? hidden.delete(k) : hidden.add(k); hidden = hidden; localStorage.setItem('bhtop_hidden', JSON.stringify([...hidden])) }

  // ---- zoom / pan ----
  function clamp() {
    tx = Math.min(0, Math.max(view.w * (1 - scale), tx))
    ty = Math.min(0, Math.max(view.h * (1 - scale), ty))
  }
  function onWheel(e) {
    e.preventDefault()
    const r = svgEl.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * view.w
    const py = ((e.clientY - r.top) / r.height) * view.h
    const ns = Math.min(16, Math.max(1, scale * (e.deltaY < 0 ? 1.18 : 1 / 1.18)))
    tx = px - (px - tx) * (ns / scale); ty = py - (py - ty) * (ns / scale); scale = ns; clamp()
  }
  function onDown(e) {
    if (dramEdit && e.target.closest?.('.dchip')) return   // let a chip badge own the drag
    dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY
  }
  function onMove(e) {
    const r = svgEl.getBoundingClientRect()
    hx = e.clientX - r.left; hy = e.clientY - r.top
    if (!dragging) return
    moved = true
    tx += ((e.clientX - lastX) / r.width) * view.w; ty += ((e.clientY - lastY) / r.height) * view.h
    lastX = e.clientX; lastY = e.clientY; clamp()
  }
  function onUp() { dragging = false }
  function reset() { scale = 1; tx = 0; ty = 0 }
  function open(t) { if (!moved) selected = t }   // item 3: right-hand tile pane

  // ---- calibration ----
  function onKey(e) {
    if (!align || !box) return
    const s = e.shiftKey ? 10 : 2, b = [...box]
    if (e.key === 'ArrowLeft') { b[0] -= s; b[2] -= s }
    else if (e.key === 'ArrowRight') { b[0] += s; b[2] += s }
    else if (e.key === 'ArrowUp') { b[1] -= s; b[3] -= s }
    else if (e.key === 'ArrowDown') { b[1] += s; b[3] += s }
    else if (e.key === '+' || e.key === '=') { b[0] -= s; b[1] -= s; b[2] += s; b[3] += s }
    else if (e.key === '-') { b[0] += s; b[1] += s; b[2] -= s; b[3] -= s }
    else return
    e.preventDefault(); box = b
  }
  function resetCal() { box = [...img.package] }

  $: hoverBW = hovered ? (() => { const f = ftiles[tileKey(hovered.noc0)]; return f ? (f.noc0 || 0) + (f.noc1 || 0) : 0 })() : 0

  // ---- DRAM dashboard + PCIe ----
  $: dramInfo = fp?.dram          // {ctrls, per_ctrl_gib, total_gib}
  $: dramBW = $frame?.dram ?? {}  // {ctrl: {r, w}} bytes/s
  $: dramMax = Math.max(2e6, ...Object.values(dramBW).flatMap((d) => [d.r || 0, d.w || 0]))

  // ---- DRAM chip overlay on the board photo (item 7: guessed positions, drag to adjust) ----
  function defaultDramPos(b, n) {
    const w = b[2] - b[0], h = b[3] - b[1], mx = w * 0.15
    const fr = [0.13, 0.37, 0.63, 0.87], half = Math.ceil(n / 2), out = []
    for (let i = 0; i < n; i++) {
      const right = i >= half
      out.push([right ? b[2] + mx : b[0] - mx, b[1] + h * fr[(right ? i - half : i) % fr.length]])
    }
    return out   // guess: half the controllers flank the package left, half right
  }
  $: if (dramInfo && box && dramPos === null) {
    const saved = JSON.parse(localStorage.getItem('bhtop_dram_pos') || 'null')
    dramPos = (saved && saved.length === dramInfo.ctrls.length) ? saved : defaultDramPos(box, dramInfo.ctrls.length)
  }
  function startChip(e, i) {
    if (!dramEdit) return
    e.preventDefault(); e.stopPropagation()
    const r = svgEl.getBoundingClientRect()
    chipDrag = { i, x0: e.clientX, y0: e.clientY, px: dramPos[i][0], py: dramPos[i][1], rw: r.width, rh: r.height }
    window.addEventListener('mousemove', moveChip); window.addEventListener('mouseup', endChip)
  }
  function moveChip(e) {
    if (!chipDrag) return
    const dx = ((e.clientX - chipDrag.x0) / chipDrag.rw) * view.w / scale
    const dy = ((e.clientY - chipDrag.y0) / chipDrag.rh) * view.h / scale
    dramPos[chipDrag.i] = [chipDrag.px + dx, chipDrag.py + dy]; dramPos = dramPos
  }
  function endChip() {
    chipDrag = null
    window.removeEventListener('mousemove', moveChip); window.removeEventListener('mouseup', endChip)
    localStorage.setItem('bhtop_dram_pos', JSON.stringify(dramPos))
  }
  function resetDram() { dramPos = defaultDramPos(box, dramInfo.ctrls.length); localStorage.setItem('bhtop_dram_pos', JSON.stringify(dramPos)) }

  // centre die node of each DRAM channel (centroid of its tiles) — for the channel→chip connector
  $: dramCenters = (() => {
    if (!fp || !box) return {}
    const g = {}, ctr = {}
    for (const t of fp.tiles) if (t.kind === 'dram' && t.dram_ctrl != null) (g[t.dram_ctrl] = g[t.dram_ctrl] || []).push(pos(t))
    for (const k in g) { const a = g[k]; ctr[k] = [a.reduce((s, p) => s + p.x + p.w / 2, 0) / a.length, a.reduce((s, p) => s + p.y + p.h / 2, 0) / a.length] }
    return ctr
  })()

  // ---- click-to-trace: highlight the selected tile's NoC neighbour links + its noc0 route to DRAM ----
  const sameTile = (a, b) => a.noc0[0] === b.noc0[0] && a.noc0[1] === b.noc0[1]
  $: railByKey = (() => { const m = {}; for (const r of rails) m[`${r.l.a.noc0[0]},${r.l.a.noc0[1]}>${r.l.b.noc0[0]},${r.l.b.noc0[1]}`] = r; return m })()
  function nearestDram(t) {
    const [c, r] = fp.noc0_dims
    let best = null, bd = 1e9
    for (const d of fp.tiles) {
      if (d.kind !== 'dram') continue
      const dist = ((d.noc0[0] - t.noc0[0] + c) % c) + ((d.noc0[1] - t.noc0[1] + r) % r)   // noc0 route len (E+S)
      if (dist < bd) { bd = dist; best = d }
    }
    return best
  }
  function routeLinks(src, dst) {
    const [c, r] = fp.noc0_dims, out = []
    let x = src.noc0[0], y = src.noc0[1], g = 0
    while (x !== dst.noc0[0] && g++ < c) { const k = railByKey[`${x},${y}>${(x + 1) % c},${y}`]; if (k) out.push(k); x = (x + 1) % c }
    g = 0
    while (y !== dst.noc0[1] && g++ < r) { const k = railByKey[`${x},${y}>${x},${(y + 1) % r}`]; if (k) out.push(k); y = (y + 1) % r }
    return out
  }
  // highlight path: straight for interior links; a 180° hairpin for torus WRAP links so the
  // flow leaves + re-enters going the SAME direction (reads as a wraparound, not a back-track)
  function hiPath(r, cfg) {
    const ca = pos(r.l.a), cb = pos(r.l.b)
    const ax = ca.x + ca.w / 2, ay = ca.y + ca.h / 2, bx = cb.x + cb.w / 2, by = cb.y + cb.h / 2
    if (!r.l.wrap) return `M${ax},${ay} L${bx},${by}`
    const horiz = r.l.a.noc0[1] === r.l.b.noc0[1], fx = horiz ? 1 : 0, fy = horiz ? 0 : 1
    const ret = horiz ? cfg.noc0.h : cfg.noc0.v              // highlight traces noc0
    return hook(ax, ay, fx, fy, ret.s, ret.w, ret.a)
  }
  $: selLinks = (selected && rails.length) ? rails.filter((r) => sameTile(r.l.a, selected) || sameTile(r.l.b, selected)) : []
  $: dramTarget = (selected && fp && selected.kind !== 'dram') ? nearestDram(selected) : null
  $: dramRoute = (selected && dramTarget && railByKey) ? routeLinks(selected, dramTarget) : []
</script>

<svelte:window on:keydown={onKey} />

<div class="wrap">
  {#if fp && box}
    <svg
      bind:this={svgEl} role="application" aria-label="Blackhole chip"
      viewBox="0 0 {view.w} {view.h}" class:grabbing={dragging}
      on:wheel={onWheel} on:mousedown={onDown} on:mousemove={onMove} on:mouseup={onUp} on:mouseleave={onUp}
    >
      <defs>
        <marker id="a0" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3 L0,6 z" fill={NOC0} /></marker>
        <marker id="a1" markerWidth="7" markerHeight="7" refX="5" refY="3" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L7,3 L0,6 z" fill={NOC1} /></marker>
        <filter id="glow" x="-60%" y="-60%" width="220%" height="220%">
          <feGaussianBlur stdDeviation={0.9 / scale} result="b" />
          <feMerge><feMergeNode in="b" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
        <!-- bold direction arrows for the click-to-trace highlight -->
        <marker id="hia" markerWidth="11" markerHeight="11" refX="7" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5.5 L0,11 z" fill="#ffd24a" stroke="#000" stroke-width="0.7" /></marker>
        <marker id="hiw" markerWidth="11" markerHeight="11" refX="7" refY="5.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0 L11,5.5 L0,11 z" fill="#fff" stroke="#000" stroke-width="0.7" /></marker>
      </defs>

      <g transform="translate({tx} {ty}) scale({scale})">
        <image href={img.src} x="0" y="0" width={img.w} height={img.h} />
        <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]}
          fill="none" stroke={align ? '#ffcc44' : '#ffffff22'} stroke-width={sw * (align ? 1.6 : 1)} stroke-dasharray={align ? sw * 4 : 0} />

        <!-- accurate routing rails: NoC0 purple (a→b, E/S), NoC1 cyan (b→a, W/N).
             item 5: high-contrast — dark halo + bright core, strong base opacity so the network
             always reads over the board photo; activity adds width + opacity on top -->
        {#if cfg.noc0.show || cfg.noc1.show}
          <g opacity={selected ? 0.18 : 1}>
          {#each rails as r}
            {#if cfg.noc0.show}
              {@const a = linkAct(r, 0)}
              <path d={r.d0} fill="none" stroke="#000" stroke-width={sw * (6 + 4 * a)} stroke-opacity="0.85" stroke-linecap="round" stroke-dasharray={r.l.wrap ? sw * 5 : 0} />
              <path d={r.d0} fill="none" stroke={NOC0} stroke-width={sw * (3.2 + 4 * a)} stroke-linecap="round" marker-end="url(#a0)" stroke-dasharray={r.l.wrap ? sw * 5 : 0} opacity={((r.l.wrap ? 0.8 : 0.74) + 0.26 * a) * cfg.noc0.op} filter={a > 0.04 ? 'url(#glow)' : null} />
            {/if}
            {#if cfg.noc1.show}
              {@const a = linkAct(r, 1)}
              <path d={r.d1} fill="none" stroke="#000" stroke-width={sw * (6 + 4 * a)} stroke-opacity="0.85" stroke-linecap="round" stroke-dasharray={r.l.wrap ? sw * 5 : 0} />
              <path d={r.d1} fill="none" stroke={NOC1} stroke-width={sw * (3.2 + 4 * a)} stroke-linecap="round" marker-end="url(#a1)" stroke-dasharray={r.l.wrap ? sw * 5 : 0} opacity={((r.l.wrap ? 0.8 : 0.74) + 0.26 * a) * cfg.noc1.op} filter={a > 0.04 ? 'url(#glow)' : null} />
            {/if}
          {/each}
          </g>
        {/if}

        <!-- click-to-trace: the selected tile's NoC neighbour links + its noc0 route to DRAM.
             Shows even when the overlay is off; hidden once you zoom into the cartoons. -->
        {#if selected && cartoonOp < 0.8}
          <!-- neighbour links: black halo + bright white core + noc0 direction arrow -->
          {#each selLinks as r}
            <path d={hiPath(r, cfg)} fill="none" stroke="#000" stroke-width={sw * 7} stroke-opacity="0.92" stroke-linecap="round" />
            <path d={hiPath(r, cfg)} fill="none" stroke="#ffffff" stroke-width={sw * 3.6} stroke-linecap="round" marker-end="url(#hiw)" />
          {/each}
          <!-- route to DRAM: black halo + bright amber + glow + a bold arrow INTO the target -->
          {#each dramRoute as r, i}
            <path d={hiPath(r, cfg)} fill="none" stroke="#000" stroke-width={sw * 8} stroke-opacity="0.92" stroke-linecap="round" />
            <path d={hiPath(r, cfg)} fill="none" stroke="#ffd24a" stroke-width={sw * 4.2} stroke-linecap="round" filter="url(#glow)" marker-end={i === dramRoute.length - 1 ? 'url(#hia)' : null} />
          {/each}
          <rect x={pos(selected).x} y={pos(selected).y} width={pos(selected).w} height={pos(selected).h} rx={Math.min(pos(selected).w, pos(selected).h) * 0.16} fill="none" stroke="#ffffff" stroke-width={sw * 3} />
          {#if dramTarget}
            <rect x={pos(dramTarget).x} y={pos(dramTarget).y} width={pos(dramTarget).w} height={pos(dramTarget).h} rx={Math.min(pos(dramTarget).w, pos(dramTarget).h) * 0.16} fill="none" stroke="#ffd24a" stroke-width={sw * 3} filter="url(#glow)" />
          {/if}
        {/if}

        <!-- tiles as fading labels — D#/E#/C#/… and tensix 00,01… — over a transparent hit-area
             for hover + click. Labels are the resting view and fade out as you ZOOM in. -->
        {#each fp.tiles as t (tileKey(t.noc0))}
          {#if !hidden.has(t.kind)}
            {@const p = pos(t)}
            <rect x={p.x} y={p.y} width={p.w} height={p.h} fill="transparent" class="tile" class:empty={t.kind === 'empty'}
              on:mouseenter={() => (hovered = t)} on:mouseleave={() => (hovered = null)} on:click={() => t.kind !== 'empty' && open(t)} />
            {#if t.kind !== 'empty' && labelOp > 0.02}
              <text x={p.x + p.w / 2} y={p.y + p.h / 2} text-anchor="middle" dominant-baseline="central"
                font-size={Math.min(p.h * 0.9, p.w * 1.27 / Math.max(2, tileLabel[tileKey(t.noc0)].length))}
                font-weight="700" class="lbl glyph" fill={labelFill(t, cfg)} opacity={labelOp * kindOp(t, cfg)}>{tileLabel[tileKey(t.noc0)]}</text>
            {/if}
          {/if}
        {/each}

        <!-- zoom past ~6× and the labels give way to each in-view tile's detailed cartoon -->
        {#if cartoonOp > 0}
          {#each visibleTiles as t (tileKey(t.noc0) + 'c')}
            {@const p = pos(t)}
            <svg x={p.x} y={p.y} width={p.w} height={p.h} viewBox="0 0 220 162" preserveAspectRatio="xMidYMid meet" opacity={cartoonOp} class="cartoon">
              <TileCartoon tile={t} bw0={bw(t, 0)} bw1={bw(t, 1)} dram={t.dram_ctrl != null ? (dramBW[String(t.dram_ctrl)] ?? { r: 0, w: 0 }) : { r: 0, w: 0 }} />
            </svg>
          {/each}
        {/if}

        <!-- connector: each channel's centre die node → its GDDR6 chip badge (per-channel colour) -->
        {#if dramInfo && dramPos}
          {#each dramInfo.ctrls as c, i}
            {#if dramCenters[c]}
              <line x1={dramCenters[c][0]} y1={dramCenters[c][1]} x2={dramPos[i][0]} y2={dramPos[i][1]}
                stroke={dramColor(c, cfg)} stroke-width={sw * 1.6} stroke-opacity={0.5 * cfg.dram.op * (dramEdit ? 1 : labelOp)} stroke-dasharray={sw * 4} class="lbl" />
            {/if}
          {/each}
        {/if}

        <!-- NIU stops, numbered N1,N2,… in noc0 order (NoC0 NIU top-left, NoC1 NIU bottom-right) -->
        {#if cfg.niu0.show || cfg.niu1.show}
          {#each fp.tiles as t (tileKey(t.noc0) + 'n')}
            {#if !hidden.has(t.kind) && t.kind !== 'empty'}
              {@const p = pos(t)}
              {#if cfg.niu0.show}<text x={p.x + p.w * 0.26} y={p.y + p.h * 0.27} font-size={Math.min(p.w, p.h) * 0.26} fill={cfg.niu0.color} opacity={cfg.niu0.op} font-weight="700" text-anchor="middle" dominant-baseline="central" class="lbl glyph">N{niuIdx[tileKey(t.noc0)]}</text>{/if}
              {#if cfg.niu1.show}<text x={p.x + p.w * 0.74} y={p.y + p.h * 0.73} font-size={Math.min(p.w, p.h) * 0.26} fill={cfg.niu1.color} opacity={cfg.niu1.op} font-weight="700" text-anchor="middle" dominant-baseline="central" class="lbl glyph">N{niuIdx[tileKey(t.noc0)]}</text>{/if}
            {/if}
          {/each}
        {/if}

        <!-- GDDR6 chip badges on the board photo — per-channel colour + thin R/W; toggle 'dram' to drag/size -->
        {#if dramInfo && dramPos}
          {#each dramInfo.ctrls as c, i}
            {@const pp = dramPos[i]}
            {@const d = dramBW[String(c)] ?? { r: 0, w: 0 }}
            {@const act = (d.r + d.w) / dramMax}
            {@const w = dramSize[0]}
            {@const h = dramSize[1]}
            <g class="dchip" class:edit={dramEdit} transform="translate({pp[0]} {pp[1]})" opacity={dramEdit ? 1 : labelOp * cfg.dram.op} on:mousedown={(e) => startChip(e, i)}>
              <rect x={-w / 2} y={-h / 2} width={w} height={h} rx="9" fill="#0b0d12cc"
                stroke={act > 0.05 ? 'var(--accent)' : (dramEdit ? '#ffcc44' : dramColor(c, cfg))} stroke-opacity={act > 0.05 || dramEdit ? 1 : 0.7} stroke-width={sw * (dramEdit ? 2.2 : 1.6)} />
              <text x="0" y={-h * 0.04} text-anchor="middle" font-size={h * 0.4} fill={dramColor(c, cfg)} font-weight="700" class="lbl">d{c}</text>
              <rect x={-w * 0.4} y={h * 0.2} width={w * 0.8} height={h * 0.06} rx="2" fill="#1b2233" class="lbl" />
              <rect x={-w * 0.4} y={h * 0.2} width={w * 0.4 * Math.min(1, d.r / dramMax)} height={h * 0.06} rx="2" fill="var(--good)" class="lbl" />
              <rect x="0" y={h * 0.2} width={w * 0.4 * Math.min(1, d.w / dramMax)} height={h * 0.06} rx="2" fill="var(--accent)" class="lbl" />
            </g>
          {/each}
        {/if}
      </g>
    </svg>

    {#if hovered}
      <div class="tip" style="left:{hx + 14}px; top:{hy + 14}px">
        <b>{glyphOf(hovered)}</b> {hovered.label} · {hovered.kind}<br />
        noc0 {hovered.noc0[0]},{hovered.noc0[1]} · die {hovered.die[0]},{hovered.die[1]}
        {#if hovered.dram_ctrl !== null}<br />GDDR6 d{hovered.dram_ctrl}{/if}<br />
        {#if safe(hovered)}<span style="color:{NOC0}">NoC0 {fmtBW(bw(hovered, 0))}</span> · <span style="color:{NOC1}">NoC1 {fmtBW(bw(hovered, 1))}</span>{:else if hovered.kind === 'empty'}<span class="muted">empty tile — router + NIU only (torus passthrough)</span>{:else}<span class="muted">not polled (mgmt)</span>{/if}
      </div>
    {/if}

    {#if selected}<TilePane tile={selected} on:close={() => (selected = null)} />{/if}

    <div class="hud">
      <div class="row">
        <button on:click={reset} disabled={scale === 1}>reset zoom</button>
        <span>{scale.toFixed(1)}×</span>
        <button class:on={align} on:click={() => (align = !align)}>align</button>
        <button class:on={dramEdit} on:click={() => (dramEdit = !dramEdit)} title="drag GDDR6 badges onto their chips">dram</button>
        <button on:click={saveDefaults} title="write the current style + placement to ui-defaults.json (git)">save ⤓</button>
        <button on:click={loadDefaults} title="pull the committed defaults from the server">load</button>
      </div>
      {#if saveMsg}<div class="savemsg">{saveMsg}</div>{/if}
      {#if dramEdit}
        <div class="cal">
          <b>place GDDR6 chips</b> — drag each <b>d#</b> badge onto its chip
          <div class="nums">
            <label>width<input type="number" bind:value={dramSize[0]} /></label>
            <label>height<input type="number" bind:value={dramSize[1]} /></label>
          </div>
          <button on:click={resetDram}>reset placement</button>
          <span class="muted">drag to place · width/height set the badge aspect · saved locally</span>
        </div>
      {/if}
      {#if styleEl}
        {@const e = styleEl}
        <div class="cal style">
          <b>{ELNAME[e] || e}</b> <span class="muted">colour / opacity{e === 'noc0' || e === 'noc1' ? ' / returns' : ''}</span>
          {#if e === 'dram'}
            <div class="srow"><span class="sn">channels</span><span class="dchs">{#each cfg.dram.ch as _, i}<input type="color" bind:value={cfg.dram.ch[i]} title="d{i}" />{/each}</span></div>
            <div class="srow"><span class="sn">opacity</span><input type="range" min="0" max="1" step="0.05" bind:value={cfg.dram.op} /></div>
          {:else}
            <div class="srow"><span class="sn">colour</span><input type="color" bind:value={cfg[e].color} /><input type="range" min="0" max="1" step="0.05" bind:value={cfg[e].op} title="opacity" /></div>
          {/if}
          {#if e === 'noc0' || e === 'noc1'}
            <div class="srow"><span class="sn">H return</span><input class="m" type="number" bind:value={cfg[e].h.s} title="stretch" /><input class="m" type="number" bind:value={cfg[e].h.w} title="width" /><input class="m" type="number" bind:value={cfg[e].h.a} title="angle°" /></div>
            <div class="srow"><span class="sn">V return</span><input class="m" type="number" bind:value={cfg[e].v.s} title="stretch" /><input class="m" type="number" bind:value={cfg[e].v.w} title="width" /><input class="m" type="number" bind:value={cfg[e].v.a} title="angle°" /></div>
          {/if}
          {#if e === 'niu0' || e === 'niu1'}
            <label class="chk"><input type="checkbox" bind:checked={cfg[e].show} /> show NIU markers on every tile</label>
          {/if}
          <div class="srow"><button on:click={() => { cfg[e] = JSON.parse(JSON.stringify(DEF_CFG[e])); cfg = cfg }}>reset</button><button on:click={() => (styleEl = null)}>done</button></div>
        </div>
      {/if}
      <!-- tile-type chips: toggle show/hide, ⚙ to style that type -->
      <div class="tlegend">
        {#each kinds as k}
          <span class="tk" class:off={hidden.has(k)}>
            <button class="tkb" on:click={() => toggleKind(k)} title="show/hide {KIND_NAME[k]} ({kindCount[k]})">
              <span class="sw" style="background:{kindSwatch(k, cfg)}"></span><b>{GLYPH[k] ?? '?'}</b>{KIND_NAME[k]}<span class="muted">{kindCount[k]}</span>
            </button>
            {#if cfg[k]}<button class="gear" class:on={styleEl === k} on:click={() => (styleEl = styleEl === k ? null : k)} title="style {KIND_NAME[k]}">⚙</button>{/if}
          </span>
        {/each}
      </div>
      <!-- NoC + NIU elements: chip toggles visibility, ⚙ styles (NoC colour/opacity/returns) -->
      <div class="tlegend">
        {#each ['noc0', 'noc1', 'niu0', 'niu1'] as k}
          <span class="tk" class:off={!cfg[k].show}>
            <button class="tkb" on:click={() => { cfg[k].show = !cfg[k].show; cfg = cfg }} title="show/hide {ELNAME[k]}">
              <span class="sw" style="background:{cfg[k].color}"></span>{ELNAME[k]}
            </button>
            <button class="gear" class:on={styleEl === k} on:click={() => (styleEl = styleEl === k ? null : k)} title="style {ELNAME[k]}">⚙</button>
          </span>
        {/each}
      </div>
      {#if align}
        <div class="cal">
          <b>align overlay</b> — arrows move · shift=fast · <kbd>+</kbd>/<kbd>-</kbd> size
          <div class="nums">
            {#each [0, 1, 2, 3] as i}
              <label>{['x0', 'y0', 'x1', 'y1'][i]}<input type="number" bind:value={box[i]} /></label>
            {/each}
          </div>
          <code>CARD_PACKAGE_PX = ({box.map((n) => Math.round(n)).join(', ')})</code>
          <button on:click={resetCal}>reset to default</button>
          <span class="muted">paste the tuple into geometry.py to make it the server default</span>
        </div>
      {/if}
    </div>
  {:else}
    <div class="loading">connecting to ttstar…</div>
  {/if}
</div>

<style>
  .wrap { position: relative; height: calc(100vh - 47px); overflow: hidden; background: #000; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; }
  svg.grabbing { cursor: grabbing; }
  .tile { cursor: pointer; }
  .tile.empty { cursor: default; }
  .tile:hover { fill: rgba(255, 255, 255, 0.08); }
  .lbl { pointer-events: none; font-family: ui-monospace, monospace; }
  .cartoon { pointer-events: none; }
  .glyph { paint-order: stroke; stroke: #05060a; stroke-width: 0.9px; }
  .dchip { pointer-events: none; }
  .dchip.edit { pointer-events: auto; cursor: grab; }
  .dchip.edit:active { cursor: grabbing; }

  .tip { position: absolute; pointer-events: none; z-index: 5; background: #0d0f14f2; border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; font-size: 12px; line-height: 1.5; max-width: 240px; }
  .tip b { color: var(--accent); }
  .muted { color: var(--muted); }
  kbd { background: var(--panel2); border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; }


  .hud { position: absolute; left: 12px; bottom: 92px; z-index: 4; background: #0d0f14e6; border: 1px solid var(--line); border-radius: 8px; padding: 9px 12px; display: flex; flex-direction: column; gap: 8px; font-size: 12px; max-width: 380px; }
  .row { display: flex; align-items: center; gap: 10px; }
  .hud button { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 9px; cursor: pointer; font: inherit; }
  .hud button.on { background: var(--accent); color: #1a1206; border-color: var(--accent); }
  .hud button:disabled { opacity: 0.4; cursor: default; }
  .tlegend { display: flex; flex-wrap: wrap; gap: 5px; border-top: 1px solid var(--line); padding-top: 7px; }
  .tk { display: flex; align-items: center; background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; }
  .tk.off { opacity: 0.4; }
  .tkb { display: flex; align-items: center; gap: 4px; font: inherit; font-size: 11px; background: none; border: none; color: var(--fg); padding: 2px 4px 2px 7px; cursor: pointer; }
  .tk .sw { width: 9px; height: 9px; border-radius: 2px; flex: none; }
  .tkb b { font-family: ui-monospace, monospace; }
  .tkb .muted { color: var(--muted); }
  .gear { background: none; border: none; border-left: 1px solid var(--line); color: var(--muted); cursor: pointer; font-size: 11px; padding: 2px 6px; }
  .gear:hover, .gear.on { color: var(--accent); }
  .savemsg { color: var(--good); font-size: 11px; }
  .style .srow .m { width: 42px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; font: inherit; }
  .chk { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 5px; }
  .cal { display: flex; flex-direction: column; gap: 6px; border-top: 1px solid var(--line); padding-top: 7px; }
  .style { max-height: 300px; overflow-y: auto; }
  .srow { display: flex; align-items: center; gap: 7px; }
  .srow .sn { width: 78px; color: var(--muted); font-size: 11px; }
  .srow input[type=color] { width: 22px; height: 18px; padding: 0; border: 1px solid var(--line); border-radius: 3px; background: none; cursor: pointer; }
  .srow input[type=range] { flex: 1; min-width: 60px; }
  .dchs { display: flex; gap: 2px; flex: 1; }
  .dchs input[type=color] { width: 15px; height: 16px; padding: 0; border: 1px solid var(--line); border-radius: 2px; cursor: pointer; }
  .cal code { color: var(--accent); user-select: all; }
  .nums { display: flex; gap: 8px; }
  .nums label { display: flex; flex-direction: column; color: var(--muted); font-size: 11px; gap: 2px; }
  .nums input { width: 56px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; font: inherit; }
  .loading { display: grid; place-items: center; height: 100%; color: var(--muted); }
</style>
