<script>
  import { push } from 'svelte-spa-router'
  import { floorplan, frame } from '../lib/stores.js'
  import { fmtBW, tileKey, getJSON, postJSON } from '../lib/api.js'

  const NOC0 = '#cf83ff' // purple — routes east + south (+ wrap)
  const NOC1 = '#36ecff' // cyan   — routes west + north (+ wrap)
  const CELL = 54, PAD = 32

  let svgEl
  let scale = 1, tx = 0, ty = 0
  let hovered = null, hx = 0, hy = 0
  let dragging = false, lastX = 0, lastY = 0, moved = false

  // 'card' = photo + accurate routing registered to the die; 'topo' = clean noc0 torus grid
  let layout = localStorage.getItem('bhtop_layout') || 'card'
  let noc = +(localStorage.getItem('bhtop_noc') ?? 0) // 0 = NoC0 only, 1 = NoC1 only, 2 = both
  let align = false
  let box = null // [x0,y0,x1,y1] live calibration of the package footprint (card layout)

  $: fp = $floorplan
  $: img = fp?.image
  $: cols = fp?.noc0_dims[0] ?? 17
  $: rows = fp?.noc0_dims[1] ?? 12
  $: ftiles = $frame?.tiles ?? {}
  $: maxBW = Math.max(1, ...Object.values(ftiles).map((t) => (t.noc0 || 0) + (t.noc1 || 0)))

  $: if (fp && box === null) box = JSON.parse(localStorage.getItem('bhtop_cal') || 'null') || [...img.package]
  $: if (box) localStorage.setItem('bhtop_cal', JSON.stringify(box))
  $: localStorage.setItem('bhtop_layout', layout)
  $: localStorage.setItem('bhtop_noc', noc)

  $: lod = scale >= 4.5 ? 'high' : scale >= 2 ? 'mid' : 'low'
  $: sw = 1 / scale
  $: view = layout === 'topo' || !img ? { w: PAD * 2 + cols * CELL, h: PAD * 2 + rows * CELL } : { w: img.w, h: img.h }

  // remap a server rect (computed for the default box) to the live-adjusted box — pure affine
  function remap(r) {
    const D = img.package, B = box
    const sx = (B[2] - B[0]) / (D[2] - D[0]), sy = (B[3] - B[1]) / (D[3] - D[1])
    return { x: B[0] + (r.x - D[0]) * sx, y: B[1] + (r.y - D[1]) * sy, w: r.w * sx, h: r.h * sy }
  }
  function pos(t) {
    if (layout === 'topo')
      return { x: PAD + t.noc0[0] * CELL + CELL * 0.12, y: PAD + t.noc0[1] * CELL + CELL * 0.12, w: CELL * 0.76, h: CELL * 0.76 }
    return remap(t.rect)
  }

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
  // straight for interior links; bowed arc for torus WRAP links (so wraparound reads)
  function railPath(xa, ya, xb, yb, wrap) {
    if (!wrap) return `M${xa},${ya} L${xb},${yb}`
    const mx = (xa + xb) / 2, my = (ya + yb) / 2
    const ux = xb - xa, uy = yb - ya, L = Math.hypot(ux, uy) || 1
    const px = -uy / L, py = ux / L, bow = Math.max(L * 0.33, Math.min(view.w, view.h) * 0.04)
    return `M${xa},${ya} Q${mx + px * bow},${my + py * bow} ${xb},${yb}`
  }
  // rail geometry (recomputes on layout/calibration change, not per frame)
  $: rails = box
    ? adj.map((l) => {
        const ca = pos(l.a), cb = pos(l.b)
        const ax = ca.x + ca.w / 2, ay = ca.y + ca.h / 2, bx = cb.x + cb.w / 2, by = cb.y + cb.h / 2
        const ux = bx - ax, uy = by - ay, L = Math.hypot(ux, uy) || 1
        const px = -uy / L, py = ux / L, o = Math.min(ca.w, ca.h) * 0.18
        return {
          l,
          d0: railPath(ax + px * o, ay + py * o, bx + px * o, by + py * o, l.wrap), // NoC0 a→b
          d1: railPath(bx - px * o, by - py * o, ax - px * o, ay - py * o, l.wrap), // NoC1 b→a
        }
      })
    : []

  function bw(t, noc) { const f = ftiles[tileKey(t.noc0)]; return f ? (noc === 0 ? f.noc0 : f.noc1) || 0 : 0 }
  function linkAct(r, noc) { return Math.max(bw(r.l.a, noc), bw(r.l.b, noc)) / maxBW }
  function safe(t) { return fp.safe_kinds.includes(t.kind) }
  function outline(t) { const c = fp.kind_rgb[t.kind] || [120, 120, 140]; return `rgb(${c[0]},${c[1]},${c[2]})` }
  // glow only the selected NoC's NIU (purple=NoC0, cyan=NoC1) so one network reads at a time
  function fill(t) {
    const a0 = noc === 1 ? 0 : bw(t, 0) / maxBW
    const a1 = noc === 0 ? 0 : bw(t, 1) / maxBW
    const r = 18 + 180 * a0 + 60 * a1, g = 20 + 120 * a0 + 214 * a1, b = 28 + 255 * a0 + 224 * a1
    return `rgb(${Math.min(255, r) | 0},${Math.min(255, g) | 0},${Math.min(255, b) | 0})`
  }
  function selBW(t) { return noc === 0 ? bw(t, 0) : noc === 1 ? bw(t, 1) : bw(t, 0) + bw(t, 1) }
  function fillOp(t) { return safe(t) ? 0.4 + 0.5 * (selBW(t) / maxBW) : 0.32 }
  function mb(v) { return v >= 1e6 ? (v / 1e6).toFixed(0) : v >= 1e3 ? (v / 1e3).toFixed(0) + 'k' : '0' }

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
  function onDown(e) { dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY }
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
  function open(t) {
    if (moved) return
    if (pickMode) { if (t.kind === 'tensix') { injSrc = [...t.noc0]; pickMode = false } return }
    push(`/tile/${t.noc0[0]}/${t.noc0[1]}`)
  }
  function setLayout(l) { layout = l; reset() }

  // ---- injection ----
  let injectOpen = false, pickMode = false
  let injSrc = null            // [x,y] noc0 source tensix
  let patterns = []
  let pattern = 'gddr6_write'
  let lengthMB = 0.25, fires = 3, stream = true
  let injBusy = false, injErr = null, injResult = null
  getJSON('/api/inject/patterns').then((p) => (patterns = p)).catch(() => {})
  $: streaming = $frame?.inject?.streaming
  $: srcTile = injSrc && fp ? fp.tiles.find((t) => t.noc0[0] === injSrc[0] && t.noc0[1] === injSrc[1]) : null

  async function fire() {
    if (!injSrc) { injErr = 'pick a source tensix tile first'; return }
    injBusy = true; injErr = null
    try {
      const r = await postJSON('/api/inject', { src: injSrc, pattern, length: Math.round(lengthMB * 1048576), fires, stream })
      injResult = r.ok ? r : null
      if (!r.ok) injErr = r.error || 'inject failed'
    } catch (e) { injErr = e.message } finally { injBusy = false }
  }
  async function stopInject() {
    injBusy = true
    try { await postJSON('/api/inject/stop'); injResult = null } catch (e) { injErr = e.message } finally { injBusy = false }
  }

  // ---- calibration (card layout) ----
  function onKey(e) {
    if (!align || layout !== 'card' || !box) return
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
  $: pcieInfo = fp?.pcie          // {link, gbps_per_dir}
  $: dramBW = $frame?.dram ?? {}  // {ctrl: {r, w}} bytes/s
  $: dramMax = Math.max(2e6, ...Object.values(dramBW).flatMap((d) => [d.r || 0, d.w || 0]))
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
      </defs>

      <g transform="translate({tx} {ty}) scale({scale})">
        {#if layout === 'card'}
          <image href={img.src} x="0" y="0" width={img.w} height={img.h} />
          <rect x={box[0]} y={box[1]} width={box[2] - box[0]} height={box[3] - box[1]}
            fill="none" stroke={align ? '#ffcc44' : '#ffffff33'} stroke-width={sw * (align ? 1.6 : 1)} stroke-dasharray={align ? sw * 4 : 0} />
        {:else}
          <rect x="0" y="0" width={view.w} height={view.h} fill="#0a0c10" />
        {/if}

        <!-- accurate routing rails: NoC0 purple (a→b, E/S), NoC1 cyan (b→a, W/N); wraps dashed in topo.
             single-NoC selection shows rails at any zoom so the interleave + wrap read clearly -->
        {#if noc !== 2 || lod !== 'low' || layout === 'topo'}
          {#each rails as r}
            {#if noc !== 1}
              {@const a = linkAct(r, 0)}
              <path d={r.d0} fill="none" stroke="#05060a" stroke-width={sw * (4 + 3.5 * a)} stroke-opacity="0.75" stroke-linecap="round" stroke-dasharray={r.l.wrap ? sw * 5 : 0} />
              <path d={r.d0} fill="none" stroke={NOC0} stroke-width={sw * (2.4 + 3.5 * a)} stroke-linecap="round" marker-end="url(#a0)" stroke-dasharray={r.l.wrap ? sw * 5 : 0} opacity={(r.l.wrap ? 0.6 : 0.48) + 0.52 * a} />
            {/if}
            {#if noc !== 0}
              {@const a = linkAct(r, 1)}
              <path d={r.d1} fill="none" stroke="#05060a" stroke-width={sw * (4 + 3.5 * a)} stroke-opacity="0.75" stroke-linecap="round" stroke-dasharray={r.l.wrap ? sw * 5 : 0} />
              <path d={r.d1} fill="none" stroke={NOC1} stroke-width={sw * (2.4 + 3.5 * a)} stroke-linecap="round" marker-end="url(#a1)" stroke-dasharray={r.l.wrap ? sw * 5 : 0} opacity={(r.l.wrap ? 0.6 : 0.48) + 0.52 * a} />
            {/if}
          {/each}
        {/if}

        {#each fp.tiles as t (tileKey(t.noc0))}
          {@const p = pos(t)}
          <rect x={p.x} y={p.y} width={p.w} height={p.h} rx={Math.min(p.w, p.h) * 0.16}
            fill={fill(t)} fill-opacity={fillOp(t)} stroke={outline(t)}
            stroke-width={sw * (safe(t) ? 2.2 : 1.4)} stroke-opacity={safe(t) ? 1 : 0.5}
            filter={safe(t) && selBW(t) / maxBW > 0.04 ? 'url(#glow)' : null}
            class="tile" class:safe={safe(t)}
            on:mouseenter={() => (hovered = t)} on:mouseleave={() => (hovered = null)} on:click={() => open(t)} />
        {/each}

        {#if srcTile}
          {@const p = pos(srcTile)}
          <rect x={p.x - p.w * 0.18} y={p.y - p.h * 0.18} width={p.w * 1.36} height={p.h * 1.36}
            rx={p.w * 0.22} fill="none" stroke="#ffd24a" stroke-width={sw * 2.6}
            class="srcmark" filter="url(#glow)" />
        {/if}

        {#if lod === 'high' || (layout === 'topo' && scale >= 1.5)}
          {#each fp.tiles as t (tileKey(t.noc0))}
            {@const p = pos(t)}
            <text x={p.x + p.w / 2} y={p.y + p.h * 0.36} font-size={sw * 5.5} fill="#fff" text-anchor="middle" dominant-baseline="central" class="lbl">{t.label}</text>
            {#if safe(t)}
              <text x={p.x + p.w / 2} y={p.y + p.h * 0.68} font-size={sw * 5} text-anchor="middle" dominant-baseline="central" class="lbl">
                {#if noc === 2}<tspan fill={NOC0}>{mb(bw(t, 0))}</tspan><tspan fill="#555">/</tspan><tspan fill={NOC1}>{mb(bw(t, 1))}</tspan>{:else}<tspan fill={noc === 1 ? NOC1 : NOC0}>{mb(selBW(t))}</tspan>{/if}
              </text>
            {/if}
          {/each}
        {/if}
      </g>
    </svg>

    {#if hovered}
      <div class="tip" style="left:{hx + 14}px; top:{hy + 14}px">
        <b>{hovered.label}</b> · {hovered.kind}<br />
        noc0 {hovered.noc0[0]},{hovered.noc0[1]} · die {hovered.die[0]},{hovered.die[1]}
        {#if hovered.dram_ctrl !== null}<br />GDDR6 d{hovered.dram_ctrl}{/if}<br />
        {#if safe(hovered)}<span style="color:{NOC0}">NoC0 {fmtBW(bw(hovered, 0))}</span> · <span style="color:{NOC1}">NoC1 {fmtBW(bw(hovered, 1))}</span>{:else}<span class="muted">not polled (mgmt)</span>{/if}
      </div>
    {/if}

    <!-- inject panel -->
    <button class="inject-toggle" class:on={injectOpen} on:click={() => (injectOpen = !injectOpen)}>⚡ inject</button>
    {#if injectOpen}
      <div class="inject">
        <div class="ihead"><b>inject traffic</b> <span class="muted">host-driven · NoC routing</span></div>

        <label class="fld">source
          <span class="src">
            {#if injSrc}<b>tensix {injSrc[0]},{injSrc[1]}</b>{:else}<span class="muted">none</span>{/if}
            <button class:on={pickMode} on:click={() => (pickMode = !pickMode)}>{pickMode ? 'click a tile…' : 'pick'}</button>
          </span>
        </label>

        <label class="fld">pattern
          <select bind:value={pattern}>
            {#each patterns as p}<option value={p.id}>{p.label}</option>{/each}
          </select>
        </label>

        <div class="params">
          <label>len MB<input type="number" step="0.05" min="0.01" bind:value={lengthMB} /></label>
          <label>fires<input type="number" min="1" bind:value={fires} /></label>
          <label class="chk"><input type="checkbox" bind:checked={stream} />stream</label>
        </div>

        <div class="acts">
          <button class="fire" on:click={fire} disabled={injBusy || !injSrc}>{streaming ? 're-fire' : 'fire'}</button>
          <button on:click={stopInject} disabled={injBusy || !streaming}>stop</button>
          {#if streaming}<span class="streaming">● streaming</span>{/if}
        </div>

        {#if injErr}<div class="ierr">{injErr}</div>{/if}
        {#if injResult}
          <div class="ires">
            moved <b>{fmtBW(injResult.moved_bytes / (injResult.secs || 1))}</b>
            · {(injResult.moved_bytes / 1e6).toFixed(2)} MB in {(injResult.secs * 1000).toFixed(1)} ms
            <table class="dram">
              <thead><tr><th>GDDR6</th><th class="num">NoC0</th><th class="num">NoC1</th></tr></thead>
              <tbody>
                {#each Object.entries(injResult.dram) as [c, d]}
                  <tr><th>d{c}</th><td class="num" style="color:{NOC0}">{(d['0'] * 64 / 1e3).toFixed(0)}k</td><td class="num" style="color:{NOC1}">{(d['1'] * 64 / 1e3).toFixed(0)}k</td></tr>
                {/each}
              </tbody>
            </table>
            <span class="muted">bytes landed per controller · watch the rails light up live</span>
          </div>
        {/if}
      </div>
    {/if}

    <!-- DRAM dashboard: 8 GDDR6 banks (capacity + live R/W) + PCIe host link -->
    {#if dramInfo}
      <div class="dram-bar">
        <div class="banks">
          {#each dramInfo.ctrls as c}
            {@const d = dramBW[String(c)] ?? { r: 0, w: 0 }}
            {@const act = (d.r + d.w) / dramMax}
            <div class="bank" class:hot={act > 0.05} title="GDDR6 d{c} — read {fmtBW(d.r)} · write {fmtBW(d.w)}">
              <div class="bars">
                <div class="bar r" style="height:{Math.max(3, (d.r / dramMax) * 100)}%"></div>
                <div class="bar w" style="height:{Math.max(3, (d.w / dramMax) * 100)}%"></div>
              </div>
              <div class="cap">{dramInfo.per_ctrl_gib}G</div>
              <div class="bid">d{c}</div>
            </div>
          {/each}
        </div>
        <div class="dram-meta">
          <div class="tot">GDDR6 · <b>{dramInfo.total_gib} GiB</b></div>
          <div class="rwkey"><i class="r"></i>read <i class="w"></i>write</div>
          {#if pcieInfo}<div class="pcie">host · PCIe <b>{pcieInfo.link}</b> <span class="muted">~{pcieInfo.gbps_per_dir} GB/s</span></div>{/if}
        </div>
      </div>
    {/if}

    <div class="hud">
      <div class="row">
        <div class="seg">
          <button class:on={layout === 'card'} on:click={() => setLayout('card')}>card</button>
          <button class:on={layout === 'topo'} on:click={() => setLayout('topo')}>topology</button>
        </div>
        <button on:click={reset} disabled={scale === 1}>reset zoom</button>
        <span>{scale.toFixed(1)}×</span>
        {#if layout === 'card'}<button class:on={align} on:click={() => (align = !align)}>align</button>{/if}
      </div>
      <div class="legend">
        <div class="seg">
          <button class:on={noc === 0} on:click={() => (noc = 0)}><i style="background:{NOC0}"></i>NoC0</button>
          <button class:on={noc === 1} on:click={() => (noc = 1)}><i style="background:{NOC1}"></i>NoC1</button>
          <button class:on={noc === 2} on:click={() => (noc = 2)}>both</button>
        </div>
        <span class="muted">{noc === 2 ? 'both NoCs' : noc === 0 ? 'NoC0 ▸▾ east+south' : 'NoC1 ◂▴ west+north'} · {layout === 'topo' ? 'dashed = wrap' : 'interleave + wrap'}</span>
      </div>
      {#if align && layout === 'card'}
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
  .tile.safe:hover { stroke: #fff !important; stroke-opacity: 1 !important; }
  .lbl { pointer-events: none; font-family: ui-monospace, monospace; }

  .tip { position: absolute; pointer-events: none; z-index: 5; background: #0d0f14f2; border: 1px solid var(--line); border-radius: 6px; padding: 6px 9px; font-size: 12px; line-height: 1.5; max-width: 240px; }
  .muted { color: var(--muted); }
  kbd { background: var(--panel2); border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; }

  .dram-bar {
    position: absolute; left: 0; right: 0; bottom: 0; z-index: 4;
    display: flex; align-items: flex-end; gap: 16px;
    background: linear-gradient(transparent, #0a0c10ee 45%); padding: 18px 16px 8px;
    pointer-events: none;
  }
  .dram-bar > * { pointer-events: auto; }
  .banks { display: flex; gap: 5px; align-items: flex-end; }
  .bank { width: 30px; display: flex; flex-direction: column; align-items: center; gap: 2px; background: #14171dcc; border: 1px solid var(--line); border-radius: 4px; padding: 4px 2px; }
  .bank.hot { border-color: var(--accent); box-shadow: 0 0 8px #ff8a4c66; }
  .bars { display: flex; gap: 2px; align-items: flex-end; height: 34px; }
  .bar { width: 6px; border-radius: 2px 2px 0 0; min-height: 3px; transition: height 0.25s; }
  .bar.r { background: var(--good); }
  .bar.w { background: var(--accent); }
  .bank .cap { font-size: 9px; color: var(--muted); }
  .bank .bid { font-size: 10px; color: var(--fg); }
  .dram-meta { display: flex; flex-direction: column; gap: 3px; font-size: 12px; padding-bottom: 6px; }
  .dram-meta .tot b { color: var(--good); }
  .rwkey { color: var(--muted); display: flex; align-items: center; gap: 5px; }
  .rwkey i { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
  .rwkey i.r { background: var(--good); }
  .rwkey i.w { background: var(--accent); margin-left: 6px; }
  .pcie b { color: var(--noc1); }

  .hud { position: absolute; left: 12px; bottom: 92px; z-index: 4; background: #0d0f14e6; border: 1px solid var(--line); border-radius: 8px; padding: 9px 12px; display: flex; flex-direction: column; gap: 8px; font-size: 12px; max-width: 380px; }
  .row { display: flex; align-items: center; gap: 10px; }
  .hud button { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 9px; cursor: pointer; font: inherit; }
  .hud button.on { background: var(--accent); color: #1a1206; border-color: var(--accent); }
  .hud button:disabled { opacity: 0.4; cursor: default; }
  .seg { display: flex; } .seg button { border-radius: 0; } .seg button:first-child { border-radius: 5px 0 0 5px; } .seg button:last-child { border-radius: 0 5px 5px 0; border-left: 0; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .lg { display: flex; align-items: center; gap: 4px; color: var(--muted); }
  .lg i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
  .cal { display: flex; flex-direction: column; gap: 6px; border-top: 1px solid var(--line); padding-top: 7px; }
  .cal code { color: var(--accent); user-select: all; }
  .nums { display: flex; gap: 8px; }
  .nums label { display: flex; flex-direction: column; color: var(--muted); font-size: 11px; gap: 2px; }
  .nums input { width: 56px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; font: inherit; }
  .loading { display: grid; place-items: center; height: 100%; color: var(--muted); }

  .srcmark { animation: pulse 1.3s ease-in-out infinite; }
  @keyframes pulse { 0%, 100% { stroke-opacity: 1; } 50% { stroke-opacity: 0.35; } }

  .inject-toggle {
    position: absolute; top: 12px; right: 12px; z-index: 6;
    background: var(--panel2); color: var(--fg); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 11px; cursor: pointer; font: inherit;
  }
  .inject-toggle.on { background: var(--accent); color: #1a1206; border-color: var(--accent); }
  .inject {
    position: absolute; top: 48px; right: 12px; z-index: 6; width: 270px;
    background: #0d0f14f2; border: 1px solid var(--line); border-radius: 8px;
    padding: 11px 13px; display: flex; flex-direction: column; gap: 9px; font-size: 12px;
  }
  .ihead { display: flex; justify-content: space-between; align-items: baseline; }
  .fld { display: flex; flex-direction: column; gap: 4px; color: var(--muted); }
  .src { display: flex; align-items: center; gap: 8px; color: var(--fg); }
  .inject select, .inject input { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 3px 5px; font: inherit; }
  .inject button { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 10px; cursor: pointer; font: inherit; }
  .inject button.on { background: var(--accent); color: #1a1206; border-color: var(--accent); }
  .inject button:disabled { opacity: 0.4; cursor: default; }
  .params { display: flex; gap: 8px; }
  .params label { display: flex; flex-direction: column; gap: 3px; color: var(--muted); font-size: 11px; }
  .params input { width: 56px; }
  .params .chk { flex-direction: row; align-items: center; gap: 4px; align-self: end; }
  .acts { display: flex; align-items: center; gap: 8px; }
  .acts .fire { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .streaming { color: var(--good); animation: pulse 1.3s infinite; }
  .ierr { color: var(--bad); background: #3a1f23; border-radius: 4px; padding: 5px 7px; }
  .ires { display: flex; flex-direction: column; gap: 6px; border-top: 1px solid var(--line); padding-top: 8px; }
  .ires b { color: var(--accent); }
  table.dram { border-collapse: collapse; }
  table.dram th, table.dram td { padding: 1px 8px 1px 0; }
  table.dram .num { text-align: right; font-variant-numeric: tabular-nums; }
</style>
