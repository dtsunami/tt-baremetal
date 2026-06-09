<script>
  import { push } from 'svelte-spa-router'
  import { floorplan, frame } from '../lib/stores.js'
  import { fmtBW, tileKey } from '../lib/api.js'

  const NOC0 = '#b478ff' // purple
  const NOC1 = '#4fd6e0' // cyan

  let svgEl
  let scale = 1, tx = 0, ty = 0
  let hovered = null, hx = 0, hy = 0
  let dragging = false, lastX = 0, lastY = 0, moved = false

  $: fp = $floorplan
  $: img = fp?.image
  $: ftiles = $frame?.tiles ?? {}
  $: maxBW = Math.max(1, ...Object.values(ftiles).map((t) => (t.noc0 || 0) + (t.noc1 || 0)))

  // level-of-detail by zoom: low = heat only, mid = + routing arrows, high = + labels/values
  $: lod = scale >= 4.5 ? 'high' : scale >= 2 ? 'mid' : 'low'
  $: sw = 1 / scale // keep strokes ~1px on screen at any zoom

  // physical-neighbour links (the real wires); dual-rail offset endpoints precomputed once
  let links = []
  $: if (fp) links = buildLinks(fp)
  function center(t) { return [t.rect.x + t.rect.w / 2, t.rect.y + t.rect.h / 2] }
  function buildLinks(fp) {
    const byDie = new Map(fp.tiles.filter((t) => t.rect).map((t) => [t.die.join(','), t]))
    const out = []
    for (const t of fp.tiles) {
      if (!t.rect) continue
      for (const [dx, dy] of [[1, 0], [0, 1]]) {
        const nb = byDie.get(`${t.die[0] + dx},${t.die[1] + dy}`)
        if (!nb) continue
        const [ax, ay] = center(t), [bx, by] = center(nb)
        const ux = bx - ax, uy = by - ay, L = Math.hypot(ux, uy) || 1
        const px = -uy / L, py = ux / L
        const o = Math.min(t.rect.w, t.rect.h) * 0.16
        out.push({
          a: t, b: nb,
          x0a: ax + px * o, y0a: ay + py * o, x0b: bx + px * o, y0b: by + py * o, // NoC0 rail a→b
          x1a: ax - px * o, y1a: ay - py * o, x1b: bx - px * o, y1b: by - py * o, // NoC1 rail b→a
        })
      }
    }
    return out
  }

  function bw(t, noc) {
    const f = ftiles[tileKey(t.noc0)]
    return f ? (noc === 0 ? f.noc0 : f.noc1) || 0 : 0
  }
  function linkAct(l, noc) { return Math.max(bw(l.a, noc), bw(l.b, noc)) / maxBW }
  function safe(t) { return fp.safe_kinds.includes(t.kind) }
  function outline(t) {
    const c = fp.kind_rgb[t.kind] || [120, 120, 140]
    return `rgb(${c[0]},${c[1]},${c[2]})`
  }
  // dark base + additive NoC0(purple)/NoC1(cyan) glow → idle tiles are dark but outlined
  function fill(t) {
    const a0 = bw(t, 0) / maxBW, a1 = bw(t, 1) / maxBW
    const r = 18 + 180 * a0 + 60 * a1
    const g = 20 + 120 * a0 + 214 * a1
    const b = 28 + 255 * a0 + 224 * a1
    return `rgb(${Math.min(255, r) | 0},${Math.min(255, g) | 0},${Math.min(255, b) | 0})`
  }
  function fillOp(t) {
    if (!safe(t)) return 0.32
    return 0.4 + 0.5 * ((bw(t, 0) + bw(t, 1)) / maxBW)
  }

  // ---- zoom / pan ----
  function clamp() {
    tx = Math.min(0, Math.max(img.w * (1 - scale), tx))
    ty = Math.min(0, Math.max(img.h * (1 - scale), ty))
  }
  function onWheel(e) {
    if (!img) return
    e.preventDefault()
    const r = svgEl.getBoundingClientRect()
    const px = ((e.clientX - r.left) / r.width) * img.w
    const py = ((e.clientY - r.top) / r.height) * img.h
    const ns = Math.min(16, Math.max(1, scale * (e.deltaY < 0 ? 1.18 : 1 / 1.18)))
    tx = px - (px - tx) * (ns / scale)
    ty = py - (py - ty) * (ns / scale)
    scale = ns
    clamp()
  }
  function onDown(e) { dragging = true; moved = false; lastX = e.clientX; lastY = e.clientY }
  function onMove(e) {
    const r = svgEl.getBoundingClientRect()
    hx = e.clientX - r.left
    hy = e.clientY - r.top
    if (!dragging || !img) return
    moved = true
    tx += ((e.clientX - lastX) / r.width) * img.w
    ty += ((e.clientY - lastY) / r.height) * img.h
    lastX = e.clientX; lastY = e.clientY
    clamp()
  }
  function onUp() { dragging = false }
  function reset() { scale = 1; tx = 0; ty = 0 }
  function open(t) { if (!moved) push(`/tile/${t.noc0[0]}/${t.noc0[1]}`) }

  $: hoverBW = hovered ? (() => { const f = ftiles[tileKey(hovered.noc0)]; return f ? (f.noc0 || 0) + (f.noc1 || 0) : 0 })() : 0
</script>

<div class="wrap">
  {#if fp}
    <svg
      bind:this={svgEl}
      role="application" aria-label="Blackhole chip floorplan"
      viewBox="0 0 {img.w} {img.h}"
      class:grabbing={dragging}
      on:wheel={onWheel} on:mousedown={onDown} on:mousemove={onMove}
      on:mouseup={onUp} on:mouseleave={onUp}
    >
      <defs>
        <marker id="a0" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L4,2 L0,4 z" fill={NOC0} />
        </marker>
        <marker id="a1" markerWidth="4" markerHeight="4" refX="3.2" refY="2" orient="auto" markerUnits="userSpaceOnUse">
          <path d="M0,0 L4,2 L0,4 z" fill={NOC1} />
        </marker>
      </defs>

      <g transform="translate({tx} {ty}) scale({scale})">
        <image href={img.src} x="0" y="0" width={img.w} height={img.h} />
        <rect
          x={img.package[0]} y={img.package[1]}
          width={img.package[2] - img.package[0]} height={img.package[3] - img.package[1]}
          fill="none" stroke="#ffffff33" stroke-width={sw}
        />

        <!-- routing rails: NoC0 purple (a→b), NoC1 cyan (b→a) — appear when zoomed in -->
        {#if lod !== 'low'}
          {#each links as l}
            <line x1={l.x0a} y1={l.y0a} x2={l.x0b} y2={l.y0b}
              stroke={NOC0} stroke-width={sw * 1.3} stroke-linecap="round"
              opacity={0.12 + 0.85 * linkAct(l, 0)} marker-end="url(#a0)" />
            <line x1={l.x1b} y1={l.y1b} x2={l.x1a} y2={l.y1a}
              stroke={NOC1} stroke-width={sw * 1.3} stroke-linecap="round"
              opacity={0.12 + 0.85 * linkAct(l, 1)} marker-end="url(#a1)" />
          {/each}
        {/if}

        <!-- tiles: bright kind outline always on (legibility) + per-NoC glow fill -->
        {#each fp.tiles as t (tileKey(t.noc0))}
          {#if t.rect}
            <rect
              x={t.rect.x} y={t.rect.y} width={t.rect.w} height={t.rect.h}
              fill={fill(t)} fill-opacity={fillOp(t)}
              stroke={outline(t)} stroke-width={sw * 1.4}
              stroke-opacity={safe(t) ? 0.95 : 0.45}
              class="tile" class:safe={safe(t)}
              on:mouseenter={() => (hovered = t)}
              on:mouseleave={() => (hovered = null)}
              on:click={() => open(t)}
            />
          {/if}
        {/each}

        <!-- labels + per-NoC values at high zoom -->
        {#if lod === 'high'}
          {#each fp.tiles as t (tileKey(t.noc0))}
            {#if t.rect}
              <text x={t.rect.x + t.rect.w / 2} y={t.rect.y + t.rect.h * 0.42}
                font-size={sw * 6.5} fill="#fff" text-anchor="middle" class="lbl">{t.label}</text>
              {#if safe(t)}
                <text x={t.rect.x + t.rect.w / 2} y={t.rect.y + t.rect.h * 0.78}
                  font-size={sw * 5} text-anchor="middle" class="lbl">
                  <tspan fill={NOC0}>{(bw(t, 0) / 1e6).toFixed(0)}</tspan><tspan fill="#666">/</tspan><tspan fill={NOC1}>{(bw(t, 1) / 1e6).toFixed(0)}</tspan>
                </text>
              {/if}
            {/if}
          {/each}
        {/if}
      </g>
    </svg>

    {#if hovered}
      <div class="tip" style="left:{hx + 14}px; top:{hy + 14}px">
        <b>{hovered.label}</b> · {hovered.kind}<br />
        noc0 {hovered.noc0[0]},{hovered.noc0[1]} · die {hovered.die[0]},{hovered.die[1]}
        {#if hovered.dram_ctrl !== null}<br />GDDR6 ctrl d{hovered.dram_ctrl}{/if}
        <br />
        {#if safe(hovered)}
          <span style="color:{NOC0}">NoC0 {fmtBW(bw(hovered, 0))}</span> ·
          <span style="color:{NOC1}">NoC1 {fmtBW(bw(hovered, 1))}</span>
        {:else}<span class="muted">not polled (mgmt)</span>{/if}
      </div>
    {/if}

    <div class="hud">
      <div class="zoom">
        <button on:click={reset} disabled={scale === 1}>reset</button>
        <span>{scale.toFixed(1)}×</span>
        <span class="muted">scroll zoom · drag pan · click tile · {lod === 'low' ? 'zoom in for routing' : lod === 'mid' ? 'NoC0/NoC1 rails' : 'rails + values'}</span>
      </div>
      <div class="legend">
        <span class="lg"><i style="background:{NOC0}"></i>NoC0 ▸▾</span>
        <span class="lg"><i style="background:{NOC1}"></i>NoC1 ◂▴</span>
        <span class="sep">·</span>
        {#each Object.entries(fp.kind_rgb) as [k, rgb]}
          <span class="lg"><i style="border:1.5px solid rgb({rgb[0]},{rgb[1]},{rgb[2]});background:transparent"></i>{k}</span>
        {/each}
      </div>
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

  .tip {
    position: absolute; pointer-events: none; z-index: 5;
    background: #0d0f14f2; border: 1px solid var(--line); border-radius: 6px;
    padding: 6px 9px; font-size: 12px; line-height: 1.5; max-width: 240px;
  }
  .muted { color: var(--muted); }

  .hud {
    position: absolute; left: 12px; bottom: 12px; z-index: 4;
    background: #0d0f14e0; border: 1px solid var(--line); border-radius: 8px;
    padding: 9px 12px; display: flex; flex-direction: column; gap: 7px; font-size: 12px;
  }
  .zoom { display: flex; align-items: center; gap: 10px; }
  .zoom button { background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 9px; cursor: pointer; font: inherit; }
  .zoom button:disabled { opacity: 0.4; cursor: default; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
  .lg { display: flex; align-items: center; gap: 4px; color: var(--muted); }
  .lg i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
  .sep { color: var(--line); }
  .loading { display: grid; place-items: center; height: 100%; color: var(--muted); }
</style>
