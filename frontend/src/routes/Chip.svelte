<script>
  import { push } from 'svelte-spa-router'
  import { floorplan, frame } from '../lib/stores.js'
  import { fmtBW, tileKey } from '../lib/api.js'

  let svgEl
  let scale = 1, tx = 0, ty = 0
  let hovered = null, hx = 0, hy = 0
  let dragging = false, lastX = 0, lastY = 0

  $: fp = $floorplan
  $: img = fp?.image
  $: ftiles = $frame?.tiles ?? {}
  // normalize heat against the busiest tile in the current frame
  $: maxBW = Math.max(
    1,
    ...Object.values(ftiles).map((t) => (t.noc0 || 0) + (t.noc1 || 0))
  )

  function heat(t) {
    const f = ftiles[tileKey(t.noc0)]
    return f ? ((f.noc0 || 0) + (f.noc1 || 0)) / maxBW : 0
  }
  function safe(t) {
    return fp.safe_kinds.includes(t.kind)
  }
  function fill(t) {
    const rgb = fp.kind_rgb[t.kind] || [120, 120, 140]
    const h = heat(t)
    const g = 30 + 225 * h // glow brighter with bandwidth
    return `rgb(${Math.min(255, rgb[0] * 0.25 + g)},${Math.min(255, rgb[1] * 0.25 + g)},${Math.min(255, rgb[2] * 0.25 + g * 0.55)})`
  }
  function opacity(t) {
    if (!safe(t)) return 0.22 // management tiles: static, never polled
    return 0.3 + 0.62 * heat(t)
  }

  // ---- zoom / pan (transform a <g>; viewBox is image-native px) ----
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
    const ns = Math.min(14, Math.max(1, scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15)))
    tx = px - (px - tx) * (ns / scale)
    ty = py - (py - ty) * (ns / scale)
    scale = ns
    clamp()
  }
  function onDown(e) {
    dragging = true
    lastX = e.clientX
    lastY = e.clientY
  }
  function onMove(e) {
    const r = svgEl.getBoundingClientRect()
    hx = e.clientX - r.left
    hy = e.clientY - r.top
    if (!dragging || !img) return
    tx += ((e.clientX - lastX) / r.width) * img.w
    ty += ((e.clientY - lastY) / r.height) * img.h
    lastX = e.clientX
    lastY = e.clientY
    clamp()
  }
  function onUp() {
    dragging = false
  }
  function reset() {
    scale = 1
    tx = 0
    ty = 0
  }
  function open(t) {
    push(`/tile/${t.noc0[0]}/${t.noc0[1]}`)
  }

  $: hoverBW = hovered
    ? (() => {
        const f = ftiles[tileKey(hovered.noc0)]
        return f ? (f.noc0 || 0) + (f.noc1 || 0) : 0
      })()
    : 0
</script>

<div class="wrap">
  {#if fp}
    <svg
      bind:this={svgEl}
      role="application"
      aria-label="Blackhole chip floorplan"
      viewBox="0 0 {img.w} {img.h}"
      class:grabbing={dragging}
      on:wheel={onWheel}
      on:mousedown={onDown}
      on:mousemove={onMove}
      on:mouseup={onUp}
      on:mouseleave={onUp}
    >
      <g transform="translate({tx} {ty}) scale({scale})">
        <image href={img.src} x="0" y="0" width={img.w} height={img.h} />
        <!-- package footprint outline (overlay sits here; tiles aren't visible under the lid) -->
        <rect
          x={img.package[0]} y={img.package[1]}
          width={img.package[2] - img.package[0]}
          height={img.package[3] - img.package[1]}
          fill="none" stroke="#ffffff22" stroke-width="1"
        />
        {#each fp.tiles as t (tileKey(t.noc0))}
          {#if t.rect}
            <rect
              x={t.rect.x} y={t.rect.y} width={t.rect.w} height={t.rect.h}
              fill={fill(t)} opacity={opacity(t)}
              stroke="#00000055" stroke-width="0.4"
              class="tile" class:safe={safe(t)}
              on:mouseenter={() => (hovered = t)}
              on:mouseleave={() => (hovered = null)}
              on:click={() => open(t)}
            />
          {/if}
        {/each}
      </g>
    </svg>

    {#if hovered}
      <div class="tip" style="left:{hx + 14}px; top:{hy + 14}px">
        <b>{hovered.label}</b> · {hovered.kind}<br />
        noc0 {hovered.noc0[0]},{hovered.noc0[1]} · die {hovered.die[0]},{hovered.die[1]}
        {#if hovered.dram_ctrl !== null}<br />GDDR6 ctrl d{hovered.dram_ctrl}{/if}
        <br />
        {#if safe(hovered)}<span class="bw">{fmtBW(hoverBW)}</span>{:else}<span class="muted">not polled (mgmt)</span>{/if}
      </div>
    {/if}

    <div class="hud">
      <div class="zoom">
        <button on:click={reset} disabled={scale === 1}>reset</button>
        <span>{scale.toFixed(1)}×</span>
        <span class="muted">scroll = zoom · drag = pan · click a tile</span>
      </div>
      <div class="legend">
        {#each Object.entries(fp.kind_rgb) as [k, rgb]}
          <span class="lg"><i style="background:rgb({rgb[0]},{rgb[1]},{rgb[2]})"></i>{k}</span>
        {/each}
      </div>
      <div class="note">overlay registered to the die package footprint — tiles aren't visible under the lid</div>
    </div>
  {:else}
    <div class="loading">connecting to ttstar…</div>
  {/if}
</div>

<style>
  .wrap { position: relative; height: calc(100vh - 47px); overflow: hidden; background: #000; }
  svg { width: 100%; height: 100%; display: block; cursor: grab; }
  svg.grabbing { cursor: grabbing; }
  .tile { cursor: pointer; transition: opacity 0.15s; }
  .tile.safe:hover { stroke: #fff; stroke-width: 1.2; }

  .tip {
    position: absolute; pointer-events: none; z-index: 5;
    background: #0d0f14ee; border: 1px solid var(--line); border-radius: 6px;
    padding: 6px 9px; font-size: 12px; line-height: 1.5; max-width: 230px;
  }
  .tip .bw { color: var(--accent); font-weight: 600; }
  .muted { color: var(--muted); }

  .hud {
    position: absolute; left: 12px; bottom: 12px; z-index: 4;
    background: #0d0f14d8; border: 1px solid var(--line); border-radius: 8px;
    padding: 9px 12px; display: flex; flex-direction: column; gap: 7px; font-size: 12px;
  }
  .zoom { display: flex; align-items: center; gap: 10px; }
  .zoom button {
    background: var(--panel2); color: var(--fg); border: 1px solid var(--line);
    border-radius: 5px; padding: 2px 9px; cursor: pointer; font: inherit;
  }
  .zoom button:disabled { opacity: 0.4; cursor: default; }
  .legend { display: flex; flex-wrap: wrap; gap: 10px; }
  .lg { display: flex; align-items: center; gap: 4px; color: var(--muted); }
  .lg i { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
  .note { color: var(--muted); font-size: 11px; max-width: 360px; }
  .loading { display: grid; place-items: center; height: 100%; color: var(--muted); }
</style>
