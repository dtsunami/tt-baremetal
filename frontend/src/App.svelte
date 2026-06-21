<script>
  import Router from 'svelte-spa-router'
  import Chip from './routes/Chip.svelte'
  import TileDetail from './routes/TileDetail.svelte'
  import DeviceBrowser from './routes/DeviceBrowser.svelte'
  import { connected, frame } from './lib/stores.js'
  import { fmtBW } from './lib/api.js'

  const routes = {
    '/': Chip,                     // chip floorplan (NoC explorer)
    '/tile/:x/:y': TileDetail,
    '/rv': DeviceBrowser,          // RV Kernels: tensix Bootloader cockpit lives here (TensixObserve)
  }

  $: mode = $frame?.mode ?? '—'
  $: resetNeeded = $frame?.reset_needed

  // ---- live NoC monitor in the header (bhtop-style) ----
  let n0 = 0, n1 = 0, n0h = [], n1h = []
  $: if ($frame) mon($frame)
  function mon(f) {
    const tl = f.tiles || {}
    let a = 0, b = 0
    for (const k in tl) { a += tl[k].noc0 || 0; b += tl[k].noc1 || 0 }
    n0 = a; n1 = b; n0h = [...n0h, a].slice(-48); n1h = [...n1h, b].slice(-48)
  }
  function spark(h, w = 62, ht = 16) {
    if (h.length < 2) return ''
    const mx = Math.max(...h, 1)
    return h.map((v, i) => `${(i / (h.length - 1)) * w},${(ht - 1) - (v / mx) * (ht - 2)}`).join(' ')
  }
</script>

<header>
  <h1><a href="#/">bhtop</a> <span class="sub">Blackhole NoC explorer</span></h1>
  <nav>
    <a href="#/">chip</a>
    <a href="#/rv">RV Kernels</a>
  </nav>
  <a class="mon" href="#/" title="live NoC bandwidth — click for the chip view">
    <span class="ml n0">NoC0</span>
    <svg class="spk" viewBox="0 0 62 16" preserveAspectRatio="none"><polyline points={spark(n0h)} style="fill:none;stroke:var(--noc0);stroke-width:1.3" /></svg>
    <span class="mv">{fmtBW(n0)}</span>
    <span class="ml n1">NoC1</span>
    <svg class="spk" viewBox="0 0 62 16" preserveAspectRatio="none"><polyline points={spark(n1h)} style="fill:none;stroke:var(--noc1);stroke-width:1.3" /></svg>
    <span class="mv">{fmtBW(n1)}</span>
  </a>
  <div class="status">
    <span class="dot" class:on={$connected}></span>
    {$connected ? 'live' : 'reconnecting…'} · {mode}
  </div>
</header>

<style>
  .mon { display: flex; align-items: center; gap: 7px; margin-left: 20px; padding: 3px 11px; border: 1px solid var(--line); border-radius: 7px; background: var(--panel2); text-decoration: none; }
  .mon:hover { border-color: var(--muted); }
  .ml { font-size: 10px; font-weight: 600; letter-spacing: 0.03em; }
  .ml.n0 { color: var(--noc0); }
  .ml.n1 { color: var(--noc1); }
  .mv { font-size: 11px; color: var(--fg); min-width: 58px; font-variant-numeric: tabular-nums; }
  .spk { width: 62px; height: 16px; display: block; }
</style>

{#if resetNeeded}
  <div class="alert">
    ⚠ NoC hang detected — run <code>tt-smi -r 0</code> on ttstar to recover, then reload.
  </div>
{/if}

<main>
  <Router {routes} />
</main>
