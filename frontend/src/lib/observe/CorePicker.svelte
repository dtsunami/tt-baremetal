<!-- CorePicker — shared compact Tensix core selector for the unified Deploy tab (overlay/LLK/metal).
     A dense floorplan grid (one cell per core, laid out by x,y) colored by live NoC bandwidth from the
     global telemetry frame + residency. Hover a cell for context; click to toggle it in the selection
     SET (deploy a kernel to / stop a group of cores). One selected core → the parent shows that core's
     dashboard; many → group actions. Binds `selected` (array of {x,y}). -->
<script>
  import { onMount } from 'svelte'
  import { getJSON } from '../api.js'
  import { frame } from '../stores.js'

  export let selected = []        // [{x,y}] — two-way bound (the deploy set / dashboard target)

  let scan = null, busy = false, err = ''
  async function doScan() {
    busy = true; err = ''
    try { scan = await getJSON('/api/tensix/bl/scan') }
    catch (e) { err = String(e.message || e) } finally { busy = false }
  }
  onMount(doScan)

  const key = (c) => `${c.x},${c.y}`
  const kindOf = (c) => c.kind || (c.resident ? 'ttmetal' : 'idle')
  const isSel = (c) => selected.some((s) => s.x === c.x && s.y === c.y)
  function toggle(c) {
    selected = isSel(c) ? selected.filter((s) => !(s.x === c.x && s.y === c.y))
                        : [...selected, { x: c.x, y: c.y }]
  }
  const clear = () => selected = []
  function selectKind(k) { selected = (scan?.cores || []).filter((c) => kindOf(c) === k).map((c) => ({ x: c.x, y: c.y })) }

  // live NoC bandwidth heat per core, from the global telemetry frame (status/telemetry from global view)
  const heat = (c) => { const t = $frame?.tiles?.[key(c)]; return t ? (t.noc0 || 0) + (t.noc1 || 0) : 0 }
  $: heatMax = $frame ? Math.max(1, ...Object.values($frame.tiles || {}).map((t) => (t.noc0 || 0) + (t.noc1 || 0))) : 1
  const fmtBW = (v) => v >= 1e9 ? (v / 1e9).toFixed(1) + ' GB/s' : v >= 1e6 ? (v / 1e6).toFixed(0) + ' MB/s' : v ? (v / 1e3).toFixed(0) + ' kB/s' : 'idle'

  $: ys = scan ? [...new Set(scan.cores.map((c) => c.y))].sort((a, b) => a - b) : []
  const ctx = (c) => `(${c.x},${c.y}) · ${kindOf(c)}${c.loaded ? ' · ' + c.loaded : ''} · ${fmtBW(heat(c))}`
</script>

<div class="cp">
  <div class="hd">
    <b>Cores</b>
    {#if scan}<span class="dim">{scan.n_bootloader ?? 0} bl · {scan.n_ttmetal ?? 0} metal</span>{/if}
    <span class="sp"></span>
    {#if selected.length}<span class="selcount">{selected.length} selected</span>
      <button class="mini" on:click={clear}>clear</button>{/if}
    <button class="mini" on:click={() => selectKind('idle')} title="select all idle cores">+idle</button>
    <button class="mini" on:click={() => selectKind('bootloader')} title="select all bootloader cores">+bl</button>
    <button class="mini" on:click={doScan} disabled={busy} title="rescan">{busy ? '…' : '⟳'}</button>
  </div>
  {#if err}<div class="err">{err}</div>{/if}
  {#if scan}
    <div class="grid">
      {#each ys as y}
        <div class="row">
          {#each scan.cores.filter((c) => c.y === y).sort((a, b) => a.x - b.x) as c (key(c))}
            <button class="cell {kindOf(c)}" class:sel={isSel(c)} class:err={c.error}
                    style="--h:{Math.min(1, heat(c) / heatMax)}"
                    title={ctx(c)} on:click={() => toggle(c)}>{c.x},{c.y}</button>
          {/each}
        </div>
      {/each}
    </div>
    <div class="legend dim">
      <span class="sw bootloader"></span>bootloader <span class="sw ttmetal"></span>metal <span class="sw idle"></span>idle
      · cell fill = live NoC bandwidth · hover for context · click to select
    </div>
  {:else if !busy}
    <div class="dim pad">no cores — hit ⟳</div>
  {/if}
</div>

<style>
  .cp { border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); padding: 8px 10px; }
  .hd { display: flex; align-items: center; gap: 8px; }
  .hd .sp { flex: 1; }
  .dim { color: var(--muted); font-size: 11px; }
  .selcount { font-size: 11px; color: var(--accent); }
  .mini { font-size: 11px; padding: 1px 7px; cursor: pointer; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; color: var(--muted); }
  .mini:hover { color: var(--fg); border-color: var(--muted); }
  .err { color: #e07a77; font-size: 11px; margin-top: 4px; }
  .pad { padding: 6px 2px; }
  .grid { display: flex; flex-direction: column; gap: 2px; margin: 8px 0 6px; overflow: auto; }
  .row { display: flex; gap: 2px; }
  .cell { position: relative; font-family: ui-monospace, monospace; font-size: 8.5px; min-width: 30px; padding: 3px 1px; border-radius: 3px;
          cursor: pointer; color: var(--muted); border: 1px solid var(--line);
          background: rgba(235,110,70, calc(var(--h, 0) * 0.55)); transition: background .25s; }
  .cell:hover { color: var(--fg); border-color: var(--muted); }
  .cell.bootloader { border-left: 2px solid var(--good); }
  .cell.ttmetal { border-left: 2px solid var(--noc1); }
  .cell.sel { outline: 2px solid var(--accent); outline-offset: 1px; color: var(--fg); z-index: 1; }
  .cell.err { opacity: .45; }
  .legend { display: flex; align-items: center; gap: 5px; flex-wrap: wrap; font-size: 10px; }
  .sw { width: 8px; height: 8px; border-radius: 2px; display: inline-block; }
  .sw.bootloader { background: var(--good); } .sw.ttmetal { background: var(--noc1); } .sw.idle { background: var(--muted); }
</style>
