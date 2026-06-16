<!-- NocObserve — the NOC (data-movement) right pane: Build / Run + Docs / Debug / Telemetry
     (per-NoC footprint). Ported from KernelLab; actions live here. The gtest to run is picked
     to match the section's selected project (active.sel). -->
<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import DocsPane from '../DocsPane.svelte'
  import { getJSON, postJSON, fmtBW, pollJob } from '../api.js'
  import { frame } from '../stores.js'

  export let active           // {engine, key, name, sel}
  export let dirty = false
  export let onSave = () => {}     // parent saves its editor content before Build/Run

  const dispatch = createEventDispatcher()

  let tab = 'telem'
  let tests = [], runName = '', running = false, runResult = null
  let building = false, buildResult = null
  let dprint = false, dprintCores = '0,0'
  let cancelRun = null, cancelBuild = null, status = 'ready'

  $: project = active?.sel || ''
  $: mode = $frame?.mode ?? '—'
  $: busy = mode === 'busy'
  $: resetNeeded = $frame?.reset_needed

  // footprint rollups
  $: foot = runResult?.foot
  $: foot0 = foot ? Object.values(foot['0'] || {}).reduce((a, b) => a + b, 0) : 0
  $: foot1 = foot ? Object.values(foot['1'] || {}).reduce((a, b) => a + b, 0) : 0
  $: footTotal = foot0 + foot1
  $: counterBW = runResult?.agg?.wall_cycles && footTotal ? footTotal / (runResult.agg.wall_cycles / 1.35e9) : 0
  $: topTiles = (() => {
    if (!foot) return []
    const m = {}
    for (const n of ['0', '1']) for (const [k, v] of Object.entries(foot[n] || {})) m[k] = (m[k] || 0) + v
    return Object.entries(m).sort((a, b) => b[1] - a[1]).slice(0, 6)
  })()
  $: live = (() => {
    let a = 0, b = 0
    const tl = $frame?.tiles || {}
    for (const k in tl) { a += tl[k].noc0 || 0; b += tl[k].noc1 || 0 }
    return { a, b }
  })()

  const norm = (s) => s.toLowerCase().replace(/[^a-z0-9]/g, '')
  function fmtBytes(b) {
    if (!b) return '0 B'
    if (b >= 1e9) return (b / 1e9).toFixed(2) + ' GB'
    if (b >= 1e6) return (b / 1e6).toFixed(1) + ' MB'
    if (b >= 1e3) return (b / 1e3).toFixed(1) + ' kB'
    return b + ' B'
  }

  onMount(() => { loadKernels(); refreshBuild(); refreshRun() })
  onDestroy(() => { cancelRun?.(); cancelBuild?.() })

  async function loadKernels() {
    try { const k = await getJSON('/api/kernels'); tests = k.tests || []; const c = tests.filter((x) => norm(x).includes(norm(project))); runName = c[0] || tests[0] || '' }
    catch (e) { tests = [] }
  }
  $: if (project && tests.length) { const c = tests.filter((x) => norm(x).includes(norm(project))); if (c.length) runName = c[0] }

  async function build() {
    if (building || busy) return
    if (dirty) await onSave()
    try { const r = await postJSON('/api/lab/build', {}); if (!r.ok) { status = 'build: ' + (r.error || 'failed'); return } building = true; status = 'building ' + r.started + ' …'; tab = 'debug'; pollBuild() }
    catch (e) { status = 'build error: ' + e }
  }
  function pollBuild() {
    cancelBuild = pollJob('/api/lab/build/last', (d) => {
      building = false
      if (d.error) { status = 'build poll error: ' + d.error; return }
      buildResult = d.result
      status = d.result?.ok ? `build ok (${d.result.secs}s${d.result.no_work ? ', no work' : ''})` : `build FAILED (${d.result?.errors?.length || 0} errors)`
    }, 2500)
  }
  async function refreshBuild() { try { const d = await getJSON('/api/lab/build/last'); buildResult = d.result; if (d.running) { building = true; pollBuild() } } catch (e) {} }

  async function run() {
    if (running || busy || !runName || resetNeeded) return
    if (dirty) await onSave()
    try { const r = await postJSON('/api/kernels/run', { name: runName, dprint, dprint_cores: dprintCores }); if (!r.ok) { status = 'run: ' + (r.error || 'failed'); return } running = true; runResult = null; status = 'running ' + runName + ' … (device owned; live polling paused)'; tab = 'telem'; pollRun() }
    catch (e) { status = 'run error: ' + e }
  }
  function pollRun() {
    cancelRun = pollJob('/api/kernels/last', (d) => {
      running = false
      if (d.error) { status = 'run poll error: ' + d.error; return }
      runResult = d.result
      status = d.result?.passed ? `run PASSED (${d.result.secs}s)` : `run done (${d.result?.secs ?? '?'}s)`
      dispatch('ran')
    }, 2500)
  }
  async function refreshRun() { try { const d = await getJSON('/api/kernels/last'); runResult = d.result; if (d.running) { running = true; pollRun() } } catch (e) {} }
</script>

<div class="tabs">
  <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
  <button class:on={tab === 'debug'} on:click={() => tab = 'debug'}>Debug</button>
  <button class:on={tab === 'telem'} on:click={() => tab = 'telem'}>Telemetry</button>
  <span class="sp"></span>
  <label class="dp" title="capture on-device DPRINT"><input type="checkbox" bind:checked={dprint} /> DPRINT</label>
  <button on:click={build} disabled={building || busy}>{building ? 'Building…' : 'Build'}</button>
  <button class="run" on:click={run} disabled={running || busy || !runName || resetNeeded} title={runName}>{running ? 'Running…' : 'Run ▸'}</button>
</div>
<div class="tabbody">
  {#if tab === 'docs'}
    <DocsPane docsUrl="/api/lab/docs" docUrl={(id) => `/api/lab/doc/${id}`} imgUrl={(n) => `/api/lab/uarch/${n}`} />

  {:else if tab === 'debug'}
    <h4>Build</h4>
    {#if buildResult}
      <div class="line">status: <b class={buildResult.ok ? 'good' : 'bad'}>{buildResult.ok ? 'ok' : 'FAILED'}</b>
        {#if buildResult.no_work}<span class="dim">(no work)</span>{/if}{#if buildResult.secs}<span class="dim">· {buildResult.secs}s</span>{/if}</div>
      {#if buildResult.errors?.length}<ul class="errs">{#each buildResult.errors as e}<li><b>{e.file}:{e.line}:{e.col}</b> {e.msg}</li>{/each}</ul>{/if}
      {#if buildResult.log_tail}<details><summary>ninja log</summary><pre class="log">{buildResult.log_tail}</pre></details>{/if}
    {:else}<div class="dim">no build yet — edit a host file and hit Build</div>{/if}

    <h4>Run</h4>
    {#if runResult}
      <div class="line">result: <b class={runResult.passed ? 'good' : (runResult.ok ? '' : 'bad')}>{runResult.passed ? 'PASSED' : (runResult.ok ? 'done' : 'FAILED')}</b>{#if runResult.secs}<span class="dim">· {runResult.secs}s</span>{/if}</div>
      {#if runResult.error}<div class="bad">{runResult.error}</div>{/if}
      {#if runResult.agg}<div class="line">profiler: <b>{(runResult.agg.bw / 1e12).toFixed(2)} TB/s</b> <span class="dim">· {runResult.agg.cores} cores · {fmtBytes(runResult.agg.total_bytes)}</span></div>{/if}
      <h5>DPRINT</h5>
      {#if runResult.dprint?.length}<pre class="log">{runResult.dprint.join('\n')}</pre>{:else}<div class="dim">none — enable DPRINT and re-run</div>{/if}
    {:else}<div class="dim">no run yet</div>{/if}

  {:else}
    {#if running}<div class="note">kernel owns the device — live counters paused; footprint appears when it finishes.</div>{/if}
    {#if foot}
      <h4>Per-NoC footprint <span class="dim">(silicon NIU counters)</span></h4>
      <table>
        <tr><th>NoC0 <span class="sw n0"></span></th><td class="num">{fmtBytes(foot0)}</td></tr>
        <tr><th>NoC1 <span class="sw n1"></span></th><td class="num">{fmtBytes(foot1)}</td></tr>
        <tr><th>total</th><td class="num"><b>{fmtBytes(footTotal)}</b></td></tr>
      </table>
      <h5>Bandwidth — reconcile</h5>
      <table>
        <tr><th>profiler aggregate</th><td class="num">{runResult.agg ? (runResult.agg.bw / 1e12).toFixed(2) + ' TB/s' : '—'}</td></tr>
        <tr><th>NIU counters</th><td class="num">{counterBW ? (counterBW / 1e12).toFixed(2) + ' TB/s' : '—'}</td></tr>
      </table>
      <h5>Top tiles</h5>
      <table>{#each topTiles as [k, v]}<tr><th>{k}</th><td class="num">{fmtBytes(v)}</td></tr>{/each}</table>
      <a class="gh" href="#/">view footprint on chip ▸</a>
    {:else}<div class="dim">run a kernel to see its NoC footprint here</div>{/if}

    <h4>Live <span class="dim">(when idle)</span></h4>
    {#if $frame && !busy}
      <table>
        <tr><th>NoC0 <span class="sw n0"></span></th><td class="num">{fmtBW(live.a)}</td></tr>
        <tr><th>NoC1 <span class="sw n1"></span></th><td class="num">{fmtBW(live.b)}</td></tr>
      </table>
    {:else}<div class="dim">{busy ? 'paused (device busy)' : 'connecting…'}</div>{/if}
  {/if}
  <div class="foot dim">{status}{#if runName} · run: {runName}{/if}</div>
</div>

<style>
  .tabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); align-items: center; }
  .tabs button { font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .tabs .sp { flex: 1; }
  .tabs .dp { font-size: 10.5px; color: var(--muted); display: flex; align-items: center; gap: 3px; padding: 0 6px; }
  .tabs > button:not(.on):last-of-type, .tabs .run { background: var(--accent); color: #1a1206; border: 1px solid var(--accent); border-radius: 5px; padding: 4px 10px; margin: 0 6px 0 0; font-weight: 600; }
  .tabs .run:disabled { opacity: 0.4; cursor: default; }
  .tabbody { overflow: auto; padding: 12px 14px; flex: 1; min-height: 0; }
  .tabbody h4 { margin: 14px 0 6px; font-size: 12px; }
  .tabbody h4:first-child { margin-top: 0; }
  .tabbody h5 { margin: 10px 0 4px; font-size: 11px; color: var(--muted); font-weight: 500; }
  .line { margin: 3px 0; } .good { color: var(--good); } .bad { color: var(--bad); } .dim { color: var(--muted); }
  .note { background: #1a1206; border: 1px solid var(--accent); color: #ffd24a; border-radius: 5px; padding: 6px 9px; margin-bottom: 8px; font-size: 12px; }
  .errs { margin: 4px 0; padding-left: 16px; color: var(--bad); font-size: 12px; }
  .errs li { margin: 2px 0; }
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 240px; font-size: 11px; line-height: 1.45; white-space: pre; }
  details summary { cursor: pointer; color: var(--muted); font-size: 11px; margin: 4px 0; }
  .sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px; vertical-align: middle; }
  .sw.n0 { background: var(--noc0); } .sw.n1 { background: var(--noc1); }
  .gh { color: var(--accent); font-size: 11px; display: inline-block; margin-top: 6px; }
  .foot { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--line); font-size: 11px; }
</style>
