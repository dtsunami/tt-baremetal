<script>
  import { onMount, onDestroy } from 'svelte'
  import CodeEditor from '../lib/CodeEditor.svelte'
  import DocsPane from '../lib/DocsPane.svelte'
  import { getJSON, postJSON, fmtBW, pollJob } from '../lib/api.js'
  import { frame } from '../lib/stores.js'

  // ---- workspace ----
  let info = null
  let project = null
  let files = []
  let current = null
  let role = 'device'
  let content = ''
  let savedContent = ''
  let hasBackup = false
  let err = null
  let status = 'ready'
  let saving = false

  // ---- build / run ----
  let tests = []
  let runName = ''
  let running = false
  let runResult = null
  let building = false
  let buildResult = null
  let dprint = false
  let dprintCores = '0,0'
  let cancelRun = null
  let cancelBuild = null


  let tab = 'docs'

  // ---- CodeMirror ----
  let editor                    // <CodeEditor> ref (bind:this)

  $: dirty = content !== savedContent
  $: mode = $frame?.mode ?? '—'
  $: busy = mode === 'busy'
  $: resetNeeded = $frame?.reset_needed

  // footprint rollups
  $: foot = runResult?.foot
  $: foot0 = foot ? Object.values(foot['0'] || {}).reduce((a, b) => a + b, 0) : 0
  $: foot1 = foot ? Object.values(foot['1'] || {}).reduce((a, b) => a + b, 0) : 0
  $: footTotal = foot0 + foot1
  $: counterBW = runResult?.agg?.wall_cycles && footTotal
    ? footTotal / (runResult.agg.wall_cycles / 1.35e9) : 0
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

  onMount(async () => {
    try {
      info = await getJSON('/api/lab/projects')
      if (!info.available) { err = 'tt-metal not found — set TT_METAL_HOME or build ~/tt-metal'; return }
      project = info.default
      await Promise.all([loadFiles(), loadKernels(), refreshBuild(), refreshRun()])
      if (files.length) await openFile((files.find((f) => f.role === 'device') || files[0]).path)
    } catch (e) { err = String(e) }
  })
  onDestroy(() => { cancelRun?.(); cancelBuild?.() })

  // ---- editor (shared <CodeEditor>) ----
  function setDoc(text) { savedContent = text; editor?.setDoc(text) }   // onChange syncs `content`

  // ---- files ----
  async function loadFiles() {
    files = await getJSON(`/api/lab/files?project=${encodeURIComponent(project)}`)
  }
  async function switchProject() {
    if (dirty && !confirm('Discard unsaved changes?')) return
    runResult = null
    await Promise.all([loadFiles(), loadKernels()])
    if (files.length) await openFile((files.find((f) => f.role === 'device') || files[0]).path)
  }
  async function openFile(path) {
    if (path !== current && dirty && !confirm('Discard unsaved changes?')) return
    const f = await getJSON(`/api/lab/file?path=${encodeURIComponent(path)}`)
    current = f.path; role = f.role; hasBackup = f.has_backup
    setDoc(f.content)
    status = role === 'device' ? 'device kernel — edit then Run (JIT, no rebuild)' : 'host program — edit then Build, then Run'
  }
  async function save() {
    if (!dirty || saving) return
    saving = true
    try {
      await postJSON('/api/lab/file', { path: current, content })
      savedContent = content; hasBackup = true; status = 'saved ' + current
      await loadFiles()
    } catch (e) { status = 'save failed: ' + e } finally { saving = false }
  }
  async function revert() {
    if (!hasBackup || !confirm('Restore the as-shipped version of this file?')) return
    try { const f = await postJSON('/api/lab/file/revert', { path: current }); setDoc(f.content); status = 'reverted to .orig' }
    catch (e) { status = 'revert failed: ' + e }
  }
  async function duplicate() {
    if (!current) return
    const base = current.split('/').pop()
    const name = prompt('Duplicate to (new file name, same dir):', base.replace(/\.(cpp|cc|h|hpp)$/, '_v2.$1'))
    if (!name) return
    try { const f = await postJSON('/api/lab/file/duplicate', { src: current, name }); await loadFiles(); await openFile(f.path); status = 'duplicated → ' + f.path }
    catch (e) { status = 'duplicate: ' + e }
  }

  // ---- build ----
  async function build() {
    if (building || busy) return
    try {
      const r = await postJSON('/api/lab/build', {})
      if (!r.ok) { status = 'build: ' + (r.error || 'failed to start'); return }
      building = true; status = 'building ' + r.started + ' …'; tab = 'debug'; pollBuild()
    } catch (e) { status = 'build error: ' + e }
  }
  function pollBuild() {
    cancelBuild = pollJob('/api/lab/build/last', (d) => {
      building = false
      if (d.error) { status = 'build poll error: ' + d.error; return }
      buildResult = d.result
      status = d.result?.ok ? `build ok (${d.result.secs}s${d.result.no_work ? ', no work' : ''})` : `build FAILED (${d.result?.errors?.length || 0} errors)`
    }, 2500)
  }
  async function refreshBuild() {
    const d = await getJSON('/api/lab/build/last'); buildResult = d.result
    if (d.running) { building = true; pollBuild() }
  }

  // ---- run ----
  async function loadKernels() {
    try { const k = await getJSON('/api/kernels'); tests = k.tests || []; const c = tests.filter((x) => norm(x).includes(norm(project))); runName = c[0] || tests[0] || '' }
    catch (e) { tests = [] }
  }
  async function run() {
    if (running || busy || !runName || resetNeeded) return
    try {
      const r = await postJSON('/api/kernels/run', { name: runName, dprint, dprint_cores: dprintCores })
      if (!r.ok) { status = 'run: ' + (r.error || 'failed to start'); return }
      running = true; runResult = null; status = 'running ' + runName + ' … (device owned; live polling paused)'; tab = 'telem'; pollRun()
    } catch (e) { status = 'run error: ' + e }
  }
  function pollRun() {
    cancelRun = pollJob('/api/kernels/last', (d) => {
      running = false
      if (d.error) { status = 'run poll error: ' + d.error; return }
      runResult = d.result
      status = d.result?.passed ? `run PASSED (${d.result.secs}s)` : `run done (${d.result?.secs ?? '?'}s)`
    }, 2500)
  }
  async function refreshRun() {
    const d = await getJSON('/api/kernels/last'); runResult = d.result
    if (d.running) { running = true; pollRun() }
  }

</script>

{#if err}
  <div class="lab-msg">⚠ {err} — <a href="#/">back to chip</a></div>
{:else if !info}
  <div class="lab-msg">loading workspace…</div>
{:else}
<div class="lab">
  <!-- files -->
  <aside class="files">
    {#if info.projects.length > 1}
      <select class="proj" bind:value={project} on:change={switchProject}>
        {#each info.projects as p}<option value={p.name}>{p.name}</option>{/each}
      </select>
    {:else}<div class="proj one">{project}</div>{/if}
    <ul>
      {#each files as f}
        <li>
          <button class="file" class:active={f.path === current} on:click={() => openFile(f.path)}>
            <span class="fn">{f.name}</span><span class="role {f.role}">{f.role}</span>
          </button>
        </li>
      {/each}
    </ul>
    <div class="hint"><b class="role device">device</b> JIT — just Run · <b class="role host">host</b> needs Build</div>
  </aside>

  <!-- editor -->
  <section class="editor">
    <div class="toolbar">
      <span class="cur">{current ?? '—'}{#if dirty}<b class="dt">●</b>{/if}</span>
      {#if current}<span class="role {role}">{role}</span>{/if}
      <span class="sp"></span>
      {#if tests.length > 1}
        <select class="runsel" bind:value={runName} title="gtest to run">
          {#each tests as tn}<option value={tn}>{tn}</option>{/each}
        </select>
      {/if}
      <label class="dp" title="capture on-device DPRINT (TT_METAL_DPRINT_CORES)"><input type="checkbox" bind:checked={dprint} /> DPRINT</label>
      {#if current}<button on:click={duplicate} title="duplicate this kernel into a new variation">⧉</button>{/if}
      {#if hasBackup}<button on:click={revert} title="restore .orig">⟲</button>{/if}
      <button on:click={save} disabled={!dirty || saving}>Save</button>
      <button on:click={build} disabled={building || busy}>{building ? 'Building…' : 'Build'}</button>
      <button class="run" on:click={run} disabled={running || busy || !runName || resetNeeded}>{running ? 'Running…' : 'Run ▸'}</button>
    </div>
    <div class="code-wrap"><CodeEditor bind:this={editor} lang="cpp" onChange={(text) => content = text} onSave={save} /></div>
    <div class="statusbar">
      <span class="st">{status}</span><span class="sp"></span>
      {#if runName}<span class="dim">run: {runName}</span>{/if}
      <span class="mode" class:busy={busy} class:bad={resetNeeded}>mode: {resetNeeded ? 'reset-needed' : mode}</span>
    </div>
  </section>

  <!-- docs / debug / telemetry -->
  <aside class="side">
    <div class="tabs">
      <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
      <button class:on={tab === 'debug'} on:click={() => tab = 'debug'}>Debug</button>
      <button class:on={tab === 'telem'} on:click={() => tab = 'telem'}>Telemetry</button>
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
          {#if runResult.dprint?.length}<pre class="log">{runResult.dprint.join('\n')}</pre>{:else}<div class="dim">none — enable the DPRINT toggle and re-run</div>{/if}
          {#if runResult.log_tail}<details><summary>log tail</summary><pre class="log">{runResult.log_tail}</pre></details>{/if}
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
    </div>
  </aside>
</div>
{/if}

<style>
  .lab-msg { padding: 24px; color: var(--muted); }
  .lab { display: grid; grid-template-columns: 200px 1fr minmax(340px, 40%); height: calc(100vh - 47px); overflow: hidden; }

  .files { border-right: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
  .proj { padding: 8px 10px; border-bottom: 1px solid var(--line); background: var(--panel); width: 100%; color: var(--fg); }
  .proj.one { color: var(--accent); font-weight: 600; }
  select.proj { font-family: inherit; font-size: 12px; border: none; border-bottom: 1px solid var(--line); }
  .files ul { list-style: none; margin: 0; padding: 6px; overflow: auto; flex: 1; }
  .file { display: flex; width: 100%; align-items: center; gap: 6px; padding: 5px 8px; background: none; border: none; color: var(--fg); cursor: pointer; border-radius: 5px; text-align: left; font-family: inherit; font-size: 12px; }
  .file:hover { background: var(--panel2); }
  .file.active { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .file .fn { flex: 1; overflow: hidden; text-overflow: ellipsis; }
  .hint { padding: 8px 10px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }
  .role { font-size: 10px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); }
  .role.device { color: var(--noc1); border-color: var(--noc1); }
  .role.host { color: var(--accent); border-color: var(--accent); }

  .editor { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
  .toolbar .cur { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .toolbar .dt { color: var(--accent); margin-left: 4px; }
  .sp { flex: 1; }
  .toolbar button { font-family: inherit; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 4px 10px; cursor: pointer; }
  .toolbar button:hover:not(:disabled) { border-color: var(--muted); }
  .toolbar button:disabled { opacity: 0.4; cursor: default; }
  .toolbar .run { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .toolbar .dp { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 3px; }
  .runsel { max-width: 200px; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px; }
  .code-wrap { flex: 1; overflow: hidden; min-height: 0; background: #0a0c10; }
  .statusbar { display: flex; align-items: center; gap: 10px; padding: 4px 10px; border-top: 1px solid var(--line); background: var(--panel); font-size: 11px; color: var(--muted); }
  .statusbar .st { color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .statusbar .mode.busy { color: var(--accent); }
  .statusbar .mode.bad { color: var(--bad); }
  .dim { color: var(--muted); }

  .side { border-left: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
  .tabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); }
  .tabs button { flex: 1; font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .tabbody { overflow: auto; padding: 12px 14px; flex: 1; min-height: 0; }
  .tabbody h4 { margin: 14px 0 6px; font-size: 12px; }
  .tabbody h4:first-child { margin-top: 0; }
  .tabbody h5 { margin: 10px 0 4px; font-size: 11px; color: var(--muted); font-weight: 500; }
  .line { margin: 3px 0; }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  .note { background: #1a1206; border: 1px solid var(--accent); color: #ffd24a; border-radius: 5px; padding: 6px 9px; margin-bottom: 8px; font-size: 12px; }
  .errs { margin: 4px 0; padding-left: 16px; color: var(--bad); font-size: 12px; }
  .errs li { margin: 2px 0; }
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 240px; font-size: 11px; line-height: 1.45; white-space: pre; }
  details summary { cursor: pointer; color: var(--muted); font-size: 11px; margin: 4px 0; }
  .sw { display: inline-block; width: 8px; height: 8px; border-radius: 2px; vertical-align: middle; }
  .sw.n0 { background: var(--noc0); }
  .sw.n1 { background: var(--noc1); }
</style>
