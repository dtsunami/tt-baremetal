<script>
  import { onMount, onDestroy } from 'svelte'
  import CodeEditor from '../lib/CodeEditor.svelte'
  import DocsPane from '../lib/DocsPane.svelte'
  import { getJSON, postJSON, pollJob } from '../lib/api.js'
  import { renderDisasm } from '../lib/riscv.js'
  import { frame } from '../lib/stores.js'

  // ---- examples + workspace ----
  let examples = [], example = '', available = true, err = null
  let files = [], current = null, role = 'compute'
  let content = '', savedContent = '', hasBackup = false, saving = false, editor

  // ---- run + observe ----
  let running = false, result = null, status = 'ready', cancelRun = null
  let tab = 'occ'
  let deployed = {}          // Tensix core "x,y" -> {kernel, math_occ} (which kernel on which tile)
  let disasm = null          // per-engine disassembly of the last run
  let tip = { show: false, x: 0, y: 0, text: '' }

  $: mode = $frame?.mode ?? '—'
  $: busy = mode === 'busy'
  $: resetNeeded = $frame?.reset_needed
  $: dirty = content !== savedContent
  $: compute = result?.compute
  $: cores = compute ? Object.entries(compute.cores) : []
  $: ranName = result ? short(result.name) : null

  const ENGINES = [
    { key: 'reader', label: 'RISC-V 1 · reader', sub: 'Router 0', col: 'var(--noc1)' },
    { key: 'unpack', label: 'RISC-V 2 · UNPACK', sub: 'compute', col: '#8b93a7' },
    { key: 'math',   label: 'RISC-V 3 · MATH',   sub: 'compute', col: 'var(--accent)' },
    { key: 'pack',   label: 'RISC-V 4 · PACK',   sub: 'compute', col: '#8b93a7' },
    { key: 'writer', label: 'RISC-V 5 · writer', sub: 'Router 1', col: 'var(--noc1)' },
  ]
  const pct = (c, w) => w ? Math.min(100, 100 * c / w) : 0
  const fmtCyc = (c) => c >= 1e6 ? (c / 1e6).toFixed(2) + 'M' : c >= 1e3 ? (c / 1e3).toFixed(1) + 'k' : '' + (c || 0)
  const short = (n) => n.replace(/^metal_example_/, '')

  onMount(async () => {
    try {
      const e = await getJSON('/api/tlab/examples')
      available = e.available; examples = e.examples || []
      example = examples.find((x) => x.includes('matmul_single')) || examples[0] || ''
      if (example) await loadFiles()
      await loadStatus()
      const d = await getJSON('/api/tlab/last'); result = d.result
      if (d.running) { running = true; poll() }
    } catch (ex) { err = String(ex) }
  })
  onDestroy(() => cancelRun?.())

  async function loadFiles() {
    files = await getJSON(`/api/tlab/files?example=${encodeURIComponent(example)}`)
    const first = files.find((f) => f.role === 'compute') || files[0]
    if (first) await openFile(first.path)
  }
  async function loadStatus() { try { deployed = (await getJSON('/api/tlab/status')).deployed || {} } catch (e) {} }
  async function loadDisasm() { try { disasm = await getJSON('/api/tlab/disasm') } catch (e) { disasm = { ok: false, error: String(e) } } }
  async function duplicate() {
    if (!current) return
    const name = prompt('Duplicate to (new file name):', current.split('/').pop().replace(/\.(cpp|cc|h|hpp)$/, '_v2.$1'))
    if (!name) return
    try { const f = await postJSON('/api/tlab/file/duplicate', { src: current, name }); await loadFiles(); await openFile(f.path); status = 'duplicated → ' + f.path }
    catch (e) { status = 'duplicate: ' + e }
  }
  function onTip(e) {
    const el = e.target.closest('[data-tip]')
    if (el) tip = { show: true, x: e.clientX, y: e.clientY, text: el.getAttribute('data-tip') }
    else if (tip.show) tip = { show: false, x: 0, y: 0, text: '' }
  }
  async function switchExample() {
    if (dirty && !confirm('Discard unsaved changes?')) return
    current = null; await loadFiles()
  }
  async function openFile(path) {
    if (path !== current && dirty && !confirm('Discard unsaved changes?')) return
    const f = await getJSON(`/api/tlab/file?path=${encodeURIComponent(path)}`)
    current = f.path; role = f.role; hasBackup = f.has_backup
    savedContent = f.content; editor?.setDoc(f.content)
    status = role === 'host' ? 'host program — edit needs a rebuild (view)' : `${role} kernel — edit then Run (JIT, no rebuild)`
  }
  async function save() {
    if (!current || !dirty || saving) return
    saving = true
    try { await postJSON('/api/tlab/file', { path: current, content }); savedContent = content; hasBackup = true; status = 'saved ' + current; await loadFiles() }
    catch (e) { status = 'save failed: ' + e } finally { saving = false }
  }
  async function revert() {
    if (!hasBackup || !confirm('Restore the as-shipped version?')) return
    try { const f = await postJSON('/api/tlab/file/revert', { path: current }); savedContent = f.content; editor?.setDoc(f.content); status = 'reverted to .orig' }
    catch (e) { status = 'revert failed: ' + e }
  }
  async function run() {
    if (running || busy || !example || resetNeeded) return
    if (dirty) await save()
    try {
      const r = await postJSON('/api/tlab/run', { name: example })
      if (!r.ok) { status = 'run: ' + (r.error || 'failed to start'); return }
      running = true; result = null; status = `running ${short(example)}… (JIT-recompiles edited kernels; tt-metal resets the x280 harts)`; tab = 'occ'; poll()
    } catch (e) { status = 'run error: ' + e }
  }
  function poll() {
    cancelRun = pollJob('/api/tlab/last', (d) => {
      running = false
      if (d.error) { status = 'poll error: ' + d.error; return }
      result = d.result
      status = result?.passed ? `passed (${result.secs}s)` : `done (${result?.secs ?? '?'}s)`
      loadStatus(); if (tab === 'disasm') loadDisasm()
    }, 2500)
  }
</script>

{#if err}
  <div class="msg">⚠ {err} — <a href="#/">back to chip</a></div>
{:else if !available}
  <div class="msg">tt-metal not found — set TT_METAL_HOME or build ~/tt-metal. <a href="#/">back</a></div>
{:else}
<div class="lab">
  <!-- left: example + its kernel files -->
  <aside class="rail">
    <select class="exsel" bind:value={example} on:change={switchExample}>
      {#each examples as e}<option value={e}>{short(e)}</option>{/each}
    </select>
    <ul>
      {#each files as f}
        <li><button class="file" class:active={f.path === current} on:click={() => openFile(f.path)}>
          <span class="fn">{f.name}</span><span class="role {f.role}">{f.role}</span></button></li>
      {/each}
    </ul>
    <div class="hint"><b class="role compute">compute</b>/<b class="role dataflow">dataflow</b> JIT — edit then Run · <b class="role host">host</b> needs rebuild</div>
  </aside>

  <!-- center: editor -->
  <section class="editor">
    <div class="toolbar">
      <span class="cur">{current ?? '—'}{#if dirty}<b class="dt">●</b>{/if}</span>
      {#if current}<span class="role {role}">{role}</span>{/if}
      <span class="sp"></span>
      {#if current}<button on:click={duplicate} title="duplicate this kernel into a new variation">⧉</button>{/if}
      {#if hasBackup}<button on:click={revert} title="restore .orig">⟲</button>{/if}
      <button on:click={save} disabled={!dirty || saving}>Save</button>
      <button class="run" on:click={run} disabled={running || busy || resetNeeded} title="JIT-recompile + run on Tensix (⌘⏎)">{running ? 'Running…' : 'Run ▸'}</button>
    </div>
    <div class="code-wrap"><CodeEditor bind:this={editor} lang="cpp" onChange={(t) => content = t} onSave={save} onSubmit={run} /></div>
    <div class="statusbar"><span class="st">{status}</span><span class="sp"></span>
      <span class="mode" class:busy={busy} class:bad={resetNeeded}>mode: {resetNeeded ? 'reset-needed' : mode}</span></div>
  </section>

  <!-- right: occupancy + docs -->
  <aside class="side">
    <div class="tabs">
      <button class:on={tab === 'occ'} on:click={() => tab = 'occ'}>Occupancy</button>
      <button class:on={tab === 'disasm'} on:click={() => { tab = 'disasm'; if (!disasm) loadDisasm() }}>Disasm</button>
      <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
    </div>
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="tabbody" on:mousemove={onTip} on:mouseleave={() => tip.show = false}>
      {#if tab === 'occ'}
        {#if Object.keys(deployed).length}
          <div class="status">
            <span class="dim">Tensix tiles · which kernel ran where:</span>
            {#each Object.entries(deployed) as [core, d]}
              <span class="tchip" title="MATH {(d.math_occ * 100).toFixed(0)}%">({core}) <b>{d.kernel}</b> <span class="dim">{(d.math_occ * 100).toFixed(0)}%</span></span>
            {/each}
          </div>
        {/if}
        {#if running}
          <div class="note">running on Tensix… per-engine profile appears when it finishes.</div>
        {:else if compute && cores.length}
          <div class="ranhd">ran <b>{ranName}</b> on <b>{compute.n_cores}</b> Tensix core{compute.n_cores > 1 ? 's' : ''} · avg MATH <b class="acc">{(compute.avg_math_occ * 100).toFixed(1)}%</b></div>
          {#each cores as [core, c]}
            <div class="core">
              <div class="corehd">core ({core}) <span class="mathocc" class:hot={c.math_occ > 0.5}>MATH {(c.math_occ * 100).toFixed(1)}%</span>
                <span class="dim">{c.math_occ > 0.5 ? 'compute-bound' : 'memory-bound'}</span></div>
              {#each ENGINES as eng}
                {@const cyc = c.engines[eng.key] || 0}
                <div class="erow" class:math={eng.key === 'math'}>
                  <span class="elabel">{eng.label}<span class="esub">{eng.sub}</span></span>
                  <span class="ebar"><span class="efill" style="width:{pct(cyc, c.wall)}%; background:{eng.col}"></span></span>
                  <span class="ecyc">{fmtCyc(cyc)}</span>
                </div>
              {/each}
            </div>
          {/each}
          {#if result?.dprint?.length}<h5>DPRINT</h5><pre class="log">{result.dprint.join('\n')}</pre>{/if}
          {#if result?.log_tail}<details><summary>log tail</summary><pre class="log">{result.log_tail}</pre></details>{/if}
        {:else if result}<div class="dim pad">Ran, no compute zones parsed. {result.error || ''}</div>
        {:else}<div class="dim pad">Edit a <b>compute</b> kernel, then <b>Run ▸</b> — you'll see per-engine busy cycles for each Tensix core (MATH occupancy = compute- vs memory-bound). Editing a compute/dataflow kernel JIT-recompiles on the next Run.</div>{/if}

      {:else if tab === 'disasm'}
        <h4>Kernel disassembly <span class="dim">· JIT-compiled, hover any op/register</span></h4>
        {#if disasm?.ok}
          <div class="dim" style="margin-bottom:6px">kernel <b>{disasm.kernel}</b> · the 3 compute engines (reader/writer are separate DM kernels)</div>
          {#each disasm.engines.filter((e) => e.present) as e}
            <details class="eng" open={e.role === 'math'}>
              <summary><b class:acc={e.role === 'math'}>{e.label}</b> <span class="dim">{e.disasm.split('\n').length} lines</span></summary>
              <div class="disasm">{@html renderDisasm(e.disasm)}</div>
            </details>
          {/each}
          {#if !disasm.engines.some((e) => e.present)}<div class="dim pad">No compiled ELFs yet — Run a compute example.</div>{/if}
        {:else}<div class="dim pad">{disasm?.error || 'Run a compute example, then the JIT-compiled per-engine disassembly appears here.'}</div>{/if}

      {:else}
        <DocsPane docsUrl="/api/tlab/docs" docUrl={(id) => `/api/tlab/doc/${id}`} />
      {/if}
    </div>
  </aside>
</div>
{#if tip.show}<div class="tip" style="left:{tip.x + 14}px; top:{tip.y + 16}px">{tip.text}</div>{/if}
{/if}

<style>
  .msg { padding: 24px; color: var(--muted); }
  .lab { display: grid; grid-template-columns: 210px 1fr minmax(330px, 38%); height: calc(100vh - 47px); overflow: hidden; }
  .rail { border-right: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
  .exsel { padding: 8px 10px; border: none; border-bottom: 1px solid var(--line); background: var(--panel); color: var(--fg); font-family: inherit; font-size: 12px; width: 100%; }
  .rail ul { list-style: none; margin: 0; padding: 6px; overflow: auto; flex: 1; }
  .file { display: flex; width: 100%; align-items: center; gap: 6px; padding: 5px 8px; background: none; border: none; color: var(--fg); cursor: pointer; border-radius: 5px; text-align: left; font-family: inherit; font-size: 12px; }
  .file:hover { background: var(--panel2); }
  .file.active { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .file .fn { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .role { font-size: 9.5px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); }
  .role.compute { color: var(--accent); border-color: var(--accent); }
  .role.dataflow { color: var(--noc1); border-color: var(--noc1); }
  .role.host { color: var(--muted); }
  .hint { padding: 8px 10px; border-top: 1px solid var(--line); color: var(--muted); font-size: 11px; }

  .editor { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
  .cur { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dt { color: var(--accent); margin-left: 4px; }
  .sp { flex: 1; }
  .toolbar button { font-family: inherit; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 4px 10px; cursor: pointer; }
  .toolbar button:hover:not(:disabled) { border-color: var(--muted); }
  .toolbar button:disabled { opacity: 0.4; cursor: default; }
  .toolbar .run { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .code-wrap { flex: 1; overflow: hidden; min-height: 0; background: #0a0c10; }
  .statusbar { display: flex; align-items: center; gap: 10px; padding: 4px 10px; border-top: 1px solid var(--line); background: var(--panel); font-size: 11px; color: var(--muted); }
  .statusbar .st { color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .mode.busy { color: var(--accent); } .mode.bad { color: var(--bad); }

  .side { border-left: 1px solid var(--line); display: flex; flex-direction: column; min-height: 0; }
  .tabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); }
  .tabs button { flex: 1; font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .tabbody { overflow: auto; padding: 12px 14px; flex: 1; min-height: 0; }
  .dim { color: var(--muted); } .pad { padding: 8px 2px; } .acc { color: var(--accent); }
  .note { background: #1a1206; border: 1px solid var(--accent); color: #ffd24a; border-radius: 6px; padding: 8px 11px; }
  .ranhd { font-size: 12px; margin-bottom: 10px; }
  .status { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin-bottom: 12px; padding-bottom: 10px; border-bottom: 1px solid var(--line); }
  .tchip { font-size: 11px; padding: 2px 7px; border-radius: 10px; border: 1px solid var(--line); background: var(--panel); }
  .tchip b { color: var(--accent); }
  details.eng { margin-bottom: 8px; }
  details.eng > summary { cursor: pointer; font-size: 12px; padding: 3px 0; }
  details.eng .acc { color: var(--accent); }
  .disasm { background: #0a0c10; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; overflow: auto; max-height: 320px; font-size: 10.5px; line-height: 1.6; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; margin-top: 4px; }
  .disasm :global(.dlabel) { color: var(--noc1); margin: 6px 0 2px; }
  .disasm :global(.dline) { white-space: pre; }
  .disasm :global(.da) { color: var(--muted); } .disasm :global(.dhex) { color: #525a68; }
  .disasm :global(.dm) { color: #ffd24a; } .disasm :global(.dm[data-tip]), .disasm :global(.dr) { cursor: help; }
  .disasm :global(.dm[data-tip]):hover, .disasm :global(.dr):hover { text-decoration: underline dotted; }
  .disasm :global(.dr) { color: var(--noc0); } .disasm :global(.dcom) { color: #69707f; font-style: italic; }
  .tip { position: fixed; z-index: 60; max-width: 320px; background: #0b0d12; border: 1px solid var(--accent); color: var(--fg); border-radius: 6px; padding: 6px 9px; font-size: 11.5px; line-height: 1.45; pointer-events: none; box-shadow: 0 4px 18px rgba(0,0,0,0.55); white-space: pre-line; }
  .core { border: 1px solid var(--line); border-radius: 8px; padding: 9px 11px; margin-bottom: 10px; background: var(--panel); }
  .corehd { font-weight: 600; margin-bottom: 7px; display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .mathocc { font-size: 11px; padding: 1px 7px; border-radius: 10px; border: 1px solid var(--line); color: var(--muted); }
  .mathocc.hot { color: var(--accent); border-color: var(--accent); }
  .erow { display: grid; grid-template-columns: 140px 1fr 50px; align-items: center; gap: 8px; padding: 2px 0; }
  .erow.math .elabel { color: var(--accent); font-weight: 600; }
  .elabel { font-size: 11px; display: flex; flex-direction: column; line-height: 1.2; }
  .esub { font-size: 9px; color: var(--muted); }
  .ebar { height: 11px; background: #0a0c10; border: 1px solid var(--line); border-radius: 3px; overflow: hidden; }
  .efill { display: block; height: 100%; }
  .ecyc { font-size: 11px; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
  h5 { margin: 12px 0 4px; font-size: 11px; color: var(--muted); font-weight: 500; }
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 200px; font-size: 11px; line-height: 1.45; white-space: pre; }
  details summary { cursor: pointer; color: var(--muted); font-size: 11px; margin: 6px 0; }
</style>
