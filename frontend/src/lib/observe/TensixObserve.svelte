<!-- TensixObserve — the TENSIX (compute) right pane: Run + per-engine Occupancy / Disasm / Docs.
     Ported from ComputeLab; actions live here (the editor stays generic). The example to run is
     the section selection carried on `active.sel`. -->
<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import DocsPane from '../DocsPane.svelte'
  import TensixLaunch from '../TensixLaunch.svelte'
  import BootloaderPanel from './BootloaderPanel.svelte'
  import CorePicker from './CorePicker.svelte'
  import { getJSON, postJSON, pollJob } from '../api.js'
  import { renderDisasm } from '../riscv.js'
  import { loadIsa, bits } from '../isa.js'
  import { frame } from '../stores.js'

  export let active           // {engine, key, name, sel}
  export let preselect = null  // bootloader overlay name picked from the tree
  export let content = ''      // live editor buffer (overlay source) — deploy compiles this
  export let dirty = false
  export let onSave = () => {}     // parent saves its editor content before Run (JIT reads disk)

  const dispatch = createEventDispatcher()

  // kernel type drives every tab's content (Deploy/Build/Disasm are context-aware; ISA/Docs shared).
  $: kind = active?.overlay ? 'overlay' : active?.llk ? 'llk' : 'metal'
  $: if (active) { tab = 'deploy' }     // selecting a kernel opens its Deploy tab

  let running = false, result = null, status = 'ready', cancelRun = null
  let tab = 'deploy', deployed = {}, disasm = null, examples = []
  let tip = { show: false, x: 0, y: 0, text: '' }
  let metalSel = [], occOpen = true   // metal core selection set + collapsible Occupancy

  // (core selection is now the shared CorePicker; metal/llk bind their own {x,y}.)

  // Map the selected file (folder-browser key like 'matmul/matmul_single_core/kernels/…') to its
  // runnable example binary: the example whose short-name is a path segment of the key (handles
  // nested examples). Falls back to the selector hint for the legacy/flat path.
  function pickExample(key, sel, exs) {
    if (key) {
      const segs = key.split('/')
      // most-specific wins: longest example short-name that is a path segment (nested examples
      // like contributed/vecadd resolve to 'vecadd', not a broader container)
      const hits = exs.filter((e) => segs.includes(short(e))).sort((a, b) => short(b).length - short(a).length)
      if (hits.length) return hits[0]
    }
    if (sel && exs.includes(sel)) return sel
    return sel ? (sel.startsWith('metal_example_') ? sel : 'metal_example_' + sel) : ''
  }
  $: example = pickExample(active?.key, active?.sel, examples)
  $: mode = $frame?.mode ?? '—'
  $: busy = mode === 'busy'
  $: resetNeeded = $frame?.reset_needed
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
    await loadStatus()
    try { examples = (await getJSON('/api/tlab/examples')).examples || [] } catch (e) {}
    try { const d = await getJSON('/api/tlab/last'); result = d.result; if (d.running) { running = true; poll() } } catch (e) {}
  })
  onDestroy(() => cancelRun?.())

  async function loadStatus() { try { deployed = (await getJSON('/api/tlab/status')).deployed || {} } catch (e) {} }
  async function loadDisasm() { try { disasm = await getJSON('/api/tlab/disasm') } catch (e) { disasm = { ok: false, error: String(e) } } }

  // ---- Build tab: verbose JIT compile log + source path + force rebuild ----
  let buildlog = null, rebuilding = false
  async function loadBuildLog() { try { buildlog = await getJSON('/api/tlab/buildlog') } catch (e) { buildlog = { ok: false, error: String(e) } } }
  async function forceRebuild() {
    rebuilding = true
    try {
      const r = await postJSON('/api/tlab/rebuild', {})
      status = r.ok ? `cleared ${r.removed?.length || 0} cached build(s) — hit Run ▸ to recompile from source` : (r.error || 'rebuild failed')
    } catch (e) { status = 'rebuild failed: ' + e } finally { rebuilding = false }
  }

  // ---- standalone build: extract to bhtop + replay tt-metal's compile (no device) ----
  let recipeInfo = null, exResult = null, exBusy = false
  $: exShort = short(example || '')
  async function loadRecipe() {
    if (!exShort) { recipeInfo = null; return }
    try { recipeInfo = await getJSON(`/api/tlab/recipe?example=${encodeURIComponent(exShort)}`) } catch (e) { recipeInfo = null }
  }
  async function doExtract() {
    exBusy = true
    try {
      const r = await postJSON('/api/tlab/extract', { example: exShort })
      status = r.ok ? `extracted ${r.files.length} file(s) → ${r.dir}` : 'extract: no sources found (Run the example first)'
    } catch (e) { status = 'extract failed: ' + e } finally { exBusy = false }
  }
  async function doBuild() {
    exBusy = true; exResult = null
    try {
      exResult = await postJSON('/api/tlab/build', { example: exShort })
      status = exResult.ok ? `built ${exResult.units.length} kernel(s) ✓` : (exResult.error || 'build had errors — see below')
    } catch (e) { status = 'build failed: ' + e } finally { exBusy = false }
  }
  // ---- ISA tab: searchable assembly.yaml reference (opcode + per-operand bit-fields) ----
  let isa = null, isaQ = '', isaCat = 'all', isaOpen = {}
  async function loadIsaTab() { if (!isa) isa = await loadIsa() }
  $: isaList = isa?.mnemonics ? Object.values(isa.mnemonics).sort((a, b) => a.name.localeCompare(b.name)) : []
  $: isaCats = ['all', ...Array.from(new Set(isaList.map((i) => i.category))).sort()]
  $: isaShown = isaList.filter((i) => {
    if (isaCat !== 'all' && i.category !== isaCat) return false
    const q = isaQ.trim().toLowerCase()
    if (!q) return true
    return i.name.toLowerCase().includes(q) || (i.desc || '').toLowerCase().includes(q)
      || (i.opcode != null && ('0x' + i.opcode.toString(16)).includes(q))
  })
  const hx = (v) => v == null ? '—' : '0x' + (v >>> 0).toString(16)

  // ---- LLK: build (on llk_lib) + load/run a perf kernel on a SET of Tensix cores ----
  let llkMeta = null, llkSel = [], llkTiles = 16, llkRunType = ''   // llkSel = [{x,y}] deploy set
  // what each PERF_RUN_TYPE isolates in the unpack(T0)→math(T1)→pack(T2) pipeline
  const RUN_TYPE_DESC = {
    MATH_ISOLATE: 'MATH_ISOLATE — only the MATH thread (T1) runs the compute loop; unpack just sets src-valids and pack returns early. Measures the FPU/SFPU math rate in isolation (the engine ceiling).',
    UNPACK_ISOLATE: 'UNPACK_ISOLATE — only the UNPACK thread (T0) runs; math clears valids, pack returns. Measures L1→SrcA/B unpack throughput in isolation.',
    PACK_ISOLATE: 'PACK_ISOLATE — only the PACK thread (T2) runs; unpack and math return early. Measures Dest→L1 pack throughput in isolation.',
    L1_TO_L1: 'L1_TO_L1 — the full pipeline unpack→math→pack end-to-end (the realistic tile flow); measures combined steady-state throughput.',
    L1_CONGESTION: 'L1_CONGESTION — unpack and pack hammer L1 concurrently (math clears valids) to expose L1 bandwidth contention.',
  }
  $: runTypeTip = 'Perf run type — isolate one stage of the unpack→math→pack pipeline to measure it.\n\n'
    + (RUN_TYPE_DESC[llkRunType] || 'pick a mode to see what it measures.')
  let llkBuilding = false, llkRunning = false, llkBuild = null, llkRuns = [], llkErr = ''
  $: if (active?.llk) loadLlkMeta(active.llk)
  async function loadLlkMeta(name) {
    try {
      const d = await getJSON('/api/tensix/llk')
      llkMeta = (d.kernels || []).find((k) => k.name === name) || null
      llkRunType = llkMeta?.default_run_type || (llkMeta?.perf_run_types || [])[0] || ''
      llkBuild = null; llkRuns = []; llkErr = ''
    } catch (e) { llkMeta = null }
  }

  // ---- overlay metadata (Build tab artifacts/flags/telemetry for bootloader overlays) ----
  let ovMeta = null
  $: if (active?.overlay) loadOvMeta(active.overlay)
  async function loadOvMeta(name) {
    try { const d = await getJSON('/api/tensix/bl/overlays'); ovMeta = (d.overlays || []).find((o) => o.name === name) || null }
    catch (e) { ovMeta = null }
  }

  // ---- context-aware Disasm (overlay .elf / llk ELFs / metal by build hash) ----
  let ovDisasm = null, llkDisasm = null
  async function loadCtxDisasm() {
    if (kind === 'metal') { if (!disasm) loadDisasm(); return }
    if (kind === 'overlay' && active?.overlay) { ovDisasm = null; try { ovDisasm = await getJSON(`/api/tensix/bl/disasm?name=${encodeURIComponent(active.overlay)}`) } catch (e) { ovDisasm = { ok: false, error: String(e) } } }
    if (kind === 'llk' && active?.llk) { llkDisasm = null; try { llkDisasm = await getJSON(`/api/tensix/llk/${encodeURIComponent(active.llk)}/disasm`) } catch (e) { llkDisasm = { ok: false, error: String(e) } } }
  }
  async function llkDoBuild() {
    if (!active?.llk) return
    llkBuilding = true; llkErr = ''; llkRuns = []
    try { llkBuild = await postJSON('/api/tensix/llk/build', { name: active.llk, run_type: llkRunType || null }) }
    catch (e) { llkErr = String(e.message || e) } finally { llkBuilding = false }
  }
  async function llkDoRun() {
    if (!active?.llk || !llkSel.length) { llkErr = 'select one or more cores ▲'; return }
    llkRunning = true; llkErr = ''; llkRuns = []
    try {
      for (const c of llkSel) {                       // build+load+run on each selected core
        const r = await postJSON('/api/tensix/llk/run', { name: active.llk, x: c.x, y: c.y, tile_cnt: +llkTiles, run_type: llkRunType || null })
        llkRuns = [...llkRuns, { core: `${c.x},${c.y}`, ...r }]
        if (r.build_log && !llkBuild) llkBuild = { ok: r.stage !== 'build', log: r.build_log }
      }
    } catch (e) { llkErr = String(e.message || e) } finally { llkRunning = false }
  }

  function onTip(e) {
    const el = e.target.closest('[data-tip]')
    if (el) tip = { show: true, x: e.clientX, y: e.clientY, text: el.getAttribute('data-tip') }
    else if (tip.show) tip = { show: false, x: 0, y: 0, text: '' }
  }
  async function run() {
    if (running || busy || !example || resetNeeded) return
    if (dirty) await onSave()
    try {
      const r = await postJSON('/api/tlab/run', { name: example })
      if (!r.ok) { status = 'run: ' + (r.error || 'failed to start'); return }
      running = true; result = null; status = `running ${short(example)}… (JIT-recompiles edited kernels)`; tab = 'deploy'; occOpen = true; poll()
    } catch (e) { status = 'run error: ' + e }
  }
  function poll() {
    cancelRun = pollJob('/api/tlab/last', (d) => {
      running = false
      if (d.error) { status = 'poll error: ' + d.error; return }
      result = d.result
      status = result?.passed ? `passed (${result.secs}s)` : `done (${result?.secs ?? '?'}s)`
      loadStatus(); if (tab === 'disasm') loadDisasm(); dispatch('ran')
    }, 2500)
  }
</script>

<div class="tabs">
  <button class:on={tab === 'deploy'} on:click={() => tab = 'deploy'}>Deploy</button>
  <button class:on={tab === 'build'} on:click={() => { tab = 'build'; if (kind === 'metal') { if (!buildlog) loadBuildLog(); loadRecipe() } }}>Build</button>
  <button class:on={tab === 'disasm'} on:click={() => { tab = 'disasm'; loadCtxDisasm() }}>Disasm</button>
  <button class:on={tab === 'isa'} on:click={() => { tab = 'isa'; loadIsaTab() }}>ISA</button>
  <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
  <span class="sp"></span>
  <span class="kindtag {kind}">{kind}</span>
  {#if kind === 'metal'}<button class="run" on:click={run} disabled={running || busy || resetNeeded} title="JIT-recompile + run on Tensix">{running ? 'Running…' : `Run ▸ ${short(example)}`}</button>{/if}
</div>
<!-- svelte-ignore a11y-no-static-element-interactions -->
<div class="tabbody" on:mousemove={onTip} on:mouseleave={() => tip.show = false}>

  <!-- ============================ DEPLOY ============================ -->
  {#if tab === 'deploy'}
    {#if kind === 'overlay'}
      <div style="height:100%;min-height:520px"><BootloaderPanel {preselect} {content} {dirty} onSave={onSave} /></div>

    {:else if kind === 'llk'}
      <h4>Deploy LLK kernel <span class="dim">· build on llk_lib → load onto a Tensix core → run (TRISC boot)</span></h4>
      {#if llkMeta}
        <div class="llkmeta">
          {#each Object.entries(llkMeta.trisc || {}) as [t, d]}
            <div class="thr"><span class="thrn">{t}</span><span class="dim">{(d.llk_headers || []).join(', ')}</span></div>
          {/each}
        </div>
      {/if}
      {#if llkMeta?.buildable === false}<div class="note">⚠ this kernel needs a per-variant build.h (Operand / special types) — the auto-generated default doesn't compile it yet.</div>{/if}
      <CorePicker bind:selected={llkSel} />
      <div class="llkbar">
        <label class="param" data-tip={runTypeTip}><span>run type</span>
          <select bind:value={llkRunType}>{#each llkMeta?.perf_run_types || [] as rt}<option value={rt}>{rt}</option>{/each}</select>
          <span class="help" data-tip={runTypeTip}>ⓘ</span></label>
        <label class="param"><span>tiles</span><input type="number" bind:value={llkTiles} /></label>
        <button class="run go" on:click={llkDoRun} disabled={llkRunning || busy || resetNeeded || !llkSel.length}>{llkRunning ? 'running…' : `Build + Load + Run ▸ (${llkSel.length} core${llkSel.length === 1 ? '' : 's'})`}</button>
      </div>
      {#if resetNeeded}<div class="note">NoC hang pending — run <code>tt-smi -r 0</code> + restart server.</div>{/if}
      {#if llkErr}<div class="msg bad">{llkErr}</div>{/if}
      {#each llkRuns as r}
        <div class="runhd">core <b>{r.core}</b> · {r.tile_cnt ?? llkTiles} tiles · <span class:acc={r.ok} class:bad={!r.ok}>{r.status || (r.stage === 'build' ? 'build failed' : 'error')}</span></div>
        {#if r.threads}
          <div class="thrcards">
            {#each Object.entries(r.threads) as [t, s]}
              <div class="thrcard" class:done={s.done} class:fail={!s.done}>
                <div class="tcn">{t}</div><div class="tcs">{s.done ? 'KERNEL_COMPLETE' : 'no-ack'}</div><div class="tcm dim">mbox {s.mailbox_hex}</div>
              </div>
            {/each}
          </div>
        {/if}
      {/each}

    {:else}
      <!-- metal: pick a core(s); single → RTA/go dashboard; Run (JIT, top-right); collapsible Occupancy -->
      <CorePicker bind:selected={metalSel} />
      {#if metalSel.length === 1}
        <div class="selhd">core <b>{metalSel[0].x},{metalSel[0].y}</b> <span class="dim">· poke runtime args + re-go, watch L1</span></div>
        {#key `${metalSel[0].x},${metalSel[0].y}`}<TensixLaunch x={metalSel[0].x} y={metalSel[0].y} />{/key}
      {:else if metalSel.length > 1}
        <div class="dim pad">{metalSel.length} cores selected — pick a <b>single</b> core for the runtime-arg dashboard. Use <b>Run ▸</b> (top-right) to JIT-run the example.</div>
      {:else}
        <div class="dim pad">Pick a core ▲ for its runtime-arg dashboard; or hit <b>Run ▸</b> (top-right) to JIT-run <b>{short(example) || 'the example'}</b>.</div>
      {/if}
      <details class="occ" bind:open={occOpen}>
        <summary>Occupancy <span class="dim">· per-engine busy cycles from the last Run</span></summary>
        {#if running}
          <div class="note">running on Tensix… per-engine profile appears when it finishes.</div>
        {:else if compute && cores.length}
          <div class="ranhd">ran <b>{ranName}</b> on <b>{compute.n_cores}</b> core{compute.n_cores > 1 ? 's' : ''} · avg MATH <b class="acc">{(compute.avg_math_occ * 100).toFixed(1)}%</b></div>
          {#each cores as [core, c]}
            <div class="core">
              <div class="corehd">core ({core}) <span class="mathocc" class:hot={c.math_occ > 0.5}>MATH {(c.math_occ * 100).toFixed(1)}%</span>
                <span class="dim">{c.math_occ > 0.5 ? 'compute-bound' : 'memory-bound'}</span></div>
              {#each ENGINES as engd}
                {@const cyc = c.engines[engd.key] || 0}
                <div class="erow" class:math={engd.key === 'math'}>
                  <span class="elabel">{engd.label}<span class="esub">{engd.sub}</span></span>
                  <span class="ebar"><span class="efill" style="width:{pct(cyc, c.wall)}%; background:{engd.col}"></span></span>
                  <span class="ecyc">{fmtCyc(cyc)}</span>
                </div>
              {/each}
            </div>
          {/each}
          {#if result?.dprint?.length}<h5>DPRINT</h5><pre class="log">{result.dprint.join('\n')}</pre>{/if}
        {:else if result}<div class="dim pad">Ran, no compute zones parsed. {result.error || ''}</div>
        {:else}<div class="dim pad">Edit a <b>compute</b> kernel, then <b>Run ▸</b> — per-engine busy cycles appear here.</div>{/if}
      </details>
    {/if}

  <!-- ============================ BUILD ============================ -->
  {:else if tab === 'build'}
    {#if kind === 'overlay'}
      <h4>Build <span class="dim">· bootloader overlay · BRISC blob</span></h4>
      {#if ovMeta}
        <div class="artifacts">
          <div class="art"><span class="al">engine</span><span>{ovMeta.engine}</span></div>
          <div class="art"><span class="al">verified</span><span class="vb {ovMeta.verified}">{ovMeta.verified}</span></div>
          <div class="art"><span class="al">artifact</span><span>{ovMeta.built ? `${ovMeta.bytes} B` : 'not built'}</span></div>
          <div class="art"><span class="al">hash</span><span class="hash">{ovMeta.hash || '—'}</span></div>
        </div>
        {#if ovMeta.params?.length}<h5>params <span class="dim">(flags poked at deploy)</span></h5>
          <table class="iargs"><tr><th>#</th><th>name</th><th>default</th><th>desc</th></tr>
            {#each ovMeta.params as p}<tr><td class="bits">{p.i}</td><td class="fn">{p.name}</td><td>{p.default}</td><td class="fd">{p.desc}</td></tr>{/each}
          </table>{/if}
        {#if ovMeta.telemetry?.length}<h5>telemetry</h5>
          <table class="iargs"><tr><th>slot</th><th>name</th><th>kind</th><th>desc</th></tr>
            {#each ovMeta.telemetry as t}<tr><td class="bits">{t.slot}</td><td class="fn">{t.name}</td><td class="ft dim">{t.kind}</td><td class="fd">{t.desc}</td></tr>{/each}
          </table>{/if}
        <div class="dim pad">Compile + live telemetry are on the <b>Deploy</b> tab (edits in the editor recompile on Run).</div>
      {:else}<div class="dim pad">overlay metadata loading…</div>{/if}

    {:else if kind === 'llk'}
      <h4>Build <span class="dim">· on llk_lib · per-thread TRISC ELFs</span></h4>
      <div class="llkbar">
        <label class="param" data-tip={runTypeTip}><span>run type</span>
          <select bind:value={llkRunType}>{#each llkMeta?.perf_run_types || [] as rt}<option value={rt}>{rt}</option>{/each}</select>
          <span class="help" data-tip={runTypeTip}>ⓘ</span></label>
        <button class="run go" on:click={llkDoBuild} disabled={llkBuilding}>{llkBuilding ? 'building…' : 'Build ⚙'}</button>
      </div>
      {#if llkMeta?.defines?.length}<div class="dim defs">defines: {llkMeta.defines.join(', ')}</div>{/if}
      {#if llkBuild}
        {#if llkBuild.flags}<h5>flags</h5><pre class="log small">{llkBuild.flags}</pre>{/if}
        {#if llkBuild.artifacts?.length}<h5>artifacts</h5>
          <table class="iargs"><tr><th>thread</th><th>file</th><th>bytes</th><th>sha256[:12]</th></tr>
            {#each llkBuild.artifacts as a}<tr><td class="fn">{a.thread}</td><td>{a.file}</td><td>{a.bytes}</td><td class="hash">{a.sha}</td></tr>{/each}
          </table>{/if}
        <details class="eng" open={!llkBuild.ok}>
          <summary><b class:acc={llkBuild.ok} class:bad={!llkBuild.ok}>build log {llkBuild.ok ? '✓' : '✗'}</b></summary>
          <pre class="log">{llkBuild.log}</pre>
        </details>
      {:else}<div class="dim pad">Hit <b>Build ⚙</b> — compiles+links the {Object.keys(llkMeta?.trisc || {}).length} TRISC ELFs on llk_lib (host, no device).</div>{/if}

    {:else}
      <h4>Build <span class="dim">· verbose JIT compile of the last Run</span>
        <button class="mini" on:click={loadBuildLog} title="refresh">⟳</button>
        <button class="rebuild" on:click={forceRebuild} disabled={rebuilding} title="delete cached builds so the next Run recompiles from source">{rebuilding ? '…' : '♻ Force rebuild'}</button>
      </h4>
      <div class="note">tt-metal <b>JIT-compiles at Run</b> from the programming_examples you edit in place. If an edit seems ignored it was served from cache → <b>Force rebuild</b>.</div>
      <div class="sb">
        <div class="sbhd">Standalone build <span class="dim">· {exShort || '—'} · compile from your bhtop copy, no device</span></div>
        <div class="sbrow">
          <button on:click={doExtract} disabled={exBusy || !exShort}>Extract → bhtop</button>
          <button class="run" on:click={doBuild} disabled={exBusy || !exShort}>{exBusy ? '…' : 'Build standalone ▸'}</button>
          {#if recipeInfo}<span class="dim">{recipeInfo.have ? `recipe ✓ (${recipeInfo.units} unit${recipeInfo.units === 1 ? '' : 's'})` : 'no recipe — Run once to capture it'}</span>{/if}
        </div>
        {#if exResult?.units}
          {#each exResult.units as u}
            <details class="eng" open={!u.ok}>
              <summary><b class:acc={u.ok} class:bad={!u.ok}>{u.target}</b> {u.ok ? '✓' : '✗'}{#if u.source}<span class="src"> · {u.source.split('/').pop()}</span>{/if}{#if u.elf}<span class="dim"> · {u.elf.split('/').slice(-3).join('/')}</span>{/if}</summary>
              {#if u.log}<h5>build log</h5><pre class="log">{u.log}</pre>{/if}
              {#if u.symbols}<h5>symbols</h5><pre class="log">{u.symbols}</pre>{/if}
              {#if u.disasm}<h5>disassembly</h5><pre class="log">{u.disasm}</pre>{/if}
            </details>
          {/each}
        {:else if exResult && !exResult.ok}<div class="dim pad">{exResult.error}</div>{/if}
      </div>
      <h5>cache build logs <span class="dim">· from the last Run's JIT</span></h5>
      {#if buildlog?.ok}
        <div class="dim" style="margin:8px 0 4px">program <b>{buildlog.program_id}</b> · {buildlog.kernels.length} kernel(s)</div>
        {#each buildlog.kernels as k}
          <details class="eng" open={k.log && !k.log.startsWith('(')}>
            <summary><b>{k.name}</b> <span class="src">{k.source}</span> <span class="hash">{k.hash?.slice(0, 8)}</span></summary>
            <pre class="log">{k.log}</pre>
          </details>
        {/each}
      {:else}<div class="dim pad">{buildlog?.error || 'Run a compute example, then the verbose compile log appears here.'}</div>{/if}
    {/if}

  <!-- ============================ DISASM ============================ -->
  {:else if tab === 'disasm'}
    {#if kind === 'overlay'}
      <h4>Disassembly <span class="dim">· overlay {active?.overlay} · hover any op/register</span></h4>
      {#if ovDisasm?.ok}<div class="disasm">{@html renderDisasm(ovDisasm.text)}</div>
      {:else}<div class="dim pad">{ovDisasm?.error || 'Build the overlay (kernels/tensix/overlays/build.sh), then objdump appears here.'}</div>{/if}

    {:else if kind === 'llk'}
      <h4>Disassembly <span class="dim">· {active?.llk} · per TRISC thread</span></h4>
      {#if llkDisasm?.ok}
        {#each Object.entries(llkDisasm.threads) as [t, text]}
          <details class="eng" open={t === 'MATH'}>
            <summary><b class:acc={t === 'MATH'}>{t}</b> <span class="dim">{text.split('\n').length} lines</span></summary>
            <div class="disasm">{@html renderDisasm(text)}</div>
          </details>
        {/each}
      {:else}<div class="dim pad">{llkDisasm?.error || 'Build the kernel first (Build tab / Deploy), then per-thread disasm appears here.'}</div>{/if}

    {:else}
      <h4>Kernel disassembly <span class="dim">· by build hash · hover any op/register</span></h4>
      {#if disasm?.ok}
        <div class="dim" style="margin-bottom:6px">kernel <b>{disasm.kernel}</b>{#if disasm.program_id != null} · program {disasm.program_id}{/if}</div>
        {#each disasm.engines.filter((e) => e.present) as e}
          <details class="eng" open={e.role === 'math'}>
            <summary><b class:acc={e.role === 'math'}>{e.label}</b> <span class="dim">{e.disasm.split('\n').length} lines</span></summary>
            <div class="disasm">{@html renderDisasm(e.disasm)}</div>
          </details>
        {/each}
        {#if !disasm.engines.some((e) => e.present)}<div class="dim pad">No compiled ELFs yet — Run a compute example.</div>{/if}
      {:else}<div class="dim pad">{disasm?.error || 'Run a compute example, then per-engine disassembly appears here.'}</div>{/if}
    {/if}

  {:else if tab === 'isa'}
    <h4>Tensix ISA <span class="dim">· {isa?.count ?? '…'} instructions from assembly.yaml · hover TT_OP_* in the editor too</span></h4>
    {#if isa && !isa.available}
      <div class="dim pad">{isa.error || 'assembly.yaml not found — set TT_METAL_HOME / install tt-metal.'}</div>
    {:else if isa}
      <div class="isabar">
        <input class="isaq" placeholder="search name / desc / opcode…" bind:value={isaQ} />
        <div class="isacats">
          {#each isaCats as c}
            <button class="catchip" class:on={isaCat === c} on:click={() => isaCat = c}>{c}</button>
          {/each}
        </div>
      </div>
      <div class="dim" style="margin:2px 0 6px">{isaShown.length} shown</div>
      <div class="isalist">
        {#each isaShown as i (i.name)}
          <div class="isarow" class:open={isaOpen[i.name]}>
            <button class="isahd" on:click={() => isaOpen = { ...isaOpen, [i.name]: !isaOpen[i.name] }}>
              <span class="im">{i.name}</span>
              <span class="iop">{hx(i.opcode)}</span>
              <span class="iu">{i.unit || ''}</span>
              <span class="icat">{i.category}</span>
              <span class="idesc dim" data-tip={i.desc}>{i.desc}</span>
            </button>
            {#if isaOpen[i.name]}
              {#if i.args?.length}
                <table class="iargs">
                  <tr><th>bits</th><th>field</th><th>type</th><th>description</th></tr>
                  {#each i.args as a}
                    <tr><td class="bits">{bits(a)}</td><td class="fn">{a.name}</td><td class="ft dim">{a.field_type || ''}</td><td class="fd">{a.desc}</td></tr>
                  {/each}
                </table>
              {:else}<div class="dim pad">no operands</div>{/if}
            {/if}
          </div>
        {/each}
      </div>
    {:else}<div class="dim pad">loading ISA…</div>{/if}

  {:else}
    <DocsPane docsUrl="/api/tlab/docs" docUrl={(id) => `/api/tlab/doc/${id}`} />
  {/if}
  <div class="foot dim">{status}</div>
</div>
{#if tip.show}<div class="tip" style="left:{tip.x + 14}px; top:{tip.y + 16}px">{tip.text}</div>{/if}

<style>
  .tabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); align-items: center; }
  .tabs button { font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .tabs .sp { flex: 1; }
  .tabs .run { margin: 0 6px; background: var(--accent); color: #1a1206; border: 1px solid var(--accent); border-radius: 5px; padding: 4px 10px; font-weight: 600; border-bottom: 1px solid var(--accent); }
  .tabs .run:disabled { opacity: 0.4; cursor: default; }
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
  .disasm { background: #0a0c10; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; overflow: auto; max-height: 320px; font-size: 10.5px; line-height: 1.6; font-family: ui-monospace, Menlo, Consolas, monospace; margin-top: 4px; }
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
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 340px; font-size: 11px; line-height: 1.45; white-space: pre; }
  .mini { background: var(--panel2); border: 1px solid var(--line); color: var(--fg); border-radius: 4px; cursor: pointer; font-size: 11px; line-height: 1; padding: 2px 6px; font-family: inherit; }
  .rebuild { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 2px 9px; cursor: pointer; margin-left: auto; }
  .rebuild:disabled { opacity: 0.5; cursor: default; }
  details.eng .src { color: var(--muted); font-family: ui-monospace, monospace; font-size: 10.5px; }
  details.eng .hash { color: var(--good); font-family: ui-monospace, monospace; font-size: 10px; }
  details.eng .bad { color: var(--bad); }
  .sb { border: 1px solid var(--accent); border-radius: 6px; padding: 9px 11px; margin: 10px 0; }
  .sbhd { font-size: 12px; font-weight: 600; margin-bottom: 6px; }
  .sbrow { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
  .sbrow button { font-family: inherit; font-size: 11.5px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 10px; cursor: pointer; }
  .sbrow .run { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .sbrow button:disabled { opacity: 0.5; cursor: default; }
  .foot { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--line); font-size: 11px; }

  /* unified shell */
  .kindtag { font-size: 9px; text-transform: uppercase; letter-spacing: .04em; padding: 1px 7px; border-radius: 10px; border: 1px solid var(--line); color: var(--muted); margin-right: 8px; }
  .kindtag.overlay { color: var(--good); border-color: var(--good); }
  .kindtag.llk { color: #4fd6e0; border-color: #4fd6e0; }
  .kindtag.metal { color: var(--noc1); border-color: var(--noc1); }
  .artifacts { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }
  .art { border: 1px solid var(--line); border-radius: 6px; padding: 6px 10px; display: flex; flex-direction: column; gap: 2px; background: var(--panel); }
  .art .al { font-size: 9px; color: var(--muted); text-transform: uppercase; }
  .art .hash { font-family: ui-monospace, monospace; color: var(--good); font-size: 11px; }
  .art .vb { font-size: 10px; }
  details.occ { margin-top: 12px; border: 1px solid var(--line); border-radius: 8px; padding: 8px 11px; background: var(--panel); }
  details.occ > summary { cursor: pointer; font-weight: 600; font-size: 12px; }
  .log.small { font-size: 10px; max-height: 80px; }
  .vb { font-size: 9px; padding: 1px 5px; border-radius: 3px; text-transform: uppercase; border: 1px solid var(--line); color: var(--muted); }
  .vb.ok { color: var(--good); border-color: var(--good); }
  .vb.wedges { color: #e07a77; border-color: #c0504d; }
  .vb.untested { color: #d8a23a; border-color: #d8a23a; }

  /* LLK tab */
  .llkhd { display: flex; align-items: baseline; gap: 10px; margin: 6px 0; }
  .llkmeta { border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; margin-bottom: 10px; background: var(--panel); }
  .thr { display: flex; gap: 8px; font-size: 11px; padding: 1px 0; }
  .thrn { color: #4fd6e0; font-family: ui-monospace, monospace; width: 60px; flex: none; text-transform: lowercase; }
  .defs { margin-top: 4px; font-size: 10.5px; }
  .llkbar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; border: 1px solid var(--accent); border-radius: 6px; padding: 10px; margin: 6px 0; }
  .llkbar .sep { color: var(--muted); }
  .llkbar .param { display: flex; align-items: center; gap: 6px; font-size: 12px; }
  .llkbar .param span { color: var(--muted); }
  .llkbar .param input { width: 52px; font-family: inherit; background: #0a0c10; color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 3px 5px; }
  .llkbar .param select { font-family: inherit; font-size: 11.5px; background: #0a0c10; color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 3px 5px; }
  .llkbar .param .help { color: var(--muted); cursor: help; font-size: 11px; }
  .llkbar .param .help:hover { color: var(--accent); }
  .llkbar .run { background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 4px 11px; cursor: pointer; font-family: inherit; font-size: 12px; }
  .llkbar .run.go { background: var(--accent); color: #1a1206; font-weight: 600; }
  .llkbar .run:disabled { opacity: .5; cursor: default; }
  .runhd { font-size: 12px; margin: 10px 0 6px; }
  .runhd .acc { color: var(--good); } .runhd .bad { color: #e07a77; }
  .thrcards { display: flex; gap: 8px; flex-wrap: wrap; }
  .thrcard { border: 1px solid var(--line); border-radius: 6px; padding: 8px 12px; min-width: 100px; }
  .thrcard.done { border-color: var(--good); }
  .thrcard.fail { border-color: #c0504d; }
  .tcn { font-family: ui-monospace, monospace; font-size: 11px; color: var(--accent); }
  .tcs { font-size: 11px; margin-top: 2px; }
  .thrcard.done .tcs { color: var(--good); } .thrcard.fail .tcs { color: #e07a77; }
  .tcm { font-size: 10px; margin-top: 2px; }

  /* ISA tab */
  .isabar { display: flex; flex-direction: column; gap: 7px; margin: 8px 0; }
  .isaq { font-family: inherit; font-size: 12px; background: #0a0c10; color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 5px 9px; }
  .isacats { display: flex; flex-wrap: wrap; gap: 4px; }
  .catchip { font-family: inherit; font-size: 10.5px; background: var(--panel2); color: var(--muted); border: 1px solid var(--line); border-radius: 10px; padding: 1px 9px; cursor: pointer; }
  .catchip.on { color: var(--accent); border-color: var(--accent); }
  .isalist { display: flex; flex-direction: column; gap: 3px; }
  .isarow { border: 1px solid var(--line); border-radius: 6px; background: var(--panel); }
  .isarow.open { border-color: var(--accent); }
  .isahd { width: 100%; display: grid; grid-template-columns: 110px 56px 60px 90px 1fr; align-items: baseline; gap: 8px; text-align: left; background: none; border: none; color: var(--fg); cursor: pointer; font-family: inherit; padding: 5px 9px; }
  .isahd:hover { background: rgba(255,255,255,0.03); }
  .im { font-family: ui-monospace, monospace; font-weight: 600; color: var(--accent); font-size: 12px; }
  .iop { font-family: ui-monospace, monospace; color: #ff8a4c; font-size: 11px; }
  .iu { color: var(--muted); font-size: 10.5px; }
  .icat { color: #4fd6e0; font-size: 10px; }
  .idesc { font-size: 10.5px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .iargs { width: 100%; border-collapse: collapse; font-size: 11px; margin: 2px 0 6px; }
  .iargs th { text-align: left; color: var(--muted); font-weight: 500; font-size: 10px; padding: 2px 8px; border-bottom: 1px solid var(--line); }
  .iargs td { padding: 2px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
  .iargs .bits { font-family: ui-monospace, monospace; color: #4fd6e0; white-space: nowrap; }
  .iargs .fn { font-family: ui-monospace, monospace; color: #ffd24a; white-space: nowrap; }
  .iargs .fd { color: #c8ced8; }

  /* Launch tab — core picker grid */
  .lhead { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .scanbtn { font-family: inherit; font-size: 11.5px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 3px 11px; cursor: pointer; }
  .scanbtn:disabled { opacity: 0.5; cursor: default; }
  .sw { display: inline-block; width: 9px; height: 9px; border-radius: 2px; vertical-align: middle; }
  .sw.res { border: 1px solid var(--good); background: rgba(121,212,121,0.25); }
  .cgrid { display: flex; flex-direction: column; gap: 3px; overflow: auto; padding-bottom: 6px; }
  .crow { display: flex; gap: 3px; }
  .cell { position: relative; font-family: ui-monospace, monospace; font-size: 9.5px; min-width: 34px; padding: 4px 2px; border-radius: 4px; cursor: pointer; color: var(--muted);
          border: 1px solid var(--line); background: rgba(235,110,70, calc(var(--h, 0) * 0.55)); transition: background 0.25s; }
  .cell:hover { color: var(--fg); border-color: var(--accent); }
  .cell.resident { border-color: var(--good); color: var(--fg); box-shadow: inset 0 0 0 1px var(--good); }
  .cell.infra { border-color: var(--muted); color: var(--muted); border-style: dashed; }
  .cell.err { border-color: var(--bad); opacity: 0.5; }
  .cell.sel { outline: 2px solid var(--accent); outline-offset: 1px; color: var(--fg); }
  .selhd { margin: 12px 0 6px; font-size: 12px; } .selhd b { color: var(--accent); }
  .rlhd { margin: 12px 0 5px; }
  .reslist { display: flex; flex-direction: column; gap: 3px; }
  .resrow { display: flex; align-items: baseline; gap: 10px; text-align: left; font-family: inherit; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; padding: 4px 9px; cursor: pointer; color: var(--fg); }
  .resrow:hover { border-color: var(--accent); }
  .resrow.sel { border-color: var(--accent); background: rgba(235,110,70,0.1); }
  .resrow .rxy { font-family: ui-monospace, monospace; font-size: 11px; color: var(--muted); width: 44px; flex: none; }
  .resrow .rk { color: var(--good); font-family: ui-monospace, monospace; font-size: 11.5px; flex: 1; }
  .resrow.infra .rk { color: var(--muted); }
  .resrow .itag { font-size: 9px; color: var(--muted); border: 1px solid var(--line); border-radius: 3px; padding: 0 4px; }
  .resrow .rgo { color: var(--muted); font-size: 10.5px; }
</style>
