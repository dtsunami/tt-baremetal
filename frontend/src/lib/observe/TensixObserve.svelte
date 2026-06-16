<!-- TensixObserve — the TENSIX (compute) right pane: Run + per-engine Occupancy / Disasm / Docs.
     Ported from ComputeLab; actions live here (the editor stays generic). The example to run is
     the section selection carried on `active.sel`. -->
<script>
  import { onMount, onDestroy, createEventDispatcher } from 'svelte'
  import DocsPane from '../DocsPane.svelte'
  import { getJSON, postJSON, pollJob } from '../api.js'
  import { renderDisasm } from '../riscv.js'
  import { frame } from '../stores.js'

  export let active           // {engine, key, name, sel}
  export let dirty = false
  export let onSave = () => {}     // parent saves its editor content before Run (JIT reads disk)

  const dispatch = createEventDispatcher()

  let running = false, result = null, status = 'ready', cancelRun = null
  let tab = 'occ', deployed = {}, disasm = null
  let tip = { show: false, x: 0, y: 0, text: '' }

  $: example = active?.sel || ''
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
    try { const d = await getJSON('/api/tlab/last'); result = d.result; if (d.running) { running = true; poll() } } catch (e) {}
  })
  onDestroy(() => cancelRun?.())

  async function loadStatus() { try { deployed = (await getJSON('/api/tlab/status')).deployed || {} } catch (e) {} }
  async function loadDisasm() { try { disasm = await getJSON('/api/tlab/disasm') } catch (e) { disasm = { ok: false, error: String(e) } } }
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
      running = true; result = null; status = `running ${short(example)}… (JIT-recompiles edited kernels)`; tab = 'occ'; poll()
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
  <button class:on={tab === 'occ'} on:click={() => tab = 'occ'}>Occupancy</button>
  <button class:on={tab === 'disasm'} on:click={() => { tab = 'disasm'; if (!disasm) loadDisasm() }}>Disasm</button>
  <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
  <span class="sp"></span>
  <button class="run" on:click={run} disabled={running || busy || resetNeeded} title="JIT-recompile + run on Tensix">{running ? 'Running…' : `Run ▸ ${short(example)}`}</button>
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
    {:else}<div class="dim pad">Edit a <b>compute</b> kernel, then <b>Run ▸</b> — per-engine busy cycles per Tensix core appear here (MATH occupancy = compute- vs memory-bound).</div>{/if}

  {:else if tab === 'disasm'}
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
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 200px; font-size: 11px; line-height: 1.45; white-space: pre; }
  .foot { margin-top: 12px; padding-top: 8px; border-top: 1px solid var(--line); font-size: 11px; }
</style>
