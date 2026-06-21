<!-- DeviceBrowser — the unified device-hierarchy browser. One LabShell (resizable):
       left   = DeviceTree (NOC / X280 / TENSIX / DRAM·ETH soon, files-only, caps-aware ops)
       center = shared CodeEditor (Save + dirty + lang)
       right  = the selected engine's observe pane (actions + tabs) — hart-lab style
     Selecting a file switches the right pane to that engine's view. Running ● badges in the
     tree come from /api/running (kernels matched by JIT build hash). -->
<script>
  import { onMount, onDestroy } from 'svelte'
  import LabShell from '../lib/LabShell.svelte'
  import DeviceTree from '../lib/DeviceTree.svelte'
  import CodeEditor from '../lib/CodeEditor.svelte'
  import NocObserve from '../lib/observe/NocObserve.svelte'
  import TensixObserve from '../lib/observe/TensixObserve.svelte'
  import HartObserve from '../lib/observe/HartObserve.svelte'
  import ParamPanel from '../lib/ParamPanel.svelte'
  import { getJSON, postJSON } from '../lib/api.js'
  import { byKey } from '../lib/engines.js'
  import { loadIsa, lookup, tipHtml } from '../lib/isa.js'

  let deploying = false

  // Tensix ISA hover tooltips: load the opcode/bit-field map once, then make TT_OP_* mnemonics
  // in the editor self-describe (only while a Tensix source is open).
  let isaMnem = null
  loadIsa().then((d) => { isaMnem = d.mnemonics || {} })
  $: isaHover = (active?.engine === 'tensix' && isaMnem)
    ? (word) => tipHtml(lookup(isaMnem, word))
    : null

  let running = {}          // by_source map from /api/running
  let active = null         // {engine, key, name, sel}
  let content = '', saved = '', lang = 'cpp', role = ''
  let editor, status = 'select a file from the device tree', saving = false
  let runTick = 0           // bump to nudge the observe pane to refresh after a run

  $: dirty = content !== saved
  $: eng = active ? byKey[active.engine] : null
  $: stale = !!(active && running[active.name] && dirty)   // edited since the running build

  let pollT = null
  onMount(() => { loadRunning(); pollT = setInterval(loadRunning, 4000) })
  onDestroy(() => clearInterval(pollT))

  async function loadRunning() {
    try { const r = await getJSON('/api/running'); running = r.by_source || {} } catch (e) { /* offline */ }
  }
  function afterRun() { runTick++; loadRunning() }

  async function onSelect(e) {
    const { engine, key, name, sel, overlay, llk } = e.detail
    if (!engine) { active = null; return }
    if (dirty && !confirm('Discard unsaved changes?')) return
    if (llk) {                                        // LLK perf kernel: load its .cpp into the editor
      try {
        const f = await getJSON(`/api/tensix/llk/${encodeURIComponent(llk)}`)
        active = { engine: 'tensix', key, name, sel, llk }
        content = f.source; saved = f.source; lang = 'cpp'; role = 'llk'
        editor?.setDoc(f.source); editor?.setLang('cpp')
        status = `LLK perf kernel · ${name} · built on llk_lib (build.sh)`
      } catch (ex) {
        active = { engine: 'tensix', key, name, sel, llk }; content = ''; saved = ''
        status = 'open LLK kernel failed: ' + ex
      }
      return
    }
    if (overlay) {                                   // bootloader overlay: load its .c into the editor
      try {
        const f = await getJSON(`/api/tensix/bl/source?name=${encodeURIComponent(overlay)}`)
        active = { engine: 'tensix', key, name, sel, overlay }
        content = f.source; saved = f.source; lang = 'c'; role = 'overlay'
        editor?.setDoc(f.source); editor?.setLang('c')
        status = `bootloader overlay · ${name}`
      } catch (ex) {
        active = { engine: 'tensix', key, name, sel, overlay }; content = ''; saved = ''
        status = 'open overlay failed: ' + ex
      }
      return
    }
    const en = byKey[engine]
    try {
      const f = await getJSON(en.fileUrl(key))
      active = { engine, key, name, sel }
      content = f.content; saved = f.content
      role = f.role ?? f.lang ?? ''
      lang = en.lang(f) || f.lang || 'cpp'
      editor?.setDoc(f.content); editor?.setLang(lang)
      status = `${name} · ${engine.toUpperCase()}`
    } catch (ex) { status = 'open failed: ' + ex }
  }
  function onSel(e) { if (active && e.detail.engine === active.engine) active = { ...active, sel: e.detail.sel } }

  async function save() {
    if (!active || !dirty || saving) return
    saving = true
    try {
      if (active.overlay) await postJSON('/api/tensix/bl/source', { name: active.overlay, source: content })
      else if (active.engine === 'x280') await postJSON('/api/l2/file', { name: active.key, content })
      else if (active.engine === 'noc') await postJSON('/api/lab/file', { path: active.key, content })
      else await postJSON('/api/tlab/file', { path: active.key, content })
      saved = content; status = 'saved ' + active.name
    } catch (e) { status = 'save failed: ' + e } finally { saving = false }
  }

  // Deploy from the params panel: compile the editor buffer with the kernel's define-params,
  // load to the selected hart grouping (deploy_all takes any subset), then fire its live
  // mailbox ops. (X280 path; the panel only shows for engines with a paramsUrl.)
  async function onParamDeploy(e) {
    const { routed, done } = e.detail
    const report = (m) => { status = m; done?.(m) }
    if (active?.engine !== 'x280') return
    if (dirty) await save()
    const dep = routed.deploy, tile = dep.tile ?? 0
    const harts = (Array.isArray(dep.hart) ? dep.hart : [dep.hart]).filter((h) => h != null)
    if (!harts.length) return report('pick at least one hart')
    deploying = true; status = 'deploying…'
    try {
      const r = await postJSON('/api/l2/deploy_all', { tile, harts, content, lang,
        addr: dep.addr ?? 0x30008000, name: active.name, defines: routed.defines })
      if (r.ok === false) return report('deploy: ' + (r.error || r.stage || 'failed'))
      for (const m of routed.mailbox)
        for (const h of harts)
          await postJSON('/api/l2/cmd', { tile, hart: h, op: m.op, arg0: m.arg0, arg1: 0 })
      report(`deployed ${active.name} → tile ${tile} hart ${harts.join(',')}`); afterRun()
    } catch (ex) { report('deploy failed: ' + ex) } finally { deploying = false }
  }
</script>

<LabShell storeKey="devbrowser.layout">
  <DeviceTree slot="left" {running} activeKey={active?.key} on:select={onSelect} on:selchange={onSel} />

  <svelte:fragment slot="editor">
    <div class="toolbar">
      <span class="cur">{active?.name ?? '—'}{#if dirty}<b class="dt">●</b>{/if}</span>
      {#if active}<span class="role {role}">{role}</span>{/if}
      {#if stale}<span class="stale" title="this source was edited since the build that's running — re-Run to load it">stale vs running</span>{/if}
      <span class="sp"></span>
      <button on:click={save} disabled={!dirty || saving}>Save</button>
    </div>
    {#if active && eng?.paramsUrl}
      <ParamPanel {eng} fileKey={active.key} busy={deploying} on:deploy={onParamDeploy} />
    {/if}
    <div class="code-wrap"><CodeEditor bind:this={editor} {lang} hover={isaHover} onChange={(t) => content = t} onSave={save} /></div>
    <div class="statusbar"><span class="st">{status}</span></div>
  </svelte:fragment>

  <svelte:fragment slot="right">
    {#if !active}
      <div class="empty">
        <h3>RV Kernels — device hierarchy</h3>
        <p>Pick a source file on the left. The view switches to that engine:</p>
        <ul>
          <li><b class="noc">NOC</b> — data-movement kernels · footprint + telemetry</li>
          <li><b class="x280">X280</b> — L2CPU harts · deploy + per-hart telemetry</li>
          <li><b class="tensix">TENSIX</b> — compute engines · occupancy + disasm</li>
        </ul>
        <p class="dim">● = source is running now (matched by tt-metal build hash).</p>
      </div>
    {:else if active.engine === 'noc'}
      <NocObserve {active} {dirty} onSave={save} on:ran={afterRun} />
    {:else if active.engine === 'tensix'}
      <TensixObserve {active} preselect={active.overlay} {content} {dirty} onSave={save} on:ran={afterRun} />
    {:else if active.engine === 'x280'}
      <HartObserve {active} {content} {lang} {dirty} onSave={save} on:ran={afterRun} />
    {/if}
  </svelte:fragment>
</LabShell>

<style>
  .toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
  .cur { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .dt { color: var(--accent); margin-left: 4px; }
  .sp { flex: 1; }
  .toolbar button { font-family: inherit; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 4px 10px; cursor: pointer; }
  .toolbar button:disabled { opacity: 0.4; cursor: default; }
  .role { font-size: 9.5px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); }
  .stale { font-size: 10px; color: var(--bad); border: 1px solid var(--bad); border-radius: 3px; padding: 1px 6px; }
  .code-wrap { flex: 1; overflow: hidden; min-height: 0; background: #0a0c10; }
  .statusbar { display: flex; align-items: center; gap: 10px; padding: 4px 10px; border-top: 1px solid var(--line); background: var(--panel); font-size: 11px; color: var(--fg); }
  .empty { padding: 22px 18px; color: var(--muted); }
  .empty h3 { color: var(--fg); margin: 0 0 8px; font-size: 14px; }
  .empty ul { padding-left: 16px; line-height: 1.9; }
  .empty b.noc { color: var(--noc0); } .empty b.x280 { color: var(--accent); } .empty b.tensix { color: var(--noc1); }
  .empty .dim { margin-top: 10px; }
</style>
