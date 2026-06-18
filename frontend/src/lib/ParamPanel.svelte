<!-- ParamPanel — the per-kernel meta-param / config strip above the source (inline, collapsible).
     Two views: FORM (the kernel.json params grouped by KIND, with descriptions; x280 routes them on
     Deploy — define -> -D, deploy -> tile/hart(s)/addr, mailbox -> live cmd) and JSON (a raw editor
     for the kernel.json, edit + save). Engines without a deploy path (NOC/TENSIX) show no Deploy
     button and open straight into the JSON editor — their kernel.json is a bhtop overlay that's
     auto-created with a default the first time you open it. -->
<script>
  import { createEventDispatcher } from 'svelte'
  import { getJSON, postJSON, routeParams } from './api.js'

  export let eng                 // engine descriptor (must have paramsUrl)
  export let fileKey = null      // selected file key
  export let busy = false        // parent deploy in-flight

  const dispatch = createEventDispatcher()
  const GROUPS = [
    ['define', 'Compile-time', 'baked in at build via -D (recompiles on Deploy)'],
    ['ctarg', 'Compile-time args', 'tt-metal get_compile_time_arg_val(i) — set by the host program'],
    ['rtarg', 'Runtime args', 'tt-metal get_arg_val(i) — set by the host at run'],
    ['deploy', 'Deploy target', 'where + how the kernel loads'],
    ['mailbox', 'Live', 'mailbox ops — sent after load; tweak on the fly'],
  ]

  let meta = null, entry = null, kernel = null, values = {}, status = ''
  let open = true, view = 'form', jsonText = '', jsonStatus = '', merging = false
  $: canDeploy = !!eng?.cmdUrl                 // only x280 deploys from here
  $: hasConfig = !!eng?.configUrl
  $: canMerge = !!eng?.mergeUrl                // parse source -> populate params

  $: if (fileKey) load(fileKey)
  const clone = (v) => (Array.isArray(v) ? [...v] : v)

  async function load(key) {
    status = ''; jsonText = ''; jsonStatus = ''
    try {
      const d = await getJSON(eng.paramsUrl(key))
      meta = d.meta; entry = d.entry; kernel = d.kernel
      values = {}; for (const p of meta.params) values[p.name] = clone(p.default)
      view = meta.params.length ? 'form' : 'json'      // no form params -> straight to JSON editor
      if (view === 'json') loadJson()
    } catch (e) { status = 'load failed: ' + e; meta = null }
  }

  async function loadJson() {
    if (!hasConfig) return
    try { jsonText = (await getJSON(eng.configUrl(fileKey))).json } catch (e) { jsonStatus = 'load failed: ' + e }
  }
  function setView(v) { view = v; if (v === 'json' && !jsonText) loadJson() }
  async function saveJson() {
    try { await postJSON(eng.configSaveUrl, { key: fileKey, text: jsonText }); jsonStatus = 'saved ✓'; await load(fileKey) }
    catch (e) { jsonStatus = 'save failed: ' + e }
  }

  const byKind = (k) => (meta?.params || []).filter((p) => p.kind === k)
  $: routed = meta ? routeParams(meta.params, values) : { defines: {}, deploy: {}, mailbox: [] }

  function toggleMulti(p, c) {
    const arr = Array.isArray(values[p.name]) ? values[p.name] : []
    values[p.name] = arr.includes(c) ? arr.filter((x) => x !== c) : [...arr, c].sort((a, b) => a - b)
    values = values
  }
  const inArr = (p, c) => Array.isArray(values[p.name]) && values[p.name].includes(c)

  $: summary = meta ? (() => {
    if (canDeploy) {
      const d = routed.deploy, h = Array.isArray(d.hart) ? (d.hart.length ? d.hart.join(',') : 'none') : d.hart
      const defs = Object.entries(routed.defines).map(([k, v]) => `${k}=${v}`).join(' ')
      return `tile ${d.tile} · hart ${h}${defs ? ' · ' + defs : ''}`
    }
    return kernel || ''
  })() : ''

  function onDeploy() { status = 'deploying…'; dispatch('deploy', { routed, values, entry, kernel, done: (m) => (status = m) }) }
  async function sendLive(p) {
    const one = routeParams([p], values).mailbox[0]
    const d = routed.deploy, harts = Array.isArray(d.hart) ? d.hart : [d.hart]
    try {
      for (const h of (harts.length ? harts : [0]))
        await postJSON(eng.cmdUrl, { tile: d.tile ?? 0, hart: h, op: one.op, arg0: one.arg0, arg1: 0 })
      status = `sent ${p.name} = ${values[p.name]}`
    } catch (e) { status = 'send failed: ' + e }
  }
  async function saveDefaults() {
    try { await postJSON(eng.paramsSaveUrl, { key: fileKey, values }); status = 'saved as defaults ✓' }
    catch (e) { status = 'save failed: ' + e }
  }
  let mergeMsg = ''            // shown in the bar, so it's visible from EITHER view (Form or JSON)
  async function doMerge() {
    if (!canMerge || !fileKey) return
    merging = true; mergeMsg = 'merging…'
    try {
      const r = await postJSON(eng.mergeUrl, { key: fileKey })
      await load(fileKey)
      mergeMsg = r.count ? `merged ${r.count}: ${r.added.join(', ')}` : 'no new params found in source'
      status = mergeMsg
    } catch (e) { mergeMsg = 'merge failed: ' + e; status = mergeMsg }
    finally { merging = false }
  }
</script>

{#if meta}
  <div class="panel" class:open>
    <div class="bar">
      <button class="toggle" on:click={() => (open = !open)} title={open ? 'collapse' : 'expand'}>
        <span class="caret" class:open>▸</span> <b>Params</b>
        <span class="sm">{summary}</span>
      </button>
      {#if open && hasConfig}
        <div class="seg">
          <button class:on={view === 'form'} on:click={() => setView('form')}>Form</button>
          <button class:on={view === 'json'} on:click={() => setView('json')}>JSON</button>
        </div>
      {/if}
      {#if open && canMerge}
        {#if mergeMsg}<span class="mergemsg" title={mergeMsg}>{mergeMsg}</span>{/if}
        <button class="merge" on:click={doMerge} disabled={merging}
                title="parse the kernel source(s) and merge discovered params into kernel.json (idempotent — your edits are kept)">
          {merging ? 'merging…' : 'Merge ⤵'}
        </button>
      {/if}
      {#if canDeploy}
        <button class="run" on:click={onDeploy} disabled={busy} title="compile with these defines, load to the target, then send live ops">
          {busy ? 'deploying…' : 'Deploy ▸'}
        </button>
      {/if}
    </div>

    {#if open}
      <div class="body">
        {#if view === 'json'}
          <textarea class="json" bind:value={jsonText} spellcheck="false"
                    placeholder={hasConfig ? 'kernel.json' : 'no config for this engine'}></textarea>
          <div class="foot">
            <span class="st">{jsonStatus}</span><span class="sp"></span>
            <button on:click={saveJson} disabled={!hasConfig} title="validate + save kernel.json">Save JSON</button>
          </div>
        {:else}
          {#each GROUPS as [kind, title, blurb]}
            {#if byKind(kind).length}
              <div class="group">
                <div class="ghead">{title} <span class="dim">— {blurb}</span></div>
                {#each byKind(kind) as p}
                  <div class="prow">
                    <span class="pl" title={p.desc}>{p.name}{#if p.index !== undefined}<span class="ix">[{p.index}]</span>{/if}</span>
                    <div class="pin">
                      {#if kind === 'rtarg' || kind === 'ctarg'}
                        <!-- tt-metal args are host-owned: show discovered value read-only, no fake apply -->
                        <span class="roval">{p.default ?? '— (set by host)'}</span>
                      {:else if p.multi}
                        <div class="chips">{#each p.choices as c}<button class="chip" class:on={inArr(p, c)} on:click={() => toggleMulti(p, c)}>{c}</button>{/each}</div>
                      {:else if p.type === 'bool'}
                        <input type="checkbox" bind:checked={values[p.name]} />
                      {:else if p.type === 'enum' || (p.type === 'int' && p.choices)}
                        <select bind:value={values[p.name]}>{#each p.choices as c}<option value={c}>{c}</option>{/each}</select>
                      {:else if p.type === 'int'}
                        <input type="number" min={p.min} max={p.max} bind:value={values[p.name]} />
                      {:else}
                        <input type="text" bind:value={values[p.name]} spellcheck="false" />
                      {/if}
                      {#if kind === 'mailbox'}<button class="send" on:click={() => sendLive(p)} title="send live (no redeploy)">send</button>{/if}
                    </div>
                    <div class="pd">{p.desc || ''}</div>
                  </div>
                {/each}
              </div>
            {/if}
          {/each}
          {#if !meta.params.length}<div class="dim pad">No params yet — {#if canMerge}click <b>Merge ⤵</b> to import from source, or {/if}switch to <b>JSON</b> to add some.</div>{/if}
          <div class="foot">
            <span class="st">{status}</span><span class="sp"></span>
            {#if eng?.paramsSaveUrl}
              <button on:click={saveDefaults} title="persist these values as the kernel's defaults">Save defaults</button>
            {/if}
          </div>
        {/if}
      </div>
    {/if}
  </div>
{/if}

<style>
  .panel { border-bottom: 1px solid var(--line); background: var(--panel); }
  .bar { display: flex; align-items: center; gap: 8px; padding: 4px 10px; }
  .toggle { display: flex; flex: 1; min-width: 0; align-items: center; gap: 6px; background: none; border: none; color: var(--fg); cursor: pointer; font-family: inherit; font-size: 12px; text-align: left; padding: 2px 0; }
  .caret { color: var(--fg); font-size: 11px; transition: transform 0.12s; display: inline-block; }
  .caret.open { transform: rotate(90deg); }
  .sm { color: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .seg { display: flex; border: 1px solid var(--line); border-radius: 5px; overflow: hidden; flex: none; }
  .seg button { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--muted); border: none; padding: 3px 9px; cursor: pointer; }
  .seg button.on { background: var(--accent); color: #1a1206; font-weight: 600; }
  .run { font-family: inherit; font-size: 11.5px; background: var(--accent); color: #1a1206; border: 1px solid var(--accent); border-radius: 5px; padding: 3px 12px; cursor: pointer; font-weight: 600; flex: none; }
  .run:disabled { opacity: 0.5; cursor: default; }
  .merge { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 3px 10px; cursor: pointer; flex: none; }
  .merge:disabled { opacity: 0.5; cursor: default; }
  .mergemsg { color: var(--muted); font-size: 10.5px; max-width: 40%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: none; }
  .ix { color: var(--muted); font-size: 10px; margin-left: 2px; }
  .roval { font-family: ui-monospace, monospace; font-size: 12px; color: var(--fg); background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 2px 8px; }
  .body { padding: 2px 12px 8px; max-height: 44vh; overflow: auto; }
  .group { margin: 4px 0 8px; }
  .ghead { font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em; color: var(--accent); border-bottom: 1px solid var(--line); padding-bottom: 3px; margin-bottom: 4px; }
  .ghead .dim { text-transform: none; letter-spacing: 0; }
  .prow { display: grid; grid-template-columns: 110px 1fr; gap: 2px 10px; align-items: center; padding: 3px 0; }
  .pl { font-size: 12px; color: var(--fg); font-family: ui-monospace, monospace; }
  .pin { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
  .pin input[type=text], .pin input[type=number], .pin select { font-family: inherit; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 6px; }
  .pin input[type=number] { width: 96px; }
  .chips { display: flex; gap: 4px; flex-wrap: wrap; }
  .chip { font-family: inherit; font-size: 11px; min-width: 24px; background: var(--panel2); color: var(--muted); border: 1px solid var(--line); border-radius: 5px; padding: 2px 8px; cursor: pointer; }
  .chip.on { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .send { font-family: inherit; font-size: 10.5px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 2px 8px; cursor: pointer; }
  .pd { grid-column: 2; color: var(--muted); font-size: 10.5px; margin-top: -1px; }
  .json { width: 100%; box-sizing: border-box; min-height: 180px; max-height: 38vh; resize: vertical; font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 12px; background: #0a0c10; color: var(--fg); border: 1px solid var(--line); border-radius: 6px; padding: 8px; }
  .foot { display: flex; align-items: center; gap: 8px; padding-top: 6px; border-top: 1px solid var(--line); margin-top: 4px; }
  .foot .st { color: var(--muted); font-size: 11px; } .sp { flex: 1; }
  .foot button { font-family: inherit; font-size: 11.5px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 10px; cursor: pointer; }
  .foot button:disabled { opacity: 0.5; cursor: default; }
  .dim { color: var(--muted); } .pad { padding: 8px 2px; }
</style>
