<script>
  // BootloaderPanel — bootloader (bare-metal overlay) cockpit, brought to x280/HartObserve parity:
  // a live floorplan CARTOON selects a core (CorePicker, single-select), then a tabbed observe
  // (Deploy / Telemetry / Plot) streams that core's /ws/bootloader telemetry inline with per-slot
  // labels and a time-series plot. The device-validated deploy path (pickCore→openWs→compileBuffer→
  // stage→setParams→exec) is UNCHANGED — only additive UI around it. Labels come in-frame
  // (telemetry.fields) + from the loaded overlay's kernel.json schema (slot→name for the raw/plot).
  import { onMount, onDestroy } from 'svelte'
  import { getJSON, postJSON } from '../api.js'
  import CorePicker from './CorePicker.svelte'

  export let preselect = null   // overlay name chosen in the tree
  export let content = ''       // live editor buffer (overlay source)
  export let dirty = false      // editor edited since load → compile-from-buffer on deploy
  export let onSave = null      // persist the source before compiling

  // NB: `picked` here is the selected OVERLAY object (from the tree); `sel` is the focused CORE coord
  // {x,y}. They are different concepts — note CorePicker's `picked` prop below is a core coord (=sel).
  let scan = null, overlays = [], sel = null, picked = null, params = {}
  let tele = null, ws = null, rate = null, _prev = null
  let force = false, busy = '', msg = '', log = ''
  let grid = '2x2', launching = false, launchMsg = ''
  let obsTab = 'deploy'                 // deploy | telem | plot
  let histFrames = [], plotSlot = 0, plotRate = false   // client-side telemetry history ring

  $: blcores = (scan?.cores || []).filter((c) => c.kind === 'bootloader')

  // per-slot labels for the raw grid + plot: prefer the overlay actually LOADED on the core, fall
  // back to the tree-picked overlay. Each overlay's kernel.json telemetry is [{slot,name,kind,desc}].
  $: labelsSrc = (tele?.loaded?.A?.overlay && overlays.find((o) => o.name === tele.loaded.A.overlay)) || picked
  $: teleLabels = labelsSrc?.telemetry ? Object.fromEntries(labelsSrc.telemetry.map((t) => [t.slot, t.name])) : {}
  $: slotChoices = labelsSrc?.telemetry || []

  async function deploy() {
    launching = true; launchMsg = 'deploying… JIT build + multicast (a few seconds)'
    try {
      const r = await postJSON('/api/tensix/bl/launch', { grid })
      if (r.ok === false) { launchMsg = r.error || 'launch failed'; return }
      for (let i = 0; i < 12; i++) {
        await new Promise((res) => setTimeout(res, 2500))
        await doScan()
        const n = (scan?.cores || []).filter((c) => c.kind === 'bootloader').length
        if (n) { launchMsg = `deployed to ${n} core(s)` ; return }
      }
      launchMsg = 'launched — hit ⟳ Scan if cores not shown yet'
    } catch (e) { launchMsg = String(e.message || e) } finally { launching = false }
  }
  async function stopDeploy() {
    launching = true; launchMsg = 'stopping…'
    try { await postJSON('/api/tensix/bl/launch/stop', {}); await doScan(); launchMsg = 'stopped — cores reset' }
    catch (e) { launchMsg = String(e.message || e) } finally { launching = false }
  }

  async function loadOverlays() { overlays = (await getJSON('/api/tensix/bl/overlays')).overlays }
  // doScan(silent): silent background polls keep residency current without flashing the ⟳ button.
  async function doScan(silent = false) {
    if (!silent) busy = 'scan'
    try { scan = await getJSON('/api/tensix/bl/scan') } finally { if (!silent) busy = '' }
  }
  // Live refresh while mounted (Deploy tab visible) — bootloader residency drifts as cores deploy/stop.
  let pollT = null
  onMount(async () => { await loadOverlays(); await doScan(); pollT = setInterval(() => doScan(true), 4000) })
  onDestroy(() => { clearInterval(pollT); if (ws) ws.close() })

  // adopt the tree-selected overlay (prime its params from the schema)
  $: if (preselect && overlays.length && (!picked || picked.name !== preselect)) {
    const o = overlays.find((o) => o.name === preselect)
    if (o) { picked = o; params = {}; for (const p of o.params || []) params[p.name] = p.default }
  }

  // cartoon → focus a core. Only bootloader cores stream telemetry; nudge otherwise.
  function onPick(e) {
    const c = (scan?.cores || []).find((k) => k.x === e.detail.x && k.y === e.detail.y)
    if (c && c.kind === 'bootloader') pickCore(c)
    else { msg = `core ${e.detail.x},${e.detail.y} isn't running the bootloader — deploy it first` }
  }
  function pickCore(c) { sel = { x: c.x, y: c.y }; tele = null; rate = null; _prev = null; histFrames = []; openWs() }
  function openWs() {
    if (ws) ws.close()
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws/bootloader`)
    ws.onopen = () => ws.send(JSON.stringify({ x: sel.x, y: sel.y, hz: 5 }))
    ws.onmessage = (e) => {
      const f = JSON.parse(e.data)
      if (_prev && f.heartbeat != null && f.ts != null) {
        const dh = (f.heartbeat - _prev.hb) >>> 0, dt = f.ts - _prev.ts
        if (dt > 0 && dh > 0) rate = dh / dt
      }
      _prev = { hb: f.heartbeat, ts: f.ts }; tele = f
      // accumulate a client-side history ring for the Plot tab (telem_raw is the slot-indexed array)
      if (Array.isArray(f.telem_raw) && f.ts != null) histFrames = [...histFrames, { ts: f.ts, raw: f.telem_raw }].slice(-90)
    }
  }

  // Compile the live editor buffer under the picked overlay's name. Returns true on success.
  async function compileBuffer() {
    log = 'compiling…'
    if (onSave && dirty) { try { await onSave() } catch (e) {} }
    const r = await postJSON('/api/tensix/bl/compile', { name: picked.name, source: content })
    if (!r.ok) { log = 'compile error:\n' + (r.log || 'unknown'); return false }
    log = `compiled ✓ ${r.name} · ${r.hash} · ${r.bytes} B`; await loadOverlays(); return true
  }

  async function act(fn, label) { busy = label; msg = ''; try { await fn() } catch (e) { msg = String(e.message || e) } finally { busy = '' } }
  const setParams = async () => { for (const p of picked.params || []) await postJSON('/api/tensix/bl/param', { x: sel.x, y: sel.y, index: p.i, value: Number(params[p.name]) }) }
  const compileOnly = () => act(async () => { await compileBuffer() }, 'compile')
  const run = () => act(async () => {
    if (dirty) { const ok = await compileBuffer(); if (!ok) return }      // deploy YOUR edits
    await postJSON('/api/tensix/bl/stage', { x: sel.x, y: sel.y, overlay: picked.name, slot: 'A' })
    await setParams()
    const r = await postJSON('/api/tensix/bl/exec', { x: sel.x, y: sel.y, slot: 'A', wait: true, force })
    msg = r.ok ? `ran ${picked.name} · ovl_ret=${r.ovl_ret}` : `no-ack (status ${r.status}) — core may be wedged`
  }, 'exec')
  const halt = () => act(async () => { await postJSON('/api/tensix/bl/halt', { x: sel.x, y: sel.y }); msg = 'halted' }, 'halt')

  const fmt = (v) => v == null ? '—' : (Math.abs(v) >= 1000 ? v.toExponential(2) : (Number.isInteger(v) ? v : v.toFixed(3)))
  const gated = (o) => o && (o.verified === 'wedges' || o.verified === 'untested')
  const mrate = (r) => r == null ? '—' : (r >= 1e6 ? (r / 1e6).toFixed(0) + ' Mspin/s' : (r / 1e3).toFixed(0) + ' kspin/s')
  const hex = (v) => '0x' + (v >>> 0).toString(16)
  const fmtv = (v) => v >= 1e6 ? (v / 1e6).toFixed(1) + 'M' : v >= 1e3 ? (v / 1e3).toFixed(1) + 'k' : String(Math.round(v))
  $: vbadge = { ok: 'good', wedges: 'bad', untested: 'warn', custom: 'cust' }
  const stcol = (s) => s === 'IDLE' ? 'good' : s === 'OVERLAY' ? 'warn' : 'bad'

  // --- Plot tab: a single slot over time for the focused core ---
  $: plotPts = (() => {
    if (histFrames.length < 2) return []
    let pts = histFrames.map((fr) => ({ t: fr.ts, v: (fr.raw?.[plotSlot] ?? 0) >>> 0 }))
    if (plotRate) {
      const r = []
      for (let i = 1; i < pts.length; i++) { let d = pts[i].v - pts[i - 1].v; if (d < 0) d += 0x100000000; const dt = pts[i].t - pts[i - 1].t; r.push({ t: pts[i].t, v: dt > 0 ? d / dt : 0 }) }
      pts = r
    }
    return pts
  })()
  $: plotMax = Math.max(1, ...plotPts.map((p) => p.v))
  const plotPoints = (pts, w = 300, ht = 140) => pts.length < 2 ? '' :
    pts.map((p, i) => `${(i / (pts.length - 1)) * w},${ht - (p.v / plotMax) * (ht - 4) - 2}`).join(' ')
</script>

<div class="bl">
  <!-- live floorplan cartoon: residency at a glance (bootloader=green) + single-select focus -->
  <CorePicker mode="single" picked={sel} on:pick={onPick} />

  {#if !blcores.length}
    <div class="deploybox">
      <div class="dim">No bootloader resident. Deploy it to a block of cores:</div>
      <div class="drow">
        <label class="param"><span>grid</span><input bind:value={grid} placeholder="2x2" /></label>
        <button class="run" on:click={deploy} disabled={launching}>{launching ? 'deploying…' : 'Deploy ▶'}</button>
      </div>
      <div class="dim">WxH block (e.g. 2x2 = 4 cores) or <code>all</code>. Each resident core busy-spins (power) — keep it small.</div>
      {#if launchMsg}<div class="lmsg">{launchMsg}</div>{/if}
    </div>
  {:else}
    <div class="counts">{scan.n_bootloader} bootloader · {scan.n_ttmetal} tt-metal
      <button class="mini" on:click={stopDeploy} disabled={launching} title="stop the resident bootloader (resets cores)">⏻ stop</button></div>
  {/if}

  <section class="main">
    {#if !picked}
      <div class="empty">Pick a <b>⚡ bootloader overlay</b> in the tree ◀ — its source opens in the editor; then click a <b class="good">green</b> core in the cartoon ▲ to deploy.</div>
    {:else}
      <div class="ohead">
        <h3>{picked.title}</h3>
        <span class="vb {vbadge[picked.verified] || 'warn'}">{picked.verified}</span>
        <span class="dim">{picked.engine}</span>
        {#if dirty}<span class="edited">edited — Run compiles your buffer</span>{/if}
      </div>

      {#if !sel}
        <div class="dim pad">click a resident (green) core in the cartoon ▲ to deploy onto + observe.</div>
      {:else}
        <div class="rhead">
          <span class="ctag">core {sel.x},{sel.y}</span>
          {#if tele && !tele.error}
            <span class="st {stcol(tele.status_name)}">{tele.status_name}</span>
            <span class="dim">{mrate(rate)}</span>
            {#if tele.loaded?.A}<span class="dim">running {tele.loaded.A.overlay} · #{tele.loaded.A.hash}</span>{/if}
          {/if}
          <button class="halt" on:click={halt}>halt</button>
        </div>

        <div class="tabs">
          <button class:on={obsTab === 'deploy'} on:click={() => obsTab = 'deploy'}>Deploy</button>
          <button class:on={obsTab === 'telem'} on:click={() => obsTab = 'telem'}>Telemetry</button>
          <button class:on={obsTab === 'plot'} on:click={() => obsTab = 'plot'}>Plot</button>
        </div>

        {#if obsTab === 'deploy'}
          <div class="deploy">
            <div class="prow">
              {#each picked.params || [] as p}
                <label class="param"><span>{p.name}</span><input type="number" bind:value={params[p.name]} title={p.desc} /></label>
              {/each}
            </div>
            {#if gated(picked)}<label class="warnbox"><input type="checkbox" bind:checked={force} /> can <b>wedge the core</b> (recover: tt-smi -r 0)</label>{/if}
            <div class="actions">
              <button on:click={compileOnly} disabled={!!busy}>Compile</button>
              <button class="run" on:click={run} disabled={!!busy || (gated(picked) && !force)}>{dirty ? 'Compile + Stage + Run ▶' : 'Stage + Run ▶'}</button>
            </div>
          </div>
          {#if log}<pre class="log">{log}</pre>{/if}
          {#if msg}<div class="msg">{msg}</div>{/if}

        {:else if obsTab === 'telem'}
          {#if tele && tele.telemetry}
            <div class="telem">
              <div class="derived">
                <div class="metric"><div class="mv">{mrate(rate)}</div><div class="mk">spin rate</div></div>
                {#each tele.telemetry.derived as d}<div class="metric"><div class="mv">{fmt(d.value)}</div><div class="mk">{d.name}</div></div>{/each}
              </div>
              <table class="tt">
                {#each tele.telemetry.fields as f}
                  <tr><td class="fn">{f.name}</td>
                    <td class="fv">{f.kind === 'hex' || f.kind === 'marker' ? hex(f.value) : f.value}</td>
                    <td class="fd dim">{f.desc}</td></tr>
                {/each}
              </table>
              {#if Array.isArray(tele.telem_raw)}
                <h5>raw slots <span class="dim">(labelled from the loaded overlay)</span></h5>
                <table class="tt">
                  {#each tele.telem_raw as v, i}
                    <tr><td class="fn">{teleLabels[i] || 'slot ' + i}</td><td class="fv">{hex(v)}</td></tr>
                  {/each}
                </table>
              {/if}
            </div>
          {:else if tele && tele.error}<div class="msg bad">read error: {tele.error}</div>
          {:else}<div class="dim pad">waiting for telemetry…</div>{/if}

        {:else if obsTab === 'plot'}
          <h4>Telemetry plot <span class="dim">· one slot over time, this core</span></h4>
          <div class="plotctl">
            <label>slot <input type="number" min="0" max="7" bind:value={plotSlot} /></label>
            {#each slotChoices as t}<button class:on={plotSlot === t.slot} on:click={() => plotSlot = t.slot}>{t.slot} {t.name}</button>{/each}
            <label class="rt"><input type="checkbox" bind:checked={plotRate} /> rate</label>
          </div>
          {#if plotPts.length}
            <svg class="plot" viewBox="0 0 300 140" preserveAspectRatio="none">
              <polyline points={plotPoints(plotPts)} style="fill:none;stroke:var(--accent);stroke-width:1.4" />
            </svg>
            <div class="dim">y-max {plotRate ? fmtv(plotMax) + ' Δ/s' : hex(plotMax)} · {histFrames.length} samples · slot {plotSlot}{teleLabels[plotSlot] ? ' (' + teleLabels[plotSlot] + ')' : ''}</div>
          {:else}<div class="dim pad">Collecting… (needs ≥2 frames on a resident bootloader core; switching cores resets history)</div>{/if}
        {/if}
      {/if}
    {/if}
  </section>
</div>

<style>
  .bl { display: flex; flex-direction: column; gap: 12px; height: 100%; min-height: 0; }
  .main { border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); padding: 10px; overflow: auto; flex: 1; min-height: 0; }
  .mini { font-size: 11px; padding: 1px 7px; cursor: pointer; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; color: var(--muted); margin-left: 8px; }
  .mini:hover { color: #e07a77; border-color: #c0504d; }
  .counts { font-size: 10px; color: var(--muted); }
  .deploybox { display: flex; flex-direction: column; gap: 8px; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); }
  .drow { display: flex; gap: 8px; align-items: center; }
  .drow .param input { width: 64px; }
  .lmsg { font-size: 11px; color: var(--accent); }
  .dim { color: var(--muted); font-size: 11px; }
  .pad { padding: 6px 2px; line-height: 1.5; }
  .ohead { display: flex; align-items: center; gap: 10px; }
  .ohead h3 { margin: 0; }
  .edited { font-size: 10px; color: #d8a23a; border: 1px solid #d8a23a; border-radius: 3px; padding: 1px 6px; }
  .rhead { display: flex; align-items: center; gap: 10px; margin-top: 12px; }
  .ctag { font-weight: 600; }
  .st { font-size: 11px; padding: 2px 7px; border-radius: 4px; }
  .st.good { background: rgba(70,170,90,.2); color: var(--good); }
  .st.warn { background: rgba(216,162,58,.2); color: #d8a23a; }
  .st.bad { background: rgba(192,80,77,.2); color: #e07a77; }
  .halt { margin-left: auto; font-size: 11px; }
  .tabs { display: flex; gap: 4px; margin: 12px 0 10px; border-bottom: 1px solid var(--line); }
  .tabs button { background: none; border: none; border-bottom: 2px solid transparent; color: var(--muted); font-size: 12px; padding: 5px 10px; cursor: pointer; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .deploy { border: 1px solid var(--line); border-radius: 6px; padding: 12px; }
  .prow { display: flex; flex-wrap: wrap; gap: 12px; }
  .param { display: flex; align-items: center; gap: 8px; font-size: 12px; }
  .param span { color: var(--muted); }
  .param input { width: 120px; }
  .vb { font-size: 9px; padding: 1px 5px; border-radius: 3px; text-transform: uppercase; }
  .vb.good { background: rgba(70,170,90,.2); color: var(--good); }
  .vb.bad { background: rgba(192,80,77,.2); color: #e07a77; }
  .vb.warn { background: rgba(216,162,58,.2); color: #d8a23a; }
  .vb.cust { background: rgba(74,144,216,.2); color: #4a90d8; }
  .warnbox { display: flex; gap: 7px; font-size: 11px; color: #e0a; margin: 10px 0; align-items: center; }
  .actions { display: flex; gap: 8px; margin-top: 10px; align-items: center; }
  .run { background: var(--accent); color: #fff; }
  .log { background: #1a1a1a; color: #d8d8d8; font-size: 11px; padding: 8px; border-radius: 5px; white-space: pre-wrap; max-height: 160px; overflow: auto; margin-top: 10px; }
  .msg { margin-top: 12px; font-size: 12px; padding: 6px 9px; border: 1px solid var(--line); border-radius: 5px; }
  .msg.bad { color: #e07a77; }
  .telem h5 { margin: 14px 0 6px; }
  .derived { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .metric { border: 1px solid var(--line); border-radius: 6px; padding: 8px 14px; text-align: center; min-width: 84px; }
  .mv { font-size: 18px; }
  .mk { font-size: 10px; color: var(--muted); }
  .tt { width: 100%; border-collapse: collapse; font-size: 12px; }
  .tt td { padding: 3px 6px; border-bottom: 1px solid var(--line); }
  .fv { text-align: right; font-family: ui-monospace, monospace; }
  .fd { font-size: 10px; }
  .plotctl { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; margin: 6px 0 10px; font-size: 12px; }
  .plotctl input[type=number] { width: 56px; }
  .plotctl button { font-size: 11px; padding: 2px 7px; cursor: pointer; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; color: var(--muted); }
  .plotctl button.on { color: var(--accent); border-color: var(--accent); }
  .plotctl .rt { margin-left: auto; color: var(--muted); }
  .plot { width: 100%; height: 160px; background: var(--panel); border: 1px solid var(--line); border-radius: 6px; }
  .empty { color: var(--muted); padding: 30px; text-align: center; line-height: 1.6; }
  .bad { color: #e07a77; }
  .good { color: var(--good); }
</style>
