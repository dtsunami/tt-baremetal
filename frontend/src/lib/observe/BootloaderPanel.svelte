<script>
  // BootloaderPanel — bootloader cockpit (default tab in TensixObserve). x280 parity: the overlay
  // SOURCE is edited in the center CodeEditor (loaded when you click an overlay in the TENSIX tree);
  // this panel picks a core (cards), then Stage+Run — which, if the source is edited, saves +
  // compiles the live editor buffer first, then deploys. Telemetry streams over /ws/bootloader.
  import { onMount, onDestroy } from 'svelte'
  import { getJSON, postJSON } from '../api.js'

  export let preselect = null   // overlay name chosen in the tree
  export let content = ''       // live editor buffer (overlay source)
  export let dirty = false      // editor edited since load → compile-from-buffer on deploy
  export let onSave = null      // persist the source before compiling

  let scan = null, overlays = [], sel = null, picked = null, params = {}
  let tele = null, ws = null, rate = null, _prev = null
  let force = false, busy = '', msg = '', log = ''
  let grid = '2x2', launching = false, launchMsg = ''

  $: blcores = (scan?.cores || []).filter((c) => c.kind === 'bootloader')

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
  async function doScan() { busy = 'scan'; try { scan = await getJSON('/api/tensix/bl/scan') } finally { busy = '' } }
  onMount(async () => { await loadOverlays(); await doScan() })
  onDestroy(() => { if (ws) ws.close() })

  // adopt the tree-selected overlay (prime its params from the schema)
  $: if (preselect && overlays.length && (!picked || picked.name !== preselect)) {
    const o = overlays.find((o) => o.name === preselect)
    if (o) { picked = o; params = {}; for (const p of o.params || []) params[p.name] = p.default }
  }

  function pickCore(c) { sel = { x: c.x, y: c.y }; tele = null; rate = null; _prev = null; openWs() }
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
  $: vbadge = { ok: 'good', wedges: 'bad', untested: 'warn', custom: 'cust' }
  const stcol = (s) => s === 'IDLE' ? 'good' : s === 'OVERLAY' ? 'warn' : 'bad'
</script>

<div class="bl">
  <aside class="cores">
    <div class="chd"><b>Cores</b>
      <span class="chd-r">
        {#if blcores.length}<button class="mini" on:click={stopDeploy} disabled={launching} title="stop the resident bootloader (resets cores)">⏻</button>{/if}
        <button class="scan" on:click={doScan} disabled={busy === 'scan'}>{busy === 'scan' ? '…' : '⟳'}</button>
      </span>
    </div>
    {#if scan}<div class="counts">{scan.n_bootloader} bootloader · {scan.n_ttmetal} tt-metal</div>{/if}
    {#if blcores.length}
      <div class="cardlist">
        {#each blcores as c}
          <button class="ccard" class:sel={sel && sel.x === c.x && sel.y === c.y} on:click={() => pickCore(c)}>
            <div class="cxy">{c.x},{c.y}</div>
            <div class="cmeta"><span class="dot {stcol(c.status)}"></span>{c.status || 'IDLE'}</div>
            <div class="cload">{c.loaded ? c.loaded : 'no overlay'}</div>
          </button>
        {/each}
      </div>
    {:else}
      <div class="deploybox">
        <div class="dim">No bootloader resident. Deploy it to a block of cores:</div>
        <div class="drow">
          <label class="param"><span>grid</span><input bind:value={grid} placeholder="2x2" /></label>
          <button class="run" on:click={deploy} disabled={launching}>{launching ? 'deploying…' : 'Deploy ▶'}</button>
        </div>
        <div class="dim">WxH block (e.g. 2x2 = 4 cores) or <code>all</code>. Each resident core busy-spins (power) — keep it small.</div>
        {#if launchMsg}<div class="lmsg">{launchMsg}</div>{/if}
      </div>
    {/if}
  </aside>

  <section class="main">
    {#if !picked}
      <div class="empty">Pick a <b>⚡ bootloader overlay</b> in the tree ◀ — its source opens in the editor; then pick a core to deploy.</div>
    {:else}
      <div class="ohead">
        <h3>{picked.title}</h3>
        <span class="vb {vbadge[picked.verified] || 'warn'}">{picked.verified}</span>
        <span class="dim">{picked.engine}</span>
        {#if dirty}<span class="edited">edited — Run compiles your buffer</span>{/if}
      </div>

      {#if !sel}
        <div class="dim pad">pick a resident core ◀ to deploy onto.</div>
      {:else}
        <div class="rhead">
          <span class="ctag">core {sel.x},{sel.y}</span>
          {#if tele && !tele.error}
            <span class="st {stcol(tele.status_name)}">{tele.status_name}</span>
            <span class="dim">{mrate(rate)}</span>
            {#if tele.loaded?.A}<span class="dim">loaded {tele.loaded.A.overlay} · {tele.loaded.A.hash}</span>{/if}
          {/if}
          <button class="halt" on:click={halt}>halt</button>
        </div>

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

        {#if tele && tele.telemetry}
          <div class="telem">
            <h4>Telemetry <span class="dim">live</span></h4>
            <div class="derived">
              <div class="metric"><div class="mv">{mrate(rate)}</div><div class="mk">spin rate</div></div>
              {#each tele.telemetry.derived as d}<div class="metric"><div class="mv">{fmt(d.value)}</div><div class="mk">{d.name}</div></div>{/each}
            </div>
            <table class="tt">
              {#each tele.telemetry.fields as f}
                <tr><td class="fn">{f.name}</td>
                  <td class="fv">{f.kind === 'hex' || f.kind === 'marker' ? '0x' + (f.value >>> 0).toString(16) : f.value}</td>
                  <td class="fd dim">{f.desc}</td></tr>
              {/each}
            </table>
          </div>
        {/if}
        {#if tele && tele.error}<div class="msg bad">read error: {tele.error}</div>{/if}
      {/if}
    {/if}
  </section>
</div>

<style>
  .bl { display: grid; grid-template-columns: 200px 1fr; gap: 12px; height: 100%; min-height: 0; }
  .cores, .main { border: 1px solid var(--line); border-radius: 8px; background: var(--panel2); padding: 10px; overflow: auto; }
  .chd { display: flex; align-items: center; justify-content: space-between; }
  .scan { font-size: 13px; padding: 2px 8px; cursor: pointer; }
  .chd-r { display: flex; gap: 4px; align-items: center; }
  .mini { font-size: 12px; padding: 2px 7px; cursor: pointer; background: var(--panel); border: 1px solid var(--line); border-radius: 5px; color: var(--muted); }
  .mini:hover { color: #e07a77; border-color: #c0504d; }
  .counts { font-size: 10px; color: var(--muted); margin: 4px 0 8px; }
  .deploybox { display: flex; flex-direction: column; gap: 8px; padding: 4px 2px; }
  .drow { display: flex; gap: 8px; align-items: center; }
  .drow .param input { width: 64px; }
  .lmsg { font-size: 11px; color: var(--accent); }
  .dim { color: var(--muted); font-size: 11px; }
  .pad { padding: 6px 2px; line-height: 1.5; }
  .cardlist { display: flex; flex-direction: column; gap: 6px; }
  .ccard { text-align: left; border: 1px solid var(--line); border-radius: 7px; padding: 7px 9px; background: var(--panel); color: var(--fg); cursor: pointer; }
  .ccard:hover { border-color: var(--muted); }
  .ccard.sel { border-color: var(--accent); box-shadow: inset 0 0 0 1px var(--accent); }
  .cxy { font-weight: 700; font-size: 14px; }
  .cmeta { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 5px; margin-top: 2px; }
  .cload { font-size: 10px; color: var(--muted); margin-top: 3px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
  .dot.good { background: var(--good); } .dot.warn { background: #d8a23a; } .dot.bad { background: #c0504d; }
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
  .deploy { border: 1px solid var(--line); border-radius: 6px; padding: 12px; margin-top: 10px; }
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
  .telem { margin-top: 16px; }
  .telem h4 { margin: 0 0 8px; }
  .derived { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 10px; }
  .metric { border: 1px solid var(--line); border-radius: 6px; padding: 8px 14px; text-align: center; min-width: 84px; }
  .mv { font-size: 18px; }
  .mk { font-size: 10px; color: var(--muted); }
  .tt { width: 100%; border-collapse: collapse; font-size: 12px; }
  .tt td { padding: 3px 6px; border-bottom: 1px solid var(--line); }
  .fv { text-align: right; }
  .fd { font-size: 10px; }
  .empty { color: var(--muted); padding: 30px; text-align: center; line-height: 1.6; }
  .bad { color: #e07a77; }
</style>
