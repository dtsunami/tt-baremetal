<script>
  import { onMount, onDestroy } from 'svelte'
  import CodeEditor from '../lib/CodeEditor.svelte'
  import DocsPane from '../lib/DocsPane.svelte'
  import { getJSON, postJSON, pollJob } from '../lib/api.js'
  import { renderDisasm, CSR, REG } from '../lib/riscv.js'

  const HARTS = 4
  const langOf = (name) => name.endsWith('.rs') ? 'rust' : (name.endsWith('.s') || name.endsWith('.S')) ? 'asm' : 'c'

  // ---- device/tiles state ----
  let tiles = []                 // [{tile, coord, released, wedged}]
  let haveRust = true
  let sel = { tile: 0, hart: 0 } // the ONE selection, synced across every pane:
                                 // deploy target AND telemetry/registers/arch focus
  let railTab = 'harts'         // left rail shows one swappable pane: 'harts' | 'files'
  let busy = null               // bringup-in-flight label

  // ---- workspace / editor ----
  let files = []
  let current = null
  let content = ''
  let savedContent = ''
  let status = 'ready'
  let saving = false
  let addrHex = '0x30001000'
  let editor                    // <CodeEditor> ref (bind:this) — setDoc / setLang
  function setDoc(text) { savedContent = text; editor?.setDoc(text) }   // onChange syncs `content`

  // ---- resizable layout (column widths + which rail pane; persisted) ----
  let leftW = 220, rightW = 430
  function loadLayout() {
    try {
      const s = JSON.parse(localStorage.getItem('hartlab.layout') || '{}')
      leftW = s.leftW || leftW; rightW = s.rightW || rightW; railTab = s.railTab || railTab
    } catch (e) { /* defaults */ }
  }
  function saveLayout() {
    try { localStorage.setItem('hartlab.layout', JSON.stringify({ leftW, rightW, railTab })) } catch (e) {}
  }
  const selectRailTab = (t) => { railTab = t; saveLayout() }
  function drag(e, sign, get, set, min, max) {
    e.preventDefault()
    const start = e.clientX, s0 = get()
    const move = (ev) => set(Math.max(min, Math.min(max, s0 + sign * (ev.clientX - start))))
    const up = () => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''; document.body.style.userSelect = ''; saveLayout()
    }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up)
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'
  }
  const dragLeft = (e) => drag(e, 1, () => leftW, (v) => leftW = v, 150, 480)
  const dragRight = (e) => drag(e, -1, () => rightW, (v) => rightW = v, 300, 760)

  // ---- telemetry / observe (PER-HART: each hart has its own 64-slot window) ----
  let teleByHart = null         // {0:[64],1:[64],2:[64],3:[64]}
  let prevSlots = []            // previous slots of the viewed hart (for change-flash)
  let deployed = {}             // {hart:{name,lang,seized}} on the focused tile (live)
  let deployedAll = {}          // {"tile,hart":{...}} every deploy, for the matrix labels
  let hartStatus = null, trigger = null
  let released = false, paused = false, wedged = false
  let hb = { v: null, ts: null, rate: 0 }
  let regs = null               // full register dump (per-hart vectors)
  let arch = null               // bh_dump_state() snapshot of the focused hart
  let lastDeploy = null
  let compileOut = null
  let tab = 'telem'

  // ---- per-hart time-series plot ----
  let histFrames = []           // rolling [{ts, byHart}] across frames
  let plotSlot = 0              // which telemetry slot to plot
  let plotRate = false          // plot Δ/sec instead of the raw value (good for counters)
  const HARTCOL = ['#ff8a4c', '#4fd6e0', '#b478ff', '#79d479'] // H0..H3
  let deployFile = ''           // kernel chosen in the Harts-pane deploy dropdown

  let ws = null, wsAlive = false, reconnectT = null

  // ---- hover tooltip (explains instructions / registers / CSRs) ----
  let tip = { show: false, x: 0, y: 0, text: '' }
  function onTip(e) {
    const el = e.target.closest('[data-tip]')
    if (el) tip = { show: true, x: e.clientX, y: e.clientY, text: el.getAttribute('data-tip') }
    else if (tip.show) tip = { show: false, x: 0, y: 0, text: '' }
  }

  $: dirty = content !== savedContent
  $: lang = current ? langOf(current) : 'c'
  $: addr = parseInt(addrHex, 16) || 0x30001000
  $: selTile = tiles.find((t) => t.tile === sel.tile)
  $: tele = teleByHart ? teleByHart[sel.hart] : null
  $: viewDeployed = deployed[sel.hart]
  $: slots = tele ? tele.map((v, i) => ({ i, v: v >>> 0, changed: (prevSlots[i] >>> 0) !== (v >>> 0) }))
                       .filter((s) => s.v !== 0 || s.i === 0) : []

  // ---- lifecycle ----
  onMount(async () => {
    loadLayout()
    await Promise.all([loadTiles(), loadFiles()])
    if (files.length) await openFile(files[0].name)
    connectWS()
  })
  onDestroy(() => { clearTimeout(reconnectT); ws?.close() })

  // ---- tiles / harts ----
  async function loadTiles() {
    try { const r = await getJSON('/api/l2/tiles'); tiles = r.tiles; haveRust = r.have_rust; busy = r.busy; deployedAll = r.deployed || {} } catch (e) { status = 'tiles: ' + e }
  }
  const hartInfo = (tile, h) => deployedAll[`${tile},${h}`]
  function hartState(tl, h) {
    if (!tl.released) return 'reset'
    return hartInfo(tl.tile, h) ? 'running' : 'parked'   // running = we deployed a kernel here
  }
  async function bringup(tile) {
    if (!confirm(`Bring up tile ${tile}? This releases its 4 harts and is ONE-SHOT — redoing it needs tt-smi -r 0.`)) return
    try {
      const r = await postJSON('/api/l2/bringup', { tile })
      if (!r.ok) { status = 'bringup: ' + (r.error || 'failed'); return }
      busy = `bringup tile ${tile}`; status = `bringing up tile ${tile}…`; pollBringup()
    } catch (e) { status = 'bringup error: ' + e }
  }
  function pollBringup() {
    pollJob('/api/l2/bringup/last', async (d) => {
      busy = null
      await loadTiles()
      const res = d.result
      status = res?.ok ? `tile ${res.tile} up — harts parked, trampoline installed` : `bringup failed: ${res?.error || d.error || '?'}`
    }, 1200)
  }
  // the single selection setter — every pane calls this, so they all stay in sync
  function selectHart(tile, hart) {
    const tileChanged = tile !== sel.tile
    sel = { tile, hart }
    prevSlots = []; hb = { v: null, ts: null, rate: 0 }
    if (tileChanged) {
      teleByHart = null; histFrames = []
      if (wsAlive) ws.send(JSON.stringify({ tile }))   // point the telemetry stream at it
      refreshRegs()
    }
    if (tab === 'arch') loadArch()
  }
  const selectTile = (tile) => selectHart(tile, sel.hart)

  // ---- files ----
  async function loadFiles() { files = await getJSON('/api/l2/files') }
  async function openFile(name) {
    if (name !== current && dirty && !confirm('Discard unsaved changes?')) return
    const f = await getJSON(`/api/l2/file?name=${encodeURIComponent(name)}`)
    current = f.name; setDoc(f.content); editor?.setLang(f.lang); status = `${f.name} · ${f.lang}`
  }
  async function newFile() {
    const name = prompt('New kernel name (e.g. blink.c / scan.rs / spin.s):')
    if (!name) return
    try { const f = await postJSON('/api/l2/file/new', { name, lang: langOf(name) }); await loadFiles(); current = f.name; setDoc(f.content); status = 'created ' + f.name }
    catch (e) { status = 'new: ' + e }
  }
  async function duplicate() {
    if (!current) return
    const name = prompt('Duplicate to (new kernel name):', current.replace(/\.(c|s|S|rs)$/, '_v2.$1'))
    if (!name) return
    try { const f = await postJSON('/api/l2/file/duplicate', { src: current, name }); await loadFiles(); await openFile(f.name); status = 'duplicated → ' + f.name }
    catch (e) { status = 'duplicate: ' + e }
  }
  async function save() {
    if (!current || !dirty || saving) return
    saving = true
    try { await postJSON('/api/l2/file', { name: current, content }); savedContent = content; status = 'saved ' + current; await loadFiles() }
    catch (e) { status = 'save failed: ' + e } finally { saving = false }
  }

  // ---- compile / deploy ----
  async function compile() {
    status = 'compiling…'
    try {
      compileOut = await postJSON('/api/l2/compile', { content, lang, addr })
      tab = 'build'
      status = compileOut.ok ? `compiled ${compileOut.bytes / 4 | 0} words (${compileOut.bytes} B)` : `compile FAILED (${compileOut.errors?.length || 0} errors)`
    } catch (e) { status = 'compile error: ' + e }
  }
  async function deploy() {
    if (!selTile?.released) { status = `tile ${sel.tile} is in reset — bring it up first`; return }
    if (lang === 'rust' && !haveRust) { status = 'rust toolchain not installed'; return }
    status = `deploying → tile ${sel.tile} hart ${sel.hart}…`
    try {
      const r = await postJSON('/api/l2/deploy', { tile: sel.tile, hart: sel.hart, content, lang, addr, name: current || '' })
      lastDeploy = r; compileOut = r.errors ? r : compileOut
      if (!r.ok) { status = (r.stage === 'compile' ? 'compile FAILED' : 'deploy: ' + (r.error || 'failed')); tab = 'build'; return }
      status = r.seized ? `loaded + seized hart ${sel.hart} ✓ (${r.words} words)` : `loaded hart ${sel.hart} (${r.words} words)`
      prevSlots = []; hb = { v: null, ts: null, rate: 0 }; await loadTiles(); if (tab === 'arch') loadArch(); tab = 'telem'
    } catch (e) { status = 'deploy error: ' + e }
  }
  async function zeroTele() {
    try { await postJSON('/api/l2/tele/zero', { tile: sel.tile, hart: sel.hart }); status = `cleared hart ${sel.hart} telemetry` } catch (e) { status = 'zero: ' + e }
  }
  async function refreshRegs() {
    try { regs = await getJSON(`/api/l2/regs?tile=${sel.tile}`) } catch (e) { regs = null }
  }
  async function loadArch() {
    try { arch = await getJSON(`/api/l2/arch?tile=${sel.tile}&hart=${sel.hart}`) } catch (e) { arch = null }
  }
  const gprTip = (g) => { const r = REG[g.abi] || REG['x' + g.x]; return r ? `${r.name} (${r.abi}) — ${r.desc}` : `x${g.x}` }
  const csrTip = (n) => CSR[n] ? `${n} — ${CSR[n]}` : n

  // ---- deploy-all / park-all (whole tile at once) ----
  async function deployAll() {
    if (!selTile?.released) { status = `tile ${sel.tile} is in reset — bring it up first`; return }
    if (lang === 'rust' && !haveRust) { status = 'rust toolchain not installed'; return }
    status = `deploying → all harts of tile ${sel.tile}…`
    try {
      const r = await postJSON('/api/l2/deploy_all', { tile: sel.tile, hart: 0, content, lang, addr, name: current || '' })
      lastDeploy = r; compileOut = r.errors ? r : compileOut
      if (!r.ok) { status = (r.stage === 'compile' ? 'compile FAILED' : 'deploy all: ' + (r.error || 'failed')); tab = 'build'; return }
      const ok = r.deployed_all.filter((o) => !o.error).length
      status = `deployed ${current || 'kernel'} → ${ok}/${r.deployed_all.length} harts of tile ${sel.tile}`
      histFrames = []; prevSlots = []; hb = { v: null, ts: null, rate: 0 }; await loadTiles(); tab = 'telem'
    } catch (e) { status = 'deploy all error: ' + e }
  }
  async function parkAll() {
    if (!selTile?.released) { status = `tile ${sel.tile} is in reset`; return }
    if (!confirm(`Park all 4 harts on tile ${sel.tile}? Stops whatever they're running.`)) return
    try {
      const r = await postJSON('/api/l2/park_all', { tile: sel.tile })
      status = r.ok ? `parked all harts on tile ${sel.tile}` : 'park: ' + (r.error || 'failed')
      await loadTiles()
    } catch (e) { status = 'park error: ' + e }
  }

  // deploy a saved workspace kernel straight from the Harts pane (no editor needed)
  $: if (!deployFile && files.length) deployFile = current || files[0].name
  async function deployPicked(all) {
    if (!selTile?.released) { status = `tile ${sel.tile} is in reset — bring it up first`; return }
    let f
    try { f = await getJSON(`/api/l2/file?name=${encodeURIComponent(deployFile)}`) } catch (e) { status = `load ${deployFile}: ${e}`; return }
    if (f.lang === 'rust' && !haveRust) { status = 'rust toolchain not installed'; return }
    const a = parseInt(addrHex, 16) || 0x30001000
    status = `deploying ${f.name} → ${all ? 'all harts' : 'hart ' + sel.hart} of tile ${sel.tile}…`
    try {
      const r = await postJSON(all ? '/api/l2/deploy_all' : '/api/l2/deploy',
        { tile: sel.tile, hart: sel.hart, content: f.content, lang: f.lang, addr: a, name: f.name })
      lastDeploy = r; compileOut = r.errors ? r : compileOut
      if (!r.ok) { status = (r.stage === 'compile' ? `compile FAILED (${f.name})` : 'deploy: ' + (r.error || 'failed')); tab = 'build'; return }
      status = all ? `deployed ${f.name} → all harts of tile ${sel.tile}`
                   : (r.seized ? `loaded + seized hart ${sel.hart} ✓ (${f.name})` : `loaded hart ${sel.hart} (${f.name})`)
      histFrames = []; prevSlots = []; hb = { v: null, ts: null, rate: 0 }; await loadTiles(); tab = 'telem'
    } catch (e) { status = 'deploy error: ' + e }
  }

  // ---- plot series (one line per hart) ----
  $: plotSeries = histFrames.length < 2 ? [] : [0, 1, 2, 3].map((h) => {
    let pts = histFrames.map((fr) => ({ t: fr.ts, v: (fr.byHart?.[h]?.[plotSlot] ?? 0) >>> 0 }))
    if (plotRate) {
      const r = []
      for (let i = 1; i < pts.length; i++) {
        let d = pts[i].v - pts[i - 1].v; if (d < 0) d += 0x100000000
        const dt = pts[i].t - pts[i - 1].t
        r.push({ t: pts[i].t, v: dt > 0 ? d / dt : 0 })
      }
      pts = r
    }
    return { hart: h, pts }
  })
  $: plotMax = Math.max(1, ...plotSeries.flatMap((s) => s.pts.map((p) => p.v)))
  const plotPoints = (pts, w = 300, ht = 140) => pts.length < 2 ? '' :
    pts.map((p, i) => `${(i / (pts.length - 1)) * w},${ht - (p.v / plotMax) * (ht - 4) - 2}`).join(' ')

  // ---- telemetry websocket ----
  function connectWS() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    ws = new WebSocket(`${proto}://${location.host}/ws/l2cpu`)
    ws.onopen = () => { wsAlive = true; ws.send(JSON.stringify({ tile: sel.tile, hz: 5 })) }
    ws.onmessage = (e) => onFrame(JSON.parse(e.data))
    ws.onclose = () => { wsAlive = false; reconnectT = setTimeout(connectWS, 1500) }
    ws.onerror = () => ws.close()
  }
  function onFrame(f) {
    if (f.tile !== sel.tile) return
    paused = !!f.paused; wedged = !!f.wedged; busy = f.busy ?? busy
    if (f.released !== undefined) released = f.released
    if (f.deployed) deployed = f.deployed
    if (f.tele_by_hart) {
      prevSlots = (teleByHart && teleByHart[sel.hart]) || []
      teleByHart = { 0: f.tele_by_hart['0'], 1: f.tele_by_hart['1'], 2: f.tele_by_hart['2'], 3: f.tele_by_hart['3'] }
      hartStatus = f.hart_status; trigger = f.trigger
      const cur = teleByHart[sel.hart] || []
      const v = (cur[0] || 0) >>> 0
      if (hb.v !== null && f.ts > hb.ts) {
        let d = v - hb.v; if (d < 0) d += 0x100000000
        hb.rate = d / (f.ts - hb.ts)
      }
      hb.v = v; hb.ts = f.ts
      histFrames = [...histFrames, { ts: f.ts, byHart: teleByHart }].slice(-90)
    }
  }

  const hex = (v, w = 8) => '0x' + (v >>> 0).toString(16).toUpperCase().padStart(w, '0')
  const hex64 = (v) => '0x' + v.toString(16).toUpperCase().padStart(10, '0')
  function fmtRate(r) {
    if (!r) return '—'
    if (r >= 1e6) return (r / 1e6).toFixed(2) + ' M/s'
    if (r >= 1e3) return (r / 1e3).toFixed(1) + ' k/s'
    return r.toFixed(0) + '/s'
  }
</script>

<div class="lab" style="grid-template-columns: {leftW}px 5px minmax(0, 1fr) 5px {rightW}px">
  <!-- left rail: Harts and Files as two swappable panes -->
  <aside class="rail">
    <div class="railtabs">
      <button class:on={railTab === 'harts'} on:click={() => selectRailTab('harts')}>Harts</button>
      <button class:on={railTab === 'files'} on:click={() => selectRailTab('files')}>Files</button>
    </div>
    <div class="railbody">
    {#if railTab === 'harts'}
    <div class="sect">
      <div class="seltag">selected <b>tile {sel.tile} · hart {sel.hart}</b></div>
      {#if selTile?.released}
        <div class="deployrow">
          <select bind:value={deployFile} title="kernel to deploy">
            {#each files as f}<option value={f.name}>{f.name}</option>{/each}
          </select>
          <button on:click={() => deployPicked(false)} title="deploy this kernel to hart {sel.hart}">▸ H{sel.hart}</button>
          <button on:click={() => deployPicked(true)} title="deploy this kernel to all 4 harts">▸ all</button>
        </div>
        <button class="parkall" on:click={parkAll} title="stop all harts on tile {sel.tile}">⏹ park all harts (tile {sel.tile})</button>
      {/if}
      {#each tiles as tl}
        <div class="tile" class:focus={sel.tile === tl.tile}>
          <button class="thead" on:click={() => selectTile(tl.tile)} title="watch this tile's telemetry">
            <b>tile {tl.tile}</b><span class="dim">({tl.coord[0]},{tl.coord[1]})</span>
            {#if tl.wedged}<span class="badge bad">wedged</span>
            {:else if tl.released}<span class="badge ok">up</span>
            {:else}<span class="badge">reset</span>{/if}
          </button>
          {#if !tl.released}
            <button class="bring" on:click={() => bringup(tl.tile)} disabled={!!busy}>{busy === `bringup tile ${tl.tile}` ? '…' : 'bring up'}</button>
          {:else}
            <div class="harts">
              {#each Array(HARTS) as _, h}
                <button class="hart {hartState(tl, h)}" class:sel={sel.tile === tl.tile && sel.hart === h}
                  on:click={() => selectHart(tl.tile, h)}
                  title={hartInfo(tl.tile, h) ? `hart ${h}: ${hartInfo(tl.tile, h).name}` : `hart ${h} — parked`}>
                  <span class="hn">H{h}</span>
                  <span class="hk">{hartInfo(tl.tile, h)?.name ?? 'parked'}</span>
                  {#if hartInfo(tl.tile, h)?.seized}<span class="hok">●</span>{/if}
                </button>
              {/each}
            </div>
          {/if}
        </div>
      {/each}
    </div>
    {:else}
    <div class="sect files">
      <div class="seltag">deploy target <b>tile {sel.tile} · hart {sel.hart}</b></div>
      <div class="filehead">Workspace <button class="mini" on:click={newFile} title="new kernel">＋</button>{#if current}<button class="mini" on:click={duplicate} title="duplicate current kernel">⧉</button>{/if}</div>
      <ul>
        {#each files as f}
          <li><button class="file" class:active={f.name === current} on:click={() => openFile(f.name)}>
            <span class="fn">{f.name}</span><span class="role {f.lang}">{f.lang}</span></button></li>
        {/each}
      </ul>
      <div class="dim hint">seeded from the bundled examples — edit freely or ＋ a new kernel</div>
    </div>
    {/if}
    </div><!-- /railbody -->
  </aside>

  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="gutter" on:pointerdown={dragLeft} title="drag to resize"></div>

  <!-- editor -->
  <section class="editor">
    <div class="toolbar">
      <span class="cur">{current ?? '—'}{#if dirty}<b class="dt">●</b>{/if}</span>
      {#if current}<span class="role {lang}">{lang}</span>{/if}
      <span class="sp"></span>
      <label class="addr" title="load address">@<input bind:value={addrHex} spellcheck="false" /></label>
      <span class="tgt" title="deploy target (synced with the selected hart)">→ T{sel.tile}/H{sel.hart}</span>
      <button on:click={save} disabled={!dirty || saving}>Save</button>
      <button on:click={compile}>Compile</button>
      <button class="run" on:click={deploy} disabled={!!busy} title="compile + load + redirect to the selected hart (⌘⏎)">Deploy ▸</button>
      <button class="runall" on:click={deployAll} disabled={!!busy} title="deploy to ALL 4 harts of tile {sel.tile}">all ▸</button>
    </div>
    <div class="code-wrap"><CodeEditor bind:this={editor} {lang} onChange={(text) => content = text} onSave={save} onSubmit={deploy} /></div>
    <div class="statusbar"><span class="st">{status}</span><span class="sp"></span>
      <span class="conn" class:on={wsAlive}>{wsAlive ? 'telemetry live' : 'connecting…'}</span></div>
  </section>

  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="gutter" on:pointerdown={dragRight} title="drag to resize"></div>

  <!-- observe: telemetry / registers / build / docs -->
  <aside class="side">
    <div class="tabs">
      <button class:on={tab === 'telem'} on:click={() => tab = 'telem'}>Telemetry</button>
      <button class:on={tab === 'plot'} on:click={() => tab = 'plot'}>Plot</button>
      <button class:on={tab === 'arch'} on:click={() => { tab = 'arch'; loadArch() }}>Arch</button>
      <button class:on={tab === 'regs'} on:click={() => { tab = 'regs'; refreshRegs() }}>Registers</button>
      <button class:on={tab === 'build'} on:click={() => tab = 'build'}>Build</button>
      <button class:on={tab === 'docs'} on:click={() => tab = 'docs'}>Docs</button>
    </div>
    <!-- svelte-ignore a11y-no-static-element-interactions -->
    <div class="tabbody" on:mousemove={onTip} on:mouseleave={() => tip.show = false}>
      {#if tab === 'telem'}
        <div class="obshead">
          <div>watching <b>tile {sel.tile}</b> {#if selTile}<span class="dim">({selTile.coord[0]},{selTile.coord[1]})</span>{/if}</div>
          <div class="state">
            {#if paused}<span class="badge">paused · {selTile ? 'device busy' : ''}</span>
            {:else if wedged}<span class="badge bad">wedged — tt-smi -r 0</span>
            {:else if !released}<span class="badge">in reset</span>
            {:else}<span class="badge ok">live</span>{/if}
          </div>
        </div>
        <div class="hartpick">
          {#each Array(HARTS) as _, h}
            <button class:on={sel.hart === h} on:click={() => selectHart(sel.tile, h)} title={deployed[h]?.name || `hart ${h} (parked)`}>
              H{h}{#if deployed[h]}<i class="run"></i>{/if}
            </button>
          {/each}
        </div>
        {#if viewDeployed}
          <div class="kname">running <b>{viewDeployed.name}</b> <span class="role {viewDeployed.lang}">{viewDeployed.lang}</span>{#if viewDeployed.seized}<span class="good"> · seized ✓</span>{/if}</div>
        {:else if released}<div class="kname dim">hart {sel.hart}: parked — no kernel deployed</div>{/if}
        {#if released && tele}
          <div class="hbcard">
            <div class="hbnum">{hex(tele[0])}</div>
            <div class="hbsub">hart {sel.hart} · slot 0 · <b>{fmtRate(hb.rate)}</b></div>
          </div>
          <div class="tgrid">
            {#each slots as s (s.i)}
              <div class="slot" class:changed={s.changed} class:hb={s.i === 0}>
                <span class="si">{s.i}</span>
                <span class="sv">{hex(s.v)}</span>
                <span class="sd">{s.v}</span>
              </div>
            {/each}
          </div>
          <div class="obsfoot">
            <button on:click={zeroTele}>clear slots</button>
            <span class="dim">{slots.length} live · HART_STATUS {hartStatus != null ? hex(hartStatus, 4) : '—'} · TRIGGER {trigger != null ? hex(trigger) : '—'}</span>
          </div>
        {:else if !released}
          <div class="dim pad">Tile {sel.tile} is in reset. Bring it up from the Harts pane, then deploy a kernel.</div>
        {:else}
          <div class="dim pad">waiting for telemetry…</div>
        {/if}

      {:else if tab === 'regs'}
        <h4>Tile {sel.tile} registers <button class="mini" on:click={refreshRegs} title="refresh">⟳</button></h4>
        {#if regs?.released}
          <table>
            <tr><th>L2CPU_RESET</th><td class="num">{hex(regs.l2cpu_reset)}</td></tr>
            <tr><th>HART_STATUS</th><td class="num">{hex(regs.hart_status, 4)} <span class="dim">{regs.hart_status ? '' : '(all parked)'}</span></td></tr>
            <tr><th>TRIGGER</th><td class="num">{hex(regs.trigger)} <span class="dim">{regs.trigger ? 'seize pending' : 'idle'}</span></td></tr>
          </table>
          <h5>per-hart vectors</h5>
          <table class="regtab">
            <tr><th></th><th>reset_vec</th><th>rnmi_trap</th></tr>
            {#each regs.harts as h}
              <tr><th>hart {h.hart}</th><td class="num" class:good={h.reset_vec === 0x30001000}>{hex64(h.reset_vec)}</td><td class="num dim">{hex(h.rnmi_trap)}</td></tr>
            {/each}
          </table>
          <p class="dim">reset_vec = where the hart runs; <b class="good">green</b> = redirected into your loaded code (0x30001000).</p>
        {:else if regs}<div class="dim pad">tile {sel.tile} is in reset.</div>
        {:else}<div class="dim pad">no register read yet.</div>{/if}

      {:else if tab === 'plot'}
        <h4>Per-hart plot <span class="dim">· a slot over time, one line per hart</span></h4>
        <div class="plotctl">
          <label>slot <input type="number" min="0" max="63" bind:value={plotSlot} /></label>
          <button class:on={plotSlot === 0} on:click={() => plotSlot = 0}>0 hb</button>
          <button class:on={plotSlot === 63} on:click={() => plotSlot = 63}>63 retired</button>
          <button class:on={plotSlot === 62} on:click={() => plotSlot = 62}>62 cycles</button>
          <label class="rt" title="plot Δ/second (throughput) instead of the raw counter"><input type="checkbox" bind:checked={plotRate} /> rate</label>
        </div>
        {#if plotSeries.some((s) => s.pts.length)}
          <svg class="plot" viewBox="0 0 300 140" preserveAspectRatio="none">
            {#each plotSeries as s}
              {#if deployed[s.hart] || s.pts.some((p) => p.v)}
                <polyline points={plotPoints(s.pts)} style="fill:none;stroke:{HARTCOL[s.hart]};stroke-width:1.4" />
              {/if}
            {/each}
          </svg>
          <div class="plotlegend">
            {#each [0, 1, 2, 3] as h}
              <span class="lg" class:off={!deployed[h]}><i style="background:{HARTCOL[h]}"></i>H{h}{#if deployed[h]} · {deployed[h].name}{/if}</span>
            {/each}
          </div>
          <div class="dim">y-max {plotRate ? fmtRate(plotMax) : hex(plotMax)} · {histFrames.length} samples · slot {plotSlot}{plotRate ? ' (Δ/s)' : ''}</div>
        {:else}<div class="dim pad">Collecting samples… deploy a kernel that writes slot {plotSlot}. For <b>retired/sec per hart</b>, deploy <b>perf.c</b> to all harts (it writes slot 63), pick <b>63 retired</b> + <b>rate</b>.</div>{/if}

      {:else if tab === 'arch'}
        <h4>Arch state <span class="dim">· tile {sel.tile} hart {sel.hart}</span> <button class="mini" on:click={loadArch} title="refresh">⟳</button></h4>
        {#if arch?.valid}
          <h5>CSRs <span class="dim">· hover for what each holds</span></h5>
          <table class="archcsr">
            {#each Object.entries(arch.csr) as [name, val]}
              {#if name !== 'magic'}<tr><th data-tip={csrTip(name)}>{name}</th><td class="num">{val}</td></tr>{/if}
            {/each}
          </table>
          <h5>registers <span class="dim">· x0–x31 · hover for ABI role</span></h5>
          <div class="gprgrid">
            {#each arch.gpr as g}
              <div class="gpr" class:zero={g.val === '0x0000000000000000'} data-tip={gprTip(g)}>
                <span class="ga">{g.abi}</span><span class="gx">x{g.x}</span><span class="gv">{g.val}</span>
              </div>
            {/each}
          </div>
        {:else}
          <div class="dim pad">No snapshot for hart {sel.hart} yet. Add <code>bh_dump_state();</code> to your kernel (see the <b>dumpstate.c</b> example), deploy it, then hit ⟳. The host can't read a CPU's registers directly — the hart writes them to DRAM for us.</div>
        {/if}

      {:else if tab === 'build'}
        {#if lastDeploy?.ok}
          <div class="line">deploy: <b class="good">{lastDeploy.seized ? 'seized ✓' : 'loaded'}</b>
            <span class="dim">· {lastDeploy.words} words · {lastDeploy.bytes} B · {hex(lastDeploy.addr)} · T{lastDeploy.tile}/H{lastDeploy.hart}</span></div>
        {/if}
        {#if compileOut && !compileOut.ok}
          <div class="line bad">compile failed</div>
          {#if compileOut.errors?.length}<ul class="errs">{#each compileOut.errors as e}<li><b>{e.file}:{e.line}:{e.col}</b> {e.msg}</li>{/each}</ul>{/if}
          <details><summary>compiler output</summary><pre class="log">{compileOut.error}</pre></details>
        {:else if compileOut?.ok || lastDeploy?.disasm}
          <div class="line">compiled <b>{(compileOut?.bytes ?? lastDeploy?.bytes) / 4 | 0}</b> words</div>
          <h5>disassembly <span class="dim">· hover any instruction or register</span></h5>
          <div class="disasm">{@html renderDisasm(compileOut?.disasm || lastDeploy?.disasm || '')}</div>
        {:else}<div class="dim pad">Compile or Deploy to see the image + disassembly here.</div>{/if}

      {:else}
        <DocsPane docsUrl="/api/l2/docs" docUrl={(id) => `/api/l2/doc/${id}`} />
      {/if}
    </div>
  </aside>
</div>

{#if tip.show}<div class="tip" style="left:{tip.x + 14}px; top:{tip.y + 16}px">{tip.text}</div>{/if}

<style>
  .lab { display: grid; height: calc(100vh - 47px); overflow: hidden; }  /* columns set inline (resizable) */

  /* drag handles between columns */
  .gutter { background: var(--line); cursor: col-resize; }
  .gutter:hover { background: var(--accent); }

  /* left rail: Harts / Files as swappable tabs */
  .rail { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .railtabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); }
  .railtabs button { flex: 1; font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .railtabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .railbody { overflow: auto; flex: 1; min-height: 0; }
  .seltag { font-size: 10.5px; color: var(--muted); background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 3px 7px; margin-bottom: 8px; }
  .seltag b { color: var(--accent); }
  .deployrow { display: flex; gap: 4px; margin-bottom: 6px; }
  .deployrow select { flex: 1; min-width: 0; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 5px; }
  .deployrow button { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 3px 7px; cursor: pointer; white-space: nowrap; }
  .deployrow button:hover { background: rgba(255,138,76,0.12); }
  .parkall { width: 100%; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 4px; cursor: pointer; margin-bottom: 8px; }
  .parkall:hover { border-color: var(--bad); color: var(--bad); }
  .filehead { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; display: flex; align-items: center; gap: 6px; margin-bottom: 7px; }
  .sect { padding: 8px 10px; }
  .mini { background: var(--panel2); border: 1px solid var(--line); color: var(--fg); border-radius: 4px; cursor: pointer; font-size: 11px; line-height: 1; padding: 2px 6px; font-family: inherit; }
  .tile { margin-bottom: 7px; border: 1px solid var(--line); border-radius: 6px; padding: 5px 6px; background: var(--panel); }
  .tile.focus { border-color: var(--accent); box-shadow: 0 0 0 1px var(--accent) inset; }
  .thead { display: flex; align-items: center; gap: 5px; width: 100%; background: none; border: none; color: var(--fg); cursor: pointer; font-family: inherit; font-size: 12px; padding: 1px; text-align: left; }
  .thead b { font-weight: 600; }
  .badge { font-size: 9.5px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); margin-left: auto; }
  .badge.ok { color: var(--good); border-color: var(--good); }
  .badge.bad { color: var(--bad); border-color: var(--bad); }
  .bring { margin-top: 5px; width: 100%; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 3px; cursor: pointer; }
  .bring:disabled { opacity: 0.4; cursor: default; }
  .harts { display: flex; flex-direction: column; gap: 3px; margin-top: 6px; }
  .hart { display: flex; align-items: center; gap: 6px; font-family: inherit; font-size: 11px; padding: 4px 7px; border-radius: 4px; cursor: pointer; border: 1px solid var(--line); background: var(--panel2); color: var(--muted); text-align: left; }
  .hart .hn { font-weight: 600; color: var(--fg); width: 20px; flex: none; }
  .hart .hk { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--muted); }
  .hart.running { border-color: var(--good); background: rgba(121,212,121,0.08); }
  .hart.running .hk { color: var(--good); }
  .hart.reset { opacity: 0.4; }
  .hart.sel { outline: 2px solid var(--accent); outline-offset: -1px; }
  .hart .hok { color: var(--good); font-size: 8px; flex: none; }

  .files ul { list-style: none; margin: 0 0 4px; padding: 0; }
  .file { display: flex; width: 100%; align-items: center; gap: 6px; padding: 4px 7px; background: none; border: none; color: var(--fg); cursor: pointer; border-radius: 5px; text-align: left; font-family: inherit; font-size: 12px; }
  .file:hover { background: var(--panel2); }
  .file.active { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .file.ghost { color: var(--muted); }
  .file .fn { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .files .hint { padding: 4px 7px; font-size: 10.5px; line-height: 1.4; }
  .role { font-size: 9.5px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); }
  .role.c { color: var(--noc1); border-color: var(--noc1); }
  .role.rust { color: var(--accent); border-color: var(--accent); }
  .role.asm { color: var(--noc0); border-color: var(--noc0); }

  /* editor */
  .editor { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
  .toolbar { display: flex; align-items: center; gap: 8px; padding: 6px 10px; border-bottom: 1px solid var(--line); background: var(--panel); }
  .toolbar .cur { font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .toolbar .dt { color: var(--accent); margin-left: 4px; }
  .sp { flex: 1; }
  .addr { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 2px; }
  .addr input { width: 88px; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 3px 5px; }
  .tgt { font-size: 11px; color: var(--accent); border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; }
  .toolbar button { font-family: inherit; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 4px 10px; cursor: pointer; }
  .toolbar button:hover:not(:disabled) { border-color: var(--muted); }
  .toolbar button:disabled { opacity: 0.4; cursor: default; }
  .toolbar .run { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .toolbar .runall { color: var(--accent); border-color: var(--accent); }
  .toolbar .runall:hover:not(:disabled) { background: rgba(255,138,76,0.12); }
  .code-wrap { flex: 1; overflow: hidden; min-height: 0; background: #0a0c10; }
  .statusbar { display: flex; align-items: center; gap: 10px; padding: 4px 10px; border-top: 1px solid var(--line); background: var(--panel); font-size: 11px; color: var(--muted); }
  .statusbar .st { color: var(--fg); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .conn { color: var(--bad); }
  .conn.on { color: var(--good); }

  /* observe pane */
  .side { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .tabs { display: flex; border-bottom: 1px solid var(--line); background: var(--panel); }
  .tabs button { flex: 1; font-family: inherit; font-size: 12px; background: none; border: none; color: var(--muted); padding: 8px; cursor: pointer; border-bottom: 2px solid transparent; }
  .tabs button.on { color: var(--fg); border-bottom-color: var(--accent); }
  .tabbody { overflow: auto; padding: 12px 14px; flex: 1; min-height: 0; }
  .tabbody h4 { margin: 0 0 7px; font-size: 12px; display: flex; align-items: center; gap: 6px; }
  .tabbody h5 { margin: 12px 0 5px; font-size: 11px; color: var(--muted); font-weight: 500; }
  .pad { padding: 8px 2px; }
  .dim { color: var(--muted); }
  .good { color: var(--good); }
  .bad { color: var(--bad); }
  .line { margin: 3px 0; }

  .obshead { display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; }
  .hartpick { display: flex; gap: 4px; margin-bottom: 8px; }
  .hartpick button { flex: 1; font-family: inherit; font-size: 11px; padding: 4px 0; border-radius: 4px; cursor: pointer; border: 1px solid var(--line); background: var(--panel2); color: var(--muted); display: flex; align-items: center; justify-content: center; gap: 4px; }
  .hartpick button.on { color: var(--fg); border-color: var(--accent); background: rgba(255,138,76,0.1); }
  .hartpick .run { width: 5px; height: 5px; border-radius: 50%; background: var(--good); display: inline-block; }
  .kname { font-size: 11.5px; margin-bottom: 10px; }
  .kname b { color: var(--fg); }
  .hbcard { background: linear-gradient(180deg, #14171d, #0e1014); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }
  .hbnum { font-size: 26px; font-weight: 600; color: var(--accent); font-variant-numeric: tabular-nums; letter-spacing: 0.02em; }
  .hbsub { font-size: 11px; color: var(--muted); margin-top: 3px; }
  .hbsub b { color: var(--good); }
  .tgrid { display: flex; flex-direction: column; gap: 2px; }
  .slot { display: grid; grid-template-columns: 30px 1fr auto; align-items: baseline; gap: 8px; padding: 3px 7px; border-radius: 4px; border: 1px solid transparent; background: var(--panel); font-variant-numeric: tabular-nums; }
  .slot.hb { border-color: rgba(255,138,76,0.4); }
  .slot.changed { background: rgba(79,214,224,0.13); border-color: rgba(79,214,224,0.35); }
  .slot .si { color: var(--muted); font-size: 11px; text-align: right; }
  .slot .sv { color: var(--fg); }
  .slot .sd { color: var(--muted); font-size: 11px; }
  .obsfoot { display: flex; align-items: center; gap: 10px; margin-top: 10px; flex-wrap: wrap; }
  .obsfoot button { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 9px; cursor: pointer; }

  /* highlighted, tooltipped disassembly */
  .disasm { background: #0a0c10; border: 1px solid var(--line); border-radius: 6px; padding: 8px 10px; overflow: auto; max-height: 440px; font-size: 11px; line-height: 1.65; font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; }
  .disasm :global(.dlabel) { color: var(--noc1); margin: 7px 0 2px; font-weight: 600; }
  .disasm :global(.dline) { white-space: pre; }
  .disasm :global(.da) { color: var(--muted); }
  .disasm :global(.dhex) { color: #525a68; }
  .disasm :global(.dm) { color: #ffd24a; }
  .disasm :global(.dm[data-tip]) { cursor: help; }
  .disasm :global(.dm[data-tip]):hover { text-decoration: underline dotted; }
  .disasm :global(.dr) { color: var(--noc0); cursor: help; }
  .disasm :global(.dr):hover { text-decoration: underline dotted; }
  .disasm :global(.dcom) { color: #69707f; font-style: italic; }
  .tip { position: fixed; z-index: 60; max-width: 320px; background: #0b0d12; border: 1px solid var(--accent); color: var(--fg); border-radius: 6px; padding: 6px 9px; font-size: 11.5px; line-height: 1.45; pointer-events: none; box-shadow: 0 4px 18px rgba(0,0,0,0.55); white-space: pre-line; }

  .regtab th, .regtab td { font-size: 11.5px; }

  /* per-hart plot */
  .plotctl { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-bottom: 8px; }
  .plotctl label { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 4px; }
  .plotctl input[type=number] { width: 46px; font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 4px; padding: 2px 4px; }
  .plotctl button { font-family: inherit; font-size: 10.5px; background: var(--panel2); color: var(--muted); border: 1px solid var(--line); border-radius: 4px; padding: 2px 7px; cursor: pointer; }
  .plotctl button.on { color: var(--fg); border-color: var(--accent); }
  .plot { width: 100%; height: 150px; background: #0a0c10; border: 1px solid var(--line); border-radius: 6px; display: block; }
  .plotlegend { display: flex; flex-wrap: wrap; gap: 10px; margin: 7px 0 4px; font-size: 11px; }
  .plotlegend .lg { display: flex; align-items: center; gap: 5px; color: var(--fg); }
  .plotlegend .lg i { width: 12px; height: 3px; border-radius: 2px; display: inline-block; }
  .plotlegend .lg.off { opacity: 0.4; }

  /* arch-state: CSRs + the 32-register file */
  .archcsr th { cursor: help; }
  .archcsr td { font-variant-numeric: tabular-nums; }
  .gprgrid { display: flex; flex-direction: column; gap: 2px; }
  .gpr { display: flex; align-items: baseline; gap: 7px; padding: 2px 6px; border-radius: 3px; background: var(--panel); font-size: 11px; font-variant-numeric: tabular-nums; cursor: help; }
  .gpr.zero { opacity: 0.4; }
  .gpr .ga { color: var(--accent); width: 34px; font-weight: 600; }
  .gpr .gx { color: var(--muted); width: 26px; font-size: 10px; }
  .gpr .gv { color: var(--fg); flex: 1; text-align: right; }
  .errs { margin: 4px 0; padding-left: 16px; color: var(--bad); font-size: 12px; }
  .errs li { margin: 2px 0; }
  .log { background: #0a0c10; border: 1px solid var(--line); border-radius: 5px; padding: 8px; overflow: auto; max-height: 340px; font-size: 11px; line-height: 1.5; white-space: pre; }
  details summary { cursor: pointer; color: var(--muted); font-size: 11px; margin: 6px 0; }

</style>
