<!-- TensixLaunch — the device-side launch cockpit for one Tensix worker core. Reads the live launch
     mailbox (bhtop.tensix over exalens), lets you POKE runtime-arg words straight into L1 and
     re-issue go — on-the-fly editing with no recompile. Rides a tt-metal-loaded program: runtime
     args are pokeable; compile-time args are baked in. Shown in TileDetail for kind === 'tensix'. -->
<script>
  import { onDestroy } from 'svelte'
  import { getJSON, postJSON } from './api.js'

  export let x
  export let y

  let snap = null, err = '', busy = false, status = ''
  let edits = {}                       // proc_id -> array of u32 values (numbers)

  // ---- liveness: poll the launch state + an L1 watch window ----
  let live = false, hz = 3, timer = null
  let watchAddr = '', watchN = 8, watchWords = [], prevWatch = [], watchErr = ''

  $: if (x != null && y != null) { stopLive(); load(x, y) }

  onDestroy(stopLive)

  async function load(x, y) {
    err = ''; snap = null
    try {
      snap = await getJSON(`/api/tensix/launch?x=${x}&y=${y}`)
      edits = {}
      for (const r of snap.rta || []) edits[r.proc_id] = (r.values || []).map((v) => v >>> 0)
    } catch (e) { err = String(e) }
  }

  async function poke(r) {
    busy = true; status = ''
    try {
      const values = (edits[r.proc_id] || []).map((v) => parseInt(v) >>> 0)
      const d = await postJSON('/api/tensix/rta', { x, y, proc: r.proc_id, values })
      status = `poked ${r.proc}: ${values.length} word(s) → ${d.addr}`
      await load(x, y)
    } catch (e) { status = 'poke failed: ' + e } finally { busy = false }
  }

  async function go() {
    busy = true; status = ''
    try {
      const d = await postJSON('/api/tensix/go', { x, y })
      status = `go → ${d.signal_name} @ ${d.addr}`
      await load(x, y)
    } catch (e) { status = 'go failed: ' + e } finally { busy = false }
  }

  // ---- L1 watch window (the kernel-is-running signal) ----
  const parseAddr = (s) => { const v = parseInt(String(s).trim()); return Number.isFinite(v) ? v >>> 0 : null }
  function setWatch(addrHex) { watchAddr = addrHex; peek() }

  async function peek() {
    const addr = parseAddr(watchAddr)
    if (addr == null) { watchErr = 'enter an L1 address'; return }
    watchErr = ''
    try {
      const d = await getJSON(`/api/tensix/peek?x=${x}&y=${y}&addr=${addr}&n=${watchN}`)
      prevWatch = watchWords
      watchWords = d.words.map((w) => w >>> 0)
    } catch (e) { watchErr = String(e) }
  }

  async function tick() {
    await load(x, y)
    if (parseAddr(watchAddr) != null) await peek()
  }
  function startLive() {
    stopLive(); live = true
    timer = setInterval(tick, Math.max(150, 1000 / hz))
  }
  function stopLive() { if (timer) clearInterval(timer); timer = null; live = false }
  function toggleLive() { live ? stopLive() : startLive() }

  // ---- infinite run-loop: keep re-issuing go so the one-shot kernel runs forever ----
  $: loopOn = snap?.loop?.running
  async function toggleLoop() {
    busy = true; status = ''
    try {
      const d = await postJSON('/api/tensix/loop', { x, y, on: !loopOn, hz })
      status = d.running ? `looping @ ${d.hz}Hz — kernel re-runs each tick` : 'loop stopped'
      if (d.running && !live) startLive()       // auto-watch so you SEE it run
      await load(x, y)
    } catch (e) { status = 'loop failed: ' + e } finally { busy = false }
  }

  const hex = (n) => '0x' + (n >>> 0).toString(16)
  $: resident = snap && snap.enables && snap.enables !== '0x0'
  // prefill the watch box with the first processor's RTA address once known
  $: if (!watchAddr && snap && snap.rta && snap.rta[0]) watchAddr = snap.rta[0].addr
</script>

<div class="panel">
  <h3>Tensix launch <span class="muted">runtime-arg poke + re-go (exalens, no rebuild)</span></h3>

  {#if err}
    <div class="err">{err}</div>
  {:else if !snap}
    <div class="muted">reading launch mailbox…</div>
  {:else}
    <div class="meta">
      rd_ptr <b>{snap.rd_ptr}</b> · active <b>#{snap.active_index}</b> · mode <b>{snap.mode}</b>
      · enables <b>{snap.enables}</b> · prog_id <b>{hex(snap.host_assigned_id)}</b>
      · go <b class="sig">{snap.go?.signal_name}</b>
    </div>

    {#if snap.kernel_names?.length}
      <div class="kbar">kernel{snap.kernel_names.length > 1 ? 's' : ''}: {#each snap.kernel_names as kn}<span class="kchip">{kn}</span>{/each}</div>
    {/if}

    {#if !resident}
      <div class="note">No resident program on this core (enables = 0). Run a tt-metal kernel on
        ({x},{y}) first — then its runtime args become pokeable here.</div>
    {:else}
      {#each snap.rta as r}
        <div class="proc">
          <div class="phead"><b>{r.proc}</b>
            {#if r.kernel}<span class="kn" title={r.kernel.source + '  ·  hash ' + r.kernel.hash}>{r.kernel.name} <span class="ksrc">{r.kernel.source?.split('/').pop()}</span></span>{/if}
            <button class="addrlink" title="watch this address live" on:click={() => setWatch(r.addr)}>rta @ {r.addr}</button></div>
          <div class="vals">
            {#each edits[r.proc_id] || [] as _, i}
              <label class="v">
                <span class="vi">[{i}]</span>
                <input type="number" bind:value={edits[r.proc_id][i]} spellcheck="false" />
              </label>
            {/each}
            <button class="poke" on:click={() => poke(r)} disabled={busy}>Poke</button>
          </div>
        </div>
      {/each}
    {/if}

    <div class="watch">
      <div class="whead">
        <b>L1 watch</b>
        <input class="waddr" bind:value={watchAddr} placeholder="0x… L1 addr" spellcheck="false" />
        <label class="wn">×<input type="number" min="1" max="64" bind:value={watchN} /></label>
        <button on:click={peek} disabled={busy} title="read this L1 window once">Read</button>
        <button class="livebtn" class:on={live} on:click={toggleLive}
                title="poll the launch state + this window — watch the kernel run">{live ? '◼ Live' : '▶ Live'}</button>
        <label class="hzl">{hz}Hz<input type="range" min="1" max="10" bind:value={hz}
                on:change={() => live && startLive()} /></label>
      </div>
      {#if watchErr}<div class="err">{watchErr}</div>{/if}
      {#if watchWords.length}
        <div class="words">
          {#each watchWords as w, i}
            <span class="w" class:chg={prevWatch[i] !== undefined && prevWatch[i] !== w}
                  title="[{i}] = {w} ({hex(w)})">{w.toString(16).padStart(8, '0')}</span>
          {/each}
        </div>
      {/if}
    </div>

    <div class="foot">
      <span class="st">{status}</span><span class="sp"></span>
      {#if live}<span class="livedot" title="polling">● live</span>{/if}
      {#if loopOn}<span class="loopn" title="re-go iterations">↻ {snap.loop.n ?? ''}</span>{/if}
      <button on:click={() => load(x, y)} disabled={busy} title="re-read the launch mailbox">Refresh</button>
      <button class="run" on:click={go} disabled={busy || !resident}
              title="run once — set go to GO, re-runs the resident kernel with the current RTAs">Go ▸</button>
      <button class="loopbtn" class:on={loopOn} on:click={toggleLoop} disabled={busy || !resident}
              title="infinite: keep re-issuing go so the kernel runs continuously (Go in a loop)">{loopOn ? '◼ Loop' : 'Loop ↻'}</button>
    </div>
  {/if}
</div>

<style>
  .panel { border: 1px solid var(--line); background: var(--panel); border-radius: 6px; padding: 12px; }
  h3 { margin: 0 0 8px; font-size: 13px; }
  h3 .muted { font-weight: 400; }
  .muted { color: var(--muted); font-size: 12px; }
  .meta { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
  .meta b { color: var(--fg); font-variant-numeric: tabular-nums; }
  .sig { color: var(--accent); }
  .note { color: var(--muted); font-size: 12px; background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 8px; }
  .proc { padding: 6px 0; border-top: 1px solid var(--line); }
  .phead { font-family: ui-monospace, monospace; font-size: 12px; margin-bottom: 4px; }
  .vals { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
  .v { display: flex; align-items: center; gap: 3px; }
  .vi { color: var(--muted); font-size: 10px; font-family: ui-monospace, monospace; }
  .vals input { width: 110px; font-family: ui-monospace, monospace; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 6px; }
  .poke { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 2px 10px; cursor: pointer; }
  .addrlink { font-family: ui-monospace, monospace; font-size: 11px; background: none; border: none; color: var(--muted); cursor: pointer; padding: 0; text-decoration: underline dotted; }
  .addrlink:hover { color: var(--accent); }
  .kbar { font-size: 11px; color: var(--muted); margin-bottom: 8px; display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .kchip { background: rgba(121,212,121,0.14); border: 1px solid var(--good); color: var(--good); border-radius: 10px; padding: 1px 8px; font-family: ui-monospace, monospace; }
  .kn { font-size: 11px; color: var(--good); font-family: ui-monospace, monospace; }
  .kn .ksrc { color: var(--muted); }
  .watch { border-top: 1px solid var(--line); margin-top: 6px; padding-top: 8px; }
  .whead { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; font-size: 12px; }
  .waddr { width: 130px; font-family: ui-monospace, monospace; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 6px; }
  .wn input { width: 48px; font-size: 12px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 4px; margin-left: 2px; }
  .whead button { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 2px 9px; cursor: pointer; }
  .livebtn.on { background: var(--accent); color: #1a1206; border-color: var(--accent); font-weight: 600; }
  .hzl { color: var(--muted); font-size: 10px; display: flex; align-items: center; gap: 3px; }
  .hzl input { width: 70px; }
  .words { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
  .words .w { font-family: ui-monospace, monospace; font-size: 11.5px; background: var(--panel2); border: 1px solid var(--line); border-radius: 4px; padding: 2px 6px; color: var(--fg); transition: background 0.25s, color 0.25s; }
  .words .w.chg { background: var(--accent); color: #1a1206; font-weight: 600; }
  .livedot { color: var(--accent); font-size: 11px; }
  .loopn { color: var(--good); font-size: 11px; font-variant-numeric: tabular-nums; }
  .loopbtn { font-family: inherit; font-size: 11.5px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 10px; cursor: pointer; }
  .loopbtn.on { background: var(--good); color: #07140a; border-color: var(--good); font-weight: 600; }
  .loopbtn:disabled { opacity: 0.5; cursor: default; }
  .foot { display: flex; align-items: center; gap: 8px; padding-top: 8px; margin-top: 6px; border-top: 1px solid var(--line); }
  .foot .st { color: var(--muted); font-size: 11px; } .sp { flex: 1; }
  .foot button { font-family: inherit; font-size: 11.5px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 10px; cursor: pointer; }
  .run { background: var(--accent) !important; color: #1a1206 !important; border-color: var(--accent) !important; font-weight: 600; }
  .foot button:disabled { opacity: 0.5; cursor: default; }
  .err { color: var(--bad); font-size: 12px; }
</style>
