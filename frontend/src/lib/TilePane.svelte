<!-- TilePane — the right-hand tile inspector (item 3). Slides in when a tile is clicked on the
     chip. Shows a LIVE hand-drawn cartoon of the tile's arch (NoC routers/NIUs + kind-specific
     core, lit by real bandwidth) over the detailed NIU counters, DRAM affinity and fold-seam
     neighbours, plus a link to the official tt-isa diagram. -->
<script>
  import { createEventDispatcher } from 'svelte'
  import { getJSON, tileKey } from './api.js'
  import { frame } from './stores.js'
  import TileCartoon from './TileCartoon.svelte'

  export let tile                 // floorplan tile {noc0, die, kind, label, dram_ctrl}
  const dispatch = createEventDispatcher()

  const GLYPH = { tensix: 'T', dram: 'D', eth: 'E', arc: 'A', pcie: 'P', l2cpu: 'C', security: 'S', empty: '·' }
  const glyphOf = (t) => t.kind === 'dram' && t.dram_ctrl != null ? `D${t.dram_ctrl}` : (GLYPH[t.kind] || '?')
  const ISA = 'https://github.com/tenstorrent/tt-isa-documentation/tree/main/BlackholeA0'

  let detail = null, error = null
  $: x = tile.noc0[0]
  $: y = tile.noc0[1]
  $: load(x, y)
  async function load(x, y) { detail = null; error = null; try { detail = await getJSON(`/api/tile/${x}/${y}`) } catch (e) { error = e.message } }

  // live bandwidth for the cartoon
  $: live = $frame?.tiles?.[tileKey(tile.noc0)]
  $: bw0 = live?.noc0 || 0
  $: bw1 = live?.noc1 || 0
  $: dram = tile.dram_ctrl != null ? ($frame?.dram?.[String(tile.dram_ctrl)] ?? { r: 0, w: 0 }) : { r: 0, w: 0 }

  // counters worth surfacing first (throughput), rest collapsed — same as the old TileDetail
  const KEY = [
    'MST_NONPOSTED_WR_DATA_WORD_SENT', 'MST_POSTED_WR_DATA_WORD_SENT',
    'MST_RD_DATA_WORD_RECEIVED', 'SLV_RD_DATA_WORD_SENT',
    'SLV_NONPOSTED_WR_DATA_WORD_RECEIVED', 'SLV_POSTED_WR_DATA_WORD_RECEIVED',
  ]
  function rows(counters) {
    const k = KEY.filter((n) => n in counters).map((n) => [n, counters[n]])
    const rest = Object.entries(counters).filter(([n]) => !KEY.includes(n) && counters[n])
    return { key: k, rest }
  }
</script>

<aside class="pane">
  <header>
    <span class="g {tile.kind}">{glyphOf(tile)}</span>
    <div class="hd">
      <b>{detail?.label ?? tile.label}</b> <span class="kind">{tile.kind}</span>
      <div class="sub">noc0 {x},{y} · die {tile.die[0]},{tile.die[1]}{#if tile.dram_ctrl != null} · GDDR6 d{tile.dram_ctrl}{/if}</div>
    </div>
    <button class="x" on:click={() => dispatch('close')} title="close">✕</button>
  </header>

  <!-- live cartoon -->
  <svg class="cartoon" viewBox="0 0 220 162" preserveAspectRatio="xMidYMid meet">
    <TileCartoon {tile} {bw0} {bw1} {dram} />
  </svg>

  <a class="isalink" href={ISA} target="_blank" rel="noopener">full arch diagram ▸ <span class="dim">tt-isa-documentation</span></a>

  <!-- detailed counters / affinity / neighbours -->
  <div class="body">
    {#if error}
      <div class="err">tile ({x},{y}): {error}</div>
    {:else if !detail}
      <div class="dim">loading counters…</div>
    {:else}
      {#if detail.nius}
        {#each Object.entries(detail.nius) as [n, niu]}
          {@const r = rows(niu.counters)}
          <details class="ctr" open={n === '0'}>
            <summary class:n0={n === '0'} class:n1={n === '1'}>NIU {n} → NoC{n} <span class="dim">{r.key.length + r.rest.length} counters</span></summary>
            <table>
              {#each r.key as [name, val]}<tr class="hi"><th>{name}</th><td class="num">{val.toLocaleString()}</td></tr>{/each}
              {#each r.rest as [name, val]}<tr><th>{name}</th><td class="num">{val.toLocaleString()}</td></tr>{/each}
            </table>
          </details>
        {/each}
      {:else}
        <div class="dim pad">Management tile — never polled (NoC0 hang hazard).</div>
      {/if}

      {#if detail.dram_affinity}
        <div class="aff">DRAM affinity — phys <b>d{detail.dram_affinity.phys_ctrl}</b> · logical <b>d{detail.dram_affinity.logi_ctrl}</b>
          {#if detail.dram_affinity.agree}<span class="ok">agree</span>{:else}<span class="bad">differ</span>{/if}</div>
      {/if}
      {#if detail.fold_seam?.length}
        <div class="seam">{#each detail.fold_seam as s}<span class="chip" class:far={s.hops > 3}>{s.dir} {s.neighbor[0]},{s.neighbor[1]} · {s.hops}h</span>{/each}</div>
      {/if}
    {/if}
  </div>
</aside>

<style>
  .pane { position: absolute; top: 0; right: 0; bottom: 0; width: 340px; z-index: 7; background: #0d0f14f5; border-left: 1px solid var(--line); display: flex; flex-direction: column; box-shadow: -8px 0 24px #00000066; }
  header { display: flex; align-items: flex-start; gap: 9px; padding: 11px 12px; border-bottom: 1px solid var(--line); }
  .g { font-family: ui-monospace, monospace; font-weight: 700; font-size: 16px; width: 26px; height: 26px; display: grid; place-items: center; border-radius: 5px; background: #11151d; border: 1px solid var(--line); flex: none; }
  .g.tensix { color: #f0825a; } .g.dram { color: #96cdaa; } .g.l2cpu { color: #b478ff; } .g.eth { color: #aab0e6; } .g.pcie { color: #9aa0e0; } .g.arc { color: #c8645a; } .g.security { color: #c8c86e; }
  .hd { flex: 1; min-width: 0; }
  .hd b { font-size: 14px; }
  .kind { color: var(--muted); font-size: 12px; }
  .sub { color: var(--muted); font-size: 11px; margin-top: 2px; }
  .x { background: var(--panel2); border: 1px solid var(--line); color: var(--fg); border-radius: 5px; cursor: pointer; width: 24px; height: 24px; flex: none; }
  .x:hover { border-color: var(--muted); }

  .cartoon { width: 100%; display: block; padding: 10px 12px 2px; box-sizing: border-box; }

  .isalink { display: block; padding: 4px 14px 10px; font-size: 11px; color: var(--accent); text-decoration: none; }
  .isalink:hover { text-decoration: underline; }
  .isalink .dim { color: var(--muted); }

  .body { overflow: auto; flex: 1; min-height: 0; padding: 4px 12px 14px; }
  .ctr { margin-bottom: 8px; }
  .ctr > summary { cursor: pointer; font-size: 12px; padding: 4px 0; }
  .ctr > summary.n0 { color: var(--noc0); } .ctr > summary.n1 { color: var(--noc1); }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; font-weight: 400; color: var(--muted); font-size: 10.5px; padding: 1px 0; }
  td.num { text-align: right; font-variant-numeric: tabular-nums; font-size: 11px; }
  tr.hi th { color: var(--fg); } tr.hi td { color: var(--accent); }
  .aff { font-size: 12px; margin: 8px 0; } .aff b { color: var(--fg); }
  .ok { color: var(--good); margin-left: 5px; } .bad { color: var(--bad); margin-left: 5px; }
  .seam { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 2px 7px; font-size: 11px; }
  .chip.far { border-color: var(--bad); color: #ffb3ba; }
  .dim { color: var(--muted); } .pad { padding: 8px 2px; } .err { color: var(--bad); }
</style>
