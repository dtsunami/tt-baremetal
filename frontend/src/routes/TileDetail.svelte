<script>
  import { getJSON, fmtBW, tileKey } from '../lib/api.js'
  import { frame } from '../lib/stores.js'

  export let params = {}
  $: x = +params.x
  $: y = +params.y

  let detail = null
  let error = null

  // refetch counters whenever the route params change
  $: if (!Number.isNaN(x) && !Number.isNaN(y)) load(x, y)

  async function load(x, y) {
    detail = null
    error = null
    try {
      detail = await getJSON(`/api/tile/${x}/${y}`)
    } catch (e) {
      error = e.message
    }
  }

  // live bandwidth for this tile from the telemetry stream
  $: live = $frame?.tiles?.[`${x},${y}`]

  // counters worth surfacing first (throughput), rest collapsed
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

<div class="page">
  <a href="#/">← chip</a>

  {#if error}
    <div class="panel err">tile ({x},{y}): {error}</div>
  {:else if !detail}
    <div class="panel">loading tile ({x},{y})…</div>
  {:else}
    <h2>{detail.label} <span class="kind">{detail.kind}</span></h2>
    <div class="meta">
      noc0 <b>{detail.noc0[0]},{detail.noc0[1]}</b> · die <b>{detail.die[0]},{detail.die[1]}</b>
      {#if detail.dram_ctrl !== null} · GDDR6 controller <b>d{detail.dram_ctrl}</b>{/if}
      {#if live} · live <span class="bw">{fmtBW((live.noc0 || 0) + (live.noc1 || 0))}</span>{/if}
    </div>

    {#if detail.nius}
      <div class="grid2">
        {#each Object.entries(detail.nius) as [noc, niu]}
          {@const r = rows(niu.counters)}
          <div class="panel">
            <h3 class:n0={noc === '0'} class:n1={noc === '1'}>NIU {noc} → NoC{noc}</h3>
            <table>
              <tbody>
                {#each r.key as [name, val]}
                  <tr class="hi"><th>{name}</th><td class="num">{val.toLocaleString()}</td></tr>
                {/each}
                {#each r.rest as [name, val]}
                  <tr><th>{name}</th><td class="num">{val.toLocaleString()}</td></tr>
                {/each}
              </tbody>
            </table>
          </div>
        {/each}
      </div>
    {:else}
      <div class="panel muted">Management tile — never polled (NoC0 hang hazard).</div>
    {/if}

    {#if detail.dram_affinity}
      <div class="panel">
        <h3>DRAM affinity <span class="muted">physical vs logical</span></h3>
        nearest controller — physical <b>d{detail.dram_affinity.phys_ctrl}</b>
        ({detail.dram_affinity.phys_cells} die cells) · logical
        <b>d{detail.dram_affinity.logi_ctrl}</b> ({detail.dram_affinity.logi_hops} noc0 hops)
        {#if detail.dram_affinity.agree}<span class="ok">agree</span>{:else}<span class="bad">differ</span>{/if}
      </div>
    {/if}

    {#if detail.fold_seam?.length}
      <div class="panel">
        <h3>Physical neighbors <span class="muted">die-adjacent → noc0 hops (fold seam)</span></h3>
        <div class="seam">
          {#each detail.fold_seam as s}
            <span class="chip" class:far={s.hops > 3}>{s.dir} {s.neighbor[0]},{s.neighbor[1]} · {s.hops}h</span>
          {/each}
        </div>
      </div>
    {/if}
  {/if}
</div>

<style>
  .page { padding: 16px; max-width: 980px; margin: 0 auto; display: flex; flex-direction: column; gap: 12px; }
  h2 { margin: 6px 0 0; }
  h2 .kind { color: var(--muted); font-weight: 400; font-size: 14px; }
  h3 { margin: 0 0 8px; font-size: 13px; }
  h3.n0 { color: var(--noc0); }
  h3.n1 { color: var(--noc1); }
  .meta { color: var(--muted); }
  .meta b { color: var(--fg); }
  .bw { color: var(--accent); }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  tr.hi th { color: var(--fg); }
  tr.hi td { color: var(--accent); }
  .seam { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { background: var(--panel2); border: 1px solid var(--line); border-radius: 5px; padding: 2px 8px; }
  .chip.far { border-color: var(--bad); color: #ffb3ba; }
  .ok { color: var(--good); margin-left: 6px; }
  .bad { color: var(--bad); margin-left: 6px; }
  .muted { color: var(--muted); font-weight: 400; font-size: 12px; }
  .err { color: var(--bad); }
  @media (max-width: 720px) { .grid2 { grid-template-columns: 1fr; } }
</style>
