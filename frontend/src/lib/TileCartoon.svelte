<!-- TileCartoon — the live hand-drawn architecture of one tile, in a 0..220 × 0..162 coordinate
     box (NO outer <svg>, so it embeds anywhere). Used by TilePane (the click-in inspector) AND
     by the chip view, which fades these in per-tile as you zoom down. Props: tile + live bw. -->
<script>
  import { fmtBW } from './api.js'
  export let tile
  export let bw0 = 0
  export let bw1 = 0
  export let dram = { r: 0, w: 0 }

  const NOC0 = '#cf83ff', NOC1 = '#36ecff'
  const rail = (v) => 1.5 + Math.min(6, Math.log10(1 + v) * 0.7)
  const lit = (v) => v > 1e4
  const RV = [
    { n: 'RV1', r: 'reader · NCRISC', c: NOC1 },
    { n: 'RV2', r: 'UNPACK', c: '#8b93a7' },
    { n: 'RV3', r: 'MATH', c: '#ff8a4c' },
    { n: 'RV4', r: 'PACK', c: '#8b93a7' },
    { n: 'RV5', r: 'writer · BRISC', c: NOC1 },
  ]
  const HART = [0, 1, 2, 3]
</script>

<rect x="3" y="3" width="214" height="156" rx="10" fill="#0b0d12" stroke="#1b2030" />

<!-- NoC0 rail (top) + NoC1 rail (bottom): width + opacity ∝ live bytes/s -->
<line x1="8" y1="24" x2="212" y2="24" stroke={NOC0} stroke-width={rail(bw0)} stroke-linecap="round" opacity={lit(bw0) ? 1 : 0.4} />
<line x1="8" y1="138" x2="212" y2="138" stroke={NOC1} stroke-width={rail(bw1)} stroke-linecap="round" opacity={lit(bw1) ? 1 : 0.4} />
<circle cx="40" cy="24" r="6" fill="#0b0d12" stroke={NOC0} stroke-width="1.6" />
<circle cx="180" cy="138" r="6" fill="#0b0d12" stroke={NOC1} stroke-width="1.6" />
<text x="12" y="18" class="t s" fill={NOC0}>NoC0 ▸ {fmtBW(bw0)}</text>
<text x="208" y="152" class="t s end" fill={NOC1}>NoC1 ◂ {fmtBW(bw1)}</text>

<!-- generic NIUs (the dram / l2cpu tiles draw their own internal NIUs, so skip them there) -->
{#if tile.kind !== 'dram' && tile.kind !== 'l2cpu'}
  <rect x="54" y="30" width="20" height="9" rx="2" fill={lit(bw0) ? NOC0 : '#222838'} opacity={lit(bw0) ? 0.9 : 0.6} />
  <text x="64" y="37" class="t xs mid" fill="#0b0d12">NIU0</text>
  <rect x="146" y="123" width="20" height="9" rx="2" fill={lit(bw1) ? NOC1 : '#222838'} opacity={lit(bw1) ? 0.9 : 0.6} />
  <text x="156" y="130" class="t xs mid" fill="#0b0d12">NIU1</text>
{/if}

{#if tile.kind === 'tensix'}
  {#each RV as e, i}
    <rect x="22" y={44 + i * 15} width="104" height="13" rx="3" fill="#11151d" stroke={e.c} stroke-width="1" />
    <text x="27" y={53 + i * 15} class="t s" fill={e.c}>{e.n} · {e.r}</text>
  {/each}
  <rect x="134" y="44" width="62" height="74" rx="4" fill="#11151d" stroke="#2b3242" />
  <text x="165" y="76" class="t s mid" fill="#9aa3b4">L1 SRAM</text>
  <text x="165" y="88" class="t s mid" fill="#6b7385">1.5 MB</text>

{:else if tile.kind === 'l2cpu'}
  <!-- faithful to "x280 to NoC and DRAM": 4 harts (I$/D$ + MMU + L2$) → L3$ → DMA/TLBs → NoC NIUs -->
  {#each HART as h}
    <rect x="6" y={42 + h * 22} width="46" height="20" rx="3" fill="#11151d" stroke="#b478ff" stroke-width="1" />
    <text x="10" y={51 + h * 22} class="t xs" fill="#b478ff">x280 Hart {h}</text>
    <text x="10" y={59 + h * 22} class="t xs" fill="#6b7385">I$/D$ 32K · L2$ 128K</text>
  {/each}
  <rect x="55" y="42" width="5" height="86" rx="2" fill="#d9c98f" opacity="0.7" />
  <rect x="64" y="46" width="34" height="78" rx="3" fill="#1a1622" stroke="#caa6d6" />
  <text x="81" y="83" class="t s mid" fill="#caa6d6">L3$</text>
  <text x="81" y="94" class="t s mid" fill="#9aa3b4">2 MiB</text>
  <rect x="104" y="46" width="44" height="16" rx="3" fill="#11151d" stroke="#2b3242" /><text x="126" y="56" class="t xs mid" fill="#9aa3b4">DMA engine</text>
  <rect x="104" y="66" width="44" height="16" rx="3" fill="#11151d" stroke="#2b3242" /><text x="126" y="76" class="t xs mid" fill="#9aa3b4">TLBs</text>
  <rect x="154" y="46" width="56" height="17" rx="3" fill={lit(bw0) ? '#3a2860' : '#11151d'} stroke={NOC0} /><text x="182" y="57" class="t xs mid" fill={NOC0}>NoC0 NIU</text>
  <rect x="154" y="66" width="56" height="17" rx="3" fill={lit(bw1) ? '#123038' : '#11151d'} stroke={NOC1} /><text x="182" y="77" class="t xs mid" fill={NOC1}>NoC1 NIU</text>
  <rect x="104" y="92" width="106" height="15" rx="3" fill="#11151d" stroke="#2b3242" /><text x="157" y="102" class="t xs mid" fill="#9aa3b4">external periph · scratch · MSI</text>
  <text x="157" y="120" class="t xs mid" fill="#6b7385">→ DRAM tile</text>

{:else if tile.kind === 'dram'}
  <!-- faithful to the tt-isa DRAM tile: off-chip DRAM → bank controller → xbar → cores -->
  <text x="110" y="45" class="t s mid" fill="#9aa3b4">off-chip DRAM</text>
  <text x="64" y="45" class="t xs end" fill="#79d479">R {fmtBW(dram.r)}</text>
  <text x="156" y="45" class="t xs" fill="#ff8a4c">W {fmtBW(dram.w)}</text>
  <line x1="110" y1="48" x2="110" y2="55" stroke={(dram.r + dram.w) > 1e5 ? '#96cdaa' : '#3a4250'} stroke-width="2.6" stroke-linecap="round" />
  <rect x="78" y="55" width="64" height="15" rx="3" fill="#11151d" stroke="#96cdaa" stroke-width={(dram.r + dram.w) > 1e5 ? 1.8 : 1} />
  <text x="110" y="65" class="t xs mid" fill="#96cdaa">DRAM bank ctrl</text>
  <line x1="110" y1="70" x2="110" y2="76" stroke="#5b6480" stroke-width="1.6" />
  <rect x="24" y="76" width="172" height="9" rx="2" fill="#1b2233" stroke="#5b6480" />
  <text x="110" y="83" class="t xs mid" fill="#aab3c8">xbar</text>
  {#each [0, 1, 2] as i}
    <line x1={53 + i * 57} y1="85" x2={53 + i * 57} y2="91" stroke="#5b6480" stroke-width="1.4" />
    <rect x={28 + i * 57} y="91" width="50" height="38" rx="4" fill="#0f131b" stroke="#2b3242" />
    <rect x={32 + i * 57} y="95" width="20" height="15" rx="2" fill="#2a1614" stroke="#f0825a" /><text x={42 + i * 57} y="105" class="t xs mid" fill="#f0825a">RV1</text>
    <rect x={32 + i * 57} y="112" width="20" height="14" rx="2" fill="#171326" stroke="#b478ff" /><text x={42 + i * 57} y="121" class="t xs mid" fill="#b478ff">R0</text>
    <rect x={54 + i * 57} y="95" width="20" height="31" rx="2" fill="#11201a" stroke="#96cdaa" /><text x={64 + i * 57} y="112" class="t xs mid" fill="#96cdaa">L1</text>
  {/each}

{:else if tile.kind === 'eth'}
  <rect x="40" y="50" width="140" height="26" rx="4" fill="#11151d" stroke="#aab0e6" /><text x="110" y="66" class="t s mid" fill="#aab0e6">Ethernet MAC</text>
  <rect x="40" y="84" width="140" height="26" rx="4" fill="#11151d" stroke="#2b3242" /><text x="110" y="100" class="t s mid" fill="#9aa3b4">SerDes PHY · chip-to-chip</text>

{:else if tile.kind === 'pcie'}
  <rect x="40" y="50" width="140" height="26" rx="4" fill="#11151d" stroke="#9aa0e0" /><text x="110" y="66" class="t s mid" fill="#9aa0e0">PCIe Gen5 x16 · host</text>
  <rect x="40" y="84" width="140" height="26" rx="4" fill="#11151d" stroke="#2b3242" /><text x="110" y="100" class="t s mid" fill="#9aa3b4">DMA engines</text>

{:else if tile.kind === 'arc'}
  <rect x="50" y="56" width="120" height="48" rx="6" fill="#11151d" stroke="#c8645a" /><text x="110" y="76" class="t s mid" fill="#c8645a">ARC management core</text><text x="110" y="92" class="t xs mid" fill="#6b7385">power · clocks · boot</text>

{:else if tile.kind === 'security'}
  <rect x="50" y="56" width="120" height="48" rx="6" fill="#11151d" stroke="#c8c86e" /><text x="110" y="80" class="t s mid" fill="#c8c86e">⚿ Security block</text>

{:else}
  <text x="110" y="84" class="t s mid" fill="#6b7385">router + NIU only — torus passthrough</text>
{/if}

<style>
  .t { font-family: ui-monospace, monospace; }
  .s { font-size: 7px; } .xs { font-size: 5.5px; }
  .mid { text-anchor: middle; } .end { text-anchor: end; }
</style>
