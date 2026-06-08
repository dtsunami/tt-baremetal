<script>
  import Router from 'svelte-spa-router'
  import Chip from './routes/Chip.svelte'
  import TileDetail from './routes/TileDetail.svelte'
  import { connected, frame } from './lib/stores.js'

  const routes = {
    '/': Chip,
    '/tile/:x/:y': TileDetail,
  }

  $: mode = $frame?.mode ?? '—'
  $: resetNeeded = $frame?.reset_needed
</script>

<header>
  <h1><a href="#/">bhtop</a> <span class="sub">Blackhole NoC explorer</span></h1>
  <nav>
    <a href="#/">chip</a>
  </nav>
  <div class="status">
    <span class="dot" class:on={$connected}></span>
    {$connected ? 'live' : 'reconnecting…'} · {mode}
  </div>
</header>

{#if resetNeeded}
  <div class="alert">
    ⚠ NoC hang detected — run <code>tt-smi -r 0</code> on ttstar to recover, then reload.
  </div>
{/if}

<main>
  <Router {routes} />
</main>
