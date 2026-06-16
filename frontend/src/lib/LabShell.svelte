<!-- LabShell — the resizable 3-column lab chassis (device tree | editor | observe), extracted
     from HartLab's persisted resizable grid so every engine shares one layout. Slots: left,
     editor, right. Column widths persist under `storeKey`. -->
<script>
  export let storeKey = 'labshell.layout'
  export let leftW = 250
  export let rightW = 440

  function load() {
    try {
      const s = JSON.parse(localStorage.getItem(storeKey) || '{}')
      if (s.leftW) leftW = s.leftW
      if (s.rightW) rightW = s.rightW
    } catch (e) { /* defaults */ }
  }
  load()
  function persist() {
    try { localStorage.setItem(storeKey, JSON.stringify({ leftW, rightW })) } catch (e) {}
  }
  function drag(e, sign, get, set, min, max) {
    e.preventDefault()
    const start = e.clientX, s0 = get()
    const move = (ev) => set(Math.max(min, Math.min(max, s0 + sign * (ev.clientX - start))))
    const up = () => {
      window.removeEventListener('pointermove', move); window.removeEventListener('pointerup', up)
      document.body.style.cursor = ''; document.body.style.userSelect = ''; persist()
    }
    window.addEventListener('pointermove', move); window.addEventListener('pointerup', up)
    document.body.style.cursor = 'col-resize'; document.body.style.userSelect = 'none'
  }
  const dragLeft = (e) => drag(e, 1, () => leftW, (v) => leftW = v, 180, 480)
  const dragRight = (e) => drag(e, -1, () => rightW, (v) => rightW = v, 300, 780)
</script>

<div class="shell" style="grid-template-columns: {leftW}px 5px minmax(0, 1fr) 5px {rightW}px">
  <aside class="left"><slot name="left" /></aside>
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="gutter" on:pointerdown={dragLeft} title="drag to resize"></div>
  <section class="center"><slot name="editor" /></section>
  <!-- svelte-ignore a11y-no-static-element-interactions -->
  <div class="gutter" on:pointerdown={dragRight} title="drag to resize"></div>
  <aside class="right"><slot name="right" /></aside>
</div>

<style>
  .shell { display: grid; height: calc(100vh - 47px); overflow: hidden; }
  .gutter { background: var(--line); cursor: col-resize; }
  .gutter:hover { background: var(--accent); }
  .left, .right { display: flex; flex-direction: column; min-height: 0; overflow: hidden; }
  .center { display: flex; flex-direction: column; min-width: 0; min-height: 0; }
</style>
