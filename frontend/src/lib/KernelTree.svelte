<!-- KernelTree — recursive folder/file tree for an engine's kernels (replaces the flat list +
     project dropdown). Dirs collapse/expand and carry a ⚙ badge when they're a kernel folder
     (have kernel.json); files keep the running ● + role badge + row ops. Folder rows expose
     new-file / duplicate / rename / delete (capability-gated). Wiring is via callback props
     (onSelect / onFileOp / onFolderOp) so it forwards cleanly through the recursion. -->
<script>
  import Self from './KernelTree.svelte'

  export let nodes = []
  export let eng
  export let activeKey = null
  export let running = {}
  export let open = {}          // shared key->bool dir-open map (mutated in place)
  export let depth = 0
  export let onSelect = () => {}
  export let onFileOp = () => {}
  export let onFolderOp = () => {}

  const isRunning = (f) => !!running[f.name]
  const isOpen = (k) => open[k] ?? false   // folders start collapsed; expand to drill in
  const toggle = (k) => { open[k] = !isOpen(k); open = open }
  const pad = (extra) => `padding-left:${depth * 11 + extra}px`
</script>

<ul class="kt">
  {#each nodes as n (n.key)}
    {#if n.type === 'dir' && n.kernel && n.children?.length === 1 && n.children[0].type === 'file'}
      <!-- single-file kernel: collapse the folder hierarchy to one row (kernel name = folder) -->
      {@const f = n.children[0]}
      <li class="frow" class:active={activeKey === f.key}>
        <button class="file" style={pad(6)} on:click={() => onSelect(f)} title={f.key}>
          {#if running[f.name]}<span class="live" title="source is running"></span>{:else}<span class="live off"></span>{/if}
          <span class="kone" title="kernel folder">⚙</span>
          <span class="fn">{n.name}</span>
          <span class="role {f.role}">{f.role}</span>
        </button>
        <span class="rowops">
          {#if eng.caps.new}<button class="mini" title="add a file (un-collapses)" on:click|stopPropagation={() => onFolderOp('newfile', n)}>＋</button>{/if}
          {#if eng.caps.folder}
            <button class="mini" title="duplicate kernel" on:click|stopPropagation={() => onFolderOp('dup', n)}>⧉</button>
            <button class="mini" title="rename kernel" on:click|stopPropagation={() => onFolderOp('rename', n)}>✎</button>
            <button class="mini del" title="delete kernel" on:click|stopPropagation={() => onFolderOp('delete', n)}>🗑</button>
          {:else if eng.caps.duplicate}
            <button class="mini" title="duplicate" on:click|stopPropagation={() => onFileOp('dup', f)}>⧉</button>
          {/if}
        </span>
      </li>
    {:else if n.type === 'dir'}
      <li class="drow">
        <div class="dhead" class:kernel={n.kernel}>
          <button class="dtoggle" style={pad(6)} on:click={() => toggle(n.key)}>
            <span class="caret" class:open={open[n.key]}>▸</span>
            <span class="dname">{n.name}</span>
            {#if n.kernel}<span class="kbadge" title="kernel folder (has kernel.json)">⚙</span>{/if}
          </button>
          <span class="rowops">
            {#if eng.caps.new}<button class="mini" title="new file here" on:click|stopPropagation={() => onFolderOp('newfile', n)}>＋</button>{/if}
            {#if eng.caps.folder}
              <button class="mini" title="duplicate folder" on:click|stopPropagation={() => onFolderOp('dup', n)}>⧉</button>
              <button class="mini" title="rename folder" on:click|stopPropagation={() => onFolderOp('rename', n)}>✎</button>
              <button class="mini del" title="delete folder" on:click|stopPropagation={() => onFolderOp('delete', n)}>🗑</button>
            {/if}
          </span>
        </div>
        {#if open[n.key]}
          {#if n.children?.length}
            <Self nodes={n.children} {eng} {activeKey} {running} {open} depth={depth + 1}
                  {onSelect} {onFileOp} {onFolderOp} />
          {:else}<div class="empty" style={pad(20)}>empty</div>{/if}
        {/if}
      </li>
    {:else}
      <li class="frow" class:active={activeKey === n.key}>
        <button class="file" style={pad(10)} on:click={() => onSelect(n)} title={n.key}>
          {#if running[n.name]}<span class="live" title="source is running"></span>{:else}<span class="live off"></span>{/if}
          <span class="fn">{n.name}</span>
          <span class="role {n.role}">{n.role}</span>
        </button>
        <span class="rowops">
          {#if eng.caps.duplicate}<button class="mini" title="duplicate" on:click|stopPropagation={() => onFileOp('dup', n)}>⧉</button>{/if}
          {#if eng.caps.rename}<button class="mini" title="rename" on:click|stopPropagation={() => onFileOp('rename', n)}>✎</button>{/if}
          {#if eng.caps.delete}<button class="mini del" title="delete" on:click|stopPropagation={() => onFileOp('delete', n)}>🗑</button>{/if}
        </span>
      </li>
    {/if}
  {/each}
</ul>

<style>
  ul.kt { list-style: none; margin: 0; padding: 0; }
  .drow, .frow { display: flex; flex-direction: column; }
  .frow { flex-direction: row; align-items: center; border-radius: 5px; }
  .frow:hover { background: var(--panel2); }
  .frow.active { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .dhead { display: flex; align-items: center; }
  .dhead:hover { background: var(--panel2); border-radius: 5px; }
  .dtoggle { display: flex; flex: 1; min-width: 0; align-items: center; gap: 5px; background: none; border: none; color: var(--fg); cursor: pointer; font-family: inherit; font-size: 12px; padding: 4px 6px; text-align: left; }
  .caret { color: var(--fg); font-size: 12px; transition: transform 0.12s; display: inline-block; width: 12px; }
  .caret.open { transform: rotate(90deg); }
  .dname { font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .dhead.kernel .dname { color: var(--fg); }
  .kbadge { font-size: 14px; color: var(--accent); }
  .kone { font-size: 14px; color: var(--accent); flex: none; }
  .file { display: flex; flex: 1; min-width: 0; align-items: center; gap: 6px; background: none; border: none; color: var(--fg); cursor: pointer; text-align: left; font-family: inherit; font-size: 12px; padding: 4px 7px; }
  .live { width: 6px; height: 6px; border-radius: 50%; background: var(--good); flex: none; box-shadow: 0 0 5px var(--good); }
  .live.off { background: transparent; box-shadow: none; border: 1px solid var(--line); }
  .fn { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .role { font-size: 9px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); flex: none; }
  .role.device, .role.dataflow, .role.c { color: var(--noc1); border-color: var(--noc1); }
  .role.compute, .role.rust { color: var(--accent); border-color: var(--accent); }
  .role.host { color: var(--muted); } .role.asm { color: var(--noc0); border-color: var(--noc0); }
  .rowops { display: flex; gap: 3px; padding-right: 5px; opacity: 0.8; }   /* always visible (not hover-only) */
  .drow > .dhead:hover .rowops, .frow:hover .rowops { opacity: 1; }
  .mini { background: none; border: none; color: var(--fg); cursor: pointer; font-size: 15px; padding: 2px 4px; border-radius: 4px; line-height: 1; }
  .mini:hover { color: var(--accent); background: var(--panel2); }
  .mini.del:hover { color: var(--bad); }
  .empty { color: var(--muted); font-size: 10.5px; padding: 2px 0; font-style: italic; }
</style>
