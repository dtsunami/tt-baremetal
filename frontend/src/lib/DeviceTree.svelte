<!-- DeviceTree — the unified device-hierarchy browser (left pane). Engines with a `treeUrl`
     (X280) render a hierarchical FOLDER browser (KernelTree) with new-folder / duplicate-folder
     / regenerate; engines still on a `selUrl` (NOC / TENSIX) keep the project/example dropdown
     + flat list. Capability-aware ops throughout. Files whose source is live show a running ●
     (matched by basename against /api/running). Selecting a file dispatches `select`. -->
<script>
  import { onMount, createEventDispatcher } from 'svelte'
  import { getJSON, postJSON } from './api.js'
  import { ENGINES, SOON, langOf } from './engines.js'
  import KernelTree from './KernelTree.svelte'

  export let running = {}      // by_source map from /api/running: basename -> running build
  export let activeKey = null  // currently-open file key (for highlight)

  const dispatch = createEventDispatcher()

  // per-section UI state: {open, sel, options, files, tree, treeOpen, available, error, loading}
  let st = {}
  ENGINES.forEach((e) => st[e.key] = { open: false, sel: null, options: [], files: [],
                                       tree: [], treeOpen: {}, available: true, error: null, loading: false })

  onMount(() => { ENGINES.forEach(initSection) })

  async function initSection(eng) {
    if (eng.treeUrl) return loadTree(eng)
    const s = st[eng.key]
    if (eng.selUrl) {
      try {
        const d = await getJSON(eng.selUrl)
        s.available = d.available !== false
        if (eng.selKind === 'project') {
          s.options = (d.projects || []).map((p) => ({ value: p.name, label: p.name }))
          s.sel = s.sel || d.default || s.options[0]?.value || null
        } else {
          const ex = d.examples || []
          s.options = ex.map((e) => ({ value: e, label: e.replace(/^metal_example_/, '') }))
          s.sel = s.sel || ex.find((x) => x.includes('matmul_single')) || ex[0] || null
        }
      } catch (e) { s.error = String(e); s.available = false }
    }
    await loadFiles(eng)
  }

  async function loadTree(eng) {
    const s = st[eng.key]
    s.loading = true; st = st
    try {
      const d = await getJSON(eng.treeUrl)
      s.available = d.available !== false
      s.tree = d.tree || []
    } catch (e) { s.tree = []; s.available = false; s.error = String(e) }
    s.loading = false; st = st
  }

  async function loadFiles(eng) {
    const s = st[eng.key]
    if (s.available === false) { s.files = []; st = st; return }
    s.loading = true; st = st
    try { s.files = await getJSON(eng.listUrl(s.sel)) } catch (e) { s.files = []; s.error = String(e) }
    s.loading = false; st = st
  }

  const reload = (eng) => eng.treeUrl ? loadTree(eng) : loadFiles(eng)
  const toggle = (k) => { st[k].open = !st[k].open; st = st }
  async function changeSel(eng) { dispatch('selchange', { engine: eng.key, sel: st[eng.key].sel }); await loadFiles(eng) }
  const isRunning = (eng, f) => !!running[eng.nameOf(f)]
  // for the folder browser the run-target (project/example) is the top-level folder of the key;
  // the flat (legacy) path still carries the dropdown selection.
  const selectKey = (eng, key, name) => dispatch('select',
    { engine: eng.key, key, name, sel: eng.treeUrl ? key.split('/')[0] : st[eng.key].sel })
  const select = (eng, f) => selectKey(eng, eng.keyOf(f), eng.nameOf(f))

  // ---- file ops (shared by flat + tree; key/name come from the node or list row) ----
  async function fileDup(eng, key, name) {
    const dst = prompt('Duplicate to (new name):', name.replace(/(\.[^.]+)$/, '_v2$1'))
    if (!dst) return
    try { const r = await postJSON(eng.dupUrl, { src: key, name: dst }); await reload(eng); selectKey(eng, r.key ?? r.path ?? r.name, r.name ?? dst) }
    catch (e) { alert('duplicate: ' + e) }
  }
  async function fileRename(eng, key, name) {
    const dst = prompt('Rename to:', name)
    if (!dst || dst === name) return
    try { const r = await postJSON(eng.renameUrl, { src: key, name: dst }); await reload(eng); selectKey(eng, r.key ?? r.name, r.name ?? dst) }
    catch (e) { alert('rename: ' + e) }
  }
  async function fileDelete(eng, key, name) {
    if (!confirm(`Delete ${name}? This removes it from the workspace.`)) return
    try { await postJSON(eng.delUrl, { name: key, content: '' }); if (activeKey === key) dispatch('select', { engine: null }); await reload(eng) }
    catch (e) { alert('delete: ' + e) }
  }

  // ---- flat (legacy) op handlers ----
  async function opNew(eng) {
    const name = prompt('New kernel name (e.g. blink.c / scan.rs / spin.s):')
    if (!name) return
    try { const f = await postJSON(eng.newUrl, { name, lang: langOf(name) }); await reload(eng); selectKey(eng, f.key ?? f.name, f.name) }
    catch (e) { alert('new: ' + e) }
  }

  // ---- tree-mode handlers (callback props for KernelTree) ----
  const treeSelect = (eng) => (n) => selectKey(eng, n.key, n.name)
  const treeFileOp = (eng) => (kind, n) =>
    kind === 'dup' ? fileDup(eng, n.key, n.name)
    : kind === 'rename' ? fileRename(eng, n.key, n.name)
    : fileDelete(eng, n.key, n.name)
  const treeFolderOp = (eng) => async (kind, dir) => {
    try {
      if (kind === 'newfile') {
        const name = prompt(`New file in ${dir.name}/ (e.g. kernel.c / probe.rs):`)
        if (!name) return
        const f = await postJSON(eng.newUrl, { name: `${dir.key}/${name}`, lang: langOf(name) })
        st[eng.key].treeOpen[dir.key] = true; await reload(eng); selectKey(eng, f.key ?? f.name, f.name)
      } else if (kind === 'dup') {
        const name = prompt('Duplicate folder to (new name):', dir.name + '_v2')
        if (!name) return
        await postJSON(eng.folderDupUrl, { src: dir.key, name }); await reload(eng)
      } else if (kind === 'rename') {
        const name = prompt('Rename folder to:', dir.name)
        if (!name || name === dir.name) return
        await postJSON(eng.folderRenameUrl, { src: dir.key, name }); await reload(eng)
      } else if (kind === 'delete') {
        if (!confirm(`Delete folder ${dir.name}/ and everything in it?`)) return
        await postJSON(eng.folderDelUrl, { path: dir.key }); await reload(eng)
      }
    } catch (e) { alert(kind + ' folder: ' + e) }
  }
  async function newFolder(eng) {
    const name = prompt('New folder name:')
    if (!name) return
    try { await postJSON(eng.folderNewUrl, { path: name }); await reload(eng) }
    catch (e) { alert('new folder: ' + e) }
  }
  async function regenerate(eng) {
    const reverts = eng.regenUrl.includes('restore')   // noc/tensix revert .orig; x280 re-seeds canonical
    const msg = reverts
      ? 'Revert ALL your in-place edits in this engine to the shipped originals? This discards your changes.'
      : 'Restore the bundled example kernels to pristine? Your own folders are kept.'
    if (!confirm(msg)) return
    try {
      const r = await postJSON(eng.regenUrl, {}); await reload(eng); dispatch('select', { engine: null })
      const n = r.refreshed?.length ?? r.reverted ?? 0
      alert(reverts ? `reverted ${n} edited file(s) to original` : `restored ${n} example kernels`)
    } catch (e) { alert('restore: ' + e) }
  }
</script>

<div class="tree">
  {#each ENGINES as eng (eng.key)}
    {@const s = st[eng.key]}
    <section class="sect">
      <button class="head" on:click={() => toggle(eng.key)} title={eng.sub}>
        <span class="caret" class:open={s.open}>▸</span>
        <b class="eg {eng.key}">{eng.label}</b>
        <span class="sub">{eng.sub}</span>
        {#if s.loading}<span class="dim">…</span>{/if}
      </button>

      {#if s.open}
        {#if eng.selUrl && s.options.length}
          <select class="sel" bind:value={st[eng.key].sel} on:change={() => changeSel(eng)} title={eng.selKind}>
            {#each s.options as o}<option value={o.value}>{o.label}</option>{/each}
          </select>
        {/if}

        <!-- capability-aware op toolbar -->
        <div class="ops">
          {#if eng.caps.new && !eng.treeUrl}<button class="op" on:click={() => opNew(eng)} title="new kernel">＋ new</button>{/if}
          {#if eng.caps.folder}<button class="op" on:click={() => newFolder(eng)} title="new top-level folder">＋ folder</button>{/if}
          {#if eng.caps.regen}<button class="op danger" on:click={() => regenerate(eng)} title="restore to pristine — destructive, discards edits">↻ {eng.regenLabel || 'restore'}</button>{/if}
          {#if eng.inPlaceNote}<span class="op note" title={eng.inPlaceNote}>edits in place ⓘ</span>{/if}
        </div>

        {#if s.error}<div class="err">{s.error}</div>{/if}
        {#if s.available === false}
          <div class="dim pad">unavailable — {eng.key === 'x280' ? 'no L2CPU workspace' : 'tt-metal not found'}</div>
        {:else if eng.treeUrl}
          {#if !s.tree.length}<div class="dim pad">{s.loading ? 'loading…' : 'no kernels'}</div>
          {:else}
            <div class="kt-wrap">
              <KernelTree nodes={s.tree} {eng} {activeKey} {running} open={st[eng.key].treeOpen}
                          onSelect={treeSelect(eng)} onFileOp={treeFileOp(eng)} onFolderOp={treeFolderOp(eng)} />
            </div>
          {/if}
        {:else if !s.files.length}
          <div class="dim pad">{s.loading ? 'loading…' : 'no files'}</div>
        {:else}
          <ul>
            {#each s.files as f (eng.keyOf(f))}
              <li class="row" class:active={activeKey === eng.keyOf(f)}>
                <button class="file" on:click={() => select(eng, f)} title={isRunning(eng, f) ? `running — ${eng.nameOf(f)}` : eng.keyOf(f)}>
                  {#if isRunning(eng, f)}<span class="live" title="source is live (matched by build hash)"></span>{:else}<span class="live off"></span>{/if}
                  <span class="fn">{eng.nameOf(f)}</span>
                  <span class="role {eng.roleOf(f)}">{eng.roleOf(f)}</span>
                </button>
                <span class="rowops">
                  {#if eng.caps.duplicate}<button class="mini" on:click|stopPropagation={() => fileDup(eng, eng.keyOf(f), eng.nameOf(f))} title="duplicate">⧉</button>{/if}
                  {#if eng.caps.rename}<button class="mini" on:click|stopPropagation={() => fileRename(eng, eng.keyOf(f), eng.nameOf(f))} title="rename">✎</button>{/if}
                  {#if eng.caps.delete}<button class="mini del" on:click|stopPropagation={() => fileDelete(eng, eng.keyOf(f), eng.nameOf(f))} title="delete">🗑</button>{/if}
                </span>
              </li>
            {/each}
          </ul>
        {/if}
      {/if}
    </section>
  {/each}

  {#each SOON as s (s.key)}
    <section class="sect soon">
      <div class="head" title={s.sub}>
        <span class="caret">▸</span><b class="eg {s.key}">{s.label}</b>
        <span class="sub">{s.sub}</span><span class="badge">soon</span>
      </div>
    </section>
  {/each}
</div>

<style>
  .tree { overflow: auto; flex: 1; min-height: 0; padding: 6px 0; }
  .sect { border-bottom: 1px solid var(--line); padding: 4px 0 8px; }
  .head { display: flex; align-items: center; gap: 6px; width: 100%; background: none; border: none; color: var(--fg); cursor: pointer; font-family: inherit; font-size: 12.5px; padding: 7px 10px; text-align: left; }
  .caret { color: var(--fg); font-size: 12px; transition: transform 0.12s; display: inline-block; width: 12px; }
  .caret.open { transform: rotate(90deg); }
  .eg { font-weight: 700; letter-spacing: 0.03em; }
  .eg.noc { color: var(--noc0); } .eg.x280 { color: var(--accent); } .eg.tensix { color: var(--noc1); }
  .eg.dram, .eg.eth { color: var(--muted); }
  .sub { color: var(--muted); font-size: 10px; flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .soon .head { cursor: default; }
  .badge { font-size: 9px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); }

  .sel { margin: 2px 10px 4px; width: calc(100% - 20px); font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--fg); border: 1px solid var(--line); border-radius: 5px; padding: 3px 5px; }
  .ops { display: flex; align-items: center; gap: 6px; padding: 0 10px 4px; flex-wrap: wrap; }
  .op { font-family: inherit; font-size: 10.5px; background: var(--panel2); color: var(--accent); border: 1px solid var(--accent); border-radius: 5px; padding: 2px 7px; cursor: pointer; }
  .op:hover { background: rgba(255,138,76,0.12); }
  .op.note { color: var(--muted); border-color: var(--line); cursor: help; }
  .op.danger { color: var(--bad); border-color: var(--bad); }
  .op.danger:hover { background: rgba(255,80,80,0.14); }

  .kt-wrap { padding: 0 6px; }
  ul { list-style: none; margin: 0; padding: 0 6px; }
  .row { display: flex; align-items: center; border-radius: 5px; }
  .row:hover { background: var(--panel2); }
  .row.active { background: var(--panel2); box-shadow: inset 2px 0 0 var(--accent); }
  .file { display: flex; flex: 1; min-width: 0; align-items: center; gap: 6px; padding: 4px 7px; background: none; border: none; color: var(--fg); cursor: pointer; text-align: left; font-family: inherit; font-size: 12px; }
  .live { width: 6px; height: 6px; border-radius: 50%; background: var(--good); flex: none; box-shadow: 0 0 5px var(--good); }
  .live.off { background: transparent; box-shadow: none; border: 1px solid var(--line); }
  .fn { flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .role { font-size: 9px; padding: 1px 5px; border-radius: 3px; border: 1px solid var(--line); color: var(--muted); flex: none; }
  .role.device, .role.dataflow, .role.c { color: var(--noc1); border-color: var(--noc1); }
  .role.compute, .role.rust { color: var(--accent); border-color: var(--accent); }
  .role.host { color: var(--muted); }
  .role.asm { color: var(--noc0); border-color: var(--noc0); }
  .rowops { display: flex; gap: 2px; padding-right: 5px; opacity: 0; }
  .row:hover .rowops { opacity: 1; }
  .mini { background: none; border: none; color: var(--muted); cursor: pointer; font-size: 11px; padding: 2px 3px; border-radius: 4px; line-height: 1; }
  .mini:hover { color: var(--fg); background: var(--panel); }
  .mini.del:hover { color: var(--bad); }
  .pad { padding: 4px 12px; } .dim { color: var(--muted); font-size: 11px; }
  .err { color: var(--bad); font-size: 11px; padding: 2px 12px; }
</style>
