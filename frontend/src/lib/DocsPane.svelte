<!-- DocsPane — the shared docs/learning pane for every lab: curated doc chips + the live
     tt-isa-documentation tree (one shared backend, isa.py) + rendered markdown/images.
     Extracted from the verbatim-duplicated docs code in Kernel Lab and Hart Lab.
     Props: docsUrl (curated index), docUrl(id)->md endpoint, imgUrl(name)->src (optional). -->
<script>
  import { onMount } from 'svelte'
  import { getJSON } from './api.js'
  import { mdToHtml } from './md.js'

  export let docsUrl                          // e.g. '/api/l2/docs' -> [{id,title,kind}]
  export let docUrl = (id) => `/api/lab/doc/${id}` // (id) -> URL returning {markdown}
  export let imgUrl = null                    // (name) -> img src; enables kind:'img' docs

  let docs = [], docId = null, docKind = 'md', docName = '', docHtml = ''
  let isaTree = [], openDirs = new Set(['BlackholeA0']), isaActive = null, isaGithub = null

  onMount(async () => {
    try { docs = await getJSON(docsUrl); if (docs[0]) await openDoc(docs[0]) } catch (e) { /* lab may be unavailable */ }
    try { isaTree = await getJSON('/api/isa/tree') } catch (e) { isaTree = [] }
  })

  async function openDoc(d) {
    isaActive = null; isaGithub = null; docId = d.id
    if (d.kind === 'img' && imgUrl) { docKind = 'img'; docName = d.id.replace('uarch/', '') }
    else {
      docKind = 'md'
      try { const r = await getJSON(docUrl(d.id)); docHtml = mdToHtml(r.markdown) }
      catch (e) { docHtml = '<p>failed to load doc</p>' }
    }
  }

  function flatten(nodes, depth = 0, prefix = '') {
    let out = []
    for (const n of nodes) {
      const id = prefix ? prefix + '/' + n.name : n.name
      if (n.type === 'dir') { out.push({ kind: 'dir', name: n.name, depth, id }); out = out.concat(flatten(n.children, depth + 1, id)) }
      else out.push({ kind: 'file', name: n.name, path: n.path, depth, id })
    }
    return out
  }
  $: isaFlat = flatten(isaTree)
  $: isaVisible = isaFlat.filter((n) => { const p = n.id.split('/'); for (let i = 1; i < p.length; i++) if (!openDirs.has(p.slice(0, i).join('/'))) return false; return true })
  function toggleDir(id) { openDirs.has(id) ? openDirs.delete(id) : openDirs.add(id); openDirs = new Set(openDirs) }
  async function openIsa(path) {
    isaActive = path; docId = 'isa:' + path; docKind = 'md'
    try {
      const r = await getJSON(`/api/isa/doc?path=${encodeURIComponent(path)}`)
      docHtml = mdToHtml(r.markdown, { rawBase: r.raw_base, repoDir: r.repo_dir }); isaGithub = r.github
      const p = path.split('/'); for (let i = 1; i < p.length; i++) openDirs.add(p.slice(0, i).join('/')); openDirs = new Set(openDirs)
    } catch (e) { docHtml = `<p class="bad">failed to load ${path}</p>` }
  }
  function onDocClick(e) { const a = e.target.closest('[data-isa]'); if (a) { e.preventDefault(); openIsa(a.getAttribute('data-isa')) } }
</script>

<div class="doclist">
  {#each docs as d}<button class="chip" class:on={d.id === docId} on:click={() => openDoc(d)}>{d.title}</button>{/each}
</div>
<details class="isa" open>
  <summary>tt-isa-documentation <span class="dim">· live from repo</span></summary>
  <div class="isatree">
    {#each isaVisible as n (n.id)}
      {#if n.kind === 'dir'}
        <button class="row dir" style="padding-left:{6 + n.depth * 12}px" on:click={() => toggleDir(n.id)}><span class="tw">{openDirs.has(n.id) ? '▾' : '▸'}</span>{n.name}</button>
      {:else}
        <button class="row file" class:on={isaActive === n.path} style="padding-left:{6 + n.depth * 12}px" on:click={() => openIsa(n.path)}>{n.name.replace(/\.md$/, '')}</button>
      {/if}
    {/each}
    {#if !isaVisible.length}<div class="dim" style="padding:6px">ISA tree unavailable (offline?)</div>{/if}
  </div>
</details>
{#if isaGithub}<a class="gh" href={isaGithub} target="_blank" rel="noopener">{isaActive} · view on GitHub ↗</a>{/if}
<!-- svelte-ignore a11y-click-events-have-key-events a11y-no-static-element-interactions -->
<div class="docview md" on:click={onDocClick}>
  {#if docKind === 'img'}<img src={imgUrl(docName)} alt={docId} />
  {:else}{@html docHtml}{/if}
</div>

<style>
  .dim { color: var(--muted); }
  .doclist { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }
  .chip { font-family: inherit; font-size: 11px; background: var(--panel2); color: var(--muted); border: 1px solid var(--line); border-radius: 12px; padding: 2px 9px; cursor: pointer; }
  .chip.on { color: var(--fg); border-color: var(--accent); }
  details.isa > summary { font-size: 11px; cursor: pointer; color: var(--muted); margin: 4px 0; }
  .isatree { max-height: 230px; overflow: auto; border: 1px solid var(--line); border-radius: 5px; margin: 4px 0 2px; }
  .isatree .row { display: block; width: 100%; text-align: left; background: none; border: none; color: var(--fg); font-family: inherit; font-size: 11.5px; padding: 3px 6px; cursor: pointer; white-space: nowrap; }
  .isatree .row:hover { background: var(--panel2); }
  .isatree .row.dir { color: var(--muted); }
  .isatree .row.file.on { color: var(--accent); }
  .isatree .tw { display: inline-block; width: 12px; color: var(--muted); }
  .gh { display: block; font-size: 11px; margin: 6px 0; }
  .docview { margin-top: 6px; }
  .docview :global(img) { max-width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #fff; }
</style>
