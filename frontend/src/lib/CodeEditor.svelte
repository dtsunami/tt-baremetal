<!-- CodeEditor — the shared CodeMirror surface for every lab (Kernel Lab, Hart Lab, …).
     Extracted from the byte-identical editor setup that was copy-pasted between the labs.
     Props: lang ('cpp'|'rust'|'asm'), onChange(text), onSave(), onSubmit() (optional ⌘⏎).
     Methods (via bind:this): setDoc(text), setLang(lang). -->
<script>
  import { onMount, onDestroy } from 'svelte'
  import { EditorView, keymap, hoverTooltip } from '@codemirror/view'
  import { EditorState, Compartment } from '@codemirror/state'
  import { basicSetup } from 'codemirror'
  import { cpp } from '@codemirror/lang-cpp'
  import { rust } from '@codemirror/lang-rust'
  import { HighlightStyle, syntaxHighlighting, StreamLanguage } from '@codemirror/language'
  import { gas } from '@codemirror/legacy-modes/mode/gas'
  import { tags as t } from '@lezer/highlight'

  export let lang = 'cpp'
  export let onChange = () => {}
  export let onSave = () => {}
  export let onSubmit = null     // optional ⌘⏎ / Ctrl+Enter handler
  // optional: (word) => htmlString|null — show a hover tooltip over an identifier (e.g. the
  // Tensix engine supplies an ISA lookup so TT_OP_* mnemonics self-describe). Read live, so the
  // map can load after mount.
  export let hover = null

  let el, view
  let hoverFn = hover
  $: hoverFn = hover
  const langComp = new Compartment()
  const langExt = (l) => l === 'rust' ? rust() : l === 'asm' ? StreamLanguage.define(gas) : cpp()

  const hl = HighlightStyle.define([
    { tag: t.keyword, color: '#cf83ff' },
    { tag: [t.typeName, t.className, t.namespace], color: '#4fd6e0' },
    { tag: [t.function(t.variableName), t.macroName], color: '#ffd24a' },
    { tag: [t.string, t.special(t.string)], color: '#79d479' },
    { tag: [t.number, t.bool, t.atom], color: '#ff8a4c' },
    { tag: t.comment, color: '#69707f', fontStyle: 'italic' },
    { tag: [t.processingInstruction, t.meta], color: '#7c8696' },
    { tag: t.operator, color: '#d8dee9' },
  ])

  export function setDoc(text) {
    if (view) view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: text } })
  }
  export function setLang(l) {
    if (view) view.dispatch({ effects: langComp.reconfigure(langExt(l)) })
  }

  onMount(() => {
    const theme = EditorView.theme({
      '&': { height: '100%', backgroundColor: '#0a0c10', color: 'var(--fg)' },
      '&.cm-focused': { outline: 'none' },
      '.cm-scroller': { fontFamily: 'ui-monospace, "SF Mono", Menlo, Consolas, monospace', fontSize: '12.5px', lineHeight: '1.5' },
      '.cm-gutters': { backgroundColor: '#0a0c10', color: 'var(--muted)', border: 'none' },
      '.cm-activeLine': { backgroundColor: 'rgba(255,255,255,0.04)' },
      '.cm-activeLineGutter': { backgroundColor: 'rgba(255,255,255,0.05)' },
      '.cm-selectionBackground, .cm-content ::selection': { backgroundColor: '#2a3550' },
      '&.cm-focused .cm-cursor': { borderLeftColor: 'var(--accent)' },
    }, { dark: true })
    const wordHover = hoverTooltip((view, pos) => {
      if (!hoverFn) return null
      const line = view.state.doc.lineAt(pos)
      const text = line.text, off = pos - line.from
      let s = off, e = off
      const w = (ch) => ch && /[A-Za-z0-9_]/.test(ch)
      while (s > 0 && w(text[s - 1])) s--
      while (e < text.length && w(text[e])) e++
      if (s === e) return null
      const html = hoverFn(text.slice(s, e))
      if (!html) return null
      return { pos: line.from + s, end: line.from + e, above: true,
        create() { const dom = document.createElement('div'); dom.className = 'cm-isa-tip'; dom.innerHTML = html; return { dom } } }
    })
    const listen = EditorView.updateListener.of((u) => { if (u.docChanged) onChange(u.state.doc.toString()) })
    const keys = [{ key: 'Mod-s', preventDefault: true, run: () => { onSave(); return true } }]
    if (onSubmit) keys.push({ key: 'Mod-Enter', preventDefault: true, run: () => { onSubmit(); return true } })
    view = new EditorView({
      parent: el,
      state: EditorState.create({ doc: '', extensions: [basicSetup, langComp.of(langExt(lang)), syntaxHighlighting(hl), theme, wordHover, listen, keymap.of(keys)] }),
    })
  })
  onDestroy(() => view?.destroy())
</script>

<div class="cm" bind:this={el}></div>

<style>
  .cm { height: 100%; }
  /* ISA hover tooltip (Tensix mnemonics) — rendered into CodeMirror's .cm-tooltip */
  .cm :global(.cm-tooltip) { background: #0b0d12; border: 1px solid var(--accent); border-radius: 6px; box-shadow: 0 4px 18px rgba(0,0,0,0.55); }
  .cm :global(.cm-isa-tip) { padding: 7px 10px; max-width: 360px; font-size: 11.5px; line-height: 1.5; color: var(--fg); font-family: ui-monospace, Menlo, Consolas, monospace; }
  .cm :global(.cm-isa-tip .h) { font-weight: 700; color: var(--accent); }
  .cm :global(.cm-isa-tip .op) { color: #ff8a4c; }
  .cm :global(.cm-isa-tip .u) { color: var(--muted); font-size: 10px; }
  .cm :global(.cm-isa-tip .d) { color: #c8ced8; margin: 3px 0 5px; font-family: system-ui, sans-serif; }
  .cm :global(.cm-isa-tip table) { border-collapse: collapse; width: 100%; }
  .cm :global(.cm-isa-tip td) { padding: 1px 6px 1px 0; vertical-align: top; }
  .cm :global(.cm-isa-tip .bits) { color: #4fd6e0; white-space: nowrap; }
  .cm :global(.cm-isa-tip .fn) { color: #ffd24a; white-space: nowrap; }
  .cm :global(.cm-isa-tip .fd) { color: var(--muted); font-family: system-ui, sans-serif; font-size: 10.5px; }
</style>
