<!-- CodeEditor — the shared CodeMirror surface for every lab (Kernel Lab, Hart Lab, …).
     Extracted from the byte-identical editor setup that was copy-pasted between the labs.
     Props: lang ('cpp'|'rust'|'asm'), onChange(text), onSave(), onSubmit() (optional ⌘⏎).
     Methods (via bind:this): setDoc(text), setLang(lang). -->
<script>
  import { onMount, onDestroy } from 'svelte'
  import { EditorView, keymap } from '@codemirror/view'
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

  let el, view
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
    const listen = EditorView.updateListener.of((u) => { if (u.docChanged) onChange(u.state.doc.toString()) })
    const keys = [{ key: 'Mod-s', preventDefault: true, run: () => { onSave(); return true } }]
    if (onSubmit) keys.push({ key: 'Mod-Enter', preventDefault: true, run: () => { onSubmit(); return true } })
    view = new EditorView({
      parent: el,
      state: EditorState.create({ doc: '', extensions: [basicSetup, langComp.of(langExt(lang)), syntaxHighlighting(hl), theme, listen, keymap.of(keys)] }),
    })
  })
  onDestroy(() => view?.destroy())
</script>

<div class="cm" bind:this={el}></div>

<style>
  .cm { height: 100%; }
</style>
