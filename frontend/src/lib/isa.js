// isa.js — Tensix ISA (assembly.yaml) client helper. Fetches the decoded opcode/bit-field map
// from /api/tensix/isa once (cached for the session) and renders it: a hover-tooltip HTML string
// for the editor (TT_OP_* mnemonics self-describe) and the data for the reference panel.

import { getJSON } from './api.js'

let _cache = null   // Promise<{available, mnemonics, count}>

export function loadIsa() {
  if (!_cache) _cache = getJSON('/api/tensix/isa').catch(() => ({ available: false, mnemonics: {} }))
  return _cache
}

// Resolve an editor identifier to an instruction record. Accepts the bare mnemonic (MVMUL),
// the TT_OP_ wrapper macro (TT_OP_MVMUL), or the lowercase ckernel form (ckernel::mvmul → mvmul).
export function lookup(mnemonics, word) {
  if (!word || !mnemonics) return null
  let w = word.replace(/^TT_OP_/, '').replace(/^TTI_/, '')
  if (mnemonics[w]) return mnemonics[w]
  const up = w.toUpperCase()
  for (const k in mnemonics) if (k.toUpperCase() === up) return mnemonics[k]
  return null
}

const esc = (s) => String(s == null ? '' : s).replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))

const bits = (a) => a.start_bit == null ? '—'
  : (a.width === 1 ? `[${a.start_bit}]` : `[${a.start_bit + (a.width || 1) - 1}:${a.start_bit}]`)

// HTML for the editor hover tooltip (rendered inside .cm-isa-tip).
export function tipHtml(info) {
  if (!info) return null
  const op = info.opcode != null ? `<span class="op">0x${info.opcode.toString(16)}</span>` : ''
  let h = `<div><span class="h">${esc(info.name)}</span> ${op} <span class="u">${esc(info.unit || '')}${info.category ? ' · ' + esc(info.category) : ''}</span></div>`
  if (info.desc) h += `<div class="d">${esc(info.desc)}</div>`
  if (info.args && info.args.length) {
    h += '<table>' + info.args.map((a) =>
      `<tr><td class="bits">${bits(a)}</td><td class="fn">${esc(a.name)}</td><td class="fd">${esc(a.desc)}</td></tr>`).join('') + '</table>'
  }
  return h
}

export { bits }
