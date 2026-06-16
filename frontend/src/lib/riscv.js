// riscv.js — a tiny RISC-V reference so the cockpit can explain itself.
// Powers hover tooltips on the disassembly and the register/arch views: what each
// instruction does, what each register is for, and what the CSRs hold. Written for
// someone reading x280 (RV64GC) assembly for the first time.

// ---- registers: ABI name + role (x0..x31) ----
const ABI = [
  ['zero', 'hardwired 0 — always reads zero, writes ignored'],
  ['ra', 'return address — where a function call (jal) jumps back to'],
  ['sp', 'stack pointer — top of the stack'],
  ['gp', 'global pointer'],
  ['tp', 'thread pointer'],
  ['t0', 'temporary 0 — caller-saved (not preserved across calls)'],
  ['t1', 'temporary 1 — caller-saved'],
  ['t2', 'temporary 2 — caller-saved'],
  ['s0', 'saved 0 / frame pointer (fp) — callee-saved (preserved across calls)'],
  ['s1', 'saved 1 — callee-saved'],
  ['a0', 'argument 0 / return value — 1st function arg, and where results come back'],
  ['a1', 'argument 1 / return value (high half)'],
  ['a2', 'argument 2'], ['a3', 'argument 3'], ['a4', 'argument 4'],
  ['a5', 'argument 5'], ['a6', 'argument 6'], ['a7', 'argument 7 — also the syscall number'],
  ['s2', 'saved 2 — callee-saved'], ['s3', 'saved 3 — callee-saved'],
  ['s4', 'saved 4 — callee-saved'], ['s5', 'saved 5 — callee-saved'],
  ['s6', 'saved 6 — callee-saved'], ['s7', 'saved 7 — callee-saved'],
  ['s8', 'saved 8 — callee-saved'], ['s9', 'saved 9 — callee-saved'],
  ['s10', 'saved 10 — callee-saved'], ['s11', 'saved 11 — callee-saved'],
  ['t3', 'temporary 3 — caller-saved'], ['t4', 'temporary 4 — caller-saved'],
  ['t5', 'temporary 5 — caller-saved'], ['t6', 'temporary 6 — caller-saved'],
]
// name -> {x, abi, desc} for every spelling (x5 and t0 both resolve)
export const REG = {}
ABI.forEach(([abi, desc], x) => {
  const info = { x, abi, name: `x${x}`, desc }
  REG[`x${x}`] = info
  REG[abi] = info
  if (abi === 's0') REG['fp'] = info
})
export const GPR = ABI.map(([abi], x) => ({ x, abi, name: `x${x}`, desc: ABI[x][1] }))

// ---- CSRs (control & status registers — only hart code can read these) ----
export const CSR = {
  mhartid: 'which hart (hardware thread) this is — 0..3',
  mcycle: 'free-running cycle counter (great cheap timer)',
  minstret: 'instructions retired so far',
  mstatus: 'machine status — interrupt enables, previous privilege, etc.',
  mtvec: 'trap-handler base address (where normal traps jump)',
  mepc: 'trap PC — the instruction address where the last trap happened',
  mcause: 'trap cause — why the last trap happened (a code)',
  mtval: 'trap value — the faulting address/value of the last trap',
  mscratch: 'scratch register for the trap handler',
  mnstatus: 'RNMI status — bit 3 (NMIE) gates resumable-NMI delivery',
  mnepc: 'RNMI PC — where a resumable NMI interrupted',
  mncause: 'RNMI cause',
  mnscratch: 'RNMI scratch register',
  pc: 'program counter — the address of the instruction being run',
}

// ---- instructions (the ones the x280 disassembler emits) ----
export const INSN = {
  lui: 'load upper immediate — set the top 20 bits of a register',
  auipc: 'add upper immediate to PC — make a PC-relative address',
  addi: 'add immediate — reg = reg + a constant (also used for `mv`/`li`)',
  add: 'add two registers', sub: 'subtract', mul: 'multiply',
  div: 'signed divide', divu: 'unsigned divide', rem: 'signed remainder', remu: 'unsigned remainder',
  and: 'bitwise AND', or: 'bitwise OR', xor: 'bitwise XOR',
  andi: 'bitwise AND with a constant', ori: 'bitwise OR with a constant', xori: 'bitwise XOR with a constant',
  sll: 'shift left logical', srl: 'shift right logical', sra: 'shift right arithmetic (sign-keeping)',
  slli: 'shift left by a constant', srli: 'shift right by a constant', srai: 'shift right arithmetic by a constant',
  slt: 'set if less-than (signed)', sltu: 'set if less-than (unsigned)', slti: 'set if less-than a constant',
  ld: 'load 64-bit doubleword from memory', lw: 'load 32-bit word', lh: 'load 16-bit halfword', lb: 'load byte',
  lwu: 'load 32-bit word (zero-extended)', lhu: 'load halfword (zero-extended)', lbu: 'load byte (zero-extended)',
  sd: 'store 64-bit doubleword to memory', sw: 'store 32-bit word', sh: 'store halfword', sb: 'store byte',
  beq: 'branch if equal', bne: 'branch if not equal',
  blt: 'branch if less-than (signed)', bge: 'branch if greater-or-equal (signed)',
  bltu: 'branch if less-than (unsigned)', bgeu: 'branch if greater-or-equal (unsigned)',
  beqz: 'branch if zero', bnez: 'branch if non-zero', blez: 'branch if ≤ 0', bgez: 'branch if ≥ 0',
  jal: 'jump and link — call a function (saves return addr in ra)',
  jalr: 'jump and link register — indirect call / return',
  j: 'jump (unconditional)', jr: 'jump to register (indirect)',
  ret: 'return from a function (jump to ra)',
  call: 'call a function', tail: 'tail-call a function',
  mv: 'move — copy one register to another', li: 'load immediate — put a constant in a register',
  la: 'load address — put a symbol address in a register', nop: 'no operation', neg: 'negate', not: 'bitwise NOT',
  csrr: 'read a CSR into a register', csrw: 'write a register to a CSR',
  csrrw: 'atomically swap register ↔ CSR', csrrs: 'set CSR bits', csrrc: 'clear CSR bits',
  csrsi: 'set CSR bits by an immediate', csrci: 'clear CSR bits by an immediate',
  fence: 'order memory accesses (make earlier ones visible first)',
  'fence.i': 'sync the instruction cache with memory — needed after writing code',
  wfi: 'wait for interrupt — pause the hart until something wakes it',
  ecall: 'environment call (into firmware/OS)', ebreak: 'breakpoint trap',
  mret: 'return from a machine trap', mnret: 'return from a resumable NMI',
  sext: 'sign-extend', zext: 'zero-extend',
}

const REG_RE = /\b(x(?:[12]?\d|3[01])|zero|ra|sp|gp|tp|fp|t[0-6]|s(?:1[01]|[0-9])|a[0-7])\b/g
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))
const insnDesc = (m) => INSN[m] || INSN[m.replace(/^c\./, '')] || (m.startsWith('f') ? 'floating-point op' : null)

function tipReg(tok) {
  const r = REG[tok]
  return r ? `${tok} = ${r.name}${r.abi !== r.name ? ` (${r.abi})` : ''} — ${r.desc}` : null
}

// wrap any register tokens in `text` with tooltip spans
function ops(text) {
  return esc(text).replace(REG_RE, (m) => {
    const tip = tipReg(m)
    return tip ? `<span class="dr" data-tip="${esc(tip)}">${m}</span>` : m
  })
}

// Render objdump output to highlighted HTML with [data-tip] hovers.
export function renderDisasm(text) {
  const out = []
  for (const raw of (text || '').split('\n')) {
    const lbl = raw.match(/^([0-9a-f]+)\s+<(.+)>:\s*$/)
    if (lbl) { out.push(`<div class="dlabel">${lbl[1]} &lt;${esc(lbl[2])}&gt;:</div>`); continue }
    const m = raw.match(/^\s*([0-9a-f]+):\s+([0-9a-f]+)\s+(\S+)(?:\s+([^#]*?))?\s*(?:#\s*(.*))?$/)
    if (!m) { if (raw.trim()) out.push(`<div class="dmisc">${esc(raw)}</div>`); continue }
    const [, addr, hex, mn, operands, comment] = m
    const d = insnDesc(mn)
    const mnHtml = d ? `<span class="dm" data-tip="${esc(`${mn} — ${d}`)}">${esc(mn)}</span>` : `<span class="dm">${esc(mn)}</span>`
    out.push(
      `<div class="dline"><span class="da">${addr}:</span> <span class="dhex">${hex}</span> ` +
      `${mnHtml}${operands ? ' <span class="dop">' + ops(operands.trim()) + '</span>' : ''}` +
      `${comment ? ' <span class="dcom"># ' + esc(comment) + '</span>' : ''}</div>`)
  }
  return out.join('')
}
