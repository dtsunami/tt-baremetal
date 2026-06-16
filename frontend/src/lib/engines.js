// engines.js — the device-hierarchy section descriptors shared by DeviceTree (left) and
// DeviceBrowser (editor + observe). One place defines each engine's files endpoint, file-op
// CAPABILITIES (capability-aware: X280 is a private workspace → full CRUD; NOC/TENSIX edit the
// real tt-metal tree in place → duplicate + revert only), and how to read a file's name/role/lang.

export const enc = encodeURIComponent

export const langOf = (name) =>
  name.endsWith('.rs') ? 'rust'
  : (name.endsWith('.s') || name.endsWith('.S')) ? 'asm'
  : /\.(cpp|cc|hpp|h)$/.test(name) ? 'cpp'
  : 'c'

export const ENGINES = [
  {
    key: 'noc', label: 'NOC', sub: 'data-movement kernels (nlab)',
    // project selector
    selUrl: '/api/lab/projects', selKind: 'project',
    listUrl: (sel) => `/api/lab/files?project=${enc(sel)}`,
    fileUrl: (key) => `/api/lab/file?path=${enc(key)}`,
    keyOf: (f) => f.path, nameOf: (f) => f.name, roleOf: (f) => f.role, lang: () => 'cpp',
    caps: { duplicate: true, revert: true },           // in-place: new/delete/rename N/A
    inPlaceNote: 'NOC kernels edit the real tt-metal source in place (Run picks up edits, no rebuild); the host loads fixed kernel paths, so new/rename/delete don’t change what Run executes. Duplicate makes a scratch variant.',
    dupUrl: '/api/lab/file/duplicate',
  },
  {
    key: 'x280', label: 'X280', sub: 'L2CPU bare-metal RISC-V (xlab)',
    listUrl: () => '/api/l2/files',
    fileUrl: (key) => `/api/l2/file?name=${enc(key)}`,
    keyOf: (f) => f.name, nameOf: (f) => f.name, roleOf: (f) => f.lang, lang: (f) => f.lang,
    caps: { new: true, duplicate: true, rename: true, delete: true },   // private workspace
    newUrl: '/api/l2/file/new', dupUrl: '/api/l2/file/duplicate',
    renameUrl: '/api/l2/file/rename', delUrl: '/api/l2/file/delete',
  },
  {
    key: 'tensix', label: 'TENSIX', sub: 'compute engines (tlab)',
    selUrl: '/api/tlab/examples', selKind: 'example',
    listUrl: (sel) => `/api/tlab/files?example=${enc(sel)}`,
    fileUrl: (key) => `/api/tlab/file?path=${enc(key)}`,
    keyOf: (f) => f.path, nameOf: (f) => f.name, roleOf: (f) => f.role, lang: () => 'cpp',
    caps: { duplicate: true, revert: true },
    inPlaceNote: 'TENSIX kernels edit the real programming_example in place (compute/dataflow JIT on the next Run); the host loads fixed kernel paths, so new/rename/delete don’t change what Run executes. Duplicate makes a scratch variant.',
    dupUrl: '/api/tlab/file/duplicate',
  },
]

export const byKey = Object.fromEntries(ENGINES.map((e) => [e.key, e]))

// future sections — shown disabled so the hierarchy reads complete
export const SOON = [
  { key: 'dram', label: 'DRAM', sub: 'GDDR6 controllers — soon' },
  { key: 'eth', label: 'ETH', sub: 'Ethernet / chip-to-chip — soon' },
]
