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
    treeUrl: '/api/lab/tree',                           // folder browser over the data_movement tree
    fileUrl: (key) => `/api/lab/file?path=${enc(key)}`,
    paramsUrl: (key) => `/api/lab/params?key=${enc(key)}`,   // kernel.json overlay (params + JSON editor)
    configUrl: (key) => `/api/lab/config?key=${enc(key)}`, configSaveUrl: '/api/lab/config',
    keyOf: (f) => f.key, nameOf: (f) => f.name, roleOf: (f) => f.role, lang: () => 'cpp',
    caps: { duplicate: true, revert: true, regen: true },   // in-place edit + restore-to-pristine
    regenUrl: '/api/lab/restore', regenLabel: 'restore',
    inPlaceNote: 'NOC kernels edit the real tt-metal source in place (Run picks up edits, no rebuild); the host loads fixed kernel paths, so the folder browses the live tree. Duplicate makes a scratch variant.',
    dupUrl: '/api/lab/file/duplicate',
  },
  {
    key: 'x280', label: 'X280', sub: 'L2CPU bare-metal RISC-V (xlab)',
    treeUrl: '/api/l2/tree',                                  // hierarchical folder browser
    listUrl: () => '/api/l2/files',                           // (legacy flat list; fallback)
    fileUrl: (key) => `/api/l2/file?name=${enc(key)}`,
    paramsUrl: (key) => `/api/l2/params?key=${enc(key)}`,     // per-kernel meta-params (kernel.json)
    paramsSaveUrl: '/api/l2/params',                          // persist chosen values as defaults
    configUrl: (key) => `/api/l2/config?key=${enc(key)}`, configSaveUrl: '/api/l2/config',  // raw kernel.json (JSON editor)
    cmdUrl: '/api/l2/cmd',                                    // live mailbox op (deploy-time + on-the-fly)
    keyOf: (f) => f.key, nameOf: (f) => f.name, roleOf: (f) => f.role || f.lang, lang: (f) => f.lang,
    // private workspace: full file CRUD + folder ops + regenerate-examples
    caps: { new: true, duplicate: true, rename: true, delete: true, folder: true, regen: true },
    newUrl: '/api/l2/file/new', dupUrl: '/api/l2/file/duplicate',
    renameUrl: '/api/l2/file/rename', delUrl: '/api/l2/file/delete',
    folderNewUrl: '/api/l2/folder/new', folderDupUrl: '/api/l2/folder/duplicate',
    folderRenameUrl: '/api/l2/folder/rename', folderDelUrl: '/api/l2/folder/delete',
    regenUrl: '/api/l2/regenerate', regenLabel: 'restore examples',
  },
  {
    key: 'tensix', label: 'TENSIX', sub: 'compute engines (tlab)',
    treeUrl: '/api/tlab/tree',                          // folder browser over programming_examples
    fileUrl: (key) => `/api/tlab/file?path=${enc(key)}`,
    paramsUrl: (key) => `/api/tlab/params?key=${enc(key)}`,
    configUrl: (key) => `/api/tlab/config?key=${enc(key)}`, configSaveUrl: '/api/tlab/config',
    keyOf: (f) => f.key, nameOf: (f) => f.name, roleOf: (f) => f.role, lang: () => 'cpp',
    caps: { duplicate: true, revert: true, regen: true },
    regenUrl: '/api/tlab/restore', regenLabel: 'restore',
    inPlaceNote: 'TENSIX kernels edit the real programming_example in place (compute/dataflow JIT on the next Run); the host loads fixed kernel paths, so the folder browses the live tree. Duplicate makes a scratch variant.',
    dupUrl: '/api/tlab/file/duplicate',
  },
]

export const byKey = Object.fromEntries(ENGINES.map((e) => [e.key, e]))

// future sections — shown disabled so the hierarchy reads complete
export const SOON = [
  { key: 'dram', label: 'DRAM', sub: 'GDDR6 controllers — soon' },
  { key: 'eth', label: 'ETH', sub: 'Ethernet / chip-to-chip — soon' },
]
