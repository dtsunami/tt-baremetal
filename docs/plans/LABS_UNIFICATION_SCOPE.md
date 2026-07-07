# Labs Unification — Scope (nlab / xlab / tlab)

Scoping for unifying **Kernel Lab (nlab, NIU/data-movement)**, **Hart Lab (xlab, L2CPU x280)**,
and a **future Tensix Compute Lab (tlab)** over a shared toolchain / telemetry / docs framework.
Produced by an 8-agent code scan (6 mappers → synthesis → adversarial critique). All claims are
grounded in real `file:line` references.

## Verdict

**Worth doing, but narrower than a "unify everything" rewrite.** Unify the **chassis** (transport
class, path-safety, error-parsing, and especially the duplicated *frontend lab shell*); keep the
**engines** separate (the device-owner core, the L2 bring-up path, the three telemetry data models,
the per-lab project models). Do the **frontend extraction first** — it's the bulk of the real
duplication and carries **zero device risk**. Treat **tlab as a separate, spike-first effort**, not
a driver of today's abstractions (its headline metric doesn't exist yet).

---

## The three labs at a glance

| | nlab (Kernel Lab) | xlab (Hart Lab) | tlab (future) |
|---|---|---|---|
| target | tt-metal data-movement kernels (drive NoC/NIU) | L2CPU x280 bare-metal RISC-V | Tensix compute engines |
| toolchain | `ninja` host build + tt-metal **JIT** of device kernels | sfpi gcc/rustc → flat image (TempDir) | tt-metal build/JIT (reuse nlab) |
| device model | tt-metal **subprocess owns the chip** (`mode=busy`, polling paused), **resets device on init** | **live ctx reuse** (`L2cpu(ctx=self.ctx)`), no reset | rides tt-metal (like nlab) |
| telemetry | NIU flit counters (per-NoC footprint) + DPRINT + profiler BW | **cooperative** per-hart DRAM slots streamed `/ws/l2cpu` + arch dump + plot | TRISC profiler zones — **does not exist yet** |
| project model | host `test_*.cpp` + `kernels/` device split, edit **in-place + `.orig`** | flat file list, **seeded private workspace** | **triad** (reader/compute/writer), one source compiled 3× |
| maturity | established | **most evolved (template)** | unbuilt |

---

## Already shared — the FROZEN core (do not touch)

- **Single-device-owner model**: one `ThreadPoolExecutor(max_workers=1)` (`device.py:43`) + the
  `_run(fn,*a)` chokepoint (`device.py:80`) serialize *every* PCIe touch (NoC poll, inject, tt-metal
  subprocess, all L2 reg/deploy/bringup/telemetry). This is load-bearing — it's what prevents NoC0
  wedges. **Only ever wrapped, never rewired.**
- **Mode state machine** (`init/polling/injecting/busy/error` + `_paused`) and **SAFE_KINDS**
  `{tensix,dram,eth}` hang-hazard gating (never NIU-read ARC/Security/PCIe/L2CPU). HTTP-409 gating at
  the route edge is the same convention everywhere.
- **CPU-vs-device split**: hardware → `_run` (device thread); filesystem/compile/build/docs →
  `asyncio.to_thread` (default pool), so live telemetry keeps running through a build.
- **`isa.py`**: one shared module (`tree()` + `doc(path)` over tt-isa-documentation, 24h cache),
  already hit by both labs at `/api/isa/*`.
- **Chip substrate**: `/api/floorplan|status|card.png|tile`, the `/ws/telemetry` frame keyed by
  noc0 `(x,y)`, `stores.js` `frame`/`connected`, the App header NoC monitor + reset banner + router.
  **noc0 (x,y) is already the universal observation key** for NoC bw, kernel footprint, AND L2CPU
  tile placement.
- **Frontend utils**: `api.js` (getJSON/postJSON/fmtBW), `md.js` (mdToHtml). Schema convention:
  Pydantic request bodies, plain-dict responses, `ValueError→HTTPException`.

## Genuinely duplicated — the REAL unify targets

**Backend (small, cheap, safe):**
- `Broadcaster`: the `set[Queue(maxsize=4)]` + subscribe + `put_nowait`-drop pattern, written 2×
  (`device.py:148-164` vs `:595-600`). ~25 real lines. ← do this.
- `safe_path(root, rel, exts)`: `lab._safe` (`lab.py:52`) and `l2lab._safe` (`l2lab.py:39`) are
  near-identical (differ only in extension set). ← do this.
- `parse_compiler_errors(log)`: `lab._ERR_RE` and `l2lab._ERR_RE` (~3-line regexes). ← fold in
  silently (not a milestone).
- The `*/last` job-poller shape (`_kernel_job`/`_build_job`/`_l2_bringup_job`) — **share only the
  guard + `_last[label]` + finally-restore**, NOT the dispatch policy (see "do not unify").

**Frontend (the bulk — ~35–45% copy-paste, zero device risk):**
- CodeMirror theme + `HighlightStyle` — **byte-identical** (`KernelLab:119-140` == `HartLab:120-141`;
  HartLab's comment literally says "same theme/highlight as Kernel Lab"). → `<CodeEditor>` (language
  via Compartment; nlab passes `cpp`).
- ISA-tree browser: `flatten`/`isaVisible`/`openIsa`/`onDocClick`/`toggleDir` — copy-pasted. →
  `useIsaTree` / `<DocsPane>`.
- The three `setTimeout`-recursion pollers (pollBuild/pollRun 2500ms, pollBringup 1200ms) →
  `pollJob(url, interval, onDone)`.
- ~26 overlapping CSS selectors, the file list, the tabbed observe pane, the resizable layout →
  `<LabShell>` (HartLab's resizable grid as template; **opt-in** for KernelLab — it's a UX change).

## Do NOT unify — real divergence to preserve

- The `_run` executor + mode machine (frozen).
- The **L2 bring-up/seize path** (PLL glide, RNMI seize, trampoline, `mnstatus.NMIE` re-arm,
  dual-transport tt-exalens NoC + pyluwen `axi_*` with `ARC_ALLOW` + address guards + canaries).
  Reset-once per ASIC reset — kept **verbatim inside the job fn**, gated on real silicon.
- The **two WebSockets**: `/ws/telemetry` is push-only + last-frame seed (`server.py:188`);
  `/ws/l2cpu` is dual-task with client steering `{tile,hz}` (`server.py:312`). Share the
  `Broadcaster` class only — keep the **control/receiver path L2-specific**. No unified envelope, no
  generic `/ws/<lab>` route.
- The **three telemetry data models**: NIU hardware counters (autonomous, noc0-keyed, 1-deep) vs
  cooperative per-hart DRAM slots (2-deep tile→hart, "paused" when idle) vs TRISC profiler zones
  (post-hoc CSV, doesn't exist). Transport unifies; the data model and frontend consumers do not.
- The **per-lab project/file models** (host/device split vs flat list vs triad) and **backing modes**
  (in-place+`.orig` vs seeded-workspace).
- The **tt-metal device-reset-on-init recovery** (`_run_kernel_blocking`, `device.py:264-277`:
  catch→`_init_device()`→retry→**poller re-baseline**; "zeroed counters ⇒ absolutes = footprint"
  inversion). Stays inside the kernel fn, never absorbed into a generic helper.

---

## Recommended sequence (reconciled with the critique)

| phase | what | device risk | priority |
|---|---|---|---|
| **0. Characterization test** | pin current behavior: mode transitions, 409 gating, **byte-identical `_frame()`/`_broadcast_mode()` dicts**, WS seeding/steering, SAFE_KINDS never violated. Add the 4 helpers *unused*. | none | mandatory net |
| **A. Frontend extraction** | `<CodeEditor>`, `useIsaTree`/`<DocsPane>`, `pollJob`; `<LabShell>` opt-in. Migrate both routes. | **none** | **do first** (highest ROI) |
| **B. nlab backend** | `safe_path`, `parse_compiler_errors`, `Broadcaster` (seeded), thin `run_job` guard. tt-metal recovery stays in the fn. | none (subprocess + host reads) | after 0 |
| **C. xlab backend** | same helpers; bring-up/seize/dual-transport **verbatim** inside the fn; keep `/ws/l2cpu` receiver. | **medium** | **only if silicon available to gate** (on-chip smoke test); else skip, do A instead |
| **D. tlab** | **separate effort.** Spike first: run one tt-metal compute example, confirm `RISCV_2/3/4` zones exist in `profile_log_device.csv`. Only then design the compute frame + triad model + MathFidelity knobs. | medium | later |
| **E. chip-as-launcher** | click Tensix→open lab; click L2CPU tile→open Hart Lab focused there; deployment/util overlays. **More net-new than it looks** (no click-to-lab nav today; no stream carries "this L2CPU tile has live harts"). | low | optional, after A–C |

**Critique's key corrections folded in:** (1) drop tlab from this initiative — its headline metric
(TRISC math utilization) **does not exist** in `aggregate_bw` today (`metal.py:97-125` computes only
wall-cycle + bytes) and depends on unverified profiler instrumentation; designing shared helpers to
fit an unbuilt lab is premature. (2) Drop the unified frame envelope / generic WS route. (3) `run_job`
is a **thin guard**, not a lifecycle manager — the three "long ops" have different (executor, pause,
mode) tuples and teardown contracts. (4) Frontend (A) is highest-ROI/lowest-risk → first or parallel
with B. (5) `parse_compiler_errors`/DocsSource are tiny — fold in, don't make them milestones.

## What unification buys
- ~35–45% less frontend code; one place to improve editor/docs/telemetry-plot for all labs.
- A consistent "develop → deploy → observe → learn" UX across engines.
- tlab later costs ~30% genuinely-new (triad model + TRISC telemetry + golden-compare) instead of a
  full lab clone — *if and only if* the chassis is shared and the compute-telemetry spike validates.

## Decisions needed from you
1. **Scope/order:** land Phases 0–B (+C if silicon) for the two real labs now, and defer tlab to a
   spike — or push for all three together? (Recommendation: defer tlab.)
2. **Silicon gate:** is a real Blackhole available to gate the xlab backend migration (Phase C), and
   can it tolerate a `tt-smi -r 0` mid-refactor? If no → skip C, do frontend only.
3. **KernelLab UX:** OK to give KernelLab HartLab's resizable/persisted layout (a visible UX change),
   or keep its fixed grid and only share the editor/docs internals?
4. **Docs:** keep three separate curated corpora (only the ISA browser is shared content), or merge
   into one index? (Merging changes product content, not just code.)
5. **Chip launcher:** when you click a Tensix tile, open Kernel *or* Compute lab — a mode toggle, or
   pick by active lab?

---
*Scan: 8 agents, ~555k tokens, 6 subsystem maps + synthesis + adversarial critique. Raw output:
`/tmp/.../tasks/wu429t9mb.output`.*
