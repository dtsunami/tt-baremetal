import sys
sys.path.insert(0, "/home/starboy/bhtop/src/bhtop/het/poc")
from _imgs import IMGS

def img(name): return f"data:image/png;base64,{IMGS[name]}"

CSS = """
:root{
  --bg:#f3f4f8; --surface:#ffffff; --surface2:#eceef4; --line:#d7dae3;
  --ink:#14161d; --ink2:#4a4f5e; --ink3:#767c8d;
  --tensix:#0f9d8f; --x280:#c9791f; --ok:#2f9e44; --accent:#0f9d8f;
  --shadow:0 1px 2px rgba(20,22,29,.05),0 6px 20px rgba(20,22,29,.06);
}
@media (prefers-color-scheme:dark){
  :root{
    --bg:#0c0e13; --surface:#14171f; --surface2:#1b1f29; --line:#272c38;
    --ink:#e7e9f2; --ink2:#a7adbd; --ink3:#6e7486;
    --tensix:#2bd4c0; --x280:#e8a24a; --ok:#4bd869; --accent:#2bd4c0;
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 30px rgba(0,0,0,.4);
  }
}
:root[data-theme="light"]{
  --bg:#f3f4f8; --surface:#ffffff; --surface2:#eceef4; --line:#d7dae3;
  --ink:#14161d; --ink2:#4a4f5e; --ink3:#767c8d;
  --tensix:#0f9d8f; --x280:#c9791f; --ok:#2f9e44; --accent:#0f9d8f;
  --shadow:0 1px 2px rgba(20,22,29,.05),0 6px 20px rgba(20,22,29,.06);
}
:root[data-theme="dark"]{
  --bg:#0c0e13; --surface:#14171f; --surface2:#1b1f29; --line:#272c38;
  --ink:#e7e9f2; --ink2:#a7adbd; --ink3:#6e7486;
  --tensix:#2bd4c0; --x280:#e8a24a; --ok:#4bd869; --accent:#2bd4c0;
  --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 30px rgba(0,0,0,.4);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:940px;margin:0 auto;padding:clamp(28px,5vw,72px) clamp(20px,4vw,40px)}
.mono{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace}
.eyebrow{font-family:ui-monospace,Menlo,monospace;font-size:12px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--ink3)}
h1{font-size:clamp(30px,5.5vw,52px);line-height:1.05;letter-spacing:-.02em;margin:.35em 0 .3em;
  text-wrap:balance;font-weight:750}
h2{font-size:clamp(19px,2.6vw,24px);letter-spacing:-.01em;margin:0 0 2px;text-wrap:balance}
.lede{font-size:clamp(17px,2.2vw,20px);color:var(--ink2);max-width:62ch;text-wrap:pretty}
.tag{display:inline-flex;align-items:center;gap:6px;font-family:ui-monospace,monospace;font-size:12px;
  padding:3px 9px;border:1px solid var(--line);border-radius:100px;color:var(--ink2);background:var(--surface)}
.tags{display:flex;flex-wrap:wrap;gap:8px;margin-top:18px}
.dot{width:7px;height:7px;border-radius:50%}
section{margin-top:clamp(40px,7vw,68px)}
.sec-head{display:flex;align-items:baseline;gap:12px;margin-bottom:20px;
  border-bottom:1px solid var(--line);padding-bottom:12px}
.sec-head .n{font-family:ui-monospace,monospace;font-size:13px;color:var(--accent)}
.sub{color:var(--ink2);max-width:64ch}
/* stat row */
.stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:22px}
.stat{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px 16px 14px;box-shadow:var(--shadow)}
.stat .v{font-family:ui-monospace,monospace;font-size:clamp(22px,3vw,28px);font-weight:600;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.stat .k{font-size:12.5px;color:var(--ink3);margin-top:3px}
/* architecture split */
.arch{display:grid;grid-template-columns:1fr auto 1fr;gap:0;align-items:stretch;
  border:1px solid var(--line);border-radius:16px;overflow:hidden;background:var(--surface);box-shadow:var(--shadow)}
.eng{padding:22px 22px 24px}
.eng h3{margin:0;font-size:15px;letter-spacing:.02em;display:flex;align-items:center;gap:8px}
.eng .role{font-family:ui-monospace,monospace;font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--ink3);margin-top:2px}
.eng ul{margin:14px 0 0;padding:0;list-style:none;display:flex;flex-direction:column;gap:9px}
.eng li{position:relative;padding-left:18px;font-size:14.5px;color:var(--ink2)}
.eng li::before{content:"";position:absolute;left:0;top:9px;width:6px;height:6px;border-radius:50%;background:currentColor;opacity:.55}
.eng.t{border-right:0}
.eng.t h3{color:var(--tensix)} .eng.t li{color:var(--ink2)}
.eng.x h3{color:var(--x280)}
.bus{display:flex;flex-direction:column;align-items:center;justify-content:center;
  padding:0 14px;background:var(--surface2);border-left:1px solid var(--line);border-right:1px solid var(--line);min-width:96px;text-align:center}
.bus .lbl{font-family:ui-monospace,monospace;font-size:11px;letter-spacing:.1em;color:var(--ink3);text-transform:uppercase}
.bus .arrows{font-size:20px;color:var(--accent);line-height:1;margin:6px 0}
/* ladder */
.ladder{display:flex;flex-direction:column;gap:0;counter-reset:step}
.rung{display:grid;grid-template-columns:auto 1fr auto;gap:16px;align-items:start;
  padding:18px 0;border-top:1px solid var(--line)}
.rung:first-child{border-top:0}
.rung .idx{font-family:ui-monospace,monospace;font-size:13px;color:var(--ink3);
  width:26px;height:26px;border:1px solid var(--line);border-radius:50%;
  display:flex;align-items:center;justify-content:center;background:var(--surface);flex:none;font-variant-numeric:tabular-nums}
.rung .body h4{margin:0 0 3px;font-size:16px;letter-spacing:-.01em}
.rung .body p{margin:0;color:var(--ink2);font-size:14.5px;max-width:70ch}
.rung .metric{font-family:ui-monospace,monospace;font-size:13px;color:var(--ink);
  background:var(--surface2);border:1px solid var(--line);border-radius:8px;padding:5px 10px;white-space:nowrap;
  align-self:center;font-variant-numeric:tabular-nums}
.rung.hi .idx{border-color:var(--accent);color:var(--accent)}
.rung.hi .metric{border-color:color-mix(in srgb,var(--accent) 45%,var(--line));color:var(--accent)}
/* renders */
.gallery{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.shot{background:var(--surface);border:1px solid var(--line);border-radius:14px;overflow:hidden;box-shadow:var(--shadow)}
.shot img{display:block;width:100%;image-rendering:pixelated;background:#0a0b0f}
.shot .cap{padding:12px 15px 14px}
.shot .cap .t{font-size:14px;font-weight:600}
.shot .cap .d{font-size:13px;color:var(--ink3);margin-top:2px}
.shot.wide{grid-column:1/-1}
.callout{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--surface);
  border-radius:0 12px 12px 0;padding:18px 20px;box-shadow:var(--shadow)}
.callout .k{font-family:ui-monospace,monospace;font-size:11.5px;letter-spacing:.12em;text-transform:uppercase;color:var(--accent)}
.callout code{font-family:ui-monospace,monospace;font-size:13px;background:var(--surface2);padding:1px 5px;border-radius:5px}
.edges{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.edge{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:15px 16px}
.edge .t{font-size:14.5px;font-weight:600;margin-bottom:2px}
.edge .d{font-size:13.5px;color:var(--ink2)}
.foot{margin-top:56px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink3);font-size:13px}
.foot .mono{color:var(--ink2)}
b.t{color:var(--tensix)} b.x{color:var(--x280)}
@media (max-width:680px){
  .stats{grid-template-columns:repeat(2,1fr)}
  .arch{grid-template-columns:1fr}
  .eng.t{border-right:0;border-bottom:1px solid var(--line)}
  .bus{flex-direction:row;gap:10px;border:0;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px}
  .bus .arrows{margin:0}
  .gallery,.edges{grid-template-columns:1fr}
  .rung{grid-template-columns:auto 1fr}.rung .metric{grid-column:2;justify-self:start;margin-top:8px}
}
@media (prefers-reduced-motion:no-preference){
  .rung,.stat,.shot{transition:transform .15s ease}
}
"""

def rung(n, title, desc, metric, hi=False):
    return f"""<div class="rung{' hi' if hi else ''}"><div class="idx">{n}</div>
      <div class="body"><h4>{title}</h4><p>{desc}</p></div>
      <div class="metric">{metric}</div></div>"""

BODY = f"""
<div class="wrap">
  <header>
    <div class="eyebrow">Blackhole p150a · bare-metal over tt-exalens · no&nbsp;tt-metal · no&nbsp;ttnn</div>
    <h1>Two engines, one frame.</h1>
    <p class="lede">A Gaussian-splatting renderer where the <b class="x">x280</b> RISC-V+RVV cores and the
      <b class="t">Tensix</b> grid cooperate as co-equal dataflow peers through GDDR — the irregular tier
      and the dense tier, each on the engine built for it. Everything below ran on silicon.</p>
    <div class="tags">
      <span class="tag"><span class="dot" style="background:var(--x280)"></span>x280 — irregular tier</span>
      <span class="tag"><span class="dot" style="background:var(--tensix)"></span>Tensix — dense tier</span>
      <span class="tag"><span class="dot" style="background:var(--ok)"></span>verified on silicon</span>
    </div>
  </header>

  <section>
    <div class="sec-head"><span class="n">01</span><div><h2>The thesis</h2></div></div>
    <p class="sub">Tenstorrent's own stack leaves the x280 idle (a silicon-bug workaround) and its JIT
      won't let the two engines share memory. This POC does the opposite: it extracts the launch and NoC
      primitives from tt-metal into a lean bare-metal harness, then puts <em>both</em> engines to work on
      one real workload — 3D Gaussian splatting — with no host arithmetic in the render loop.</p>
    <div class="stats">
      <div class="stat"><div class="v" style="color:var(--x280)">16</div><div class="k">x280 cores brought up (4 tiles × 4 harts)</div></div>
      <div class="stat"><div class="v" style="color:var(--tensix)">11</div><div class="k">MVMUL+SFPU stages, all on-device</div></div>
      <div class="stat"><div class="v">52.9<span style="font-size:15px"> dB</span></div><div class="k">hetero render vs golden</div></div>
      <div class="stat"><div class="v" style="color:var(--ok)">0</div><div class="k">tt-metal / ttnn / host math</div></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="n">02</span><div><h2>The division of labor</h2>
      <div class="sub">Dense dataflow is what Tensix is for; data-dependent, serial-decision work is what it's bad at — and what the x280 is for. They meet in GDDR.</div></div></div>
    <div class="arch">
      <div class="eng x">
        <h3><span class="dot" style="background:var(--x280)"></span>x280 &nbsp;·&nbsp; RISC-V + RVV</h3>
        <div class="role">irregular tier</div>
        <ul>
          <li>Depth sort — argsort Gaussians front-to-back</li>
          <li>Gather — sorted-gid → params <span style="opacity:.6">(next)</span></li>
          <li>Scatter-add — ordered FP32 grads <span style="opacity:.6">(backward)</span></li>
          <li>16 cores across 4 L2CPU tiles, parallel</li>
        </ul>
      </div>
      <div class="bus"><div class="lbl">GDDR</div><div class="arrows">⇄</div><div class="lbl">shared</div></div>
      <div class="eng t">
        <h3><span class="dot" style="background:var(--tensix)"></span>Tensix &nbsp;·&nbsp; matrix + vector</h3>
        <div class="role">dense tier</div>
        <ul>
          <li>Eval — whitened field as int8 / bf16 matmul (MVMUL)</li>
          <li>Transcendentals — exp / log / square (SFPU)</li>
          <li>Composite — front-to-back blend, as a matmul</li>
          <li>Cold-booted TRISCs, one shared exalens context</li>
        </ul>
      </div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="n">03</span><div><h2>The ladder — every rung on silicon</h2>
      <div class="sub">A real dependency chain: each primitive unlocks the next. Bottom rung was the longest pole; the top is the whole machine.</div></div></div>
    <div class="ladder">
      {rung(1,"Bare-metal MVMUL","Cold-boot the Tensix compute threads over exalens and run the matrix engine with no tt-metal. Bit-exact <code>A@B</code>.","0 / 1024 mismatch")}
      {rung(2,"“Frame it as ints”","The FPU multiplies integer mantissas under the hood; feeding true Int8→Int32 keeps every bit. Exact for the full ±127 range (HiFi4 + sign-magnitude).","±127 exact, 1 matmul")}
      {rung(3,"Int-matmul Gaussian eval","The whitened surrogate field v = φ·ψ as a 2-limb int8 matmul — the eval that renders each splat, on the matrix engine.","α-wtd L1 = 2.7e-4")}
      {rung(4,"SFPU transcendentals","exp · log · square · log1p on the vector unit, bare-metal. (Full-tile coverage needs ITERATIONS=32 — one face otherwise.)","1024 / 1024 datums")}
      {rung(5,"Fully on-device forward","6 MVMUL + 5 SFPU, zero host arithmetic. The serial composite reformulated as a triangular matmul so it fits the two vector engines — no eltwise-binary needed.","52.9 dB",hi=True)}
      {rung(6,"Heterogeneous render","The x280 sorts by depth; Tensix does eval+exp+composite. Both engines, one frame, one shared context.","52.9 dB",hi=True)}
      {rung(7,"Multi-hetero","Three x280 harts each sort a different scene in parallel; Tensix renders all three.","3 harts ∥")}
      {rung(8,"Full fleet","All four L2CPU tiles up — the complete 16-core x280 fleet doing the irregular tier at once.","16 cores")}
      {rung(9,"Direct handoff","x280 sorts AND gathers into depth order; Tensix NoC-reads the result straight from GDDR. The host is out of the data path entirely.","0 / 48 mismatch",hi=True)}
      {rung(10,"DRAM circular buffer","Producer ⇄ consumer through a bounded GDDR ring with produced/acked backpressure — the x280 fills while Tensix drains, both engines concurrent.","12 items · wrapped 8×")}
      {rung(11,"Fused streaming pipeline","The CB drives the render: the x280 sorts each tile and streams the order through the ring; the Tensix forward render is the consumer, acking to unblock the producer. Irregular tier → GDDR → dense tier → image.","4 tiles · 53–54 dB")}
      {rung(12,"Dense operands through the ring","The x280 gathers and tilizes every Gaussian operand — ψ, opacity, color — into shared GDDR; the Tensix render NoC-reads each straight into its matmul input. The host relays zero Gaussian data, and the frame is pixel-identical.","0 host bytes · 52.9 dB",hi=True)}
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="n">04</span><div><h2>The trick that made the composite fit</h2></div></div>
    <div class="callout">
      <div class="k">serial scan → matmul</div>
      <p style="margin:.5em 0 0;color:var(--ink2)">Front-to-back alpha compositing is a serial recurrence — the thing vector hardware hates. Reformulated in log-space it becomes matmuls with constant matrices, so it runs on the engines already proven:
      <code>T = exp( log(1−α) @ S▲ )</code>, and one triangular-plus-identity matrix folds the prefix-sum <em>and</em> the per-splat weight into a single pass. Opacity is a diagonal matmul; the −½·Σv² is a pair-sum matmul. No new kernel type — just MVMUL + SFPU.</p>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="n">05</span><div><h2>On silicon</h2>
      <div class="sub">Rendered on the bare-metal substrate (left) against the exact golden (right). 32×32 tiles, 16 Gaussians, upscaled — pixels are real device output.</div></div></div>
    <div class="gallery">
      <div class="shot wide"><img alt="Fused streaming pipeline: four tiles" src="{img('splat_streaming.png')}">
        <div class="cap"><div class="t">Fused pipeline — x280 sorts, streams through the ring, Tensix renders</div>
          <div class="d">Four tiles streamed x280→GDDR ring→Tensix with backpressure. Top: device. Bottom: golden. 53–54 dB.</div></div></div>
      <div class="shot wide"><img alt="Multi-hetero render: three tiles" src="{img('splat_multihetero.png')}">
        <div class="cap"><div class="t">Multi-hetero — three x280 harts, one Tensix grid</div>
          <div class="d">Top: each tile sorted by a different x280 hart in parallel, composited by Tensix. Bottom: golden. 52.9 / 52.0 / 54.3 dB.</div></div></div>
      <div class="shot"><img alt="Fully on-device render" src="{img('splat_ondevice.png')}">
        <div class="cap"><div class="t">Fully on-device forward</div><div class="d">6 MVMUL + 5 SFPU, no host math · 52.9 dB</div></div></div>
      <div class="shot"><img alt="Int-limb hybrid render" src="{img('splat_bare_metal.png')}">
        <div class="cap"><div class="t">Int-limb eval (exact matmul)</div><div class="d">bare-metal int8 eval + composite · 67.7 dB</div></div></div>
    </div>
  </section>

  <section>
    <div class="sec-head"><span class="n">06</span><div><h2>Honest edges — what turns the POC into the product</h2></div></div>
    <div class="edges">
      <div class="edge"><div class="t">x280 owns projection too</div><div class="d">Sort, gather, and operand layout are on the x280 now; the one-time whitening still runs host-side and hands bf16 coeffs over. Moving it on-chip needs the x280 scalar FPU enabled.</div></div>
      <div class="edge"><div class="t">The backward pass</div><div class="d">Forward is complete on the heterogeneous machine; the gradient scatter-add is the x280's next natural job.</div></div>
      <div class="edge"><div class="t">The number</div><div class="d">Correctness first, by design. A measured head-to-head vs the host-orchestrated path needs the sort and stream at full scale.</div></div>
      <div class="edge"><div class="t">Fleet at scale</div><div class="d">16 cores streaming into the ring, and the x280 seize has a reset-once quirk to work around for a clean 16/16.</div></div>
    </div>
  </section>

  <div class="foot">
    <span class="mono">bhtop / tt-splat</span> · Blackhole p150a · rendered bare-metal over tt-exalens, no tt-metal · every metric measured on silicon.
  </div>
</div>
"""

html = f"<style>{CSS}</style>\n{BODY}"
open("/home/starboy/bhtop/src/bhtop/het/poc/hetero_poc.html","w").write(html)
print("wrote hetero_poc.html", len(html), "bytes")
