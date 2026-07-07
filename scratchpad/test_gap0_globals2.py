import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
DB, DONE = 0x16000, 0x16010
A_ADDR, B_ADDR, D_ADDR = 0x21000, 0x31000, 0x61000
enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
GLOB = {"unp_ctx(U)":0x16070, "dest_off(M)":0x16074, "cfg_sid(M)":0x16078, "math_dst(M)":0x1607c,
        "pack_ptr(P)":0x16080, "dest_off(P)":0x16084, "cfg_sid(P)":0x16088}
RUNTIME=[1,1,1,1,1,128,128,0,4,4]
pr=lambda *a: print(*a,flush=True)
ctx=init_ttexalens(); coord=TensixLauncher.at(1,2,ctx=ctx).coord
A=[((i*7+k*3)%13)*0.1 for i in range(32) for k in range(32)]
B=[((k*5+j*2)%11)*0.1 for k in range(32) for j in range(32)]
D=[((i+j)%7)*0.1+0.05 for i in range(32) for j in range(32)]
ov={"ELTWISE_BINARY_OP":"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
b=llk_run.build("resident_mm_elw_perf",run_type="L1_TO_L1",fidelity="HiFi4",fp32_acc=False,overrides=ov); assert b["ok"],b["log"][-2000:]
boot_resident("resident_mm_elw_perf",coord,ctx=ctx,runtime_words=RUNTIME,clear_words=48); time.sleep(0.3)
wr(coord,A_ADDR,enc(A),context=ctx); wr(coord,B_ADDR,enc(B),context=ctx); wr(coord,D_ADDR,enc(D),context=ctx)
def readg(): return {k:rd(coord,a,context=ctx) for k,a in GLOB.items()}
snaps={}
for r in range(1,6):   # drive into ring 5 (the stall) too
    wr(coord,DB,[r],context=ctx); t0=time.time()
    while time.time()-t0<4.0 and rd(coord,DONE,context=ctx)!=r: time.sleep(0.004)
    ok=rd(coord,DONE,context=ctx)==r
    snaps[r]=readg()
    pr(f"ring {r}: {'DONE' if ok else 'STALL'}  globals={snaps[r]}")
    if not ok: break
pr("\n=== C-global drift across rings ===")
for k in GLOB:
    vals=[snaps[r][k] for r in sorted(snaps)]
    pr(f"  {k:14s} {vals}{'   <<< DRIFTS' if len(set(vals))>1 else ''}")
