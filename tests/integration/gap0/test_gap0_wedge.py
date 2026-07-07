import sys, time, re
sys.path.insert(0,"/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
DB,DONE=0x16000,0x16010; A_ADDR,B_ADDR,D_ADDR=0x21000,0x31000,0x61000
enc=lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
pr=lambda *a: print(*a,flush=True)
ctx=init_ttexalens(); coord=TensixLauncher.at(1,2,ctx=ctx).coord
blk=ctx.devices[0].get_block(coord); db=blk.get_debug_bus()
# stall/deadlock signals: what is each thread blocked on?
STALL=re.compile(r"(stalled|_busy$|semget_pending|sync_activated|data_ready|scoreboard_stall|scoreboard_pending|src.*ready|src.*gate|write_ready|dvalid_clear|bank_switch)",re.I)
sigs=[n for n in db.signal_names if STALL.search(n) and not re.search(r"icache|perf_cnt|lsq|rq_head|mailbox",n,re.I)]
A=[((i*7+k*3)%13)*0.1 for i in range(32) for k in range(32)]
B=[((k*5+j*2)%11)*0.1 for k in range(32) for j in range(32)]
D=[((i+j)%7)*0.1+0.05 for i in range(32) for j in range(32)]
ov={"ELTWISE_BINARY_OP":"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
b=llk_run.build("resident_mm_elw_perf",run_type="L1_TO_L1",fidelity="HiFi4",fp32_acc=False,overrides=ov); assert b["ok"],b["log"][-2000:]
boot_resident("resident_mm_elw_perf",coord,ctx=ctx,runtime_words=[1,1,1,1,1,128,128,0,4,4],clear_words=48); time.sleep(0.3)
wr(coord,A_ADDR,enc(A),context=ctx); wr(coord,B_ADDR,enc(B),context=ctx); wr(coord,D_ADDR,enc(D),context=ctx)
def snap(): 
    s={}
    for n in sigs:
        try: s[n]=int(db.read_signal(n))
        except Exception: pass
    return s
healthy=None
for r in range(1,7):
    wr(coord,DB,[r],context=ctx); t0=time.time()
    while time.time()-t0<4.0 and rd(coord,DONE,context=ctx)!=r: time.sleep(0.004)
    if rd(coord,DONE,context=ctx)!=r:
        pr(f"\n*** WEDGE at ring {r} ***")
        wedged=snap()
        pr("PCs:", {t:hex(blk.get_risc_debug(t).get_pc()) for t in ('trisc0','trisc1','trisc2')})
        pr("\n=== stall/busy signals: HEALTHY(ring4 quiescent) vs WEDGED(ring5) ===")
        for n in sorted(set(healthy)&set(wedged)):
            h,w=healthy[n],wedged[n]
            pr(f"  {n:52s} healthy={h}  wedged={w}{'   <<< DIFFERS' if h!=w else ''}")
        break
    healthy=snap()   # last healthy snapshot (ring r quiescent)
    pr(f"ring {r} DONE")
