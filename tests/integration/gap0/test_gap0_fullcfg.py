import sys, time
sys.path.insert(0,"/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
DB,DONE=0x16000,0x16010; A_ADDR,B_ADDR,D_ADDR=0x21000,0x31000,0x61000
RD_CNTL,RDDATA=0xFFB12058,0xFFB12078
NIDX=16384
enc=lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
pr=lambda *a: print(*a,flush=True)
ctx=init_ttexalens(); coord=TensixLauncher.at(1,2,ctx=ctx).coord
A=[((i*7+k*3)%13)*0.1 for i in range(32) for k in range(32)]; B=[((k*5+j*2)%11)*0.1 for k in range(32) for j in range(32)]; D=[((i+j)%7)*0.1+0.05 for i in range(32) for j in range(32)]
ov={"ELTWISE_BINARY_OP":"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWMUL;"}
b=llk_run.build("resident_mm_elw_perf",run_type="L1_TO_L1",fidelity="HiFi4",fp32_acc=False,overrides=ov); assert b["ok"]
boot_resident("resident_mm_elw_perf",coord,ctx=ctx,runtime_words=[1,1,1,1,1,128,128,0,4,4],clear_words=48); time.sleep(0.3)
wr(coord,A_ADDR,enc(A),context=ctx); wr(coord,B_ADDR,enc(B),context=ctx); wr(coord,D_ADDR,enc(D),context=ctx)
w32=coord.noc_write32; r32=coord.noc_read32
def full_cfg():
    out=[0]*NIDX
    for i in range(NIDX):
        w32(RD_CNTL,i); out[i]=r32(RDDATA)
    return out
snaps={}
for r in range(1,5):
    wr(coord,DB,[r],context=ctx); t0=time.time()
    while time.time()-t0<4.0 and rd(coord,DONE,context=ctx)!=r: time.sleep(0.004)
    assert rd(coord,DONE,context=ctx)==r
    time.sleep(0.02); t=time.time(); snaps[r]=full_cfg()
    pr(f"ring {r}: dumped {NIDX} config words ({time.time()-t:.1f}s)")
pr("\n=== full 16384-word config diff (same-phase even rings 1,3 vs 2,4; monotonic) ===")
nchg=0
for i in range(NIDX):
    vals=[snaps[r][i] for r in (1,2,3,4)]
    if len(set(vals))==1: continue
    nchg+=1
    even=[snaps[1][i],snaps[3][i]]; odd=[snaps[2][i],snaps[4][i]]
    mono=all(snaps[r+1][i]-snaps[r][i]==snaps[2][i]-snaps[1][i]!=0 for r in (1,2,3))
    tag="  MONOTONIC<<<" if mono else ("  same-phase-DRIFT<<<" if len(set(even))>1 or len(set(odd))>1 else "  toggles")
    pr(f"  cfg[{i:5d}] (0x{0xFFEF0000+i*4:08x})  {vals}{tag}")
pr(f"\n{nchg} of {NIDX} config words changed across rings 1-4")
