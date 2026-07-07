import sys, time
sys.path.insert(0,"/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, write_words_to_device as wr, read_words_from_device as rds
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
DB,DONE=0x16000,0x16010; A_ADDR,B_ADDR,D_ADDR,I_ADDR,OUT=0x21000,0x31000,0x61000,0x71000,0x51000
enc=lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
dec=lambda a,ctx,c: MM.untilize32(MM.unpack_bf16_words(rds(c,a,word_count=512,context=ctx)))
pr=lambda *a: print(*a,flush=True)
ctx=init_ttexalens(); coord=TensixLauncher.at(1,2,ctx=ctx).coord
A=[((i*7+k*3)%13)*0.1 for i in range(32) for k in range(32)]; B=[((k*5+j*2)%11)*0.1 for k in range(32) for j in range(32)]; D=[((i+j)%7)*0.1+0.05 for i in range(32) for j in range(32)]
C1=MM.matmul_golden(A,B); Eg=[C1[i]*D[i] for i in range(1024)]
def relerr(dev,g):
    n=sum(abs(dev[i]-g[i]) for i in range(1024)); d=sum(abs(x) for x in g)+1e-9; return n/d
ov={"ITERATIONS":"constexpr int ITERATIONS = 32;"}
b=llk_run.build("resident_mm_elw_perf",run_type="L1_TO_L1",fidelity="HiFi4",fp32_acc=False,overrides=ov)
if not b["ok"]:
    pr("BUILD FAILED:\n"+b["log"][-2500:]); sys.exit(1)
boot_resident("resident_mm_elw_perf",coord,ctx=ctx,runtime_words=[1,1,1,1,1,128,128,0,4,4],clear_words=48); time.sleep(0.3)
wr(coord,A_ADDR,enc(A),context=ctx); wr(coord,B_ADDR,enc(B),context=ctx); wr(coord,D_ADDR,enc(D),context=ctx); wr(coord,I_ADDR,enc([1.0 if i==j else 0.0 for i in range(32) for j in range(32)]),context=ctx)
npass=0
for r in range(1,51):
    wr(coord,OUT,[0xBADF00D5]*512,context=ctx); wr(coord,DB,[r],context=ctx); t0=time.time()
    while time.time()-t0<4.0 and rd(coord,DONE,context=ctx)!=r: time.sleep(0.004)
    if rd(coord,DONE,context=ctx)!=r: pr(f"[ring {r}] *** STALL ***"); break
    e=relerr(dec(OUT,ctx,coord),Eg); ok=e<0.05; npass+=ok
    pr(f"[ring {r}] DONE rel-err={e:.2e} -> {'PASS' if ok else 'CHECK'}")
pr(f"\n[sfpubin] {npass}/50 rings PASS => {'*** SFPU-BINARY IS MULTI-RING RESIDENT — GAP 0 CLOSED ON TENSIX!! ***' if npass==50 else 'stalls (SFPU path also affected)' if npass<14 else ''}")
