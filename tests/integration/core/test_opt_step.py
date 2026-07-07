"""Standalone silicon test of the x280 opt_step kernel: one Adam step (whiten-backward + un-sort + Adam),
device vs an identical host reference. Isolates the kernel logic from the Tensix backward."""
import sys, struct, math, random, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

K = 12
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_step.c"
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]     # f32 -> u32 bits
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

def host_ref(param, gradin, order, m, v, bc1, bc2, lr):
    b1,b2,eps = 0.9,0.999,1e-8
    p = [row[:] for row in param]; m = [row[:] for row in m]; v = [row[:] for row in v]
    for i in range(K):  # returns updated (p,m,v) so the host mirror persists across steps
        o = order[i]; d_sa,d_m12,d_tx,d_m22,d_ty = gradin[i][:5]
        gx,gy,a,b,c = p[o][:5]
        sa = math.sqrt(max(a,1e-8)); m12 = b/sa; t = max(c-b*b/a,1e-8); m22 = math.sqrt(t)
        Dsa = d_sa+d_tx*(-gx); Dm12 = d_m12+d_tx*(-gy); Dm22 = d_m22+d_ty*(-gy)
        g_gx = d_tx*(-sa); g_gy = d_tx*(-m12)+d_ty*(-m22)
        g_a = Dsa*(0.5/sa)+Dm12*(-0.5*b/(a*sa))+Dm22*((b*b/(a*a))/(2*m22))
        g_b = Dm12*(1.0/sa)+Dm22*(-b/(a*m22)); g_c = Dm22*(1.0/(2*m22))
        g = [g_gx,g_gy,g_a,g_b,g_c, gradin[i][5], gradin[i][6], gradin[i][7], gradin[i][8]]
        for j in range(9):
            m[o][j] = b1*m[o][j]+(1-b1)*g[j]; v[o][j] = b2*v[o][j]+(1-b2)*g[j]*g[j]
            mh = m[o][j]*bc1; vh = v[o][j]*bc2; np = p[o][j]-lr[j]*mh/(math.sqrt(vh)+eps)
            if j in (2,4): np = max(np,1e-3)
            elif j == 5: np = min(0.99,max(0.05,np))
            elif j >= 6: np = min(1.0,max(0.0,np))
            p[o][j] = np
    return p, m, v

def main():
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception as e: print("bringup:", type(e).__name__, "(already up, ok)")
    rng = random.Random(3)
    # scene params (orig order): gx,gy,a,b,c,op,c0,c1,c2
    param = []
    for _ in range(K):
        s1,s2,th = rng.uniform(2,4),rng.uniform(2,4),rng.uniform(0,math.pi)
        ct,st = math.cos(th),math.sin(th); S00=ct*ct*s1*s1+st*st*s2*s2
        S01=ct*st*(s1*s1-s2*s2); S11=st*st*s1*s1+ct*ct*s2*s2; det=S00*S11-S01*S01
        param.append([rng.uniform(3,13),rng.uniform(3,13),S11/det,-S01/det,S00/det,
                      rng.uniform(0.4,0.9),rng.random(),rng.random(),rng.random()])
    order = list(range(K)); rng.shuffle(order)
    m = [[0.0]*9 for _ in range(K)]; v = [[0.0]*9 for _ in range(K)]

    # resident init: params + zero adam state + order, ONCE; then drive steps via the doorbell
    dev.wr(0, 0x30005040, [o & 0xFFFFFFFF for o in order])
    dev.wr(0, 0x30005800, [fb(param[i][j]) for i in range(K) for j in range(9)])
    dev.wr(0, 0x30006000, [0]*(K*9)); dev.wr(0, 0x30006400, [0]*(K*9))
    dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0])           # doorbell/done cleared
    words = tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc")
    dev.load(0, 0, words); time.sleep(0.3)
    print("resident:", dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421)

    # DISTINCTIVE LRs (not the kernel's old hardcoded 0.15/2e-3/...) -> a device==host match proves the
    # kernel actually reads lr[9] from the header, i.e. hyperparameters are host-plumbed.
    LR = [0.07,0.03, 5e-3,5e-3,5e-3, 0.011, 0.09,0.09,0.09]
    ok = True
    for step in (1, 2):
        gradin = [[rng.uniform(-0.02,0.02) for _ in range(9)] for _ in range(K)]
        bc1 = 1.0/(1-0.9**step); bc2 = 1.0/(1-0.999**step)
        param, m, v = host_ref(param, gradin, order, m, v, bc1, bc2, LR)   # host mirror persists
        hdr = [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR]
        dev.wr(0, 0x30005100, [fb(gradin[i][j]) for i in range(K) for j in range(9)])
        dev.wr(0, 0x30005000, hdr)                                     # K,step,bc1,bc2,b1,b2,eps,lr[9]
        dev.wr(0, 0x30004000, [step])                                  # ring the doorbell
        got_done = False
        for _ in range(40):
            if dev.rd(0, 0x30004010) == step: got_done = True; break
            time.sleep(0.05)
        if step == 1:
            hb = dev.rdn(0, 0x30005000, 16)
            print("   header readback: K=%d step=%d bc1=%.3f b1=%.3f eps=%.1e lr0=%.4f lr5=%.4f lr6=%.4f" % (
                hb[0], hb[1], bf(hb[2]), bf(hb[4]), bf(hb[6]), bf(hb[7]), bf(hb[12]), bf(hb[13])))
        got = [[bf(u) for u in dev.rdn(0, 0x30005800 + i*9*4, 9)] for i in range(K)]
        if step == 1:
            print("   dev param[0][:3]=", [round(x,4) for x in got[0][:3]], " host=", [round(x,4) for x in param[0][:3]])
        maxrel = max(abs(got[i][j]-param[i][j])/(abs(param[i][j])+1e-6) for i in range(K) for j in range(9))
        print(f"  step {step}: done={got_done}  max_rel_err={maxrel:.2e}  {'PASS' if got_done and maxrel<1e-3 else 'FAIL'}")
        ok = ok and got_done and maxrel < 1e-3
    print("doorbell + 2-step persistence:", "PASS" if ok else "FAIL")

if __name__ == "__main__":
    main()
