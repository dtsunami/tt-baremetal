"""Complete geometry backward: full analytic gradient dL/dC -> dL/d(mean gx,gy; cov a,b,c; opacity; color),
grad-checked vs finite differences, then train the whole Gaussian to fit a target — positions and shapes
move, not just color. The trained scene is rendered on the bare-metal Tensix pipeline to confirm.

Forward (matches the device pipeline's math): v1=sa(px-gx)+m12(py-gy), v2=m22(py-gy),
E=-0.5(v1^2+v2^2), ar=exp(E), a=op*ar, front-to-back composite C=sum_i T_i a_i col_i, T_i=prod_{j<i}(1-a_j).
"""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")

K, size = 12, 16
PIX = [(px, py) for py in range(size) for px in range(size)]
P = len(PIX)

def whiten(a, b, c):
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa; m22 = math.sqrt(max(c - b*b/a, 1e-8))
    return sa, m12, m22

def forward(prm, order, need_interm=False):
    gx, gy, a, b, c, op, col = prm
    W = [whiten(a[i], b[i], c[i]) for i in range(K)]
    C = [[0.0, 0.0, 0.0] for _ in range(P)]
    interm = {"v1": [[0.0]*K for _ in range(P)], "v2": [[0.0]*K for _ in range(P)],
              "ar": [[0.0]*K for _ in range(P)], "al": [[0.0]*K for _ in range(P)],
              "w": [[0.0]*K for _ in range(P)], "T": [[0.0]*K for _ in range(P)]}
    for p, (px, py) in enumerate(PIX):
        T = 1.0
        for i in order:
            sa, m12, m22 = W[i]
            v1 = sa*(px-gx[i]) + m12*(py-gy[i]); v2 = m22*(py-gy[i])
            ar = math.exp(max(-0.5*(v1*v1+v2*v2), -60.0)); al = op[i]*ar
            w = T*al
            for ch in range(3): C[p][ch] += w*col[i][ch]
            if need_interm:
                interm["v1"][p][i]=v1; interm["v2"][p][i]=v2; interm["ar"][p][i]=ar
                interm["al"][p][i]=al; interm["w"][p][i]=w; interm["T"][p][i]=T
            T *= (1.0 - al)
    return (C, interm) if need_interm else C

def loss_of(C, target):
    return sum((C[p][ch]-target[p][ch])**2 for p in range(P) for ch in range(3)) / (P*3)

def backward(prm, order, interm, dLdC):
    gx, gy, a, b, c, op, col = prm
    W = [whiten(a[i], b[i], c[i]) for i in range(K)]
    g = {k: [0.0]*K for k in ("gx","gy","a","b","c","op")}
    gcol = [[0.0]*3 for _ in range(K)]
    for p, (px, py) in enumerate(PIX):
        # dL/dw_i and dL/dcolor
        dLdw = {}
        for i in order:
            dLdw[i] = sum(dLdC[p][ch]*col[i][ch] for ch in range(3))
            for ch in range(3): gcol[i][ch] += interm["w"][p][i]*dLdC[p][ch]
        # composite backward: dL/dalpha via suffix-sum over sorted order
        S = 0.0
        for i in reversed(order):
            al = interm["al"][p][i]; T = interm["T"][p][i]; w = interm["w"][p][i]
            dLdal = dLdw[i]*T - S/max(1.0-al, 1e-6)
            S += dLdw[i]*w
            ar = interm["ar"][p][i]
            g["op"][i] += dLdal*ar                     # alpha = op*ar
            dLdE = (dLdal*op[i])*ar                     # ar=exp(E) -> dL/dE = dL/dar*ar
            v1 = interm["v1"][p][i]; v2 = interm["v2"][p][i]
            dLdv1 = dLdE*(-v1); dLdv2 = dLdE*(-v2)      # E=-0.5(v1^2+v2^2)
            sa, m12, m22 = W[i]
            dsa  = dLdv1*(px-gx[i]); dm12 = dLdv1*(py-gy[i]); dm22 = dLdv2*(py-gy[i])
            g["gx"][i] += dLdv1*(-sa)
            g["gy"][i] += dLdv1*(-m12) + dLdv2*(-m22)
            # whitening backward (sa,m12,m22) -> (a,b,c)
            ai, bi = a[i], b[i]
            g["a"][i] += dsa*(0.5/sa) + dm12*(-0.5*bi/(ai*sa)) + dm22*((bi*bi/(ai*ai))/(2*m22))
            g["b"][i] += dm12*(1.0/sa) + dm22*(-bi/(ai*m22))
            g["c"][i] += dm22*(1.0/(2*m22))
    return g, gcol

def scene(seed):
    rnd = random.Random(seed)
    gx=[rnd.uniform(3,13) for _ in range(K)]; gy=[rnd.uniform(3,13) for _ in range(K)]
    a=[];b=[];c=[]
    for _ in range(K):
        s1=rnd.uniform(2,4); s2=rnd.uniform(2,4); th=rnd.uniform(0,math.pi)
        ct,st=math.cos(th),math.sin(th); S00=ct*ct*s1*s1+st*st*s2*s2
        S01=ct*st*(s1*s1-s2*s2); S11=st*st*s1*s1+ct*ct*s2*s2; det=S00*S11-S01*S01
        a.append(S11/det); b.append(-S01/det); c.append(S00/det)
    op=[rnd.uniform(0.4,0.9) for _ in range(K)]
    col=[[rnd.random(),rnd.random(),rnd.random()] for _ in range(K)]
    z=[rnd.random() for _ in range(K)]
    return [gx,gy,a,b,c,op,col], z

def main():
    tprm, z = scene(11)                                  # target
    order = sorted(range(K), key=lambda i: z[i])
    target = forward(tprm, order)

    # ---- grad-check the analytic backward vs finite differences ----
    prm0, _ = scene(22)
    C0, im0 = forward(prm0, order, need_interm=True)
    dLdC = [[2.0*(C0[p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
    g, gcol = backward(prm0, order, im0, dLdC)
    def fd(setter, base):
        eps=1e-3
        p2=[list(x) if isinstance(x,list) else x for x in prm0]  # shallow copy of param arrays
        p2=[list(prm0[k]) for k in range(6)]+[[list(cc) for cc in prm0[6]]]
        setter(p2, base+eps); Lp=loss_of(forward(p2, order), target)
        setter(p2, base-eps); Lm=loss_of(forward(p2, order), target)
        return (Lp-Lm)/(2*eps)
    checks=[]
    for name,idx,arr in [("gx",0,0),("gy",1,3),("a",2,1),("b",3,2),("c",4,5),("op",5,4)]:
        an=g[name][arr]
        num=fd(lambda p,val,i=idx,j=arr: p[i].__setitem__(j,val), prm0[idx][arr])
        checks.append((name,an,num,abs(an-num)/(abs(num)+1e-9)))
    ancol=gcol[3][1]
    numcol=fd(lambda p,val: p[6][3].__setitem__(1,val), prm0[6][3][1])
    checks.append(("color",ancol,numcol,abs(ancol-numcol)/(abs(numcol)+1e-9)))
    print("GRAD-CHECK (analytic vs finite-diff):")
    for nm,an,num,rel in checks:
        print(f"  {nm:6s} analytic={an:+.5e} numeric={num:+.5e} rel={rel:.1e} {'OK' if rel<1e-2 else 'BAD'}")
    ok = all(rel<1e-2 for *_,rel in checks)
    print("  -> geometry backward", "CORRECT" if ok else "MISMATCH")

    # ---- train the full Gaussian to fit the target ----
    prm, _ = scene(22)
    m={k:[0.0]*K for k in ("gx","gy","a","b","c","op")}; v={k:[0.0]*K for k in ("gx","gy","a","b","c","op")}
    mc=[[0.0]*3 for _ in range(K)]; vc=[[0.0]*3 for _ in range(K)]
    b1,b2,eps=0.9,0.999,1e-8; lr={"gx":0.15,"gy":0.15,"a":2e-3,"b":2e-3,"c":2e-3,"op":0.02}
    def pos_err():
        return sum(math.hypot(prm[0][i]-tprm[0][i], prm[1][i]-tprm[1][i]) for i in range(K))/K
    print("\nTRAIN full geometry (pos + shape + opacity + color):")
    for step in range(60):
        C, im = forward(prm, order, need_interm=True)
        L = loss_of(C, target)
        dLdC=[[2.0*(C[p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
        g, gcol = backward(prm, order, im, dLdC)
        for name,idx in [("gx",0),("gy",1),("a",2),("b",3),("c",4),("op",5)]:
            for i in range(K):
                m[name][i]=b1*m[name][i]+(1-b1)*g[name][i]; v[name][i]=b2*v[name][i]+(1-b2)*g[name][i]**2
                mh=m[name][i]/(1-b1**(step+1)); vh=v[name][i]/(1-b2**(step+1))
                prm[idx][i]-=lr[name]*mh/(math.sqrt(vh)+eps)
            if name in ("a","c"):
                for i in range(K): prm[idx][i]=max(prm[idx][i],1e-3)
            if name=="op":
                for i in range(K): prm[idx][i]=min(0.99,max(0.05,prm[idx][i]))
        for i in range(K):
            for ch in range(3):
                mc[i][ch]=b1*mc[i][ch]+(1-b1)*gcol[i][ch]; vc[i][ch]=b2*vc[i][ch]+(1-b2)*gcol[i][ch]**2
                mh=mc[i][ch]/(1-b1**(step+1)); vh=vc[i][ch]/(1-b2**(step+1))
                prm[6][i][ch]=min(1.0,max(0.0, prm[6][i][ch]-0.1*mh/(math.sqrt(vh)+eps)))
        if step%10==0 or step==59:
            psnr=99 if L<1e-12 else 10*math.log10(1/L)
            print(f"  step {step:2d}: loss={L:.5f} PSNR={psnr:5.1f} dB  mean|Δpos|={pos_err():.2f}px")

    # ---- render the trained scene on the bare-metal Tensix pipeline (close the loop) ----
    from ttexalens import init_ttexalens
    from bhtop.tensix.loader import TensixLauncher
    from bhtop.tensix import splat as SP
    ctx=init_ttexalens(); coord=TensixLauncher.at(1,2,ctx=ctx).coord
    gs=[(prm[0][i],prm[1][i],prm[2][i],prm[3][i],prm[4][i],prm[5][i],
         prm[6][i][0],prm[6][i][1],prm[6][i][2], 0.0) for i in range(K)]
    r=SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=gs, order=order, verbose=False)
    dev=r["rgb"]
    mse=sum((dev[p][ch]-target[p][ch])**2 for p in range(P) for ch in range(3))/(P*3)
    dpsnr=99 if mse<1e-12 else 10*math.log10(1/mse)
    print(f"\nDEVICE render of the trained scene vs target: {dpsnr:.1f} dB "
          f"(host-trained fit was {10*math.log10(1/loss_of(forward(prm,order),target)):.1f} dB)")
    print("=> geometry backward trains the full Gaussian; the het pipeline renders the result")
    return prm, tprm, order, target

if __name__ == "__main__":
    main()
