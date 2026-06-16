# Linux on the x280 + SSH

**Verdict: yes — you can SSH into Linux running on the x280, out of the box, via `make ssh`.**
[tt-bh-linux](https://github.com/tenstorrent/tt-bh-linux) boots real Debian on the harts and
ships the entire host-side path: a userspace runner emulates virtio devices over the PCIe/NoC
link, networking is **libslirp userspace NAT** (no tap, no bridge, no root), and the host
forwards `localhost:2222 → guest:22`. The default rootfs already includes `openssh-server`.

> One caveat: this needs the **full Debian rootfs** (`build-image.sh` / the prebuilt
> `rootfs.ext4`), **not** the minimal `user-data.yaml` cloud-init path, which doesn't install sshd.

## How it works (so it fits your mental model)

The host program `console/tt-bh-linux` (built from `console/`) spawns a thread per emulated
device and talks to the guest entirely through tile DRAM over the same PCIe→NoC mapping bhtop
uses — there's no kernel networking on the host:

- **Console** = an OpenSBI "virtual UART" (ring buffers in DRAM, eye-catcher `OSBIdbug`), pumped
  by the host (`console.hpp`/`uart_loop`). Kernel bootargs: `rw console=hvc0 earlycon=sbi`. The
  `./console/tt-bh-linux …` process *is* your console (raw terminal; quit with `Ctrl-A x`).
  It is **not** a virtio-console.
- **Network** = a virtio-mmio **virtio-net** device (`network.hpp`), backed by **libslirp**
  (`-lvdeslirp -lslirp`). The virtio ring lives in tile DRAM; the host thread pumps it into
  slirp, which does DHCP/DNS/NAT in userspace. Guest gets `10.0.2.15`; host forwards
  `127.0.0.1:(2222 + l2cpu_idx + 4*ttdevice) → 10.0.2.15:22`. `boot.py` adds the matching
  `virtio,mmio` nodes + reserved-memory to the DTB (IRQ 32 net, 33 disk, via the PLIC).
- **sshd** = `build-image.sh` debootstraps `openssh-server` and creates an empty-password
  `debian` user with `PermitEmptyPasswords yes` → `make ssh` logs straight in.

## Steps

```sh
# build/install + fetch prebuilt OpenSBI/kernel/dtb/rootfs
git clone https://github.com/tenstorrent/tt-bh-linux && cd tt-bh-linux
make install_all
make install_tt_installer
make download_all

# boot Linux on a tile (loads OpenSBI@0x4000_3000_0000, kernel, dtb, rootfs; patches DTB;
# then runs the console/virtio/slirp host runner — this takes over the terminal)
make boot                 # TTDEVICE=<n> L2CPU=<idx> to pick card/tile

# in another terminal: SSH in (port = 2222 + L2CPU + 4*TTDEVICE; tile 0/card 0 -> 2222)
make ssh                  # ssh -p2222 -o User=debian ... localhost   (empty password)
```

What you'd add yourself:
- **Key-based login:** drop your pubkey into `/home/debian/.ssh/authorized_keys` in the image
  (uncomment the copy in `build-image.sh`), since the empty-password login is loopback-NAT only.
- **Cloud-init image:** if you boot `user-data.yaml` instead, add `openssh-server` to its
  `packages:` and a key to `ssh_authorized_keys` (it sets `ssh_pwauth: false`).
- **Multiple cards/cores:** each gets its own forwarded port via `2222 + l2cpu_idx + 4*ttdevice`.

## Relationship to `bhtop-l2cpu`

This loader is the **bare-metal** iterate-live path (asm/C/Rust + RNMI redirect). Full Linux is
tt-bh-linux's heavier path, but both use the same ARC PLL glide + `L2CPU_RESET` flip, so they
coexist on different tiles. Prototype/poke with `bhtop-l2cpu`; hand a tile to tt-bh-linux when
you want a shell.

(Source note: the repo moved to `tenstorrent-riscv-software/tt-bh-linux`; the old
`tenstorrent/tt-bh-linux` raw URLs still serve the same content. Key files: `console/network.hpp`,
`console/console.hpp`, `console/tt-bh-linux.cpp`, `console/Makefile`, `boot.py`, `build-image.sh`,
`user-data.yaml`, root `Makefile` `boot:`/`ssh:` targets.)
