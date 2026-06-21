"""
Pins tensix.abi against the probe-verified Blackhole launch ABI (no device needed).
Runnable WITHOUT pytest:  .venv/bin/python tests/test_tensix_abi.py
"""
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from bhtop.tensix import abi

_fails = 0


def check(name, cond):
    global _fails
    if not cond:
        _fails += 1
    print(f"  {'ok ' if cond else 'FAIL'} {name}")


def make_kcfg(base_tensix=0x10000, rta=(0x100, 0x200, 0, 0, 0), crta=(0x900,) * 5,
              enables=0b00011, host_id=0xABCD, wids=(7, 8, 0, 0, 0)):
    """Build a 112-byte kernel_config_msg_t with known fields at the probe-verified offsets."""
    buf = bytearray(abi.LAUNCH_STRIDE)
    struct.pack_into("<4I", buf, abi.KCFG_BASE_OFF, base_tensix, 0, 0, 0)
    pairs = []
    for i in range(abi.MAX_PROCS):
        pairs += [rta[i], crta[i]]
    struct.pack_into(f"<{abi.MAX_PROCS * 2}H", buf, abi.KCFG_RTA_OFF, *pairs)
    buf[abi.KCFG_MODE_OFF] = 1
    struct.pack_into("<I", buf, abi.KCFG_HOST_ID_OFF, host_id)
    struct.pack_into("<I", buf, abi.KCFG_ENABLES_OFF, enables)
    struct.pack_into(f"<{abi.MAX_PROCS}H", buf, abi.KCFG_WATCHER_IDS_OFF, *wids)
    return bytes(buf)


def test_addresses():
    check("mailbox base", abi.MEM_MAILBOX_BASE == 96)
    check("rd_ptr addr", abi.launch_rd_ptr_addr() == 96 + 12)
    check("launch[0] addr", abi.launch_addr(0) == 96 + 16)
    check("launch[1] addr (stride 112)", abi.launch_addr(1) == 96 + 16 + 112)
    check("go[0] addr", abi.go_addr(0) == 96 + 912)
    check("go[2] addr (stride 4)", abi.go_addr(2) == 96 + 912 + 8)
    check("go_index addr", abi.go_index_addr() == 96 + 960)


def test_decode_kcfg():
    kc = abi.decode_kernel_config(make_kcfg())
    check("kernel_config_base[TENSIX]", kc["kernel_config_base"][abi.CORE_TYPE_TENSIX] == 0x10000)
    check("rta_offset[DM0].rta", kc["rta_offset"][abi.PROC["DM0"]]["rta"] == 0x100)
    check("rta_offset[DM1].rta", kc["rta_offset"][abi.PROC["DM1"]]["rta"] == 0x200)
    check("crta_offset[DM0]", kc["rta_offset"][abi.PROC["DM0"]]["crta"] == 0x900)
    check("mode HOST", kc["mode"] == 1)
    check("host_assigned_id", kc["host_assigned_id"] == 0xABCD)
    check("enables", kc["enables"] == 0b11)
    check("enabled_procs", kc["enabled_procs"] == [0, 1])
    check("watcher_kernel_ids", kc["watcher_kernel_ids"][:2] == [7, 8])


def test_kernel_resolution():
    """The pure watcher_kernel_id -> kernel join (loader._resolve_kernels / _dedup_names)."""
    from bhtop.tensix import loader
    kmap = {7: {"name": "reader", "source": "x/reader.cpp", "hash": "abc", "infra": False},
            8: {"name": "reader", "source": "x/reader.cpp", "hash": "abc", "infra": False},
            9: {"name": "cq_dispatch", "source": "tt_metal/impl/dispatch/k.cpp", "hash": "d", "infra": True}}
    ks = loader._resolve_kernels([7, 8, 0, 0, 0], 0b11, kmap)
    check("resolve: 2 procs resolved", len(ks) == 2 and ks[0]["name"] == "reader")
    check("resolve: infra flag carried", ks[0]["infra"] is False)
    check("dedup names collapses repeats", loader._dedup_names(ks) == ["reader"])
    ks2 = loader._resolve_kernels([9, 0, 0, 0, 0], 0b1, kmap)
    check("resolve: infra kernel flagged", ks2[0]["infra"] is True)


def test_rta_addr():
    kc = abi.decode_kernel_config(make_kcfg(base_tensix=0x20000, rta=(0x40, 0x80, 0, 0, 0)))
    check("rta_l1_addr DM0 = base+off", abi.rta_l1_addr(kc, abi.PROC["DM0"]) == 0x20000 + 0x40)
    check("rta_l1_addr DM1 = base+off", abi.rta_l1_addr(kc, abi.PROC["DM1"]) == 0x20000 + 0x80)
    check("crta_l1_addr DM0", abi.crta_l1_addr(kc, abi.PROC["DM0"]) == 0x20000 + 0x900)


def test_go():
    w = 0x80_05_03_00 | 0  # signal=0x80, master_y=5, master_x=3, off=0  (LE: off,x,y,sig)
    # construct a real little-endian word: bytes [off, x, y, sig]
    word = 0x00 | (3 << 8) | (5 << 16) | (0x80 << 24)
    g = abi.decode_go(word)
    check("decode go signal", g["signal"] == 0x80 and g["signal_name"] == "GO")
    check("decode go master_x", g["master_x"] == 3 and g["master_y"] == 5)
    check("with_signal sets DONE keeps master", abi.with_signal(word, abi.RUN_MSG_DONE)
          == (0x00 | (3 << 8) | (5 << 16) | (0x00 << 24)))
    check("with_signal sets GO", (abi.with_signal(0, abi.RUN_MSG_GO) >> 24) == 0x80)


def test_word_byte_roundtrip():
    words = [0xDEADBEEF, 0x01020304, 0x0]
    check("words->bytes->words", abi.bytes_to_words(abi.words_to_bytes(words)) == words)


def main():
    print("tensix.abi tests")
    test_addresses()
    test_decode_kcfg()
    test_kernel_resolution()
    test_rta_addr()
    test_go()
    test_word_byte_roundtrip()
    print(f"\n{'ALL PASS' if not _fails else str(_fails) + ' FAILED'}")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
