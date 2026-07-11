# Blackhole x280 — Silicon V/F Margining Campaign

Empirical voltage/frequency characterization of the p150a x280 (SiFive RISC-V) core on host **ttstar**, via
`pyluwen` (ARC DVFS + PLL + SPIROM) and `tt-exalens` (x280 hart bring-up/load). Full mechanism reference:
memory `bh-arc-dvfs-voltage`. **2026-07-08.**

---

## >>> RESUME AFTER REBOOT — START HERE <<<

**State at hand-off:** `vdd_max=1000 mV` is written to SPIROM (`cmfwcfg`) and **persists across the reboot**.
exalens was hard-down (ETH harvest invalid — see Incident below); a **host reboot of ttstar** cold-trains ETH and
brings exalens back. Card was left idle (700 mV / 800 MHz / ~70 °C), nothing forcing voltage.

**After the reboot, verify then resume:**
1. Confirm the unlock survived + exalens is back:
   ```
   /home/starboy/bhtop/.venv/bin/python -u -c "
   from pyluwen import PciChip; vl=PciChip(pci_interface=0).as_bh().get_telemetry().vdd_limits
   print('vdd_max', (vl>>16)&0xFFFF)                      # expect 1000
   import sys; sys.path.insert(0,'/home/starboy/bhtop/src')
   from ttexalens import init_ttexalens; init_ttexalens(); print('exalens OK')"
   ```
2. Run the 1 V extension + ceiling probe (self-resets, streamed):
   ```
   cd /home/starboy/bhtop
   timeout 420 .venv/bin/python -u scratchpad/vmin_1v.py > scratchpad/vmin_1v_out.txt 2>&1 &
   ```
   Watch `scratchpad/vmin_1v_out.txt`. Part A = Vmin at 2500/2600/2700 (predict ~919/956/993 mV).
   Part B = pin 1000 mV, push fbdiv up → the true max freq at 1 V (predict ~2720 MHz).
3. Fold the new points into the plot: edit `scratchpad/vf_shmoo.html` (`VMIN`/`LOCKED` arrays), republish the
   SAME artifact (URL `912be7ad-4f62-4ba8-9f4c-a8a9e8632605`, keeps the link).
4. **RESTORE when the campaign ends** (undo the over-volt unlock — sustained >900 mV ages silicon):
   ```
   /home/starboy/bhtop/.venv/bin/python -u -c "
   from pyluwen import PciChip; bh=PciChip(pci_interface=0).as_bh()
   bh.spi_write(0x1f7000, open('/home/starboy/bhtop/scratchpad/spirom_backup/cmfwcfg_0x1f7000.bin','rb').read())
   print('restored; reboot/reset to reload')"
   ```
   then reboot → telemetry `vdd_max` back to 900.

**3 GHz?** Extrapolating the Vmin line (0.37 mV/MHz), 3000 MHz needs ~1104 mV — above the 1000 mV we unlocked,
so **not reachable at 1 V**. It would need another `vdd_max` bump (~1150 mV, more aggressive over-volt) AND the
PLL VCO to lock at 3 GHz (unverified — the ceiling probe finds the real PLL limit). Decide after the 1 V ceiling.

---

## Results so far (measured on silicon)

**Frequency frontier** (freq swept up at fixed voltage, FP-FMA virus): 765mV→2100, 823→2225, 881→2350,
899→2425 MHz. Max stable @ the old 900 mV clamp = **2425 MHz** (fbdiv-overclock, `set_fbdiv_explore`).

**Per-hart Vmin curve** (binary/step-down, voltage-leads-frequency, FP virus): 2100→**771**, 2200→**804**,
2300→**844**, 2400→**882** mV; 2500→>900 (needs ~919). **Linear ~0.37 mV/MHz. hart3 = weak die, the limiter at
every frequency** (hart0 second; both fail at 2500). Max OC within the firmware clamp = **2400 MHz @ 882 mV
(+37% over the 1750 nominal, 3× the 800 default)**. Thermals never the limiter (≤74 °C vs 90 °C throttle).

**1.0 V unlock DONE** (reversible SPIROM edit): `cmfwcfg.chip_limits.vdd_max 900→1000`, flash-verified,
CMFW-adopted (telemetry `vdd_max=1000`). Opens the band for the 2500–2700 MHz points above.

## Tooling (all in `~/bhtop/scratchpad/`, working versions)

- `vmin_1v.py` — the RESUME script (Vmin 2500/2600/2700 + 1 V ceiling probe, temp/current guards).
- `vmin_multi.py` — Vmin-vs-freq (proven linear step-down, self-reset+settle).
- `vf_freq_explore.py` — frequency frontier (fbdiv-overclock at fixed voltages).
- `vf_shmoo.py` / `vf_margin.c` / `vf_margin_fp.c` — per-voltage undervolt sweep + the two self-check viruses.
- `vf_shmoo.html` — the published Shmoo/Vmin artifact (voltage-vertical).
- `spirom_backup/cmfwcfg_0x1f7000.bin` — **the restore backup** (88 B, original vdd_max=900 table).

**New capability in `bhtop.l2cpu.L2cpu`** (committed to source): `arc_msg`, `limits`, `monitor`, `get_voltage`,
`perf_busy`/`perf_idle` (GO_BUSY/GO_IDLE), `force_aiclk`, `force_vdd(mv, allow_over=)`, `set_fbdiv_explore(fb)`.
SPIROM API is on `PciChip.as_bh()` (PciBlackhole): `decode_boot_fs_table` / `get_spirom_table_spi_addr` /
`get_spirom_table_image_size` / `spi_read` / `encode_and_write_boot_fs_table` / `spi_write`.

**Hard-won protocol (memory `bh-arc-dvfs-voltage`):** clean reset per campaign (the search wedges harts, a wedged
hart can't be re-loaded); `force_vdd` ONLY, never `perf_busy`/GO_BUSY (engages CMFW DVFS flows that confound
bare-metal + glitched the load); voltage LEADS frequency (gliding freq up at low V undervolts/wedges);
approach the edge FROM ABOVE (from-below deep-undervolts and hard-wedges weak harts on the shared rail);
self-reset+`time.sleep(18)` INSIDE python before init (chained `reset; python` races re-enumeration); stream to a
file + Monitor (piping through `tail` buffers until EOF); measured-vcore readback every probe (the clamp can't
hose the bracket).

---

## INCIDENT — first hard-down (exalens ETH harvest, 2026-07-08)

**Symptom.** After the `vdd_max=1000` flash edit + its reset (which SUCCEEDED — telemetry read 1000, card healthy),
the next run's warm reset left `init_ttexalens` failing fatally:
`UmdBaseException: Exactly 2 or 14 ETH cores should be harvested on full Blackhole`
(`~/tt-umd/device/coordinates/blackhole_coordinate_manager.cpp:71`, in `TopologyDiscovery`). Persisted across
5 init retries and **3 further `tt-smi -r 0` warm resets** (+25 s settle each). exalens unusable → x280 hart
bring-up/load (the margining path) blocked.

**Not caused by the flash edit (evidence).** pyluwen (a different transport, no ETH topology discovery) stayed
fully alive throughout: `vdd_max=1000`, vcore 714, aiclk 800, temp 70 °C. The edited `cmfwcfg` decodes with EVERY
field intact — `chip_harvesting_table={soft_harvesting:0}`, `eth_property_table={eth_disable_mask:0}`,
feature_enable/pci/dram/fan tables all correct — only `vdd_max` changed. `vdd_max` is a clamp ceiling; boot voltage
is unchanged (~700 mV), so it can't affect ETH SerDes training. The timing is coincidental.

**Root cause.** BH `tt-smi -r 0` warm reset does NOT reliably re-train the ETH SerDes (the benign
`no_wait_for_eth_training` warning). Normally the live ETH harvest count reads a valid 2 or 14; occasionally it
comes back invalid, and THIS tt-umd version's coordinate manager hard-rejects it (fatal, vs the older benign
warning). Once stuck, warm reset can't clear it. **This is the documented "BH warm-reset unreliable, keep host
reboot in reserve" hazard (`docs/plans/TTMETAL_PLAN.md`, memory `bh-noc-hang-hazard`) — now observed fatal.**

**Why first time this session.** Voltage margining needs a clean reset per campaign, so this session ran ~8+
warm resets; most produced a valid ETH count. Hitting the invalid state is stochastic — more resets, more chances.
The vdd_max reset was just the one that happened to land on it.

**Recovery.** Host reboot of ttstar → cold ETH training → valid harvest → exalens back. `vdd_max=1000` persists
(SPIROM), so the unlock is intact after reboot.

**Mitigations going forward.** (1) Minimize warm resets — one campaign per reset, batch frequencies. (2) Consider a
**pyluwen-only margining path** (`bh.noc_write/noc_read` to load the virus + read telemetry) that skips exalens
`TopologyDiscovery` entirely — resilient to ETH-harvest flakiness, and pyluwen proved rock-solid here. (3) Keep the
host-reboot expectation explicit for any reset-heavy silicon campaign.
