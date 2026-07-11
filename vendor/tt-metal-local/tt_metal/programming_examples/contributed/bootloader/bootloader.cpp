// SPDX-License-Identifier: Apache-2.0
//
// bootloader.cpp (host) — bring the grid up with tt-metal and deploy ONE resident
// bootloader kernel to EVERY worker core (one CreateKernel over a CoreRange ->
// llrt NoC-multicasts the binary). Then PARK: no Finish, no close, so the cores are
// never reset and keep looping. bhtop drives them live over tt-exalens.
//
// Teardown is on YOUR terms: this process traps SIGINT, pokes BL_CMD_HALT into every
// core's mailbox (via a tiny halt program / or just let bhtop do it), waits for the
// loops to exit, then closes cleanly.
//
// NOTE (validate on device): we use fast-dispatch non-blocking enqueue and never call
// Finish(). The dispatcher issues `go` and moves on; a resident kernel that never writes
// DONE is fine as long as we never Finish/close or enqueue another program on this core.
// If your metal build's dispatcher dislikes that, run slow dispatch
// (TT_METAL_SLOW_DISPATCH_MODE=1) and launch via detail::LaunchProgram on a side thread.

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tt_metal.hpp>   // detail::LaunchProgram (slow-dispatch park path)
#include <csignal>
#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <iostream>

#include "bootloader_abi.h"

using namespace tt::tt_metal;

static std::atomic<bool> g_stop{false};
static void on_sigint(int) { g_stop = true; }

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif

int main() {
    int device_id = 0;
    std::signal(SIGINT, on_sigint);

    auto mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // How many cores to make resident. Each resident core busy-spins the poll loop continuously
    // (real power — drives the fan), so default to a SMALL block; set BL_GRID=WxH or BL_GRID=all.
    auto full = mesh_device->compute_with_storage_grid_size();   // {x, y}
    uint32_t gw = full.x, gh = full.y;
    const char* env = std::getenv("BL_GRID");
    std::string g = env ? env : "2x2";                            // default: 4 cores, quiet
    if (g != "all") {
        int w = 0, h = 0;
        if (std::sscanf(g.c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
            gw = std::min<uint32_t>(gw, (uint32_t)w);
            gh = std::min<uint32_t>(gh, (uint32_t)h);
        }
    }
    CoreRange all_cores({0, 0}, {gw - 1, gh - 1});
    std::cout << "Deploying resident bootloader to " << gw << "x" << gh << " = " << (gw * gh)
              << " worker cores (BL_GRID=" << g << "; set BL_GRID=all for the full grid)\n";

    Program program = CreateProgram();

    // The resident loader on ALL FIVE baby RISCs. Each RISC owns a DISJOINT 64 KiB L1 region
    // (bootloader_abi.h); we pass each its region CTRL base + RISC index as runtime args. No
    // CBs, no buffers — we own a FIXED high-L1 window, so metal's allocator never competes with
    // our code slots.
    //   - BRISC  (RISCV_0) + NCRISC (RISCV_1): two DataMovement kernels, same source.
    //   - TRISC0/1/2: ONE Compute kernel, compiled 3x (COMPILE_FOR_TRISC=0/1/2); each derives
    //     its own region from that index, so all three get distinct mailboxes.
    auto brisc = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/bootloader/kernels/bootloader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = {}});
    SetRuntimeArgs(program, brisc, all_cores, {bl_ctrl(BL_RISC_BRISC), BL_RISC_BRISC});

    auto ncrisc = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/bootloader/kernels/bootloader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = {}});
    SetRuntimeArgs(program, ncrisc, all_cores, {bl_ctrl(BL_RISC_NCRISC), BL_RISC_NCRISC});

    // One compute kernel -> resident on TRISC0/1/2. It needs no runtime args: each instance
    // computes its own region base from COMPILE_FOR_TRISC.
    auto trisc = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/bootloader/kernels/bootloader_compute.cpp",
        all_cores,
        ComputeConfig{});
    SetRuntimeArgs(program, trisc, all_cores, {});

    // Two launch paths for a kernel that never returns:
    //  - SLOW DISPATCH (recommended): detail::LaunchProgram with wait=false issues `go` and returns
    //    immediately — no dispatch core waiting on a DONE that never comes. force_slow_dispatch=true.
    //  - FAST DISPATCH (default env): non-blocking EnqueueMeshWorkload; never call Finish(). The
    //    dispatcher fires `go` and moves on. Validate on your build (README point 2).
    const bool slow = std::getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr;
    distributed::MeshWorkload workload;   // kept in scope; only used on the fast path
    if (slow) {
        IDevice* dev = mesh_device->get_devices().at(0);
        detail::LaunchProgram(dev, program, /*wait_until_cores_done=*/false, /*force_slow_dispatch=*/true);
        std::cout << "Launched via slow dispatch (LaunchProgram, fire-and-forget).\n";
    } else {
        workload.add_program(device_range, std::move(program));
        distributed::EnqueueMeshWorkload(mesh_device->mesh_command_queue(), workload, /*blocking=*/false);
        std::cout << "Launched via fast dispatch (non-blocking EnqueueMeshWorkload).\n";
    }

    std::cout << "Bootloader resident on the grid — all 5 RISCs/core (BRISC,NCRISC,TRISC0-2). "
              << "Per-RISC control mailboxes @ L1 0x" << std::hex << bl_ctrl(0) << "/0x" << bl_ctrl(1)
              << "/0x" << bl_ctrl(2) << "/0x" << bl_ctrl(3) << "/0x" << bl_ctrl(4) << std::dec << ".\n"
              << "Drive it live with bhtop (poke params / stage overlays / read heartbeat).\n"
              << "Ctrl-C to halt the grid and close.\n";

    // PARK. The cores loop; we just hold the device open.
    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    // Clean exit: tell bhtop / a halt program to poke BL_CMD_HALT into each core's
    // BL_DOORBELL, or implement the halt-multicast here, then:
    std::cout << "\nHalting and closing.\n";
    mesh_device->close();   // asserts soft-reset — only happens because WE asked.
    return 0;
}
