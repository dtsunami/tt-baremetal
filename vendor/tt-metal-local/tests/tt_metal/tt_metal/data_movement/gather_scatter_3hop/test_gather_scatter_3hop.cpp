// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "multi_device_fixture.hpp"
#include "tt_metal/test_utils/stimulus.hpp"
#include "dm_common.hpp"
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <distributed/mesh_device_impl.hpp>

namespace tt::tt_metal {

using namespace std;
using namespace tt;
using namespace tt::test_utils;

namespace unit_tests::dm::gather_scatter_3hop {

constexpr uint32_t START_ID = 316;

// All-core dual-NoC 3-hop gather/scatter: every Tensix runs writer (RISCV_0 / NOC_0,
// scatter east) and reader (RISCV_1 / NOC_1, gather south->local) concurrently — the
// low-congestion neighborhood pattern from the data-movement slide, on the full grid.
struct GatherScatter3HopConfig {
    uint32_t test_id = START_ID;
    uint32_t num_of_transactions = 0;
    uint32_t pages_per_transaction = 0;
    uint32_t bytes_per_page = 0;
    uint32_t num_virtual_channels = 4;
    uint32_t hops = 3;
};

bool run_dm(const shared_ptr<distributed::MeshDevice>& mesh_device, const GatherScatter3HopConfig& cfg) {
    IDevice* device = mesh_device->impl().get_device(0);

    CoreCoord grid = device->compute_with_storage_grid_size();
    CoreRange all_cores({0, 0}, {grid.x - 1, grid.y - 1});
    CoreRangeSet all_core_set({all_cores});

    uint32_t bytes_per_transaction = cfg.pages_per_transaction * cfg.bytes_per_page;

    auto l1 = unit_tests::dm::get_l1_address_and_size(mesh_device, {0, 0});
    if (l1.size < 4 * bytes_per_transaction) {
        log_error(tt::LogTest, "Insufficient L1 for 4x{}B carve-out", bytes_per_transaction);
        return false;
    }
    // Non-overlapping quadrants: writer src / writer dst (remote) / reader dst / reader src (remote)
    uint32_t scatter_src = l1.base_address;
    uint32_t scatter_dst = scatter_src + bytes_per_transaction;
    uint32_t gather_dst = scatter_dst + bytes_per_transaction;
    uint32_t gather_src = gather_dst + bytes_per_transaction;

    Program program = CreateProgram();

    string kdir = "tests/tt_metal/tt_metal/data_movement/gather_scatter_3hop/kernels/";
    vector<uint32_t> writer_cta = {
        cfg.test_id, scatter_src, scatter_dst, cfg.num_of_transactions, bytes_per_transaction,
        cfg.num_virtual_channels};
    auto writer = CreateKernel(
        program,
        kdir + "writer.cpp",
        all_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::NOC_0, .compile_args = writer_cta});

    vector<uint32_t> reader_cta = {
        cfg.test_id, gather_dst, gather_src, cfg.num_of_transactions, bytes_per_transaction,
        cfg.num_virtual_channels};
    auto reader = CreateKernel(
        program,
        kdir + "reader.cpp",
        all_core_set,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_cta});

    // Per-core partners: writer scatters +hops east (logical wrap), reader gathers from
    // +hops south (logical wrap). Direction = NoC choice on Blackhole; same physical
    // coords for both NoCs.
    for (uint32_t x = 0; x < grid.x; x++) {
        for (uint32_t y = 0; y < grid.y; y++) {
            CoreCoord c = {x, y};
            CoreCoord wlog = {(x + cfg.hops) % grid.x, y};
            CoreCoord rlog = {x, (y + cfg.hops) % grid.y};
            CoreCoord wphys = device->worker_core_from_logical_core(wlog);
            CoreCoord rphys = device->worker_core_from_logical_core(rlog);
            SetRuntimeArgs(program, writer, c, {(uint32_t)wphys.x, (uint32_t)wphys.y});
            SetRuntimeArgs(program, reader, c, {(uint32_t)rphys.x, (uint32_t)rphys.y});
        }
    }

    log_info(tt::LogTest, "Running Test ID: {}, Run ID: {}", cfg.test_id, unit_tests::dm::runtime_host_id);
    program.set_runtime_id(unit_tests::dm::runtime_host_id++);

    // Seed both remote-read sources once (contents don't matter for a BW run)
    vector<uint32_t> seed(bytes_per_transaction / sizeof(uint32_t), 0xA5A5A5A5);
    for (uint32_t x = 0; x < grid.x; x++) {
        for (uint32_t y = 0; y < grid.y; y++) {
            detail::WriteToDeviceL1(device, {x, y}, scatter_src, seed);
            detail::WriteToDeviceL1(device, {x, y}, gather_src, seed);
        }
    }
    MetalContext::instance().get_cluster().l1_barrier(device->id());

    auto mesh_workload = distributed::MeshWorkload();
    vector<uint32_t> coord_data = {0, 0};
    auto target_devices = distributed::MeshCoordinateRange(distributed::MeshCoordinate(coord_data));
    mesh_workload.add_program(target_devices, std::move(program));
    auto& cq = mesh_device->mesh_command_queue();
    distributed::EnqueueMeshWorkload(cq, mesh_workload, false);
    Finish(cq);

    return true;
}

void directed_ideal_test(const shared_ptr<distributed::MeshDevice>& mesh_device) {
    auto [bytes_per_page, max_bytes, max_pages] = unit_tests::dm::compute_physical_constraints(mesh_device);
    GatherScatter3HopConfig cfg;
    cfg.pages_per_transaction = 256;   // 256 * 64B = 16384B = single-packet fast path
    cfg.bytes_per_page = bytes_per_page;
    cfg.num_of_transactions = 256;
    EXPECT_TRUE(run_dm(mesh_device, cfg));
}

}  // namespace unit_tests::dm::gather_scatter_3hop

TEST_F(GenericMeshDeviceFixture, TensixDataMovementGatherScatter3HopDirectedIdeal) {
    unit_tests::dm::gather_scatter_3hop::directed_ideal_test(get_mesh_device());
}

}  // namespace tt::tt_metal
