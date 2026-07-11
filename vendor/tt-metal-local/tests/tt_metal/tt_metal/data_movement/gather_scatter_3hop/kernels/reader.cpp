// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"

// Dual-NoC 3-hop GATHER on NOC1 (Y-then-X, north/west). Runs on RISCV_1.
// Same physical NoC-0 coords as the writer; routing direction is the NoC property.
void kernel_main() {
    constexpr uint32_t test_id = get_compile_time_arg_val(0);
    constexpr uint32_t dst_l1_base = get_compile_time_arg_val(1);            // local gather sink
    constexpr uint32_t src_l1_base = get_compile_time_arg_val(2);            // remote partner source
    constexpr uint32_t num_of_transactions = get_compile_time_arg_val(3);
    constexpr uint32_t bytes_per_transaction = get_compile_time_arg_val(4);
    constexpr uint32_t num_virtual_channels = get_compile_time_arg_val(5);

    uint32_t partner_x = get_arg_val<uint32_t>(0);
    uint32_t partner_y = get_arg_val<uint32_t>(1);
    uint32_t per_core_bytes = 0;

    {
        DeviceZoneScopedN("RISCV1");
        if (partner_x != 0xFFFFFFFF) {
            uint64_t src_noc_addr = get_noc_addr(partner_x, partner_y, src_l1_base);
            for (uint32_t i = 0; i < num_of_transactions; i++) {
                uint32_t vc = i % num_virtual_channels;
                noc_async_read(src_noc_addr, dst_l1_base, bytes_per_transaction, noc_index, vc);
            }
            per_core_bytes = num_of_transactions * bytes_per_transaction;
        }
        noc_async_read_barrier();  // single drain inside the timed zone
    }

    DeviceTimestampedData("Test id", test_id);
    DeviceTimestampedData("NoC Index", noc_index);  // = 1
    DeviceTimestampedData("Number of transactions", num_of_transactions);
    DeviceTimestampedData("Transaction size in bytes", bytes_per_transaction);
    DeviceTimestampedData("Number of Virtual Channels", num_virtual_channels);
    DeviceTimestampedData("Per-core bytes", per_core_bytes);
}
