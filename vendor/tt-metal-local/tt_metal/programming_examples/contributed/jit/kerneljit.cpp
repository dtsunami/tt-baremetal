// SPDX-FileCopyrightText: © 2024 Martin Chang
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/distributed.hpp>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>

using namespace tt::tt_metal;
using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

constexpr uint32_t TILE_WIDTH = 32;
constexpr uint32_t TILE_HEIGHT = 32;

std::shared_ptr<distributed::MeshBuffer> MakeBuffer(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t size, uint32_t /*page_size*/, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? size : 1;
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_tiles * tile_size, .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM)};
    const distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * size};
    return distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
}

// Allocate a buffer on DRAM or SRAM. Assuming the buffer holds BFP16 data.
// A tile on Tenstorrent is 32x32 elements, given us using BFP16, we need 2 bytes per element.
// Making the tile size 32x32x2 = 2048 bytes.
// @param device: The device to allocate the buffer on.
// @param n_tiles: The number of tiles to allocate.
// @param sram: If true, allocate the buffer on SRAM, otherwise allocate it on DRAM.
std::shared_ptr<distributed::MeshBuffer> MakeBufferBFP16(
    const std::shared_ptr<distributed::MeshDevice>& mesh_device, uint32_t n_tiles, bool sram) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    const uint32_t page_tiles = sram ? n_tiles : 1;
    const distributed::DeviceLocalBufferConfig device_local_config{
        .page_size = page_tiles * tile_size,
        .buffer_type = (sram ? BufferType::L1 : BufferType::DRAM),
        .bottom_up = false};
    const distributed::ReplicatedBufferConfig buffer_config{.size = tile_size * n_tiles};
    return distributed::MeshBuffer::create(buffer_config, device_local_config, mesh_device.get());
}

CBHandle MakeCircularBuffer(
    Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t size, uint32_t page_size, tt::DataFormat format) {
    CircularBufferConfig cb_config = CircularBufferConfig(size, {{cb, format}}).set_page_size(cb, page_size);
    return CreateCircularBuffer(program, core, cb_config);
}

// Circular buffers are Tenstorrent's way of communicating between the data movement and the compute kernels.
// kernels queue tiles into the circular buffer and takes them when they are ready. The circular buffer is
// backed by SRAM. There can be multiple circular buffers on a single Tensix core.
// @param program: The program to create the circular buffer on.
// @param core: The core to create the circular buffer on.
// @param cb: Which circular buffer to create (c_in0, c_in1, c_out0, c_out1, etc..). This is just an ID
// @param n_tiles: The number of tiles the circular buffer can hold.
CBHandle MakeCircularBufferBFP16(Program& program, const CoreSpec& core, tt::CBIndex cb, uint32_t n_tiles) {
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_WIDTH * TILE_HEIGHT;
    return MakeCircularBuffer(program, core, cb, n_tiles * tile_size, tile_size, tt::DataFormat::Float16_b);
}

std::string next_arg(int& i, int argc, char** argv) {
    if (i + 1 >= argc) {
        std::cerr << "Expected argument after " << argv[i] << std::endl;
        exit(1);
    }
    return argv[++i];
}

void help(std::string_view program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "This program demonstrates how to add two vectors using tt-Metalium.\n";
    std::cout << "\n";
    std::cout << "Options:\n";
    std::cout << "  --device, -d <device_id>  Specify the device to run the program on. Default is 0.\n";
    std::cout << "  --seed, -s <seed>         Specify the seed for the random number generator. Default is random.\n";
    exit(0);
}

#ifndef OVERRIDE_KERNEL_PREFIX
#define OVERRIDE_KERNEL_PREFIX ""
#endif
int main(int argc, char** argv) {
    int device_id = 0;
    for (int i = 1; i < argc; i++) {
        std::string_view arg = argv[i];
    }
    std::shared_ptr<distributed::MeshDevice> mesh_device = distributed::MeshDevice::create_unit_mesh(device_id);
    auto device_range = distributed::MeshCoordinateRange(mesh_device->shape());
    distributed::MeshWorkload workload;
    Program program = CreateProgram();
    CoreCoord core = {0, 0};
    const auto device_coord = distributed::MeshCoordinate(0, 0);




    // A Tensix core is made up with 5 processors. 2 data movement processors, and 3 compute processors. The 2 data
    // movement processors act independent to other cores. And the 3 compute processors act together (hence 1 kernel for
    // compute). There is no need to explicitly parallelize the compute kernels. Unlike traditional CPU/GPU style SPMD
    // programming, the 3 compute processors moves data from SRAM into the FPU(tensor engine)/SFPU(SIMD engine),
    // operates on the data, and move it back to SRAM. The data movement processors moves data from the NoC, or in our
    // case, the DRAM, into the SRAM.
    //
    // The vector add example consists of 3 kernels. `interleaved_tile_read` reads tiles from the input buffers A and B
    // into 2 circular buffers. `add` reads tiles from the circular buffers, adds them together, and dumps the result
    // into a third circular buffer. `tile_write` reads tiles from the third circular buffer and writes them to the
    // output buffer C.
    std::vector<uint32_t> reader_compile_time_args;
    auto reader = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/jit/kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = reader_compile_time_args});
    std::vector<uint32_t> writer_compile_time_args;
    auto writer = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/jit/kernels/tile_write.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1,
            .noc = NOC::RISCV_1_default,
            .compile_args = writer_compile_time_args});
    auto compute = CreateKernel(
        program,
        OVERRIDE_KERNEL_PREFIX "contributed/jit/kernels/add.cpp",
        core,
        ComputeConfig{.math_approx_mode = false, .compile_args = {}, .defines = {}});

    SetRuntimeArgs(program, reader, core, {});
    SetRuntimeArgs(program, writer, core, {});
    SetRuntimeArgs(program, compute, core, {});

    // Finally, we close the device.
    mesh_device->close();
    return 0;
}
