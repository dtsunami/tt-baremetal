cmake_minimum_required(VERSION 3.22...4.2)
project(dang_kernel_jit)

add_executable(dang_kernel_jit)
target_sources(dang_kernel_jit PRIVATE kerneljit.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(dang_kernel_jit PUBLIC TT::Metalium)
if(TARGET TT::CommonPCH)
    tt_reuse_precompile_headers(dang_kernel_jit TT::CommonPCH)
endif()
