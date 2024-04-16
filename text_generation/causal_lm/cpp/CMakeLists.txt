# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.15)
project(causal_lm)

add_subdirectory(../../../thirdparty/openvino_tokenizers/ "${CMAKE_CURRENT_BINARY_DIR}/openvino_tokenizers/")

add_executable(greedy_causal_lm greedy_causal_lm.cpp)
target_compile_definitions(greedy_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(greedy_causal_lm PRIVATE openvino::runtime)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(greedy_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(beam_search_causal_lm beam_search_causal_lm.cpp)
target_compile_definitions(beam_search_causal_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(beam_search_causal_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(beam_search_causal_lm PRIVATE openvino::runtime)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(beam_search_causal_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

add_executable(speculative_decoding_lm speculative_decoding_lm.cpp)
target_compile_definitions(speculative_decoding_lm PRIVATE OPENVINO_TOKENIZERS_PATH=\"$<TARGET_FILE:openvino_tokenizers>\")
target_include_directories(speculative_decoding_lm PRIVATE ./)
find_package(OpenVINO REQUIRED COMPONENTS Runtime)
target_link_libraries(speculative_decoding_lm PRIVATE openvino::runtime)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD 17)
set_target_properties(speculative_decoding_lm PROPERTIES CXX_STANDARD_REQUIRED ON)

find_package(OpenCL QUIET)
if(NOT OpenCL_FOUND)
    message(WARNING "OpenCL is disabled or not found, causal_lm will be built without OpenCL support. Install OpenCL.")
endif()

if(OpenCL_FOUND )
    set(OpenCL_DEFINITIONS HAVE_GPU_DEVICE_MEM_SUPPORT)

    # Append OpenCL CPP headers to C headers and use both
    if(OpenCL_HPP_INCLUDE_DIR)
        list(APPEND OpenCL_HEADERS ${OpenCL_HPP_INCLUDE_DIR})
        # the macro below is defined when opencl.hpp is found to suppress deprecation message from cl2.hpp
        list(APPEND OpenCL_DEFINITIONS OV_GPU_USE_OPENCL_HPP)
    endif()
    if(CL2_HPP_INCLUDE_DIR)
        list(APPEND OpenCL_HEADERS ${CL2_HPP_INCLUDE_DIR})
    endif()

    # cmake cannot set properties for imported targets
    get_target_property(opencl_target OpenCL::OpenCL ALIASED_TARGET)
    if(NOT TARGET ${opencl_target})
        set(opencl_target OpenCL::OpenCL)
    endif()

    set_property(TARGET ${opencl_target} APPEND PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${OpenCL_HEADERS})
    set_property(TARGET ${opencl_target} APPEND PROPERTY
        INTERFACE_COMPILE_DEFINITIONS ${OpenCL_DEFINITIONS})

    target_link_libraries(speculative_decoding_lm PRIVATE OpenCL::OpenCL)
else()
    message(WARNING "OpenCL CPP header is not found, ${TARGET_NAME} will be built without OpenCL support and you will not be able to use the '-use_device_mem' option. Please, install '<apt | yum> install opencl-headers' to enable the option")
endif()
