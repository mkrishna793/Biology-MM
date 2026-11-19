# ==============================================================================
# Biology-MM Native Acceleration Backends (Trium-XBT CPU, Vulkan, Metal, SYCL)
# ==============================================================================
cmake_minimum_required(VERSION 3.20)
project(BiologyMMNative
    VERSION 1.0.0
    DESCRIPTION "Cross-platform native CPU, Vulkan, Metal, and SYCL backends for Biology-MM"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)

if(APPLE)
    set(_TRIUM_DEFAULT_METAL ON)
else()
    set(_TRIUM_DEFAULT_METAL OFF)
endif()

option(TRIUM_ENABLE_VULKAN "Enable zero-SDK dynamic Vulkan Compute backend" ON)
option(TRIUM_ENABLE_METAL "Enable Apple Metal GPU backend on macOS" ${_TRIUM_DEFAULT_METAL})
option(TRIUM_ENABLE_SYCL "Enable SYCL 2020 (DPC++ / AdaptiveCpp) GPU backend" OFF)

set(BIOLOGY_MM_LIB_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../julia/lib")
file(MAKE_DIRECTORY "${BIOLOGY_MM_LIB_OUTPUT_DIR}")

foreach(_OUT_VAR
    CMAKE_LIBRARY_OUTPUT_DIRECTORY
    CMAKE_RUNTIME_OUTPUT_DIRECTORY
    CMAKE_ARCHIVE_OUTPUT_DIRECTORY
)
    set(${_OUT_VAR} "${BIOLOGY_MM_LIB_OUTPUT_DIR}")
    foreach(_CFG RELEASE DEBUG RELWITHDEBINFO MINSIZEREL)
        set(${_OUT_VAR}_${_CFG} "${BIOLOGY_MM_LIB_OUTPUT_DIR}")
    endforeach()
endforeach()

# ------------------------------------------------------------------------------
# 1. Trium-XBT CPU/NEON Molecular Mechanics Bridge
# ------------------------------------------------------------------------------
add_library(trium_mm_bridge SHARED
    src/trium_mm_bridge.cpp
)
target_include_directories(trium_mm_bridge PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(trium_mm_bridge PRIVATE -O3)
    if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64")
        target_compile_options(trium_mm_bridge PRIVATE -mcpu=native)
    endif()
endif()
target_link_libraries(trium_mm_bridge PRIVATE Threads::Threads)

# ------------------------------------------------------------------------------
# 2. Portable Vulkan Compute Dynamic Loader (Zero Vulkan SDK Compile Dependency)
# ------------------------------------------------------------------------------
if(TRIUM_ENABLE_VULKAN)
    add_library(trium_gpu_vulkan SHARED
        src/gpu_portable/trium_gpu_vulkan.cpp
    )
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(trium_gpu_vulkan PRIVATE -O3 -Wall -Wextra)
    endif()
    if(UNIX)
        target_link_libraries(trium_gpu_vulkan PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
    else()
        target_link_libraries(trium_gpu_vulkan PRIVATE Threads::Threads)
    endif()

    find_program(GLSLC_EXECUTABLE NAMES glslc)
    find_program(GLSLANG_EXECUTABLE NAMES glslangValidator)

    set(TRIUM_GLSL_SHADERS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_portable/shaders/xbt_lj_dsf_csr.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_portable/shaders/xbt_ternary_matvec.comp"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_portable/shaders/xbt_msm_stencil27.comp"
    )

    if(GLSLC_EXECUTABLE OR GLSLANG_EXECUTABLE)
        set(TRIUM_SPV_DIR "${BIOLOGY_MM_LIB_OUTPUT_DIR}/shaders")
        file(MAKE_DIRECTORY "${TRIUM_SPV_DIR}")
        set(TRIUM_SPV_OUTPUTS)
        foreach(SHADER_SRC ${TRIUM_GLSL_SHADERS})
            get_filename_component(SHADER_NAME "${SHADER_SRC}" NAME_WE)
            set(SPV_OUT "${TRIUM_SPV_DIR}/${SHADER_NAME}.spv")
            if(GLSLC_EXECUTABLE)
                add_custom_command(
                    OUTPUT "${SPV_OUT}"
                    COMMAND "${GLSLC_EXECUTABLE}" --target-env=vulkan1.1 -O "${SHADER_SRC}" -o "${SPV_OUT}"
                    DEPENDS "${SHADER_SRC}"
                    COMMENT "Compiling GLSL 450 -> SPIR-V: ${SHADER_NAME}.spv"
                )
            else()
                add_custom_command(
                    OUTPUT "${SPV_OUT}"
                    COMMAND "${GLSLANG_EXECUTABLE}" -V --target-env vulkan1.1 "${SHADER_SRC}" -o "${SPV_OUT}"
                    DEPENDS "${SHADER_SRC}"
                    COMMENT "Compiling GLSL 450 -> SPIR-V: ${SHADER_NAME}.spv"
                )
            endif()
            list(APPEND TRIUM_SPV_OUTPUTS "${SPV_OUT}")
        endforeach()
        add_custom_target(trium_spirv_shaders ALL DEPENDS ${TRIUM_SPV_OUTPUTS})
        add_dependencies(trium_gpu_vulkan trium_spirv_shaders)
    else()
        message(STATUS "glslc/glslangValidator not found: skipping offline SPIR-V build (GLSL sources installed in src/gpu_portable/shaders/).")
    endif()
endif()

# ------------------------------------------------------------------------------
# 3. Apple Metal GPU Backend (macOS)
# ------------------------------------------------------------------------------
if(TRIUM_ENABLE_METAL AND APPLE)
    enable_language(OBJCXX)
    add_library(trium_gpu_bridge SHARED
        src/trium_gpu_metal.mm
    )
    target_compile_options(trium_gpu_bridge PRIVATE -O3)
    target_link_libraries(trium_gpu_bridge PRIVATE
        "-framework Metal"
        "-framework Foundation"
        Threads::Threads
    )
endif()

# ------------------------------------------------------------------------------
# 4. Universal SYCL 2020 Interface Target (Intel DPC++ / AdaptiveCpp)
# ------------------------------------------------------------------------------
if(TRIUM_ENABLE_SYCL)
    add_library(trium_gpu_sycl INTERFACE)
    target_include_directories(trium_gpu_sycl INTERFACE
        "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu_portable"
    )
    find_package(IntelSYCL QUIET)
    if(IntelSYCL_FOUND)
        target_link_libraries(trium_gpu_sycl INTERFACE IntelSYCL::SYCL_CXX)
    else()
        find_package(AdaptiveCpp QUIET)
        if(AdaptiveCpp_FOUND)
            target_link_libraries(trium_gpu_sycl INTERFACE AdaptiveCpp::acpp-rt)
        else()
            target_compile_options(trium_gpu_sycl INTERFACE -fsycl)
            target_link_options(trium_gpu_sycl INTERFACE -fsycl)
        endif()
    endif()
endif()
