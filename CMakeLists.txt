cmake_minimum_required(VERSION 3.24)
# GCC shuffles and spills the ternary kernel accumulators on every step; clang keeps all 12 in registers.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER clang++)
endif()
project(parakeetogo CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# No implicit fusing: scalar float and double code rounds like the reference. FMA is used through intrinsics.
add_compile_options(-Wall -Wextra -ffp-contract=off)
# One Highway target per architecture, fixed at compile time: AVX2 on x86, NEON on ARM. Only the ternary kernel is
# picked at runtime.
add_compile_definitions(HWY_COMPILE_ONLY_STATIC)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  set(PK_TERN src/tern_neon.cc)
  set_source_files_properties(src/tern_neon.cc PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
else()
  add_compile_options(-mavx2 -mfma -mf16c -mbmi -mbmi2)
  add_compile_definitions(HWY_DISABLE_PCLMUL_AES)
  set(PK_TERN src/tern_vnni.cc src/tern_avx2.cc)
  set_source_files_properties(src/tern_vnni.cc PROPERTIES COMPILE_OPTIONS "-mavxvnni")
endif()

include(FetchContent)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
FetchContent_Declare(highway
  URL https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON)
FetchContent_MakeAvailable(highway)

option(PK_EMBED "Embed model.safetensors and tokenizer.json from PK_ASSET_DIR into the binary" OFF)
option(PK_STATIC "Link fully static" OFF)
set(PK_ASSET_DIR ${CMAKE_SOURCE_DIR}/model CACHE PATH "")

# Assets missing from PK_ASSET_DIR are downloaded at configure time. Nothing is fetched at runtime.
option(PK_FETCH_ASSETS "Download missing assets into PK_ASSET_DIR when configuring" ON)
set(PK_ASSET_URL https://huggingface.co/rodbiren/kokoro_parakeet_aio_models/resolve/766eb98b496915a437f38b1cec8e9458879cb2f3/parakeet CACHE STRING "")
if(PK_FETCH_ASSETS)
  foreach(f model.safetensors tokenizer.json)
    if(NOT EXISTS ${PK_ASSET_DIR}/${f})
      message(STATUS "Downloading ${f}")
      file(DOWNLOAD ${PK_ASSET_URL}/${f} ${PK_ASSET_DIR}/${f}.part STATUS status)
      list(GET status 0 code)
      if(code)
        file(REMOVE ${PK_ASSET_DIR}/${f}.part)
        message(WARNING "Could not download ${f}: ${status}")
      else()
        file(RENAME ${PK_ASSET_DIR}/${f}.part ${PK_ASSET_DIR}/${f})
      endif()
    endif()
  endforeach()
endif()
if(PK_EMBED)
  set(PK_ASSETS src/assets.cc)
  set_source_files_properties(src/assets.cc PROPERTIES COMPILE_DEFINITIONS "PK_ASSET_DIR=\"${PK_ASSET_DIR}\""
    OBJECT_DEPENDS "${PK_ASSET_DIR}/model.safetensors;${PK_ASSET_DIR}/tokenizer.json")
else()
  set(PK_ASSETS src/noassets.cc)
endif()
if(PK_STATIC)
  add_link_options(-static)
endif()

add_library(pk STATIC ${PK_ASSETS} src/pool.cc src/st.cc src/tern.cc ${PK_TERN} src/dense.cc src/frontend.cc
            src/model.cc src/decode.cc)
target_include_directories(pk PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(pk PUBLIC hwy)

add_executable(pk_cli src/main.cc)
set_target_properties(pk_cli PROPERTIES OUTPUT_NAME pk)
target_link_libraries(pk_cli PRIVATE pk)
