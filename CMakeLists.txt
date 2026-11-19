cmake_minimum_required(VERSION 3.25)
# GCC shuffles and spills the ternary kernel accumulators on every step; clang keeps all 12 in registers. Only picked
# when this is the top level project: a parent project's compiler is never overridden.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR AND NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX}
   AND NOT CMAKE_TOOLCHAIN_FILE)
  set(CMAKE_CXX_COMPILER clang++)
endif()
project(parakeetogo CXX)

if(PROJECT_IS_TOP_LEVEL AND NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PK_BUILD_CLI "Build the pk command line tool" ${PROJECT_IS_TOP_LEVEL})
option(PK_EMBED "Embed model.safetensors and tokenizer.json from PK_ASSET_DIR into the library" OFF)
option(PK_STATIC "Link the pk command line tool fully static" OFF)
option(PK_FETCH_ASSETS "Download missing assets into PK_ASSET_DIR when configuring" ${PROJECT_IS_TOP_LEVEL})
set(PK_ASSET_DIR ${PROJECT_SOURCE_DIR}/model CACHE PATH "Directory holding model.safetensors and tokenizer.json")
# Ultra is the default for quality: same speed as redux, but its dense weights mangle far less jargon. For the smaller
# ternary redux, set this to https://huggingface.co/rodbiren/kokoro_parakeet_aio_models/resolve/766eb98b496915a437f38b1cec8e9458879cb2f3/parakeet
set(PK_ASSET_URL https://huggingface.co/moondream/parakeet-ultra/resolve/73175eb7aeb0d82f1e2a6b53b3aabc10a90bcd0b CACHE STRING "")

# Highway, shared with kokoallovero when both are in one tree: whichever project comes first fetches it, with the
# same version and settings in both. Nothing here sets Highway's targets globally; each library fixes its own below,
# so the hwy library itself is built plain.
if(NOT TARGET hwy)
  include(FetchContent)
  set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
  set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(highway
    URL https://github.com/google/highway/archive/refs/tags/1.3.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP ON)
  FetchContent_MakeAvailable(highway)
  # Current clang warns about every AVX-512 target attribute in Highway's own sources.
  target_compile_options(hwy PRIVATE -Wno-ignored-attributes)
endif()

# Assets missing from PK_ASSET_DIR are downloaded at configure time. Nothing is fetched at runtime.
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

# One Highway target per architecture, fixed at compile time: AVX2 with FMA, F16C and BMI2 on x86 (x86-64-v3), NEON
# on ARM. kokoallovero makes the same choice. The flags stay on this library's own sources. Only the ternary kernel
# is picked at runtime.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
  set(PK_TERN src/tern_neon.cc)
  set_source_files_properties(src/tern_neon.cc PROPERTIES COMPILE_OPTIONS "-march=armv8.2-a+dotprod")
  set(PK_ARCH_OPTIONS "")
  set(PK_ARCH_DEFINITIONS "")
else()
  set(PK_TERN src/tern_vnni.cc src/tern_avx2.cc)
  set_source_files_properties(src/tern_vnni.cc PROPERTIES COMPILE_OPTIONS "-mavxvnni")
  set(PK_ARCH_OPTIONS -mavx2 -mfma -mf16c -mbmi -mbmi2)
  set(PK_ARCH_DEFINITIONS HWY_DISABLE_PCLMUL_AES)
endif()

add_library(parakeetogo STATIC ${PK_ASSETS} src/pool.cc src/st.cc src/tern.cc ${PK_TERN} src/dense.cc
            src/frontend.cc src/model.cc src/decode.cc)
add_library(parakeetogo::parakeetogo ALIAS parakeetogo)
set_target_properties(parakeetogo PROPERTIES OUTPUT_NAME pk)
target_compile_features(parakeetogo PUBLIC cxx_std_20)
target_include_directories(parakeetogo PUBLIC ${PROJECT_SOURCE_DIR})
target_link_libraries(parakeetogo PUBLIC hwy)
target_compile_definitions(parakeetogo PRIVATE HWY_COMPILE_ONLY_STATIC ${PK_ARCH_DEFINITIONS})
# No implicit fusing: scalar float and double code rounds like the reference. FMA is used through intrinsics.
target_compile_options(parakeetogo PRIVATE -Wall -Wextra -ffp-contract=off ${PK_ARCH_OPTIONS})

if(PK_BUILD_CLI)
  add_executable(pk_cli src/main.cc)
  set_target_properties(pk_cli PROPERTIES OUTPUT_NAME pk)
  target_compile_options(pk_cli PRIVATE -Wall -Wextra)
  target_link_libraries(pk_cli PRIVATE parakeetogo)
  if(PK_STATIC)
    target_link_options(pk_cli PRIVATE -static)
  endif()
endif()
