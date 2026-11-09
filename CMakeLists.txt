cmake_minimum_required(VERSION 3.20)
project(MLScratch VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags (only applied to Release/RelWithDebInfo to prevent Debug /RTC1 collisions)
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /Oi /Ot /fp:fast /arch:AVX2")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /Oi /Ot /fp:fast /arch:AVX2")
else()
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -O3 -march=native -ffast-math")
endif()

# Google Benchmark via FetchContent
include(FetchContent)
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Executable Target
add_executable(ml_project_app
    src/main.cpp
    utils/matrix.cpp
)

target_include_directories(ml_project_app PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/utils
)

target_link_libraries(ml_project_app PRIVATE
    benchmark::benchmark
)

if(MSVC)
    target_compile_options(ml_project_app PRIVATE /W3)
else()
    target_compile_options(ml_project_app PRIVATE -Wall -Wextra)
endif()