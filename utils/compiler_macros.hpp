#pragma once

// Loop tilling constant
#ifndef L1_CACHE_BLOCK_SIZE
// 32 doubles = 256 bytes per row segment. 
// A 32x32 block of doubles is 8 KB, safely fitting inside modern 32-48 KB L1D caches.
#define L1_CACHE_BLOCK_SIZE 32
#endif

// Loop vectorization hint (ignore assumed loop-carried vector dependencies)
#if defined(_MSC_VER)
#define PRAGMA_IVDEP __pragma(loop(ivdep))
#elif defined(__clang__)
#define PRAGMA_IVDEP _Pragma("clang loop vectorize(enable)")
#elif defined(__GNUC__)
#define PRAGMA_IVDEP _Pragma("GCC ivdep")
#else
#define PRAGMA_IVDEP
#endif

// Compiler-agnostic restrict pointer qualifier
// Restrict pointers allow the compiler to optimize memory access by assuming that the pointer is the only reference to that memory location, enabling better vectorization and optimization.
// It ensures no overlaps in memory access, which can lead to more efficient code generation.
#if defined(_MSC_VER)
#define RESTRICT __restrict
#elif defined(__GNUC__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT
#endif

// Branch prediction hints (C++20 provides [[likely]] / [[unlikely]], 
// but manual macros remain useful for older targets or expressions)
#if defined(__GNUC__) || defined(__clang__)
#define LIKELY(x)   __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#endif