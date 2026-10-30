#!/usr/bin/env python3
"""Backport sglang#47149 (4201b8fa) onto sglang main@7d996300.

[Perf] dsv4: Fix amax8_varlen and mega MoE pre-dispatch grids for
1024-thread SMs.

block_amax.cuh: grid-stride over key chunks with a resident-CTA-capped
grid (kWaves=2, occupancy query cached per device via sgl_kernel/once.h --
vendored in-tree at kernels/jit/include, #41147, present at this pin);
upstream measured 56-84% faster than the full-width grid on CC 10.7.
mega_moe_pre_dispatch.cuh: strided persistent form (NO-OP here: our MoE
runs b12x_next, not MegaMoE -- carried for byte-fidelity to git apply).

EXECUTES HERE: amax8_varlen is called at
srt/layers/attention/dsv4/v41_indexer/sparse_table.py:519 -- the V4.1
indexer sparse-table build, our prefill/cold path. GB10 = 48 SMs CC 12.1
(smaller win than 10.7's 148-SM full-width pathology expected). Gate =
paired cold probe, adopt only if not slower.

Applies clean with `git apply` on the pristine 7d996300 files (verified
2026-10-30). PIN-MIGRATION NOTE: absorbed at any pin >= 4201b8fa; JIT
csrc recompiles at first boot after patch (cache key = source hash).

Idempotent. Exits non-zero on anchor drift (these scripts are the
build's drift probes). Optional argv[1] overrides the sglang python
tree root (default: the in-image path).
"""
import pathlib
import sys

DEFAULT_BASE = "/sgl-workspace/sglang/python"
PR = '47149'

FILES = [
    (
        '''sglang/kernels/jit/csrc/deepseek_v4/block_amax.cuh''',
        '''#include <sgl_kernel/once.h>
''',
        [
            ('''#pragma once

#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

/// Level-one keys of the two-level indexer: the max of each kBlockTokens-score
/// block, the row's newest block forced to +inf. Scores and keys are each fp32
/// or bf16; a bf16 score's max is exact in either key type. Contract: BlockAmaxKernel.
template <typename InT, typename OutT>
struct BlockAmaxConfig {
  using DType = InT;
  using KeyT = OutT;
  static constexpr uint32_t kBlockTokens = 8;  // scores per key
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kOccupancy = 4;
  // One fp32 block is 32 B: a single load on Blackwell, two 16 B loads before it.
  // A bf16 block is one 16 B load anywhere.
  static constexpr uint32_t kBlockBytes = kBlockTokens * sizeof(DType);
  static constexpr uint32_t kVecBytes = std::min<uint32_t>(kBlockBytes, device::kMaxVecBytes);
  static constexpr uint32_t kVecSize = kVecBytes / sizeof(DType);
  static constexpr uint32_t kVecsPerBlock = kBlockTokens / kVecSize;
  static constexpr uint32_t kNumItems = 64 / kBlockBytes;  // 64B scores (16 registers) per thread
  static constexpr uint32_t kKeysPerCTA = kBlockSize * kNumItems;
  static_assert(kVecsPerBlock * kVecSize == kBlockTokens);
  using vec_t = device::AlignedVector<DType, kVecSize>;
};

template <typename InT, typename OutT>
struct BlockAmaxParams {
  const InT* __restrict__ scores;
  OutT* __restrict__ amax_scores;
  const int32_t* __restrict__ seq_len;
  int64_t stride_scores;       // in elements
  int64_t stride_amax_scores;  // in elements
  uint32_t topk;               // rows with <= topk blocks are skipped, 0 = never skip
};

/// grid = (rows, ceil(max_keys / kKeysPerCTA)); a CTA owns kKeysPerCTA consecutive
/// keys of one row, a thread kNumItems keys kBlockSize apart (coalesced loads).
template <bool kUsePDL, typename InT, typename OutT>
__global__ __launch_bounds__(BlockAmaxConfig<InT, OutT>::kBlockSize, BlockAmaxConfig<InT, OutT>::kOccupancy)  //
    void amax8_varlen_kernel(const __grid_constant__ BlockAmaxParams<InT, OutT> params) {
  using namespace device;
  using C = BlockAmaxConfig<InT, OutT>;
  using vec_t = typename C::vec_t;
  const auto bx = blockIdx.x;
  const auto by = blockIdx.y;
  const auto tx = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();  // seq_len and scores are the previous kernels' outputs
  const auto seq_len = static_cast<uint32_t>(params.seq_len[bx]);
  const auto num_keys = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  const auto first_key = by * C::kKeysPerCTA;
  if (num_keys <= params.topk || first_key >= num_keys) {
    return PDLTriggerSecondary<kUsePDL>();
  }
  const auto* __restrict__ in = params.scores + bx * params.stride_scores;
  auto* __restrict__ out = params.amax_scores + bx * params.stride_amax_scores;

  vec_t vec[C::kNumItems][C::kVecsPerBlock];
#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto idx = first_key + tx + i * C::kBlockSize;
    if (idx < num_keys) {
#pragma unroll
      for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
        vec[i][v].load(in, idx * C::kVecsPerBlock + v);
      }
    }
  }
  // The dependent grid may start its prologue now; its griddepcontrol.wait still
  // covers every store below (it waits for this grid to complete).
  PDLTriggerSecondary<kUsePDL>();

#pragma unroll
  for (uint32_t i = 0; i < C::kNumItems; ++i) {
    const auto idx = first_key + tx + i * C::kBlockSize;
    if (idx < num_keys) {
      // fmaxf and __hmax2 ignore a NaN score, torch.amax would propagate it
      if constexpr (std::is_same_v<InT, float>) {
        float key = vec[i][0][0];
#pragma unroll
        for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
#pragma unroll
          for (uint32_t j = 0; j < C::kVecSize; ++j) {
            key = fmaxf(key, vec[i][v][j]);
          }
        }
        const auto fp32_inf = std::numeric_limits<float>::infinity();
        out[idx] = cast<OutT>(idx + 1 == num_keys ? fp32_inf : key);
      } else {
        const auto vec_ptr = reinterpret_cast<bf16x2_t*>(vec[i][0].data());
        static_assert(C::kVecSize % 2 == 0);
        bf16x2_t key2 = vec_ptr[0];
#pragma unroll
        for (uint32_t v = 1; v < C::kVecsPerBlock * C::kVecSize / 2; ++v) {
          key2 = __hmax2(key2, vec_ptr[v]);
        }
        const auto key = __hmax(key2.x, key2.y);
        const auto bf16_inf = cast<bf16_t>(std::numeric_limits<float>::infinity());
        out[idx] = cast<OutT>(idx + 1 == num_keys ? bf16_inf : key);
      }
    }
  }
}

/// Host entry: `amax_scores[b, i] = max(scores[b, 8 i : 8 i + 8])` for
/// `i < ceil(seq_len[b] / 8)`, the last of them +inf; rows with at most `topk`
/// blocks untouched. `scores` (`InT`) rows must keep every block of 8 aligned
/// (stride % 8 == 0); `amax_scores` is `OutT`. Each type is fp32 or bf16.
/// The grid covers `amax_scores`' width, so the caller sizes it for the longest
/// row: `seq_len[b] <= 8 * amax_scores.shape[1]` for every row (not checked).
template <bool kPDL, typename InT, typename OutT>
struct BlockAmaxKernel {
  static void amax8_varlen(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView amax_scores,
      const uint32_t topk) {
    using namespace host;
    using C = BlockAmaxConfig<InT, OutT>;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto K = SymbolicSize{"max_keys"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({-1, 1})
        .with_dtype<typename C::DType>()
        .with_device(device_)
        .ensure_alignment(C::kBlockBytes)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, K})  // amax_scores
        .with_strides({-1, 1})
        .with_dtype<typename C::KeyT>()
        .with_device(device_)
        .verify(amax_scores);
    const auto max_keys = K.unwrap();  // ceil(longest row / 8), sized by the caller
    const auto params = BlockAmaxParams<InT, OutT>{
        .scores = static_cast<const InT*>(scores.data_ptr()),
        .amax_scores = static_cast<OutT*>(amax_scores.data_ptr()),
        .seq_len = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .stride_scores = scores.stride(0),
        .stride_amax_scores = amax_scores.stride(0),
        .topk = topk,
    };
    const auto grid = dim3(
        static_cast<uint32_t>(B.unwrap()),
        static_cast<uint32_t>(div_ceil(max_keys, static_cast<int64_t>(C::kKeysPerCTA))));
    LaunchKernel(grid, C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(amax8_varlen_kernel<kPDL, InT, OutT>, params);
  }
};

}  // namespace sglang
''',
             '''#pragma once

#include <sgl_kernel/once.h>
#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace sglang {

/// Level-one keys of the two-level indexer: the max of each kBlockTokens-score
/// block, the row's newest block forced to +inf. Scores and keys are each fp32
/// or bf16; a bf16 score's max is exact in either key type. Contract: BlockAmaxKernel.
template <typename InT, typename OutT>
struct BlockAmaxConfig {
  using DType = InT;
  using KeyT = OutT;
  static constexpr uint32_t kBlockTokens = 8;  // scores per key
  static constexpr uint32_t kBlockSize = 512;
  static constexpr uint32_t kOccupancy = 4;
  // One fp32 block is 32 B: a single load on Blackwell, two 16 B loads before it.
  // A bf16 block is one 16 B load anywhere.
  static constexpr uint32_t kBlockBytes = kBlockTokens * sizeof(DType);
  static constexpr uint32_t kVecBytes = std::min<uint32_t>(kBlockBytes, device::kMaxVecBytes);
  static constexpr uint32_t kVecSize = kVecBytes / sizeof(DType);
  static constexpr uint32_t kVecsPerBlock = kBlockTokens / kVecSize;
  static constexpr uint32_t kNumItems = 64 / kBlockBytes;  // 64B scores (16 registers) per thread
  static constexpr uint32_t kKeysPerCTA = kBlockSize * kNumItems;
  // grid.y covers this many waves of resident CTAs; 2 is unswept (56-84% faster than the full-width grid on CC 10.7).
  static constexpr uint32_t kWaves = 2;
  static_assert(kVecsPerBlock * kVecSize == kBlockTokens);
  using vec_t = device::AlignedVector<DType, kVecSize>;
};

template <typename InT, typename OutT>
struct BlockAmaxParams {
  const InT* __restrict__ scores;
  OutT* __restrict__ amax_scores;
  const int32_t* __restrict__ seq_len;
  int64_t stride_scores;       // in elements
  int64_t stride_amax_scores;  // in elements
  uint32_t topk;               // rows with <= topk blocks are skipped, 0 = never skip
};

/// grid = (rows, <= ceil(max_keys / kKeysPerCTA)); a CTA owns chunks blockIdx.y, blockIdx.y + gridDim.y, ...
/// of kKeysPerCTA consecutive keys of one row, a thread kNumItems keys kBlockSize apart (coalesced loads).
template <bool kUsePDL, typename InT, typename OutT>
__global__ __launch_bounds__(BlockAmaxConfig<InT, OutT>::kBlockSize, BlockAmaxConfig<InT, OutT>::kOccupancy)  //
    void amax8_varlen_kernel(const __grid_constant__ BlockAmaxParams<InT, OutT> params) {
  using namespace device;
  using C = BlockAmaxConfig<InT, OutT>;
  using vec_t = typename C::vec_t;
  const auto bx = blockIdx.x;
  const auto by = blockIdx.y;
  const auto tx = threadIdx.x;

  PDLWaitPrimary<kUsePDL>();  // seq_len and scores are the previous kernels' outputs
  const auto seq_len = static_cast<uint32_t>(params.seq_len[bx]);
  const auto num_keys = (seq_len + C::kBlockTokens - 1) / C::kBlockTokens;
  if (num_keys <= params.topk || by * C::kKeysPerCTA >= num_keys) {
    return PDLTriggerSecondary<kUsePDL>();
  }
  const auto* __restrict__ in = params.scores + bx * params.stride_scores;
  auto* __restrict__ out = params.amax_scores + bx * params.stride_amax_scores;

  for (uint32_t first_key = by * C::kKeysPerCTA; first_key < num_keys; first_key += gridDim.y * C::kKeysPerCTA) {
    vec_t vec[C::kNumItems][C::kVecsPerBlock];
#pragma unroll
    for (uint32_t i = 0; i < C::kNumItems; ++i) {
      const auto idx = first_key + tx + i * C::kBlockSize;
      if (idx < num_keys) {
#pragma unroll
        for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
          vec[i][v].load(in, idx * C::kVecsPerBlock + v);
        }
      }
    }
    // The dependent grid may start its prologue now; its griddepcontrol.wait still
    // covers every store below (it waits for this grid to complete).
    PDLTriggerSecondary<kUsePDL>();

#pragma unroll
    for (uint32_t i = 0; i < C::kNumItems; ++i) {
      const auto idx = first_key + tx + i * C::kBlockSize;
      if (idx < num_keys) {
        // fmaxf and __hmax2 ignore a NaN score, torch.amax would propagate it
        if constexpr (std::is_same_v<InT, float>) {
          float key = vec[i][0][0];
#pragma unroll
          for (uint32_t v = 0; v < C::kVecsPerBlock; ++v) {
#pragma unroll
            for (uint32_t j = 0; j < C::kVecSize; ++j) {
              key = fmaxf(key, vec[i][v][j]);
            }
          }
          const auto fp32_inf = std::numeric_limits<float>::infinity();
          out[idx] = cast<OutT>(idx + 1 == num_keys ? fp32_inf : key);
        } else {
          const auto vec_ptr = reinterpret_cast<bf16x2_t*>(vec[i][0].data());
          static_assert(C::kVecSize % 2 == 0);
          bf16x2_t key2 = vec_ptr[0];
#pragma unroll
          for (uint32_t v = 1; v < C::kVecsPerBlock * C::kVecSize / 2; ++v) {
            key2 = __hmax2(key2, vec_ptr[v]);
          }
          const auto key = __hmax(key2.x, key2.y);
          const auto bf16_inf = cast<bf16_t>(std::numeric_limits<float>::infinity());
          out[idx] = cast<OutT>(idx + 1 == num_keys ? bf16_inf : key);
        }
      }
    }
  }
}

/// Host entry: `amax_scores[b, i] = max(scores[b, 8 i : 8 i + 8])` for
/// `i < ceil(seq_len[b] / 8)`, the last of them +inf; rows with at most `topk`
/// blocks untouched. `scores` (`InT`) rows must keep every block of 8 aligned
/// (stride % 8 == 0); `amax_scores` is `OutT`. Each type is fp32 or bf16.
/// The grid covers `amax_scores`' width, so the caller sizes it for the longest
/// row: `seq_len[b] <= 8 * amax_scores.shape[1]` for every row (not checked).
template <bool kPDL, typename InT, typename OutT>
struct BlockAmaxKernel {
  static uint32_t max_grid_y(int device_id, uint32_t rows) {
    using C = BlockAmaxConfig<InT, OutT>;
    if (rows == 0) return 1;
    const uint32_t resident_ctas = host::init_once(device_id, [device_id] {
      int sms = 0, ctas_per_sm = 0;
      host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device_id));
      host::RuntimeDeviceCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &ctas_per_sm, amax8_varlen_kernel<kPDL, InT, OutT>, C::kBlockSize, 0));
      return static_cast<uint32_t>(sms * ctas_per_sm);
    });
    return std::max(1u, host::div_ceil(resident_ctas * C::kWaves, rows));
  }

  static void amax8_varlen(
      const tvm::ffi::TensorView scores,
      const tvm::ffi::TensorView seq_lens,
      const tvm::ffi::TensorView amax_scores,
      const uint32_t topk) {
    using namespace host;
    using C = BlockAmaxConfig<InT, OutT>;
    auto B = SymbolicSize{"batch_size"};
    auto L = SymbolicSize{"max_seq_len"};
    auto K = SymbolicSize{"max_keys"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLGPU>();
    TensorMatcher({B, L})  // scores
        .with_strides({-1, 1})
        .with_dtype<typename C::DType>()
        .with_device(device_)
        .ensure_alignment(C::kBlockBytes)
        .verify(scores);
    TensorMatcher({B})  // seq_lens
        .with_dtype<int32_t>()
        .with_device(device_)
        .verify(seq_lens);
    TensorMatcher({B, K})  // amax_scores
        .with_strides({-1, 1})
        .with_dtype<typename C::KeyT>()
        .with_device(device_)
        .verify(amax_scores);
    const auto max_keys = K.unwrap();  // ceil(longest row / 8), sized by the caller
    const auto params = BlockAmaxParams<InT, OutT>{
        .scores = static_cast<const InT*>(scores.data_ptr()),
        .amax_scores = static_cast<OutT*>(amax_scores.data_ptr()),
        .seq_len = static_cast<const int32_t*>(seq_lens.data_ptr()),
        .stride_scores = scores.stride(0),
        .stride_amax_scores = amax_scores.stride(0),
        .topk = topk,
    };
    const auto rows = static_cast<uint32_t>(B.unwrap());
    const auto chunks = static_cast<uint32_t>(div_ceil(max_keys, static_cast<int64_t>(C::kKeysPerCTA)));
    const auto grid = dim3(rows, std::min(chunks, max_grid_y(device_.unwrap().device_id, rows)));
    LaunchKernel(grid, C::kBlockSize, device_.unwrap())
        .config({.use_pdl = kPDL})
        .launch(amax8_varlen_kernel<kPDL, InT, OutT>, params);
  }
};

}  // namespace sglang
'''),
        ],
    ),
    (
        '''sglang/kernels/jit/csrc/deepseek_v4/mega_moe_pre_dispatch.cuh''',
        '''#include <sgl_kernel/once.h>
''',
        [
            ('''#include <sgl_kernel/tensor.h>
''',
             '''#include <sgl_kernel/once.h>
#include <sgl_kernel/tensor.h>
'''),
            ('''template <uint32_t kGroupSize, bool kUsePDL, bool kMultiChunk, bool kWriteSharedSF, bool kCopyTopk>
''',
             '''template <
    uint32_t kGroupSize,
    bool kUsePDL,
    bool kMultiChunk,
    uint32_t kChunksPerThread,
    bool kWriteSharedSF,
    bool kCopyTopk>
'''),
            ('''    const auto quantize_chunk = [&](uint32_t chunk) {
      InputVec in_vec;
      in_vec.load(token_in, chunk);

''',
             '''    const auto quantize_chunk = [&](uint32_t chunk, const InputVec& in_vec) {
'''),
            ('''        quantize_chunk(chunk);
''',
             '''        InputVec in_vec;
        in_vec.load(token_in, chunk);
        quantize_chunk(chunk, in_vec);
'''),
            ('''      quantize_chunk(tid);
''',
             '''      InputVec in_vecs[kChunksPerThread];
#pragma unroll
      for (uint32_t c = 0; c < kChunksPerThread; ++c) {
        in_vecs[c].load(token_in, tid + c * blockDim.x);
      }
#pragma unroll
      for (uint32_t c = 0; c < kChunksPerThread; ++c) {
        quantize_chunk(tid + c * blockDim.x, in_vecs[c]);
      }
'''),
            ('''    const uint32_t slot = pad_base + copy_bid * blockDim.x + tid;
''',
             ''''''),
            ('''
    if (slot < total_slots) {
      params.buf_topk_idx[slot] = -1;
      params.buf_topk_weights[slot] = 0.0f;
''',
             '''#pragma unroll
    for (uint32_t c = 0; c < kChunksPerThread; ++c) {
      const uint32_t slot = pad_base + (copy_bid * kChunksPerThread + c) * blockDim.x + tid;
      if (slot < total_slots) {
        params.buf_topk_idx[slot] = -1;
        params.buf_topk_weights[slot] = 0.0f;
      }
'''),
            ('''      mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL, false, kWriteSharedSF, kCopyTopk>;
''',
             '''      mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL, false, 1, kWriteSharedSF, kCopyTopk>;
  static constexpr auto kernel_two_chunks =
      mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL, false, 2, kWriteSharedSF, kCopyTopk>;
'''),
            ('''      mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL, true, kWriteSharedSF, kCopyTopk>;
''',
             '''      mega_moe_pre_dispatch_kernel<static_cast<uint32_t>(kGroupSize), kUsePDL, true, 1, kWriteSharedSF, kCopyTopk>;

  static int max_threads_per_sm(int device_id) {
    return host::init_once(device_id, [device_id] {
      int threads = 0;
      host::RuntimeDeviceCheck(cudaDeviceGetAttribute(&threads, cudaDevAttrMaxThreadsPerMultiProcessor, device_id));
      return threads;
    });
  }
'''),
            ('''    const auto block_size = std::min(num_chunks, kMaxBlockThreads);
''',
             ''''''),
            ('''    const bool one_chunk_per_thread = num_chunks <= kMaxBlockThreads;
''',
             '''    const bool one_chunk_per_thread = num_chunks <= kMaxBlockThreads;
    const bool two_chunks_per_thread =
        one_chunk_per_thread && num_chunks % (2u * device::kWarpThreads) == 0 && top_k <= num_chunks / 2u &&
        2u * num_chunks > static_cast<uint32_t>(max_threads_per_sm(device.unwrap().device_id));
    const uint32_t chunks_per_thread = two_chunks_per_thread ? 2u : 1u;
    const auto block_size = std::min(num_chunks, kMaxBlockThreads) / chunks_per_thread;
'''),
            ('''    const uint32_t num_pad_blocks = pad_slots == 0 ? 0u : ((pad_slots + block_size - 1u) / block_size);
''',
             '''    const uint32_t pad_slots_per_block = block_size * chunks_per_thread;
    const uint32_t num_pad_blocks = div_ceil(pad_slots, pad_slots_per_block);
'''),
            ('''    if (one_chunk_per_thread) {
      LaunchKernel(num_total_blocks, block_size, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel_one_chunk, params);
    } else {
      LaunchKernel(num_total_blocks, block_size, device.unwrap())  //
          .enable_pdl(kUsePDL)(kernel_multi_chunk, params);
    }
''',
             '''    const auto kernel = two_chunks_per_thread  ? kernel_two_chunks
                        : one_chunk_per_thread ? kernel_one_chunk
                                               : kernel_multi_chunk;
    LaunchKernel(num_total_blocks, block_size, device.unwrap()).enable_pdl(kUsePDL)(kernel, params);
'''),
        ],
    ),
]

def main() -> int:
    base = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE)
    rc = 0
    for rel, sentinel, patches in FILES:
        target = base / rel
        if not target.is_file():
            print("ERROR: target not found: %s (base image changed?)" % target,
                  file=sys.stderr)
            return 1
        src = target.read_text()
        if sentinel in src:
            print("already patched: %s" % target)
            continue
        for old, _ in patches:
            n = src.count(old)
            if n != 1:
                print("ERROR: %s: expected exactly 1 anchor match, found %d -- "
                      "base changed, re-derive from sglang#%s" % (target, n, PR),
                      file=sys.stderr)
                rc = 1
        if rc:
            return rc
        for old, new in patches:
            src = src.replace(old, new)
        if sentinel not in src or any(new not in src for _, new in patches):
            print("ERROR: post-patch verification failed: %s" % target,
                  file=sys.stderr)
            return 1
        target.write_text(src)
        print("patched %s (%d hunk%s)" % (target, len(patches),
              "" if len(patches) == 1 else "s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
