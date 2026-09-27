#include "kittens.cuh"

#ifndef ATTN_B
constexpr int ATTN_B = 16; // batch size
#endif

#ifndef ATTN_H
constexpr int ATTN_H = 32; // number of heads
#endif

#ifndef ATTN_H_KV
constexpr int ATTN_H_KV = 8; // number of heads for key and value
#endif

constexpr int GROUP_SIZE = ATTN_H / ATTN_H_KV; // queries per KV head group

#ifndef ATTN_N
constexpr int ATTN_N = 8192; // sequence length
#endif

#ifndef ATTN_D
constexpr int ATTN_D = 128; // dimension
#endif
#ifndef ATTN_SINK
#define ATTN_SINK 0
#endif
#if ATTN_D == 64
#define FA_VM2 "2"
#define FA_VM4 "4"
#else
#define FA_VM2 "2"
#define FA_VM4 "4"
#endif
constexpr int Q_BLOCK_SIZE = 32; // q block size
constexpr int KV_BLOCK_SIZE = 64; // kv block size
constexpr bool causal = true;
// WINDOW>0: sliding-window attention, query i attends keys in [i-WINDOW+1, i]
#ifndef WINDOW
#define WINDOW 0
#endif

#define NUM_WARPS 4
#define NUM_THREADS (kittens::WARP_THREADS * NUM_WARPS)

#define MFMA_MASK 0x08
#define VALU_MASK 0x02
#define EXP_MASK  0x400

using namespace kittens;
using _gl_QKVO = gl<bf16, -1, -1, -1, -1>;

using G = kittens::group<NUM_WARPS>;

#define SCHED_BARRIER(mask, cnt, group) __builtin_amdgcn_sched_group_barrier(mask, cnt, group)

template<int Pairs, int VALU_CNT, int Group>
__device__ __forceinline__ void sched_barrier_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(VALU_MASK, VALU_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_pairs<Pairs - 1, VALU_CNT, Group>();
}

template<int Pairs, int EXP_CNT, int Group>
__device__ __forceinline__ void sched_barrier_exp_pairs() {
    SCHED_BARRIER(MFMA_MASK, 1, Group);
    SCHED_BARRIER(EXP_MASK, EXP_CNT, Group);
    if constexpr (Pairs > 1) sched_barrier_exp_pairs<Pairs - 1, EXP_CNT, Group>();
}

template<typename T, ducks::rt_layout::all layout, ducks::rt_shape::all shape>
__device__ inline void exp2(rt_base<T, layout, shape> &dst, const rt_base<T, layout, shape> &src) {
    static_assert(std::is_same_v<shape, rt_32x32_s>, "Only 32x32 tiles are supported");

    #pragma unroll
    for(int k = 0; k < dst.packed_per_thread; k++) {
        dst.data[k] = base_ops::exp2::op(src.data[k]);
    }

}

template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using qo_tile = rt<T, Q_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using qo_tile_transposed = rt<T, D, Q_BLOCK_SIZE, L, S>;
template<int D, typename T=bf16, typename L=row_l, typename S=rt_32x16_s> using kv_tile = rt<T, KV_BLOCK_SIZE, D, L, S>;
template<int D, typename T=bf16, typename L=col_l, typename S=rt_16x32_s> using kv_tile_transposed = rt<T, D, KV_BLOCK_SIZE, L, S>;
template<typename T=float, typename L=col_l, typename S=rt_16x32_4_s> using attn_tile = rt<T, KV_BLOCK_SIZE, Q_BLOCK_SIZE, L, S>;

/**********************************************************/
template<int THR_X, int THR_Y>
__device__ inline void mask_vec2_imm(uint32_t rel_vgpr, uint32_t rel_hi_vgpr, uint32_t neg_inf_vgpr,
                                     uint32_t& x_ref, uint32_t& y_ref) {

    uint64_t x_mask, y_mask;
#if WINDOW
    // causal+window in one asm block to not disturb register allocation
    asm volatile(
        "v_cmp_lt_i32_e64 %0, %4, %5\n\t"
        "v_cmp_lt_i32_e64 %1, %4, %7\n\t"
        "v_cndmask_b32_e64 %2, %2, %6, %0\n\t"
        "v_cndmask_b32_e64 %3, %3, %6, %1\n\t"
        "v_cmp_ge_i32_e64 %0, %8, %5\n\t"
        "v_cmp_ge_i32_e64 %1, %8, %7\n\t"
        "v_cndmask_b32_e64 %2, %2, %6, %0\n\t"
        "v_cndmask_b32_e64 %3, %3, %6, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "+v"(x_ref), "+v"(y_ref)
        : "v"(rel_vgpr), "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y), "v"(rel_hi_vgpr)
        : "vcc"
    );
#else
    asm volatile(
        // x: rel < THR_X ?
        "v_cmp_lt_i32_e64 %0, %6, %7\n\t"
        // y: rel < THR_Y ?
        "v_cmp_lt_i32_e64 %1, %6, %9\n\t"
        "v_cndmask_b32_e64 %2, %4, %8, %0\n\t"
        "v_cndmask_b32_e64 %3, %5, %8, %1\n\t"
        : "=s"(x_mask), "=s"(y_mask), "=v"(x_ref), "=v"(y_ref)
        : "v"(x_ref), "v"(y_ref), "v"(rel_vgpr),
          "n"(THR_X), "v"(neg_inf_vgpr), "n"(THR_Y)
        : "vcc"
    );
#endif
}

template<ducks::rt::col_layout RT>
__device__ inline void mask_kv_tile(RT &dst, int q_abs, int k_abs, uint32_t neg_inf_v, int lane) {
    const int col  = lane & 31;                 // 0..31 column within the 32-wide col tile

    // Absolute positions
    const int q_base = q_abs * Q_BLOCK_SIZE;    // start index for this Q tile
    const int k_base = k_abs * KV_BLOCK_SIZE;   // start index for this K/V tile

    // q position for this lane's column
    const int q_pos  = q_base + col;

    #pragma unroll
    for (int i = 0; i < dst.height; ++i) {
        // Row base of the 32x* chunk produced by MFMA
        const int row_base = (i * 32) + ((lane >> 5) << 2); // multiplesof 4

        // Relative index of the FIRST element in this row-chunk w.r.t. q_pos
        // (smaller rel ⇒ more "future" keys that must be -inf)
        const int rel0 = q_pos - (k_base + row_base);
        const uint32_t rel = static_cast<uint32_t>(rel0);
        // rel-WINDOW keeps THR within the inline-constant range
        const uint32_t rel_hi = static_cast<uint32_t>(rel0 - WINDOW);

        #pragma unroll
        for (int j = 0; j < dst.width; ++j) {
            auto& d0x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].x);
            auto& d0y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[0].y);
            auto& d1x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].x);
            auto& d1y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[1].y);
            auto& d2x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].x);
            auto& d2y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[2].y);
            auto& d3x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].x);
            auto& d3y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[3].y);
            auto& d4x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].x);
            auto& d4y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[4].y);
            auto& d5x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].x);
            auto& d5y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[5].y);
            auto& d6x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].x);
            auto& d6y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[6].y);
            auto& d7x = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].x);
            auto& d7y = *reinterpret_cast<uint32_t*>(&dst.tiles[i][j].data[7].y);

            //  - reuse a single neg_inf register
            //  - keep VCC live across the pair
            //  - avoid reloading -inf or recomputing rel
            mask_vec2_imm< 0, 1 >(rel, rel_hi, neg_inf_v, d0x, d0y);
            mask_vec2_imm< 2, 3 >(rel, rel_hi, neg_inf_v, d1x, d1y);
            mask_vec2_imm< 8, 9 >(rel, rel_hi, neg_inf_v, d2x, d2y);
            mask_vec2_imm<10,11 >(rel, rel_hi, neg_inf_v, d3x, d3y);
            mask_vec2_imm<16,17 >(rel, rel_hi, neg_inf_v, d4x, d4y);
            mask_vec2_imm<18,19 >(rel, rel_hi, neg_inf_v, d5x, d5y);
            mask_vec2_imm<24,25 >(rel, rel_hi, neg_inf_v, d6x, d6y);
            mask_vec2_imm<26,27 >(rel, rel_hi, neg_inf_v, d7x, d7y);
        }
    }
}

/**********************************************************/

// GQA-fused SWA128 forward: a workgroup is one 32-query tile x one KV group. Its (at most) three in-window KV tiles
// stay resident in LDS (48 KB -> three workgroups per CU) and warp w walks query heads 2w, 2w+1 over them, so no warp
// streams K/V from global per tile. Per-tile math (QK MMA, fp32 scale-after, window mask, online max/scale/norm,
// bf16 P, PV MMA order, sink, LSE) is the pipelined kernel's, tile for tile; its fully masked tiles are not visited.
constexpr int LDS_TILES = 3, WARP_HEADS = GROUP_SIZE / NUM_WARPS;

template<int D> __launch_bounds__(NUM_THREADS, 3)
__global__ void attend_ker(bf16 *O_ptr, float *L_vec_ptr, bf16 *Q_ptr, bf16 *K_ptr, bf16 *V_ptr
#if ATTN_SINK
    , float *Sinks_ptr
#endif
    ) {
    _gl_QKVO Og{O_ptr, ATTN_B, ATTN_N, ATTN_H, ATTN_D};
    _gl_QKVO Qg{Q_ptr, ATTN_B, ATTN_N, ATTN_H, ATTN_D};
    _gl_QKVO Kg{K_ptr, ATTN_B, ATTN_N, ATTN_H_KV, ATTN_D};
    _gl_QKVO Vg{V_ptr, ATTN_B, ATTN_N, ATTN_H_KV, ATTN_D};
    gl<float, -1, -1, -1, -1> L_vec{L_vec_ptr, ATTN_B, ATTN_H, 1, ATTN_N};

    extern __shared__ alignment_dummy __shm[];
    shared_allocator al((int*)&__shm[0]);
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s> (&k_smem)[LDS_TILES] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>, LDS_TILES>();
    st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s> (&v_smem)[LDS_TILES] = al.allocate<st_bf<KV_BLOCK_SIZE, ATTN_D, st_8x32_s>, LDS_TILES>();

    const int head_idx_kv = blockIdx.x;
    // wave-uniform head keeps the Q/O/L buffer descriptors scalar (no readfirstlane waterfall loops)
    const int first_head = head_idx_kv * GROUP_SIZE + __builtin_amdgcn_readfirstlane(warpid()) * WARP_HEADS;
    const int tile_idx = blockIdx.y;
    const int batch_idx = blockIdx.z;
    const int lane = laneid();
    const int q_start_pos = tile_idx * Q_BLOCK_SIZE;
    const int first_tile = max(0, q_start_pos - WINDOW + 1) / KV_BLOCK_SIZE;
    const int last_tile = (q_start_pos + Q_BLOCK_SIZE - 1) / KV_BLOCK_SIZE;

    constexpr float TEMPERATURE_SCALE = 0.125f*1.44269504089f;
    uint32_t neg_inf_v = 0xff800000;

    using T = typename st_bf<KV_BLOCK_SIZE, ATTN_D, st_32x32_s>::dtype;
    constexpr int bytes_per_thread = st_32x32_s::template bytes_per_thread<T>();
    constexpr int bytes_per_memcpy = bytes_per_thread * NUM_THREADS;
    constexpr int memcpy_per_tile = KV_BLOCK_SIZE * ATTN_D * sizeof(T) / bytes_per_memcpy;
    uint32_t swizzled_offsets_V[memcpy_per_tile];
    uint32_t swizzled_offsets_K[memcpy_per_tile];
    G::prefill_swizzled_offsets<1, false>(k_smem[0], Kg, swizzled_offsets_K);
    G::prefill_swizzled_offsets<1, false>(v_smem[0], Vg, swizzled_offsets_V);
    for (int t = first_tile; t <= last_tile; t++) {
        G::load<1, false>(k_smem[t - first_tile], Kg, {batch_idx, t, head_idx_kv, 0}, swizzled_offsets_K);
        G::load<1, false>(v_smem[t - first_tile], Vg, {batch_idx, t, head_idx_kv, 0}, swizzled_offsets_V);
    }

    qo_tile<D, bf16> q_reg;
    qo_tile_transposed<D, bf16> q_reg_transposed;
    kv_tile<D, bf16> k_reg;
    kv_tile_transposed<D, bf16> k_reg_transposed;
    kv_tile<D, bf16, col_l, rt_16x32_4_s> v_reg;
    qo_tile_transposed<D, float, col_l, rt_32x32_s> o_reg;
    attn_tile<float, col_l, rt_32x32_s> att_block;
    attn_tile<bf16, col_l, rt_32x32_s> att_block_bf16;
    attn_tile<bf16, col_l, rt_16x32_4_s> att_block_bf16_in;
    typename attn_tile<float, col_l, rt_32x32_s>::row_vec max_vec, norm_vec, max_vec_prev, scale_vec;
    qo_tile<D, float> q_reg_fl;

    load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, Qg, {batch_idx, tile_idx, first_head, 0});
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_sched_barrier(0);
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_sched_barrier(0);

    for (int hh = 0; hh < WARP_HEADS; hh++) {
        const int head_idx = first_head + hh;
        copy(q_reg, q_reg_fl);
        transpose(q_reg_transposed, q_reg);
        if (hh + 1 < WARP_HEADS) load<1, qo_tile<D, float>, _gl_QKVO>(q_reg_fl, Qg, {batch_idx, tile_idx, head_idx + 1, 0});

        zero(o_reg);
        zero(norm_vec);
        zero(max_vec_prev);
        add(max_vec_prev, max_vec_prev, -1e4f);  // same max floor as the pipelined kernel's first tile

        for (int t = first_tile; t <= last_tile; t++) {
            __builtin_amdgcn_sched_barrier(0);
            load(k_reg, k_smem[t - first_tile]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            zero(att_block);
            transpose(k_reg_transposed, k_reg);
            mma_AtB(att_block, k_reg_transposed, q_reg_transposed, att_block);
            mul(att_block, att_block, TEMPERATURE_SCALE);
            mask_kv_tile(att_block, tile_idx, t, neg_inf_v, lane);
            col_max(max_vec, att_block, max_vec_prev);
            sub(scale_vec, max_vec_prev, max_vec);
            copy(max_vec_prev, max_vec);
            exp2(scale_vec, scale_vec);
            sub_col(att_block, att_block, max_vec);
            exp2(att_block.tiles[0][0], att_block.tiles[0][0]);
            exp2(att_block.tiles[1][0], att_block.tiles[1][0]);
            mul_col(o_reg, o_reg, scale_vec);
            mul(norm_vec, norm_vec, scale_vec);
            col_sum(norm_vec, att_block, norm_vec);
            copy(att_block_bf16, att_block);
            att_block_bf16_in = *reinterpret_cast<attn_tile<bf16, col_l, rt_16x32_4_s>*>(&att_block_bf16);
            __builtin_amdgcn_sched_barrier(0);
            load(v_reg, v_smem[t - first_tile]);
            asm volatile("s_waitcnt lgkmcnt(0)");
            __builtin_amdgcn_sched_barrier(0);
            mma_AtB(o_reg, v_reg, att_block_bf16_in, o_reg);
        }
#if ATTN_SINK
        {
            const float sink_l2 = Sinks_ptr[head_idx] * 1.44269504089f;
            typename attn_tile<float, col_l, rt_32x32_s>::row_vec sink_term;
            mul(sink_term, max_vec, -1.0f);
            add(sink_term, sink_term, sink_l2);
            exp2(sink_term, sink_term);
            add(norm_vec, norm_vec, sink_term);
        }
#endif
        div_col(o_reg, o_reg, norm_vec);
        qo_tile<D, float, row_l, rt_32x32_s> o_reg_transposed;
        transpose(o_reg_transposed, o_reg);
        store<1>(Og, o_reg_transposed, {batch_idx, tile_idx, head_idx, 0});
        mul(max_vec, max_vec, 0.69314718056f);
        log(norm_vec, norm_vec);
        add(norm_vec, norm_vec, max_vec);
        store(L_vec, norm_vec, {batch_idx, head_idx, 0, tile_idx});
    }
}

template __global__ void attend_ker<ATTN_D>(bf16*, float*, bf16*, bf16*, bf16*
#if ATTN_SINK
    , float*
#endif
);

static_assert(ATTN_D == 64 && WINDOW == 128 && NUM_WARPS == 4 && GROUP_SIZE == 8);
static_assert(sizeof(st_bf<64,64,st_32x32_s>) == 8192 && sizeof(st_bf<64,64,st_8x32_s>) == 8192);
