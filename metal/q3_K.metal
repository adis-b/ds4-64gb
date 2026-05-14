// Q3_K dense matmul/matvec kernels for the 64 GB target.
//
// Attention tensors are stored as Q8_0 in the upstream GGUF. When the model
// is requantized to fit on a 64 GB Mac the attention rows can be shrunk to
// Q3_K, which is ~3.4 bpw and ~60% smaller. Routed experts (50 GiB at IQ1_S)
// dominate the file size; cutting attention from 5 GiB to 2 GiB is what lets
// the mlock budget cover a meaningfully larger top-K expert set.
//
// The kernels here are the Q3_K equivalents of kernel_mul_mv_q8_0_f32 and
// kernel_mul_mm_q8_0_f32, ported from llama.cpp's ggml-metal.metal. They are
// not used for routed experts (those have their own _id variants in
// moe.metal); they are dispatched directly from the attention path in
// ds4_gpu_matmul_q3_K_tensor.

// moe.metal #undef's QK_K and QK_NL after its own use, so we redefine here.
#define QK_K 256
#define QK_NL 16

#define N_R0_Q3_K 2
#define N_SG_Q3_K 4

struct block_q3_K {
    uchar hmask[QK_K/8];   // high bit of the 3-bit quants
    uchar qs[QK_K/4];      // low 2 bits of the 3-bit quants
    uchar scales[12];      // per-sub-block scales, packed 6-bit
    half  d;               // super-block scale
};

// Dequantize one 16-element chunk of a Q3_K super-block at chunk index `il`
// (0 <= il < 16). Used by the matmul template, which expects (block, il)
// to fill a 4x4 register tile.
template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uchar  * q = (device const uchar  *)xb->qs;
    device const uchar  * h = (device const uchar  *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    q = q + 32 * (il/8) + 16 * (il&1);
    h = h + 16 * (il&1);
    uint8_t m = 1 << (il/2);
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) :
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    for (int i = 0; i < 16; ++i) {
        reg[i/4][i%4] = dl * (q[i] & mask) - (h[i] & m ? 0 : ml);
    }
}

// Q3_K matvec kernel. Mirrors kernel_mul_mv_q8_0_f32_impl: one row block per
// threadgroup, one super-block of 256 weights consumed across 4 simdgroup
// lanes (ix = tiisg%4). Used for decode (n_tok == 1).
template<short nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%args.ne12;
    const uint i13 = im/args.ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x  = (device const block_q3_K *) (src0 + offset0);
    device const float      * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    const short tid = tiisg/4;
    const short ix  = tiisg%4;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // Masks indexed by (ip, il/2). Pre-tabulated because the Metal compiler
    // doesn't constant-fold this trig down to two literals on its own.
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},
                           {0x0004, 0x0400, 0x0008, 0x0800},
                           {0x0010, 0x1000, 0x0020, 0x2000},
                           {0x0040, 0x4000, 0x0080, 0x8000}};

    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00},
                        {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + ix*QK_K + y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = ix; i < nb; i += 4) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        device const uint16_t * q  = (device const uint16_t *)(x[i].qs    + q_offset);
        device const uint16_t * h  = (device const uint16_t *)(x[i].hmask + l0);
        device const uint16_t * a  = (device const uint16_t *)(x[i].scales);
        device const half     * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2];
                s1 += yl[l+0]  * (qs & qm[il/2][0]);
                s2 += yl[l+1]  * (qs & qm[il/2][1]);
                s3 += ((h[l/2] & hm[0]) ? 0.f : yl[l+0])
                    + ((h[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((h[l/2] & hm[2]) ? 0.f : yl[l+16])
                    + ((h[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = q[l/2+8];
                s1 += yl[l+8]  * (qs & qm[il/2][0]);
                s2 += yl[l+9]  * (qs & qm[il/2][1]);
                s3 += ((h[l/2+8] & hm[0]) ? 0.f : yl[l+ 8])
                    + ((h[l/2+8] & hm[1]) ? 0.f : yl[l+ 9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((h[l/2+8] & hm[2]) ? 0.f : yl[l+24])
                    + ((h[l/2+8] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += 4 * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst
        + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ds4_metal_args_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ds4_metal_args_mul_mv &>(
            args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// Q3_K matmul. The dense.metal kernel_mul_mm template iterates the K dim in
// chunks of NK=32 weights and calls `dequantize_func(x, il, temp_a)` to fill
// a 4x4 register tile per chunk. For Q3_K, one super-block holds 256 weights
// = 16 chunks of 16 weights each, so the `nl` template parameter is 16.
template [[host_name("kernel_mul_mm_q3_K_f32")]]
kernel mul_mm_t kernel_mul_mm<
        half,  half4x4,   simdgroup_half8x8,
        half,  half2x4,   simdgroup_half8x8,
        block_q3_K, 16, dequantize_q3_K,
        float, float4x4, float, float2x4>;
