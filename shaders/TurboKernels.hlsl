// DybyDx HLSL SM 6.6 - (C) 2026 DirectLLM Team
// TurboKernels: dFlash (FlashAttention-2), dSpark (MoE top-K routing),
//               TurboQuant (INT4 dequant)
//
// Wave size: [WaveSize(32)] forces Wave32 on all hardware.
//   AMD RDNA 3/4 (9070 XT) supports Wave32 mode in compute via this attribute.
//   Intel Arc (Xe-LPG, Core Ultra iGPU) natively Wave32.
//   NVIDIA Ampere+ Wave32.
//   => Consistent intrinsics, same dispatch on all machines.
//
// All kernels correctly WRITE their output (prior stubs did not).

// ============================================================
//  [dflash] FlashAttention-2 (Online Softmax, Tiled QK^T V)
//  Fuses attention score + softmax + weighted V sum in one pass.
//  Avoids materialising the full N×N attention matrix in VRAM.
//  Uses SM6.6 WaveIntrinsics for parallel max/sum reduction.
// ============================================================

#define WARP_SIZE   32       // forced below with [WaveSize(32)]
#define MAX_HEAD_DIM 128     // max head dimension supported

// KV cache quant type IDs, must match DirectLLM::KVCacheQuantType enum order exactly.
#define KVQ_FP32 0
#define KVQ_FP16 1
#define KVQ_FP8  2
#define KVQ_INT8 3
#define KVQ_INT4 4

struct AttentionConstants {
    uint  BatchSize;
    uint  NumHeads;
    uint  HeadDim;        // actual head dimension (<=MAX_HEAD_DIM)
    uint  SeqLen;         // number of VALID cached positions to attend over (current cache length)
    float InvSqrtD;       // 1/sqrt(HeadDim) pre-computed
    uint  MaxSeqLen;      // allocated cache capacity; needed for stride math, != SeqLen
    uint  QuantType;      // KVQ_* : how KeyBuffer/ValueBuffer slots are packed
    uint  HeadStrideBytes;// bytes per (layer, head, seqPos) slot, from KVCacheManager::GetHeadStrideBytes()
    uint  NumLayers;      // total transformer layers sharing this cache buffer
    uint  LayerIdx;       // which layer's region this dispatch reads/writes
};

ConstantBuffer<AttentionConstants> attnConfig : register(b0);

// Query is always exactly one token (this kernel is decode-only: one new token
// attending over cached history). [Heads, HeadDim] FP32, no Batch/Seq stride needed.
StructuredBuffer<float>   QueryBuffer  : register(t0);

// KV cache buffers, bound directly from KVCacheManager::GetKeyBuffer()/GetValueBuffer().
// Head-major layout, byte-addressed because slot size/format depends on QuantType.
// Persistently UAV (KVCacheManager never leaves them in another state), so bound as UAV here too.
RWByteAddressBuffer        KeyBuffer    : register(u1);
RWByteAddressBuffer        ValueBuffer  : register(u2);

RWStructuredBuffer<float> AttnOutput   : register(u0); // [Heads, HeadDim] FP32, one token

// Dequantise element `idx` (0..HeadDim-1) of the head-slot starting at byte `rowBase`.
// Must exactly mirror KVCacheManager::QuantizeVector's CPU-side encode.
float DequantElement(RWByteAddressBuffer buf, uint rowBase, uint idx) {
    if (attnConfig.QuantType == KVQ_FP32) {
        return asfloat(buf.Load(rowBase + idx * 4));
    } else if (attnConfig.QuantType == KVQ_FP16) {
        uint wordOffset = rowBase + (idx / 2) * 4;
        uint word = buf.Load(wordOffset);
        uint half_ = (idx % 2 == 0) ? (word & 0xFFFF) : (word >> 16);
        return f16tof32(half_);
    } else if (attnConfig.QuantType == KVQ_FP8) {
        uint byteOffset = rowBase + idx;
        uint word = buf.Load(byteOffset & ~3u);
        uint shift = (byteOffset & 3u) * 8;
        uint byteVal = (word >> shift) & 0xFF;
        // Must mirror FloatToFP8: byte 0=-448, 128=0, 255=+447 (linear map)
        return ((float)byteVal / 127.5f - 1.0f) * 448.0f;
    } else if (attnConfig.QuantType == KVQ_INT8) {
        float scale = asfloat(buf.Load(rowBase));
        uint byteOffset = rowBase + 4 + idx;
        uint word = buf.Load(byteOffset & ~3u);
        uint shift = (byteOffset & 3u) * 8;
        int sv = (int)((word >> shift) & 0xFF);
        if (sv >= 128) sv -= 256; // sign-extend 8-bit
        return (float)sv * scale;
    } else { // KVQ_INT4
        float scale = asfloat(buf.Load(rowBase));
        uint nibbleByteOffset = rowBase + 4 + (idx / 2);
        uint word = buf.Load(nibbleByteOffset & ~3u);
        uint shift = (nibbleByteOffset & 3u) * 8;
        uint byteVal = (word >> shift) & 0xFF;
        uint nibble = (idx % 2 == 0) ? (byteVal & 0x0F) : (byteVal >> 4);
        return ((float)nibble - 8.0f) * scale; // same +8 bias as CPU INT4 encode
    }
}

// One wave = one (query, head) pair. Decode-only: exactly one query token per dispatch.
// Grid: Dispatch(1, NumHeads, BatchSize)
[WaveSize(32)]
[numthreads(WARP_SIZE, 1, 1)]
void FusedFlashAttentionKernel(uint3 dtid  : SV_DispatchThreadID,
                                uint3 gtid  : SV_GroupThreadID,
                                uint3 gid   : SV_GroupID) {
    uint lane    = gtid.x;
    uint headIdx = gid.y;
    uint batch   = gid.z;

    if (headIdx >= attnConfig.NumHeads) return;

    uint headDim = attnConfig.HeadDim;
    uint qBase   = headIdx * headDim; // single token: no Batch/Seq stride

    // Load Q into registers (each lane holds a slice; loop for headDim > WARP_SIZE)
    float q[MAX_HEAD_DIM / WARP_SIZE];
    uint elementsPerLane = (headDim + WARP_SIZE - 1) / WARP_SIZE;
    for (uint e = 0; e < elementsPerLane; ++e) {
        uint d = lane + e * WARP_SIZE;
        q[e] = (d < headDim) ? QueryBuffer[qBase + d] : 0.0f;
    }

    // Online softmax state
    float runningMax = -1e30f;
    float runningSum = 0.0f;

    float acc[MAX_HEAD_DIM / WARP_SIZE];
    for (uint e = 0; e < elementsPerLane; ++e) acc[e] = 0.0f;

    uint numKVTiles = (attnConfig.SeqLen + WARP_SIZE - 1) / WARP_SIZE;
    uint rowStride = attnConfig.HeadStrideBytes;

    for (uint tileIdx = 0; tileIdx < numKVTiles; ++tileIdx) {
        uint kPos = tileIdx * WARP_SIZE + lane; // cached position this lane handles

        // No causal mask needed: SeqLen is the cache's current fill length, which by
        // construction only ever contains this token and strictly earlier ones.
        float score = 0.0f;
        if (kPos < attnConfig.SeqLen) {
            uint kRowBase = (((batch * attnConfig.NumLayers + attnConfig.LayerIdx) * attnConfig.NumHeads + headIdx)
                              * attnConfig.MaxSeqLen + kPos) * rowStride;
            for (uint e = 0; e < elementsPerLane; ++e) {
                uint d = lane + e * WARP_SIZE;
                if (d < headDim) score += q[e] * DequantElement(KeyBuffer, kRowBase, d);
            }
            score = WaveActiveSum(score) * attnConfig.InvSqrtD;
        } else {
            score = -1e30f;
        }

        // ---- Online softmax update (FlashAttention-2 style) ----
        float tileMax = WaveActiveMax(score);
        float newMax  = max(runningMax, tileMax);

        float p       = exp(score - newMax);
        float rescale = exp(runningMax - newMax);
        float tileSum = WaveActiveSum(p);

        runningSum = runningSum * rescale + tileSum;
        runningMax = newMax;

        // ---- Weighted V accumulation ----
        if (kPos < attnConfig.SeqLen) {
            uint vRowBase = (((batch * attnConfig.NumLayers + attnConfig.LayerIdx) * attnConfig.NumHeads + headIdx)
                              * attnConfig.MaxSeqLen + kPos) * rowStride;
            for (uint e = 0; e < elementsPerLane; ++e) {
                uint d = lane + e * WARP_SIZE;
                float vVal = (d < headDim) ? DequantElement(ValueBuffer, vRowBase, d) : 0.0f;
                acc[e] = acc[e] * rescale + p * vVal;
            }
        } else {
            for (uint e = 0; e < elementsPerLane; ++e)
                acc[e] = acc[e] * rescale;
        }
    }

    // ---- Write normalised output ----
    float invSum = (runningSum > 0.0f) ? (1.0f / runningSum) : 0.0f;
    uint outBase = headIdx * headDim; // single token output, no Batch/Seq stride
    for (uint e = 0; e < elementsPerLane; ++e) {
        uint d = lane + e * WARP_SIZE;
        if (d < headDim)
            AttnOutput[outBase + d] = acc[e] * invSum;
    }
}


// ============================================================
//  [dspark] Sparse MoE Top-K Gating & Routing Kernel
//  Correct formulation:
//    logit[e] = dot(HiddenState[token], GateWeights[expert_e])
//    top-K selection via parallel reduction
//    softmax over top-K logits
//    writes expert indices + normalised gating weights
//
//  Supports up to 256 experts (DeepSeek V3 scale).
//  One thread group per token.
// ============================================================

#define MAX_EXPERTS 256
#define DSPARK_THREADS 64

struct MoEConstants {
    uint NumTokens;
    uint NumExperts;    // total experts (e.g. 256 for DeepSeek)
    uint ActiveK;       // top-K (e.g. 2 for Mixtral, 6 for DeepSeek)
    uint HiddenDim;     // hidden state dimension
};

ConstantBuffer<MoEConstants> moeConst : register(b1);

StructuredBuffer<float>   HiddenStates : register(t3); // [NumTokens, HiddenDim]
StructuredBuffer<float>   GateWeights  : register(t4); // [NumExperts, HiddenDim]

// Output: for each token, ActiveK expert IDs and their normalised weights
RWStructuredBuffer<uint>  ExpertIds    : register(u1); // [NumTokens, ActiveK]
RWStructuredBuffer<float> ExpertWeights : register(u2); // [NumTokens, ActiveK]

groupshared float gs_logits[MAX_EXPERTS];
groupshared uint  gs_topIds[8];       // support up to top-8
groupshared float gs_topWeights[8];

[WaveSize(32)]
[numthreads(DSPARK_THREADS, 1, 1)]
void SparseMoERoutingKernel(uint3 gid  : SV_GroupID,
                              uint3 gtid : SV_GroupThreadID) {
    uint tokenIdx = gid.x;
    uint tid      = gtid.x;

    if (tokenIdx >= moeConst.NumTokens) return;

    // ---- 1. Compute logit for every expert: logit[e] = h · W_gate[e] ----
    // Threads cooperate: each thread handles a chunk of experts.
    uint expertsPerThread = (moeConst.NumExperts + DSPARK_THREADS - 1) / DSPARK_THREADS;
    uint eStart = tid * expertsPerThread;
    uint eEnd   = min(eStart + expertsPerThread, moeConst.NumExperts);

    for (uint e = eStart; e < eEnd; ++e) {
        float dot = 0.0f;
        uint hBase = tokenIdx * moeConst.HiddenDim;
        uint gBase = e        * moeConst.HiddenDim;
        // Inner loop: dot product of hidden state with expert gate row
        for (uint d = 0; d < moeConst.HiddenDim; d += 4) {
            float h0 = HiddenStates[hBase + d + 0];
            float h1 = (d+1 < moeConst.HiddenDim) ? HiddenStates[hBase + d + 1] : 0.0f;
            float h2 = (d+2 < moeConst.HiddenDim) ? HiddenStates[hBase + d + 2] : 0.0f;
            float h3 = (d+3 < moeConst.HiddenDim) ? HiddenStates[hBase + d + 3] : 0.0f;
            dot += h0 * GateWeights[gBase + d + 0]
                 + h1 * GateWeights[gBase + d + 1]
                 + h2 * GateWeights[gBase + d + 2]
                 + h3 * GateWeights[gBase + d + 3];
        }
        gs_logits[e] = dot;
    }

    GroupMemoryBarrierWithGroupSync();

    // ---- 2. Top-K selection (thread 0 does sequential scan) ----
    // For ActiveK <= 8, this is fast. Could parallelise for K > 8.
    if (tid == 0) {
        for (uint k = 0; k < moeConst.ActiveK && k < 8; ++k) {
            float best    = -1e30f;
            uint  bestIdx = 0;
            for (uint e = 0; e < moeConst.NumExperts; ++e) {
                // Skip already selected experts
                bool skip = false;
                for (uint prev = 0; prev < k; ++prev)
                    if (gs_topIds[prev] == e) { skip = true; break; }
                if (!skip && gs_logits[e] > best) {
                    best    = gs_logits[e];
                    bestIdx = e;
                }
            }
            gs_topIds[k]     = bestIdx;
            gs_topWeights[k] = best;
        }

        // ---- 3. Softmax over top-K logits ----
        float maxW = gs_topWeights[0];
        for (uint k = 1; k < moeConst.ActiveK && k < 8; ++k)
            maxW = max(maxW, gs_topWeights[k]);

        float sumExp = 0.0f;
        for (uint k = 0; k < moeConst.ActiveK && k < 8; ++k) {
            gs_topWeights[k] = exp(gs_topWeights[k] - maxW);
            sumExp += gs_topWeights[k];
        }
        float invSum = (sumExp > 0.0f) ? (1.0f / sumExp) : 0.0f;
        for (uint k = 0; k < moeConst.ActiveK && k < 8; ++k)
            gs_topWeights[k] *= invSum;

        // ---- 4. Write outputs ----
        uint outBase = tokenIdx * moeConst.ActiveK;
        for (uint k = 0; k < moeConst.ActiveK && k < 8; ++k) {
            ExpertIds[outBase + k]     = gs_topIds[k];
            ExpertWeights[outBase + k] = gs_topWeights[k];
        }
    }
}


// ============================================================
//  [turboquant] INT4 Block-Wise Dequantization Kernel
//  Unpacks Q4_0 / Q4_K packed weights -> FP32 output.
//  Supports arbitrary matrix dimensions via constants.
//  Each thread unpacks one packed uint (8 weights).
// ============================================================

struct TurboQuantConstants {
    uint NumRows;         // weight matrix rows (vocab or hidden)
    uint NumCols;         // weight matrix cols (hidden or intermediate)
    uint GroupSize;       // quantisation group size (e.g. 32)
};

ConstantBuffer<TurboQuantConstants> tqConst : register(b2);

StructuredBuffer<uint>   TQ_PackedWeights : register(t5); // INT4 packed: 8 weights per uint
StructuredBuffer<float>  TQ_Scales        : register(t6); // one scale per group
StructuredBuffer<float>  TQ_ZeroPoints    : register(t7); // one zero-point per group (packed INT4)

RWStructuredBuffer<float> TQ_Output       : register(u3); // dequantised FP32 weights

// Each thread handles 8 weights (one uint).
// Dispatch: ceil(NumRows * NumCols / 8) threads
[WaveSize(32)]
[numthreads(64, 1, 1)]
void TurboQuantDequantKernel(uint3 dtid : SV_DispatchThreadID) {
    uint packedIdx = dtid.x; // index into PackedWeights array
    uint totalPacked = (tqConst.NumRows * tqConst.NumCols + 7) / 8;
    if (packedIdx >= totalPacked) return;

    uint packed = TQ_PackedWeights[packedIdx];

    // Unpack 8 INT4 values (nibbles)
    uint baseElemIdx = packedIdx * 8;

    for (uint i = 0; i < 8; ++i) {
        uint elemIdx = baseElemIdx + i;
        if (elemIdx >= tqConst.NumRows * tqConst.NumCols) break;

        // Extract nibble i from the packed uint
        uint nibble = (packed >> (i * 4)) & 0x0F;

        // Find which group this element belongs to
        uint groupIdx = elemIdx / tqConst.GroupSize;
        float scale = TQ_Scales[groupIdx];

        // Zero-point: packed as INT4 in TQ_ZeroPoints
        uint zpPacked = TQ_ZeroPoints[groupIdx / 2];
        float zp = (float)((groupIdx & 1) ? (zpPacked >> 4) & 0x0F : zpPacked & 0x0F);

        // Dequantize: w_float = (nibble - zp) * scale
        float dequant = ((float)nibble - zp) * scale;

        TQ_Output[elemIdx] = dequant;
    }
}
