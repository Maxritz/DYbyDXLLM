// DirectLLM HLSL SM 6.6 - (C) 2026 DirectLLM Team
// High Performance INT4 Weight-Only Matrix Multiplication (dequantization + GEMM)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define GROUP_SIZE (BLOCK_SIZE_X * BLOCK_SIZE_Y)

// Root signature bindings
struct GEMMConstants {
    uint M; // Batch dimension
    uint N; // Output channel dimension (columns of W)
    uint K; // Input channel dimension (rows of W / elements of X)
};

ConstantBuffer<GEMMConstants> config : register(b0);

// FP16 Activations (Input vector X)
StructuredBuffer<float16_t> InputX : register(t0);

// INT4 Quantized Weights (Pack 2 weights per byte)
// Weight dimensions: (K, N / 2) bytes
StructuredBuffer<uint> QuantizedWeightsW : register(t1);

// Quantization Scales (one per block of weights, e.g., group size 32 or 64)
StructuredBuffer<float16_t> WeightScales : register(t2);

// Quantization Zero Points
StructuredBuffer<uint> WeightZeroPoints : register(t3);

// Output FP16 Activations
RWStructuredBuffer<float16_t> OutputY : register(u0);

// Shared memory tiling for fast cache Reuse
groupshared float16_t tileX[BLOCK_SIZE_Y][BLOCK_SIZE_X];

[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void main(uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID, uint3 dtid : SV_DispatchThreadID) {
    uint row = dtid.y; // Output row (Token index in Batch M)
    uint col = dtid.x; // Output col (Output feature column in Output channel N)

    float accum = 0.0f;

    // Loop through the K dimension in blocks of BLOCK_SIZE_X
    uint numTiles = (config.K + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;
    
    for (uint tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // 1. Cooperative Load: Load input vector tile into fast Local Shared Memory (LDS)
        uint x_col = tileIdx * BLOCK_SIZE_X + gtid.x;
        if (row < config.M && x_col < config.K) {
            tileX[gtid.y][gtid.x] = InputX[row * config.K + x_col];
        } else {
            tileX[gtid.y][gtid.x] = (float16_t)0.0;
        }

        GroupMemoryBarrierWithGroupSync();

        // 2. Compute GEMM inside tile. Dequantize weight on-the-fly.
        // We unpack 4-bit weights, apply scale and zero point.
        for (uint k_local = 0; k_local < BLOCK_SIZE_X; ++k_local) {
            uint k_global = tileIdx * BLOCK_SIZE_X + k_local;
            if (k_global < config.K && col < config.N) {
                // Read packed byte. Since each uint contains 8 quantized weights (4-bit each):
                uint weightIndex = (k_global * config.N + col) / 8;
                uint subOffset = (k_global * config.N + col) % 8;
                
                uint packedReg = QuantizedWeightsW[weightIndex];
                uint rawInt4 = (packedReg >> (subOffset * 4)) & 0x0F;

                // Load scale & zero-point for this quantization block (assumed group size 32)
                uint qBlockIdx = k_global / 32;
                float16_t scale = WeightScales[qBlockIdx * config.N + col];
                
                // Read 4-bit zero point (packed in similar fashion)
                uint zpIndex = (qBlockIdx * config.N + col) / 8;
                uint zpOffset = (qBlockIdx * config.N + col) % 8;
                uint zpReg = WeightZeroPoints[zpIndex];
                float16_t zp = (float16_t)((zpReg >> (zpOffset * 4)) & 0x0F);

                // On-the-fly FP16 dequantization formula: W_unquant = (W_quant - zero_point) * scale
                float dequantWeight = (float)(rawInt4 - zp) * (float)scale;

                accum += (float)tileX[gtid.y][k_local] * dequantWeight;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    // 3. Write back the final float16 value to the output UAV buffer
    if (row < config.M && col < config.N) {
        OutputY[row * config.N + col] = (float16_t)accum;
    }
}
