// DirectLLM HLSL SM 6.6 - (C) 2026 DirectLLM Team
// High Performance FP32 Input / INT4 Weight-Only GEMM
// Fixed: InputX and OutputY are float32 (matching CPU upload format)

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16

struct GEMMConstants {
    uint M; // Batch dimension (1 for decode)
    uint N; // Output columns  = vocab size
    uint K; // Inner dimension = hidden size
};

ConstantBuffer<GEMMConstants> config : register(b0);

// FP32 activations - matches what ModelPipeline uploads from CPU embedding lookup
StructuredBuffer<float> InputX : register(t0);

// INT4 packed weights (2 weights per byte, 8 per uint)
StructuredBuffer<uint>  QuantizedWeightsW : register(t1);

// Per-block quantization scales (one scale per group of 32 weights)
StructuredBuffer<float> WeightScales : register(t2);

// Per-block quantization zero points (packed INT4)
StructuredBuffer<uint>  WeightZeroPoints : register(t3);

// FP32 output logits
RWStructuredBuffer<float> OutputY : register(u0);

// Shared tile for input vector X (fast LDS reuse across the K loop)
groupshared float tileX[BLOCK_SIZE_Y][BLOCK_SIZE_X];

[numthreads(BLOCK_SIZE_X, BLOCK_SIZE_Y, 1)]
void main(uint3 gid  : SV_GroupID,
          uint3 gtid : SV_GroupThreadID,
          uint3 dtid : SV_DispatchThreadID)
{
    uint row = dtid.y; // token row in batch M
    uint col = dtid.x; // output column in vocab N

    float accum = 0.0f;

    uint numTiles = (config.K + BLOCK_SIZE_X - 1) / BLOCK_SIZE_X;

    for (uint tileIdx = 0; tileIdx < numTiles; ++tileIdx) {
        // Cooperative load of X tile into LDS
        uint x_col = tileIdx * BLOCK_SIZE_X + gtid.x;
        if (row < config.M && x_col < config.K)
            tileX[gtid.y][gtid.x] = InputX[row * config.K + x_col];
        else
            tileX[gtid.y][gtid.x] = 0.0f;

        GroupMemoryBarrierWithGroupSync();

        // Dequantize and accumulate
        for (uint k_local = 0; k_local < BLOCK_SIZE_X; ++k_local) {
            uint k_global = tileIdx * BLOCK_SIZE_X + k_local;
            if (k_global < config.K && col < config.N) {
                // Unpack INT4 weight (8 weights per uint)
                uint wIdx      = (k_global * config.N + col) / 8;
                uint wSubOff   = (k_global * config.N + col) % 8;
                uint packedReg = QuantizedWeightsW[wIdx];
                uint rawInt4   = (packedReg >> (wSubOff * 4)) & 0x0F;

                // Scale (group size 32)
                uint  qBlock = k_global / 32;
                float scale  = WeightScales[qBlock * config.N + col];

                // Zero point (packed INT4)
                uint zpIdx    = (qBlock * config.N + col) / 8;
                uint zpSubOff = (qBlock * config.N + col) % 8;
                float zp      = (float)((WeightZeroPoints[zpIdx] >> (zpSubOff * 4)) & 0x0F);

                float w = (float(rawInt4) - zp) * scale;
                accum  += tileX[gtid.y][k_local] * w;
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }

    if (row < config.M && col < config.N)
        OutputY[row * config.N + col] = accum;
}