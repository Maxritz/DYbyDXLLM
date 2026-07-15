// DirectLLM HLSL SM 6.6 - (C) 2026 DirectLLM Team
// High Performance INT8 DP4A (Dot Product 4-way Accumulate) Compute Kernel

#define THREAD_GROUP_SIZE 64

struct MatrixConfig {
    uint RowCount;
    uint ColCount;
    uint InnerDim;
};

ConstantBuffer<MatrixConfig> matConfig : register(b0);

// FP16 Activations (Row major)
StructuredBuffer<float16_t> InputX : register(t0);

// INT8 Quantized weights (Packed as 4 signed 8-bit integers inside each uint descriptor)
StructuredBuffer<uint> QuantWeightsINT8 : register(t1);

// Scale per row blocks
StructuredBuffer<float16_t> ChannelScales : register(t2);

// Output buffer
RWStructuredBuffer<float16_t> OutputY : register(u0);

[numthreads(THREAD_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint rowIdx = dtid.y;
    uint colIdx = dtid.x;

    if (rowIdx >= matConfig.RowCount || colIdx >= matConfig.ColCount) return;

    float accumulator = 0.0f;
    uint packedWordCount = matConfig.InnerDim / 4;

    for (uint i = 0; i < packedWordCount; ++i) {
        // 1. Read input activations and construct temporary FP16 structures
        float16_t4 actVector;
        actVector.x = InputX[rowIdx * matConfig.InnerDim + (i * 4 + 0)];
        actVector.y = InputX[rowIdx * matConfig.InnerDim + (i * 4 + 1)];
        actVector.z = InputX[rowIdx * matConfig.InnerDim + (i * 4 + 2)];
        actVector.w = InputX[rowIdx * matConfig.InnerDim + (i * 4 + 3)];

        // 2. Load packed 4-way signed INT8 weights from unstructured descriptor
        uint packedWeights = QuantWeightsINT8[i * matConfig.ColCount + colIdx];

        // 3. Unpack to signed floating points
        float dequantW0 = (float)((int)((packedWeights >> 0) & 0xFF) - 128);
        float dequantW1 = (float)((int)((packedWeights >> 8) & 0xFF) - 128);
        float dequantW2 = (float)((int)((packedWeights >> 16) & 0xFF) - 128);
        float dequantW3 = (float)((int)((packedWeights >> 24) & 0xFF) - 128);

        // 4. Multiply activations by dequantized weight representations
        accumulator += (float)actVector.x * dequantW0;
        accumulator += (float)actVector.y * dequantW1;
        accumulator += (float)actVector.z * dequantW2;
        accumulator += (float)actVector.w * dequantW3;
    }

    // 5. Apply linear scale multiplier to restore weights range
    float16_t scaleMultiplier = ChannelScales[colIdx];
    OutputY[rowIdx * matConfig.ColCount + colIdx] = (float16_t)(accumulator * (float)scaleMultiplier);
}
