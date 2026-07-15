// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <memory>
#include <string>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    // Vendor Specific Acceleration architectures
    enum class OptimizationPipelineType {
        StandardD3D12Compute = 0,
        AMDRDNA_Wave32,       // AMD RDNA 1/2/3/4 Wave32 mode
        AMDCDNA_MatrixCore,   // AMD CDNA Matrix Cores execution bypass
        NVIDIACUDA_Native,    // NVIDIA CUDA / TensorCore PInvoke bypass
        IntelXMX_DPAS         // Intel Xe Matrix Extensions (XMX)
    };

    struct OffloadConfig {
        bool EnableSystemRamOffload;
        float SystemRamOffloadPercent;
        uint32_t ActiveExperts;
    };

    class AdvancedVendorOptimizations {
    public:
        AdvancedVendorOptimizations();
        ~AdvancedVendorOptimizations();

        bool Initialize(ID3D12Device* device, const OffloadConfig& offloadConfig);
        
        // Fused Attention (FlashAttention-style) dynamic setup
        bool DispatchFusedAttention(ID3D12GraphicsCommandList* cmdList,
                                    ID3D12Resource* query,
                                    ID3D12Resource* key,
                                    ID3D12Resource* value,
                                    ID3D12Resource* output,
                                    uint32_t batchSize,
                                    uint32_t numHeads,
                                    uint32_t headDim,
                                    uint32_t seqLen);

        // Mixture of Experts (MoE) Routing and System Memory Offload engine
        bool DispatchMoEExpertLayers(ID3D12GraphicsCommandList* cmdList,
                                     ID3D12Resource* hiddenStates,
                                     ID3D12Resource* gateWeights,
                                     ID3D12Resource* expertIds,
                                     ID3D12Resource* expertWeights,
                                     uint32_t numTokens,
                                     uint32_t numExperts,
                                     uint32_t activeK,
                                     uint32_t hiddenDim);

        // TurboQuant sub-byte block decompressions
        bool DispatchTurboQuantDequant(ID3D12GraphicsCommandList* cmdList,
                                       ID3D12Resource* packedWeights,
                                       ID3D12Resource* scales,
                                       ID3D12Resource* zeroPoints,
                                       ID3D12Resource* output,
                                       uint32_t numRows,
                                       uint32_t numCols,
                                       uint32_t groupSize);

        void LogOptimizationSpecs();

    private:
        ID3D12Device* m_device = nullptr;
        OffloadConfig m_config;
        OptimizationPipelineType m_activePipeline = OptimizationPipelineType::StandardD3D12Compute;

        // Custom handles for CUDA or HIP background worker loops (for future vendor extensions)
        // Primary target currently is DirectX 12 + Intel OpenVINO.
        void* m_cudaContextHandle = nullptr;
        void* m_amdHipDeviceContext = nullptr;

        // Pipeline States for the compiled HLSL kernels
        ComPtr<ID3D12RootSignature> m_attnRootSignature;
        ComPtr<ID3D12PipelineState> m_attnPSO;

        ComPtr<ID3D12RootSignature> m_moeRootSignature;
        ComPtr<ID3D12PipelineState> m_moePSO;

        ComPtr<ID3D12RootSignature> m_tqRootSignature;
        ComPtr<ID3D12PipelineState> m_tqPSO;

        void DetectHardwarePipeline();
        bool CompileAndBuildPipelines();
    };
}
