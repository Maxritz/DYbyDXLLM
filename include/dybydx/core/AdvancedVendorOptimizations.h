// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <vector>
#include <memory>
#include <string>

namespace DirectLLM {

    // Vendor Specific Acceleration architectures
    enum class OptimizationPipelineType {
        StandardD3D12Compute = 0,
        AMDRDNA_Wave32,       // AMD RDNA 1/2/3/4 dedicated lane sizes (Wave32 mode)
        AMDCDNA_MatrixCore,   // AMD CDNA 1/2/3/4 Matrix Cores execution bypass
        NVIDIACUDA_Native,    // NVIDIA CUDA / TensorCore PInvoke bypass
        IntelXMX_DPAS         // Intel Xe Matrix Extensions
    };

    struct OffloadConfig {
        bool EnableSystemRamOffload; // Enable offloading dense weights or Mixture of Experts (MoE) active weights
        float SystemRamOffloadPercent; // Percentage of layers stored in host CPU memory vs GPU VRAM
        uint32_t ActiveExperts;       // Active experts dispatched in MoE models (e.g. 2 active experts in Mixtral 8x7B)
    };

    class AdvancedVendorOptimizations {
    public:
        AdvancedVendorOptimizations();
        ~AdvancedVendorOptimizations();

        bool Initialize(ID3D12Device* device, const OffloadConfig& offloadConfig);
        
        // Fused Attention (FlashAttention-style) dynamic setup
        // Fuses Query-Key-Value projection, softmax scaling, causal masking, and output multiplication into a single compute pass.
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
                                     ID3D12Resource* inputTokens,
                                     const std::vector<ID3D12Resource*>& gpuExperts,
                                     const std::vector<void*>& cpuHostExpertsMemory);

        // Profile custom techniques
        void LogOptimizationSpecs();

    private:
        ID3D12Device* m_device = nullptr;
        OffloadConfig m_config;
        OptimizationPipelineType m_activePipeline = OptimizationPipelineType::StandardD3D12Compute;

        // Custom handles for CUDA or OpenVINO background worker loops
        void* m_cudaContextHandle = nullptr;
        void* m_amdHipDeviceContext = nullptr;

        void DetectHardwarePipeline();
    };
}
