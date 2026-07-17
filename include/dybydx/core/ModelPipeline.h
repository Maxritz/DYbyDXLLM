// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "dybydx/core/DirectXEngine.h"
#include "dybydx/core/KVCacheManager.h"
#include "dybydx/core/DirectStorageLoader.h"
#include "dybydx/core/IntelOpenVINOInterop.h"
#include "dybydx/core/AdvancedVendorOptimizations.h"

namespace DirectLLM {

    using float16_t = uint16_t;

    enum class ModelArchitecture {
        Llama = 0,
        DeepSeek,
        Gemma,
        Phi,
        Laguna
    };

    enum class ModelType {
        Dense,            // Standard Transformer (Llama, Phi, Gemma, Laguna)
        MixtureOfExperts  // Sparse MoE (Mixtral 8x7B, DeepSeek)
    };

    enum class QuantizationType {
        None_FP32,        // Full Single Precision (32-bit)
        None_FP16,        // Pure Half Precision (16-bit)
        Q1_K,             // 1-bit ultra-sparse weights
        Q2_K,             // 2-bit block-quantized weights
        Q3_K,             // 3-bit block-quantized weights
        Q4_0,             // 4-bit block-quantized weights (legacy)
        Q4_1,             // 4-bit block-quantized weights with linear offset
        Q4_K,             // 4-bit block-quantized weights (modern)
        Q5_0,             // 5-bit block-quantized weights
        Q5_1,             // 5-bit block-quantized weights with linear offset
        Q5_K,             // 5-bit block-quantized weights (modern)
        Q6_K,             // 6-bit block-quantized weights
        Q8_0,             // 8-bit block-quantized weights (legacy)
        Q8_K,             // 8-bit block-quantized weights (modern)
        BF16,             // bfloat16
        IQ4_NL,           // 4-bit non-linear (32/block)
        IQ4_XS            // 4-bit non-linear super-block (256/block)
    };

    enum class DeviceLocation {
        GPU_VRAM,         // Native ultra-fast High Bandwidth Memory
        CPU_SystemRAM     // System memory offloaded via PCIe/Host-Visible Heap
    };

    struct Tensor {
        std::string Name;
        std::vector<size_t> Shape;
        QuantizationType QuantType;
        DeviceLocation Location;
        
        ComPtr<ID3D12Resource> GPUResource;
        std::vector<uint8_t> CPUHostData;
        
        ComPtr<ID3D12Resource> GPUScales;
        ComPtr<ID3D12Resource> GPUZeroPoints;
        std::vector<float16_t> CPUScales;
        std::vector<uint8_t> CPUZeroPoints;

        size_t GetSizeInBytes() const;
    };

    struct ModelConfig {
        ModelArchitecture Arch = ModelArchitecture::Llama;
        ModelType Type = ModelType::Dense;
        std::wstring ModelName;
        size_t NumLayers = 32;
        size_t HiddenDim = 4096;
        size_t IntermediateDim = 11008;
        size_t NumHeads = 32;
        size_t NumKVHeads = 0;    // GQA; 0 = same as NumHeads (MHA)
        size_t HeadDim = 128;
        size_t VocabSize = 32000;
        float  RopeFreqBase = 10000.0f;
        float  RmsNormEps = 1e-5f;
        bool   RopeNeox = false;  // true: rotate (i, i+half); false: adjacent pairs
        
        QuantizationType WeightQuantType = QuantizationType::Q4_K;
        KVCacheQuantType CacheQuantType = KVCacheQuantType::None_FP16;
        
        size_t NumExperts = 8;
        size_t ActiveExpertsK = 2;
        
        float VramAllocationLimitMB = 6000.0f;
        bool EnableSystemRamOffload = true;
        bool ForceCpuOnly = false;
        bool UseMetadataOnlyLoad = false;
        bool ForceStreamMode = false;
    };

struct TransformerLayer {
        size_t LayerIndex;
        DeviceLocation PrimaryLocation;

        // Attention weights - support both fused QKV and separate Q/K/V
        Tensor QKV_Proj;
        Tensor Q_Proj;  // Separate Q projection (Llama 3.2 style)
        Tensor K_Proj;  // Separate K projection (Llama 3.2 style)
        Tensor V_Proj;  // Separate V projection (Llama 3.2 style)
        Tensor O_Proj;  // Output projection

        // Optional per-projection biases (Qwen2) and Q/K head norms (Qwen3)
        Tensor Q_Bias;
        Tensor K_Bias;
        Tensor V_Bias;
        Tensor Attn_Q_Norm;
        Tensor Attn_K_Norm;

        // RMSNorm weights
        Tensor Attn_Norm;    // Attention layer norm
        Tensor FFN_Norm;    // FFN layer norm

        // FFN weights (for non-MoE models)
        Tensor FFN_Gate_Proj;
        Tensor FFN_Up_Proj;
        Tensor FFN_Down_Proj;

        // MoE weights
        Tensor MoE_Gate;
        std::vector<Tensor> Experts;
        std::vector<DeviceLocation> ExpertLocations;
    };

    class ModelPipeline {
    public:
        ModelPipeline();
        ~ModelPipeline();

        bool Initialize(DirectXEngine* dxEngine, const ModelConfig& config);
        
        bool LoadModelWeights(const std::wstring& weightsPath);

        bool LoadOpenVinoIR(const std::string& xmlPath, const std::string& binPath, const std::string& deviceTarget = "") {
            return m_openVINO.LoadModelIR(xmlPath, binPath, deviceTarget);
        }
        
        bool RunInferenceStep(uint32_t batchSize, 
                               const std::vector<int32_t>& inputTokenIds, 
                               uint32_t currentSequenceOffset,
                               std::vector<float>& outLogits);

        void Reset();
        
        size_t GetTotalVramUsageInBytes() const { return m_vramUsageBytes; }
        size_t GetTotalSystemRamUsageInBytes() const { return m_systemRamUsageBytes; }
        float GetGpuExecutionRatio() const;

        KVCacheManager& GetKVCache() { return m_kvCache; }
        IntelOpenVINOInterop& GetOpenVINO() { return m_openVINO; }
        DirectStorageLoader& GetDirectStorage() { return m_dsLoader; }
        AdvancedVendorOptimizations& GetOptimizations() { return m_opts; }

    private:
        bool m_hasAVX512F    = false;
        bool m_hasAVX2       = false;
        bool m_hasAVX        = false;
        int  m_cpuThreadCount = 1;
        void DetectCPUCapabilities();

        DirectXEngine* m_dxEngine = nullptr;
        ModelConfig m_config;
        std::vector<TransformerLayer> m_layers;
        
        std::unordered_map<std::string, Tensor> m_weightTensors;
        
        Tensor m_embedTokens;
        Tensor m_lmHead;

        // Subsystems
        DirectStorageLoader  m_dsLoader;
        KVCacheManager       m_kvCache;
        IntelOpenVINOInterop m_openVINO;
        AdvancedVendorOptimizations m_opts;

        size_t m_vramUsageBytes = 0;
        size_t m_systemRamUsageBytes = 0;
        size_t m_gpuDispatchedOperators = 0;
        size_t m_cpuDispatchedOperators = 0;

        ComPtr<ID3D12Resource> m_gpuStagingBuffer;
        size_t m_stagingBufferSize = 1024 * 1024 * 256;

        bool AllocateTensor(Tensor& tensor, size_t sizeInBytes, DeviceLocation location);
        bool DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y);
        bool DispatchCPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y);
        bool DispatchMoERouting(const Tensor& X, const TransformerLayer& layer, std::vector<float>& expertWeights, std::vector<std::vector<uint32_t>>& expertTokens);
        
ComPtr<ID3D12CommandAllocator> m_cmdAllocator;
        ComPtr<ID3D12GraphicsCommandList> m_cmdList;
        ComPtr<ID3D12Fence> m_executionFence;
        UINT64 m_fenceValue = 0;

        ComPtr<ID3D12RootSignature> m_gemmRootSignature;
        ComPtr<ID3D12PipelineState> m_gemmPSO;
        bool m_gemmPSOReady = false;

        ComPtr<ID3D12RootSignature> m_flashAttnRootSig;
        ComPtr<ID3D12PipelineState> m_flashAttnPSO;

        ComPtr<ID3D12Resource> m_xUploadBuffer;
        ComPtr<ID3D12Resource> m_xGPUBuffer;
        ComPtr<ID3D12Resource> m_yGPUBuffer;
        ComPtr<ID3D12Resource> m_yReadbackBuffer;
        size_t m_persistentHiddenBytes = 0;
        size_t m_persistentVocabBytes  = 0;

        ComPtr<ID3D12CommandAllocator> m_inferCmdAllocator;
        ComPtr<ID3D12GraphicsCommandList> m_inferCmdList;
        ComPtr<ID3D12Fence> m_inferFence;
        UINT64 m_inferFenceValue = 0;
        HANDLE m_inferFenceEvent = nullptr;
        bool m_inferAllocUsed = false; // RDNA4: skip allocator Reset before first submit

        bool BuildGEMMPipeline();
        bool BuildFlashAttentionPipeline();
        bool EnsurePersistentBuffers(size_t hiddenBytes, size_t vocabBytes);

        float DotProductSIMD(const float* a, const float* b, int n) const;

        void WaitForGPU();

        bool DispatchCPUMatrixMultiplyQKV(const Tensor& X, const Tensor& W, Tensor& Y);
        void ComputeFlashAttentionCPU(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                                        std::vector<float>& out, int seqLen, int nHeads, int headDim, uint32_t layerIdx);
        bool DispatchGPUFlashAttention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                                        std::vector<float>& out, int seqLen, int nHeads, int headDim, uint32_t layerIdx);

        // CPU weights cache for Zen3 GEMM
        std::vector<float> m_cpuWeightsCacheFloat;
        size_t m_cpuWeightsCacheN = 0;
        size_t m_cpuWeightsCacheK = 0;
        std::string m_cpuWeightsCacheKey;
        bool m_cpuWeightsCacheValid = false;

        // CPU-side KV cache: [layer] -> seqLen * (NumKVHeads * HeadDim) floats.
        // The CPU forward pass keeps its own K/V history; the GPU KVCacheManager
        // is only used by GPU attention paths.
        std::vector<std::vector<float>> m_cpuKCache;
        std::vector<std::vector<float>> m_cpuVCache;
        uint32_t m_cpuSeqLen = 0;

        // Unified quant support: dequantize row `rowIdx` (length rowLen) of W into out
        bool DequantRow(const Tensor& W, size_t rowIdx, size_t rowLen, float* out) const;
        // y[0..N) = W(N x K) * x[0..K), multithreaded, any supported quant
        bool MatVec(const Tensor& W, const float* x, float* y, size_t N, size_t K) const;
    };
}
