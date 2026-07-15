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
        Q8_0,             // 8-bit block-quantized weights
        Q6_K,             // 6-bit block-quantized weights
        Q5_K,             // 5-bit block-quantized weights
        Q4_K,             // 4-bit block-quantized weights
        Q3_K,             // 3-bit block-quantized weights
        Q2_K,             // 2-bit block-quantized weights
        Q1_K              // 1-bit ultra-sparse weights
    };

    enum class DeviceLocation {
        GPU_VRAM,         // Native ultra-fast High Bandwidth Memory
        CPU_SystemRAM     // System memory offloaded via PCIe/Host-Visible Heap
    };

    // Representing a tensor descriptor and its backing data
    struct Tensor {
        std::string Name;
        std::vector<size_t> Shape;
        QuantizationType QuantType;
        DeviceLocation Location;
        
        // GPU Resource Handles
        ComPtr<ID3D12Resource> GPUResource;
        
        // Host RAM backup for offloaded split execution
        std::vector<uint8_t> CPUHostData;
        
        // Scale and zero points for block-wise quantization
        ComPtr<ID3D12Resource> GPUScales;
        ComPtr<ID3D12Resource> GPUZeroPoints;
        std::vector<float16_t> CPUScales;
        std::vector<uint8_t> CPUZeroPoints;

        size_t GetSizeInBytes() const;
    };

    // Architecture config for Dense and MoE weights
    struct ModelConfig {
        ModelArchitecture Arch = ModelArchitecture::Llama;
        ModelType Type = ModelType::Dense;
        std::wstring ModelName;
        size_t NumLayers = 32;
        size_t HiddenDim = 4096;
        size_t IntermediateDim = 11008; // FFN dimension
        size_t NumHeads = 32;
        size_t HeadDim = 128;
        size_t VocabSize = 32000;
        
        // Quantization Configurations
        QuantizationType WeightQuantType = QuantizationType::Q4_K;
        KVCacheQuantType CacheQuantType = KVCacheQuantType::None_FP16;
        
        // MoE parameters (only used if Type == MixtureOfExperts)
        size_t NumExperts = 8;
        size_t ActiveExpertsK = 2; // e.g. Top-2 Routing
        
        // Memory partition config (Split Layer Architecture)
        float VramAllocationLimitMB = 6000.0f; // Max VRAM budget allowed
        bool EnableSystemRamOffload = true;
    };

    // A single transformer layer containing weights split across CPU or GPU
    struct TransformerLayer {
        size_t LayerIndex;
        DeviceLocation PrimaryLocation; // Location of the main FFN/Attention layers
        
        // Attention projections
        Tensor QKV_Proj;
        Tensor O_Proj;

        // Feed-Forward Networks (Dense Models)
        Tensor FFN_Gate_Proj;
        Tensor FFN_Up_Proj;
        Tensor FFN_Down_Proj;

        // Mixture of Experts (MoE Models)
        Tensor MoE_Gate; // Router weights (usually kept in VRAM for fast evaluation)
        std::vector<Tensor> Experts; // List of active experts (dense MLP layers)
        std::vector<DeviceLocation> ExpertLocations; // Each expert can be individually pinned to CPU or GPU
    };

    class ModelPipeline {
    public:
        ModelPipeline();
        ~ModelPipeline();

        bool Initialize(DirectXEngine* dxEngine, const ModelConfig& config);
        
        // Split-layer engine: loads parts of weights to VRAM and offloads remaining to host memory
        bool LoadModelWeights(const std::wstring& weightsPath);
        
        // Forward pass evaluation: dynamically orchestrates CPU and GPU dispatches
        bool RunInferenceStep(uint32_t batchSize, 
                              const std::vector<int32_t>& inputTokenIds, 
                              uint32_t currentSequenceOffset,
                              std::vector<float>& outLogits);

        void Reset();
        
        // Diagnostic metrics
        size_t GetTotalVramUsageInBytes() const { return m_vramUsageBytes; }
        size_t GetTotalSystemRamUsageInBytes() const { return m_systemRamUsageBytes; }
        float GetGpuExecutionRatio() const;

    private:
        DirectXEngine* m_dxEngine = nullptr;
        ModelConfig m_config;
        std::vector<TransformerLayer> m_layers;
        
        // GGUF-loaded weight tensors
        std::unordered_map<std::string, Tensor> m_weightTensors;
        
        // Global word embedding and final language-model head
        Tensor m_embedTokens;
        Tensor m_lmHead;

        // Memory counters
        size_t m_vramUsageBytes = 0;
        size_t m_systemRamUsageBytes = 0;
        size_t m_gpuDispatchedOperators = 0;
        size_t m_cpuDispatchedOperators = 0;

        // Staging buffer used to stream offloaded CPU weights into VRAM dynamically if needed
        ComPtr<ID3D12Resource> m_gpuStagingBuffer;
        size_t m_stagingBufferSize = 1024 * 1024 * 256; // 256 MB streaming window

        // Helper functions
        bool AllocateTensor(Tensor& tensor, size_t sizeInBytes, DeviceLocation location);
        bool DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y);
        bool DispatchCPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y);
        bool DispatchMoERouting(const Tensor& X, const TransformerLayer& layer, std::vector<float>& expertWeights, std::vector<std::vector<uint32_t>>& expertTokens);
        
        // Command recording contexts
        ComPtr<ID3D12CommandAllocator> m_cmdAllocator;
        ComPtr<ID3D12GraphicsCommandList> m_cmdList;
        ComPtr<ID3D12Fence> m_executionFence;
        UINT64 m_fenceValue = 0;

        void WaitForGPU();
    };
}
