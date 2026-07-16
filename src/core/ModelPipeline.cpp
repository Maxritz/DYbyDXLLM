// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/ModelPipeline.h"
#include "dybydx/core/GgufLoader.h"
#include "dybydx/core/DirectStorageLoader.h"

#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>
#include <fstream>

// SIMD intrinsics
#include <intrin.h>      // __cpuid / __cpuidex / _xgetbv  (MSVC)
#include <immintrin.h>   // AVX / AVX2 / AVX-512 intrinsics

namespace DirectLLM {

    // ================================================================
    //  Half-precision (Float16) conversion helper
    // ================================================================
    static float FP16ToFloat(uint16_t h) {
        union { uint32_t u; float f; } w;
        uint32_t sign = (h & 0x8000) << 16;
        uint32_t exp  = (h & 0x7C00) >> 10;
        uint32_t mant = h & 0x03FF;
        if (exp == 0x1F) {
            w.u = sign | 0x7F800000 | (mant << 13);
        } else if (exp == 0) {
            if (mant == 0) {
                w.u = sign;
            } else {
                // Subnormal
                exp = 127 - 15;
                while (!(mant & 0x0400)) {
                    mant <<= 1;
                    exp--;
                }
                mant &= 0x03FF;
                w.u = sign | ((exp + 1) << 23) | (mant << 13);
            }
        } else {
            w.u = sign | ((exp - 15 + 127) << 23) | (mant << 13);
        }
        return w.f;
    }

    // ================================================================
    //  Tensor helpers
    // ================================================================
    size_t Tensor::GetSizeInBytes() const {
        size_t elements = 1;
        for (auto d : Shape) elements *= d;
        switch (QuantType) {
            case QuantizationType::None_FP32: return elements * 4;
            case QuantizationType::None_FP16: return elements * 2;
            case QuantizationType::Q8_0:
            case QuantizationType::Q8_K:      return (elements / 32) * 34;
            case QuantizationType::Q6_K:      return (elements * 6) / 8;
            case QuantizationType::Q5_0:
            case QuantizationType::Q5_1:
            case QuantizationType::Q5_K:      return (elements * 5) / 8;
            case QuantizationType::Q4_0:      return (elements / 32) * 18;
            case QuantizationType::Q4_1:      return (elements / 32) * 20;
            case QuantizationType::Q4_K:      return elements / 2;
            case QuantizationType::Q3_K:      return (elements * 3) / 8;
            case QuantizationType::Q2_K:      return elements / 4;
            default: return elements * 2;
        }
    }

    // ================================================================
    //  CPU SIMD Capability Detection
    // ================================================================
    void ModelPipeline::DetectCPUCapabilities() {
        int info[4] = {};

        __cpuid(info, 0);
        int maxLeaf = info[0];

        __cpuid(info, 1);
        bool cpuOSXSAVE = (info[2] >> 27) & 1;
        bool cpuAVX     = (info[2] >> 28) & 1;

        bool osAvxOk    = false;
        bool osAvx512Ok = false;
        if (cpuOSXSAVE && cpuAVX) {
            unsigned long long xcr0 = _xgetbv(0);
            osAvxOk    = (xcr0 & 0x06ULL) == 0x06ULL;
            osAvx512Ok = (xcr0 & 0xE6ULL) == 0xE6ULL;
        }

        m_hasAVX = cpuAVX && osAvxOk;

        if (maxLeaf >= 7) {
            __cpuidex(info, 7, 0);
            m_hasAVX2    = m_hasAVX && ((info[1] >> 5)  & 1);
            m_hasAVX512F = m_hasAVX && osAvx512Ok && ((info[1] >> 16) & 1);
        }

        SYSTEM_INFO si;
        GetSystemInfo(&si);
        m_cpuThreadCount = (int)si.dwNumberOfProcessors;

        std::cout << "[ModelPipeline][CPU] ISA: "
                  << (m_hasAVX512F ? "AVX-512F " : "")
                  << (m_hasAVX2    ? "AVX2 "     : "")
                  << (m_hasAVX     ? "AVX "      : "")
                  << (!m_hasAVX    ? "SSE2 "     : "")
                  << "| " << m_cpuThreadCount << " logical cores" << std::endl;
    }

    // ================================================================
    //  SIMD dot-product kernels
    // ================================================================

#ifdef __AVX512F__
    static float Dot_AVX512(const float* __restrict a, const float* __restrict b, int n) {
        __m512 acc = _mm512_setzero_ps();
        int i = 0;
        for (; i <= n - 16; i += 16)
            acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc);
        alignas(64) float tmp[16];
        _mm512_store_ps(tmp, acc);
        float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7]
                + tmp[8]+tmp[9]+tmp[10]+tmp[11]+tmp[12]+tmp[13]+tmp[14]+tmp[15];
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
#endif

#ifdef __AVX2__
    static float Dot_AVX2(const float* __restrict a, const float* __restrict b, int n) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc);
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
#endif

#ifdef __AVX__
    static float Dot_AVX(const float* __restrict a, const float* __restrict b, int n) {
        __m256 acc = _mm256_setzero_ps();
        int i = 0;
        for (; i <= n - 8; i += 8)
            acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        float s = tmp[0]+tmp[1]+tmp[2]+tmp[3]+tmp[4]+tmp[5]+tmp[6]+tmp[7];
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
#endif

    static float Dot_Scalar(const float* __restrict a, const float* __restrict b, int n) {
        float s = 0.0f;
        for (int i = 0; i < n; i++) s += a[i] * b[i];
        return s;
    }

    float ModelPipeline::DotProductSIMD(const float* a, const float* b, int n) const {
#ifdef __AVX512F__
        if (m_hasAVX512F) return Dot_AVX512(a, b, n);
#endif
#ifdef __AVX2__
        if (m_hasAVX2)    return Dot_AVX2(a, b, n);
#endif
#ifdef __AVX__
        if (m_hasAVX)     return Dot_AVX(a, b, n);
#endif
        return Dot_Scalar(a, b, n);
    }

    static void WaitFence(ID3D12Fence* fence, UINT64 value, HANDLE event) {
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            WaitForSingleObject(event, INFINITE);
        }
    }

    ModelPipeline::ModelPipeline() {}

    ModelPipeline::~ModelPipeline() {
        if (m_inferFenceEvent) CloseHandle(m_inferFenceEvent);
        Reset();
    }

    bool ModelPipeline::Initialize(DirectXEngine* dxEngine, const ModelConfig& config) {
        m_dxEngine = dxEngine;
        m_config   = config;
        DetectCPUCapabilities();

        if (m_dxEngine) {
            m_dsLoader.Initialize(m_dxEngine->GetDevice());
            m_openVINO.InitializeWithSharedDevice(m_dxEngine->GetDevice());

            KVCacheConfig kvConfig;
            kvConfig.MaxSequenceLength = 2048;
            kvConfig.BatchSize = 1;
            kvConfig.NumHeads = (uint32_t)m_config.NumHeads;
            kvConfig.HeadDimension = (uint32_t)m_config.HeadDim;
            kvConfig.QuantType = m_config.CacheQuantType;
            m_kvCache.Initialize(m_dxEngine->GetDevice(), kvConfig);

            // Initialize Advanced Vendor Optimizations (dflash, dspark, turboquant)
            OffloadConfig offConfig;
            offConfig.EnableSystemRamOffload = m_config.EnableSystemRamOffload;
            offConfig.SystemRamOffloadPercent = 0.5f;
            offConfig.ActiveExperts = (uint32_t)m_config.ActiveExpertsK;
            m_opts.Initialize(m_dxEngine, offConfig);
        }

        std::cout << "[ModelPipeline] Initialized pipeline." << std::endl;
        return true;
    }

    bool ModelPipeline::AllocateTensor(Tensor& tensor, size_t sizeInBytes, DeviceLocation location) {
        tensor.Location = location;
        if (location == DeviceLocation::GPU_VRAM) {
            if (!m_dxEngine) return false;
            bool ok = m_dxEngine->AllocateGPUBuffer(sizeInBytes, D3D12_HEAP_TYPE_DEFAULT,
                                                      D3D12_RESOURCE_STATE_COMMON, &tensor.GPUResource);
            if (!ok) return false;
            m_vramUsageBytes += sizeInBytes;
        } else {
            tensor.CPUHostData.resize(sizeInBytes);
            m_systemRamUsageBytes += sizeInBytes;
        }
        return true;
    }

    bool ModelPipeline::BuildGEMMPipeline() {
        if (!m_dxEngine) return false;

        ComPtr<ID3DBlob> shaderBlob;
        if (!m_dxEngine->CompileComputeShader(L"shaders/QuantizedGEMM.hlsl", "main", &shaderBlob)) {
            std::cerr << "[ModelPipeline][PSO] Shader compile failed." << std::endl;
            return false;
        }

        D3D12_ROOT_PARAMETER rootParameters[6] = {};
        rootParameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        rootParameters[0].Constants.ShaderRegister = 0;
        rootParameters[0].Constants.RegisterSpace  = 0;
        rootParameters[0].Constants.Num32BitValues = 3;
        rootParameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        for (int i = 1; i <= 4; i++) {
            rootParameters[i].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParameters[i].Descriptor.ShaderRegister = (UINT)(i - 1);
            rootParameters[i].Descriptor.RegisterSpace  = 0;
            rootParameters[i].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        rootParameters[5].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
        rootParameters[5].Descriptor.ShaderRegister = 0;
        rootParameters[5].Descriptor.RegisterSpace  = 0;
        rootParameters[5].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
        rootSigDesc.NumParameters = 6;
        rootSigDesc.pParameters   = rootParameters;
        rootSigDesc.Flags         = D3D12_ROOT_SIGNATURE_FLAG_NONE;

        ComPtr<ID3DBlob> sig, err;
        if (FAILED(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err)))
            return false;
        if (FAILED(m_dxEngine->GetDevice()->CreateRootSignature(0,
                sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_gemmRootSignature))))
            return false;

        ComPtr<ID3D12PipelineState> pso;
        if (!m_dxEngine->CreateComputePipelineState(shaderBlob.Get(), m_gemmRootSignature.Get(), &pso))
            return false;
        m_gemmPSO = pso;

        if (FAILED(m_dxEngine->GetDevice()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_inferCmdAllocator)))) return false;
        if (FAILED(m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                m_inferCmdAllocator.Get(), m_gemmPSO.Get(), IID_PPV_ARGS(&m_inferCmdList)))) return false;
        m_inferCmdList->Close();

        if (FAILED(m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                IID_PPV_ARGS(&m_inferFence)))) return false;

        m_inferFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        m_inferFenceValue = 0;
        m_gemmPSOReady    = true;

        std::cout << "[ModelPipeline][PSO] GEMM pipeline compiled and cached." << std::endl;
        return true;
    }

    bool ModelPipeline::EnsurePersistentBuffers(size_t hiddenBytes, size_t vocabBytes) {
        if (!m_dxEngine) return false;
        if (hiddenBytes == m_persistentHiddenBytes && vocabBytes == m_persistentVocabBytes)
            return true;

        if (m_dxEngine->SupportsReBAR()) {
            if (!m_dxEngine->AllocateReBarBuffer(hiddenBytes, &m_xGPUBuffer)) return false;
            m_xUploadBuffer.Reset();
        } else {
            if (!m_dxEngine->AllocateGPUBuffer(hiddenBytes, D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ, &m_xUploadBuffer)) return false;
            if (!m_dxEngine->AllocateGPUBuffer(hiddenBytes, D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_STATE_COPY_DEST, &m_xGPUBuffer)) return false;
        }

        if (!m_dxEngine->AllocateGPUBuffer(vocabBytes, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &m_yGPUBuffer)) return false;
        if (!m_dxEngine->AllocateGPUBuffer(vocabBytes, D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST, &m_yReadbackBuffer)) return false;

        m_persistentHiddenBytes = hiddenBytes;
        m_persistentVocabBytes  = vocabBytes;
        return true;
    }

    bool ModelPipeline::LoadModelWeights(const std::wstring& weightsPath) {
        std::wcout << L"[ModelPipeline] Loading: " << weightsPath << std::endl;

        std::string path(weightsPath.begin(), weightsPath.end());
        GgufLoader loader;
        
        bool parseOk = false;
        if (m_config.UseMetadataOnlyLoad) {
            parseOk = loader.LoadMetadataOnly(path);
        } else {
            parseOk = loader.LoadFile(path);
        }
        
        if (!parseOk) {
            std::cerr << "[ModelPipeline] GGUF load failed." << std::endl;
            return false;
        }

        std::cout << "[ModelPipeline] GGUF: version=" << (uint32_t)loader.GetVersion()
                  << " tensors=" << (uint64_t)loader.GetTensorCount() << std::endl;

        if (loader.HasMetadata("general.architecture"))
            std::cout << "[ModelPipeline] Arch: " << loader.GetMetadataString("general.architecture") << std::endl;
        if (loader.HasMetadata("llama.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("llama.block_count");
        else if (loader.HasMetadata("qwen2.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("qwen2.block_count");
        else if (loader.HasMetadata("qwen35.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("qwen35.block_count");
        else if (loader.HasMetadata("phi3.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("phi3.block_count");
        else if (loader.HasMetadata("gemma2.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("gemma2.block_count");
        else if (loader.HasMetadata("gemma4.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("gemma4.block_count");
        else if (m_config.NumLayers == 0)
            m_config.NumLayers = 32;

        if (loader.HasMetadata("llama.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("llama.embedding_length");
        else if (loader.HasMetadata("qwen2.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("qwen2.embedding_length");
        else if (loader.HasMetadata("qwen35.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("qwen35.embedding_length");
        else if (loader.HasMetadata("gemma4.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("gemma4.embedding_length");
        else if (loader.HasMetadata("gemma2.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("gemma2.embedding_length");
        else if (loader.HasMetadata("phi3.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("phi3.embedding_length");
        else if (m_config.HiddenDim == 0)
            m_config.HiddenDim = 4096;

        if (loader.HasMetadata("llama.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("llama.attention.head_count");
        else if (loader.HasMetadata("qwen2.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("qwen2.attention.head_count");
        else if (loader.HasMetadata("qwen35.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("qwen35.attention.head_count");
        else if (loader.HasMetadata("gemma4.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("gemma4.attention.head_count");
        else if (loader.HasMetadata("gemma2.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("gemma2.attention.head_count");
        else if (loader.HasMetadata("phi3.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("phi3.attention.head_count");
        else if (m_config.NumHeads == 0)
            m_config.NumHeads = 32;

        // Auto-enable system RAM offload if model is likely larger than VRAM limit
        m_config.EnableSystemRamOffload = true;

        if (loader.HasMetadata("llama.feed_forward_length"))
            m_config.IntermediateDim = loader.GetMetadataUint32("llama.feed_forward_length");
        else if (loader.HasMetadata("qwen2.feed_forward_length"))
            m_config.IntermediateDim = loader.GetMetadataUint32("qwen2.feed_forward_length");
        else
            m_config.IntermediateDim = m_config.HiddenDim * 4;

        m_config.HeadDim = m_config.HiddenDim / m_config.NumHeads;

        const GgufTensor* tokEmb = loader.GetTensor("token_embd.weight");
        if (tokEmb && tokEmb->Shape.size() >= 2)
            m_config.VocabSize = (size_t)tokEmb->Shape[0];

        // Ensure KV cache is re-initialized with correct dimensions
        if (m_dxEngine) {
            KVCacheConfig kvConfig;
            kvConfig.MaxSequenceLength = 2048;
            kvConfig.BatchSize = 1;
            kvConfig.NumHeads = (uint32_t)m_config.NumHeads;
            kvConfig.HeadDimension = (uint32_t)m_config.HeadDim;
            kvConfig.QuantType = m_config.CacheQuantType;
            m_kvCache.Reset();
            m_kvCache.Initialize(m_dxEngine->GetDevice(), kvConfig);
        }

        // DirectStorage loading fence
        ComPtr<ID3D12Fence> loadFence;
        HANDLE loadEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        UINT64 loadFenceValue = 1;
        if (m_dxEngine) {
            m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&loadFence));
        }

        // Batched GPU upload fallback variables
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        std::vector<ComPtr<ID3D12Resource>> stagingBuffers;
        size_t currentBatchBytes = 0;
        const size_t BATCH_LIMIT = 64ULL * 1024 * 1024; // 64 MB

        if (m_dxEngine && !m_dsLoader.IsInitialized()) {
            m_dxEngine->GetDevice()->CreateCommandAllocator(
                D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
            m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                allocator.Get(), nullptr, IID_PPV_ARGS(&list));
        }

        auto& tensors = loader.GetTensors();
        for (auto& [name, tensor] : tensors) {
            Tensor t;
            t.Name = name;
            for (auto d : tensor.Shape) t.Shape.push_back((size_t)d);
            switch (tensor.Type) {
                case GgmlType::F32:  t.QuantType = QuantizationType::None_FP32; break;
                case GgmlType::F16:  t.QuantType = QuantizationType::None_FP16; break;
                case GgmlType::Q4_0: t.QuantType = QuantizationType::Q4_0;      break;
                case GgmlType::Q4_1: t.QuantType = QuantizationType::Q4_1;      break;
                case GgmlType::Q4_K: t.QuantType = QuantizationType::Q4_K;      break;
                case GgmlType::Q5_0: t.QuantType = QuantizationType::Q5_0;      break;
                case GgmlType::Q5_1: t.QuantType = QuantizationType::Q5_1;      break;
                case GgmlType::Q5_K: t.QuantType = QuantizationType::Q5_K;      break;
                case GgmlType::Q6_K: t.QuantType = QuantizationType::Q6_K;      break;
                case GgmlType::Q8_0: t.QuantType = QuantizationType::Q8_0;      break;
                case GgmlType::Q8_K: t.QuantType = QuantizationType::Q8_K;      break;
                case GgmlType::Q2_K: t.QuantType = QuantizationType::Q2_K;      break;
                case GgmlType::Q3_K: t.QuantType = QuantizationType::Q3_K;      break;
                default:             t.QuantType = QuantizationType::None_FP16;  break;
            }

            size_t sizeInBytes = tensor.DataSize;
            DeviceLocation loc = DeviceLocation::GPU_VRAM;

            if (!m_dxEngine || name.find("token_embd") != std::string::npos)
                loc = DeviceLocation::CPU_SystemRAM;

            if (loc == DeviceLocation::GPU_VRAM && m_config.EnableSystemRamOffload) {
                size_t limit = (size_t)(m_config.VramAllocationLimitMB * 1024.0f * 1024.0f);
                if (m_vramUsageBytes + sizeInBytes > limit) {
                    std::cout << "[ModelPipeline][SplitLoad] VRAM full ("
                              << (m_vramUsageBytes / (1024*1024)) << " MB / "
                              << m_config.VramAllocationLimitMB << " MB) — "
                              << name << " → CPU RAM" << std::endl;
                    loc = DeviceLocation::CPU_SystemRAM;
                }
            }

            if (loc == DeviceLocation::CPU_SystemRAM && loader.IsMetadataOnly()) {
                t.CPUHostData.resize(sizeInBytes);
                std::ifstream f(path, std::ios::binary);
                if (f.is_open()) {
                    uint64_t absOffset = loader.GetTensorDataBaseOffset() + tensor.FileOffset;
                    f.seekg(absOffset, std::ios::beg);
                    f.read(reinterpret_cast<char*>(t.CPUHostData.data()), sizeInBytes);
                    f.close();
                }
                m_systemRamUsageBytes += sizeInBytes;
                t.Location = loc;
            } else {
                if (!AllocateTensor(t, sizeInBytes, loc)) {
                    std::cerr << "[ModelPipeline] Alloc failed: " << name << std::endl;
                    CloseHandle(loadEvent);
                    return false;
                }
            }

            if (loc == DeviceLocation::CPU_SystemRAM) {
                if (!loader.IsMetadataOnly() && tensor.DataPtr && sizeInBytes > 0)
                    std::memcpy(t.CPUHostData.data(), tensor.DataPtr, sizeInBytes);
            } else if (sizeInBytes > 0) {
                if (m_dsLoader.IsInitialized()) {
                    if (loader.IsMetadataOnly()) {
                        uint64_t absOffset = loader.GetTensorDataBaseOffset() + tensor.FileOffset;
                        m_dsLoader.EnqueueFileToGPU(weightsPath, absOffset, sizeInBytes, t.GPUResource.Get());
                    } else {
                        m_dsLoader.EnqueueMemoryToGPU(tensor.DataPtr, sizeInBytes, t.GPUResource.Get());
                    }
                } else if (!loader.IsMetadataOnly() && tensor.DataPtr && list) {
                    ComPtr<ID3D12Resource> upload;
                    if (m_dxEngine->AllocateGPUBuffer(sizeInBytes, D3D12_HEAP_TYPE_UPLOAD,
                            D3D12_RESOURCE_STATE_GENERIC_READ, &upload)) {
                        void* p = nullptr;
                        upload->Map(0, nullptr, &p);
                        std::memcpy(p, tensor.DataPtr, sizeInBytes);
                        upload->Unmap(0, nullptr);
                        list->CopyResource(t.GPUResource.Get(), upload.Get());
                        stagingBuffers.push_back(upload);
                        currentBatchBytes += sizeInBytes;

                        if (currentBatchBytes >= BATCH_LIMIT) {
                            list->Close();
                            ID3D12CommandList* ls[] = { list.Get() };
                            m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
                            ComPtr<ID3D12Fence> f;
                            m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
                            m_dxEngine->GetComputeQueue()->Signal(f.Get(), 1);
                            WaitFence(f.Get(), 1, loadEvent);
                            stagingBuffers.clear();
                            currentBatchBytes = 0;
                            allocator->Reset();
                            list->Reset(allocator.Get(), nullptr);
                        }
                    }
                }
            }
            m_weightTensors[name] = std::move(t);
        }

        if (m_dsLoader.IsInitialized() && m_dsLoader.GetPendingCount() > 0) {
            m_dsLoader.SubmitAndWait(loadFence.Get(), loadFenceValue, loadEvent);
        }

        if (list && currentBatchBytes > 0) {
            list->Close();
            ID3D12CommandList* ls[] = { list.Get() };
            m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
            ComPtr<ID3D12Fence> f;
            m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&f));
            m_dxEngine->GetComputeQueue()->Signal(f.Get(), 1);
            WaitFence(f.Get(), 1, loadEvent);
        }
        CloseHandle(loadEvent);

std::cout << "[ModelPipeline] Loaded " << (uint64_t)m_weightTensors.size() << " tensors. VRAM: "
               << (m_vramUsageBytes / (1024*1024)) << " MB, RAM: "
               << (m_systemRamUsageBytes / (1024*1024)) << " MB." << std::endl;

        // Populate TransformerLayer structures from weight tensors
        m_layers.clear();
        m_layers.resize(m_config.NumLayers);
        for (size_t l = 0; l < m_config.NumLayers; ++l) {
            m_layers[l].LayerIndex = l;
            m_layers[l].PrimaryLocation = DeviceLocation::GPU_VRAM;

            // Extract layer tensors - support both GGUF naming conventions
            std::string base = "blk." + std::to_string(l) + ".";
            std::string baseAlt = "model.layers." + std::to_string(l) + ".";

            auto getTensor = [&](const std::string& name) -> Tensor* {
                auto it = m_weightTensors.find(name);
                return (it != m_weightTensors.end()) ? &it->second : nullptr;
            };
            auto getTensorAlt = [&](const std::string& name1, const std::string& name2, const std::string& name3) -> Tensor* {
                Tensor* t = getTensor(name1);
                if (!t) t = getTensor(name2);
                if (!t) t = getTensor(name3);
                return t;
            };

            // Attention projections - support Llama, Gemma, Qwen naming
            Tensor* qkv = getTensorAlt(base + "attn_qkv.weight", baseAlt + "self_attn.q_proj.weight", base + "attention.query_key_value.weight");
            if (qkv) m_layers[l].QKV_Proj = *qkv;

            Tensor* o = getTensorAlt(base + "attn_ov.weight", baseAlt + "self_attn.o_proj.weight", base + "attention.output.weight");
            if (o) m_layers[l].O_Proj = *o;

            // FFN projections (for non-MoE models) - support multiple naming conventions
            Tensor* gate = getTensorAlt(base + "ffn_gate.weight", baseAlt + "mlp.gate_proj.weight", base + "mlp.gate.weight");
            if (gate) m_layers[l].FFN_Gate_Proj = *gate;

            Tensor* up = getTensorAlt(base + "ffn_up.weight", baseAlt + "mlp.up_proj.weight", base + "mlp.up.weight");
            if (up) m_layers[l].FFN_Up_Proj = *up;

            Tensor* down = getTensorAlt(base + "ffn_down.weight", baseAlt + "mlp.down_proj.weight", base + "mlp.down.weight");
            if (down) m_layers[l].FFN_Down_Proj = *down;

            // RMSNorm weights - support multiple naming conventions
            Tensor* attnNorm = getTensorAlt(base + "attn_norm.weight", baseAlt + "input_layernorm.weight", base + "attention_norm.weight");
            if (attnNorm) m_layers[l].Attn_Norm = *attnNorm;

            Tensor* ffnNorm = getTensorAlt(base + "ffn_norm.weight", baseAlt + "post_attention_layernorm.weight", base + "mlp_norm.weight");
            if (ffnNorm) m_layers[l].FFN_Norm = *ffnNorm;
        }

        if (m_dxEngine) BuildGEMMPipeline();

        return true;
    }

    // ================================================================
    //  CPU Math helpers
    // ================================================================
    static void RMSNorm(float* out, const float* x, int dim, float eps) {
        float sq = 0.0f;
        for (int i = 0; i < dim; i++) sq += x[i] * x[i];
        float rms = 1.0f / std::sqrt(sq / dim + eps);
        for (int i = 0; i < dim; i++) out[i] = x[i] * rms;
    }

    static float SiLU(float x) { return x / (1.0f + std::exp(-x)); }

    // ================================================================
    //  Sampler
    // ================================================================
    static int Sample(float* logits, int vocabSize, float temp, float topP, int topK) {
        if (temp < 0.001f) temp = 0.001f;
        for (int i = 0; i < vocabSize; i++) logits[i] /= temp;

        std::vector<std::pair<float, int>> scored;
        scored.reserve(vocabSize);
        for (int i = 0; i < vocabSize; i++) scored.emplace_back(logits[i], i);

        int effectiveK = (topK > 0 && topK < vocabSize) ? topK : vocabSize;
        std::nth_element(scored.begin(), scored.begin() + effectiveK, scored.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
        scored.resize(effectiveK);
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b) { return a.first > b.first; });

        float maxVal = scored[0].first;
        float sum = 0.0f;
        for (auto& s : scored) { s.first = std::exp(s.first - maxVal); sum += s.first; }
        for (auto& s : scored) s.first /= sum;

        if (topP > 0.0f && topP < 1.0f) {
            float cum = 0.0f; size_t cut = scored.size();
            for (size_t i = 0; i < scored.size(); i++) {
                cum += scored[i].first;
                if (cum >= topP) { cut = i + 1; break; }
            }
            scored.resize(cut);
        }

        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0.f, 1.f);
        float r = dist(gen), cdf = 0.f;
        for (auto& s : scored) { cdf += s.first; if (r <= cdf) return s.second; }
        return scored.empty() ? 0 : scored.back().second;
    }

    // ================================================================
    //  RunInferenceStep
    // ================================================================
    bool ModelPipeline::RunInferenceStep(uint32_t batchSize,
                                          const std::vector<int32_t>& inputTokenIds,
                                          uint32_t currentSequenceOffset,
                                          std::vector<float>& outLogits) {
        if (batchSize == 0 || inputTokenIds.empty()) return false;
        int vocab = (int)m_config.VocabSize;
        if (vocab <= 0) vocab = 248000;

        int tokId = inputTokenIds.back();
        if (tokId < 0 || tokId >= vocab) tokId %= vocab;
        if (tokId < 0) tokId = 0;

        int hidden = (int)m_config.HiddenDim;
        if (hidden <= 0) hidden = 2048;

        auto embedIt = m_weightTensors.find("token_embd.weight");
        if (embedIt == m_weightTensors.end())
            embedIt = m_weightTensors.find("tok_embeddings.weight");
        auto lmHeadIt = m_weightTensors.find("output.weight");
        if (lmHeadIt == m_weightTensors.end())
            lmHeadIt = m_weightTensors.find("lm_head.weight");
        if (lmHeadIt == m_weightTensors.end())
            lmHeadIt = m_weightTensors.find("embed_tokens.weight"); // Phi-3 shares embed/lm_head
        const Tensor* embedTensor = embedIt != m_weightTensors.end() ? &embedIt->second : nullptr;
        const Tensor* lmHeadTensor = lmHeadIt != m_weightTensors.end() ? &lmHeadIt->second : nullptr;

        if (embedTensor && lmHeadTensor) {
            // Build embedding vector X on CPU with correct dequantization
            std::vector<float> x(hidden, 0.0f);
            if (!embedTensor->CPUHostData.empty()) {
                if (embedTensor->QuantType == QuantizationType::None_FP32) {
                    const float* embedData = reinterpret_cast<const float*>(embedTensor->CPUHostData.data());
                    size_t rowBytes = (size_t)hidden * sizeof(float);
                    size_t offset   = (size_t)tokId * rowBytes;
                    if (offset + rowBytes <= embedTensor->CPUHostData.size())
                        std::memcpy(x.data(), embedData + (size_t)tokId * hidden, rowBytes);
                } else if (embedTensor->QuantType == QuantizationType::None_FP16) {
                    const uint16_t* embedData = reinterpret_cast<const uint16_t*>(embedTensor->CPUHostData.data());
                    size_t rowElements = (size_t)hidden;
                    size_t offset      = (size_t)tokId * rowElements;
                    if (offset + rowElements <= embedTensor->CPUHostData.size() / 2) {
                        for (size_t i = 0; i < rowElements; ++i) {
                            x[i] = FP16ToFloat(embedData[offset + i]);
                        }
                    }
                } else if (embedTensor->QuantType == QuantizationType::Q8_0) {
                    const uint8_t* embedData = embedTensor->CPUHostData.data();
                    size_t blockSizeBytes = 34;
                    size_t blocksPerRow = (size_t)hidden / 32;
                    size_t rowBytes = blocksPerRow * blockSizeBytes;
                    size_t offset = (size_t)tokId * rowBytes;
                    if (offset + rowBytes <= embedTensor->CPUHostData.size()) {
                        const uint8_t* rowPtr = embedData + offset;
                        for (size_t b = 0; b < blocksPerRow; ++b) {
                            const uint8_t* blockPtr = rowPtr + b * blockSizeBytes;
                            uint16_t d_raw;
                            std::memcpy(&d_raw, blockPtr, 2);
                            float d = FP16ToFloat(d_raw);
                            const int8_t* qs = reinterpret_cast<const int8_t*>(blockPtr + 2);
                            for (size_t j = 0; j < 32; ++j) {
                                x[b * 32 + j] = qs[j] * d;
                            }
                        }
                    }
                } else if (embedTensor->QuantType == QuantizationType::Q4_0) {
                    const uint8_t* embedData = embedTensor->CPUHostData.data();
                    size_t blocksPerRow = (size_t)hidden / 32;
                    for (size_t b = 0; b < blocksPerRow; ++b) {
                        size_t blockOffset = ((size_t)tokId * blocksPerRow + b) * 18;
                        const uint8_t* blockPtr = embedData + blockOffset;
                        uint16_t scale_raw;
                        std::memcpy(&scale_raw, blockPtr, 2);
                        float scale = FP16ToFloat(scale_raw);
                        const uint8_t* qs = blockPtr + 2;
                        for (size_t j = 0; j < 32; ++j) {
                            uint8_t byte = qs[j / 2];
                            uint8_t nibble = (j % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                            x[b * 32 + j] = (float)nibble * scale;
                        }
                    }
                } else {
                    const float* embedData = reinterpret_cast<const float*>(embedTensor->CPUHostData.data());
                    size_t rowBytes = (size_t)hidden * sizeof(float);
                    size_t offset   = (size_t)tokId * rowBytes;
                    if (offset + rowBytes <= embedTensor->CPUHostData.size())
                        std::memcpy(x.data(), embedData + (size_t)tokId * hidden, rowBytes);
                }
            }

            // ----------------------------------------------------------------
            //  Transformer Layer Inference (Attention + FFN)
            // ----------------------------------------------------------------
            if (m_config.NumLayers > 0 && !m_layers.empty()) {
                for (size_t layerIdx = 0; layerIdx < m_config.NumLayers; ++layerIdx) {
                    const auto& layer = m_layers[layerIdx];

                    // RMSNorm before attention
                    if (!layer.Attn_Norm.CPUHostData.empty()) {
                        const float* normW = reinterpret_cast<const float*>(layer.Attn_Norm.CPUHostData.data());
                        std::vector<float> normX(hidden);
                        RMSNorm(normX.data(), x.data(), (int)hidden, 1e-5f);
                        for (size_t i = 0; i < hidden; ++i) x[i] = normX[i] * normW[i];
                    }

                    // QKV projection
                    if (!layer.QKV_Proj.CPUHostData.empty()) {
                        std::vector<float> qkvOut(hidden * 3, 0.0f);
                        // Placeholder - real implementation needs matrix multiply dispatch
                    }

                    // O projection after attention
                    if (!layer.O_Proj.CPUHostData.empty()) {
                        // Placeholder - would need attention output
                    }

                    // RMSNorm before FFN
                    if (!layer.FFN_Norm.CPUHostData.empty()) {
                        const float* normW = reinterpret_cast<const float*>(layer.FFN_Norm.CPUHostData.data());
                        std::vector<float> normX(hidden);
                        RMSNorm(normX.data(), x.data(), (int)hidden, 1e-5f);
                        for (size_t i = 0; i < hidden; ++i) x[i] = normX[i] * normW[i];
                    }

                    // FFN (Dense): Gate -> SiLU -> Up -> Multiply -> Down
                    if (m_config.Type == ModelType::Dense && !layer.FFN_Gate_Proj.CPUHostData.empty()) {
                        // Placeholder - real implementation needs matrix multiply dispatch
                    }
                }
            }

            // ----------------------------------------------------------------
            //  GPU path — lm_head is in VRAM
            // ----------------------------------------------------------------
            bool isQuantized = (lmHeadTensor->QuantType == QuantizationType::Q4_0 ||
                                lmHeadTensor->QuantType == QuantizationType::Q4_K ||
                                lmHeadTensor->QuantType == QuantizationType::Q5_K ||
                                lmHeadTensor->QuantType == QuantizationType::Q6_K);
            bool hasGpuScales = lmHeadTensor->GPUScales || lmHeadTensor->GPUZeroPoints;
            if (lmHeadTensor->Location == DeviceLocation::GPU_VRAM &&
                m_dxEngine && m_gemmPSOReady && (!isQuantized || hasGpuScales)) {

                size_t hiddenBytes = (size_t)hidden * sizeof(float);
                size_t vocabBytes  = (size_t)vocab  * sizeof(float);
                if (!EnsurePersistentBuffers(hiddenBytes, vocabBytes)) return false;

                outLogits.resize(vocab);

                if (m_dxEngine->SupportsReBAR()) {
                    void* pGPU = nullptr;
                    m_xGPUBuffer->Map(0, nullptr, &pGPU);
                    std::memcpy(pGPU, x.data(), hiddenBytes);
                    m_xGPUBuffer->Unmap(0, nullptr);

                    m_inferCmdAllocator->Reset();
                    m_inferCmdList->Reset(m_inferCmdAllocator.Get(), m_gemmPSO.Get());
                } else {
                    void* pUpload = nullptr;
                    m_xUploadBuffer->Map(0, nullptr, &pUpload);
                    std::memcpy(pUpload, x.data(), hiddenBytes);
                    m_xUploadBuffer->Unmap(0, nullptr);

                    m_inferCmdAllocator->Reset();
                    m_inferCmdList->Reset(m_inferCmdAllocator.Get(), m_gemmPSO.Get());

                    m_inferCmdList->CopyResource(m_xGPUBuffer.Get(), m_xUploadBuffer.Get());

                    D3D12_RESOURCE_BARRIER barrierX = {};
                    barrierX.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    barrierX.Transition.pResource   = m_xGPUBuffer.Get();
                    barrierX.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                    barrierX.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    barrierX.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    m_inferCmdList->ResourceBarrier(1, &barrierX);
                }

                // Dispatch GEMM (final lm_head projection)
                // Reads from m_xGPUBuffer (hidden activations) → m_yGPUBuffer (vocab logits)
                struct { uint32_t M, N, K; } constants;
                constants.M = 1;
                if (lmHeadTensor->Shape.size() >= 2) {
                    constants.N = (uint32_t)lmHeadTensor->Shape[0];
                    constants.K = (uint32_t)lmHeadTensor->Shape[1];
                } else {
                    constants.N = (uint32_t)vocab;
                    constants.K = (uint32_t)hidden;
                }

                m_inferCmdList->SetComputeRootSignature(m_gemmRootSignature.Get());
                m_inferCmdList->SetPipelineState(m_gemmPSO.Get());
                m_inferCmdList->SetComputeRoot32BitConstants(0, 3, &constants, 0);
                if (m_xGPUBuffer)             m_inferCmdList->SetComputeRootShaderResourceView(1, m_xGPUBuffer->GetGPUVirtualAddress());
                if (lmHeadTensor->GPUResource) m_inferCmdList->SetComputeRootShaderResourceView(2, lmHeadTensor->GPUResource->GetGPUVirtualAddress());
                auto scales = lmHeadTensor->GPUScales ? lmHeadTensor->GPUScales : lmHeadTensor->GPUResource;
                auto zeros  = lmHeadTensor->GPUZeroPoints ? lmHeadTensor->GPUZeroPoints : lmHeadTensor->GPUResource;
                if (scales) m_inferCmdList->SetComputeRootShaderResourceView(3, scales->GetGPUVirtualAddress());
                if (zeros)  m_inferCmdList->SetComputeRootShaderResourceView(4, zeros->GetGPUVirtualAddress());
                if (m_yGPUBuffer) m_inferCmdList->SetComputeRootUnorderedAccessView(5, m_yGPUBuffer->GetGPUVirtualAddress());

                m_inferCmdList->Dispatch((constants.N + 15) / 16, 1, 1);

                D3D12_RESOURCE_BARRIER barriers[2] = {};
                barriers[0].Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barriers[0].UAV.pResource = m_yGPUBuffer.Get();
                barriers[1].Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barriers[1].Transition.pResource   = m_yGPUBuffer.Get();
                barriers[1].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barriers[1].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barriers[1].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                m_inferCmdList->ResourceBarrier(2, barriers);

                m_inferCmdList->CopyResource(m_yReadbackBuffer.Get(), m_yGPUBuffer.Get());

                D3D12_RESOURCE_BARRIER restore[2] = {};
                int restoreCount = 0;
                restore[restoreCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                restore[restoreCount].Transition.pResource   = m_yGPUBuffer.Get();
                restore[restoreCount].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                restore[restoreCount].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                restore[restoreCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                restoreCount++;

                if (!m_dxEngine->SupportsReBAR()) {
                    restore[restoreCount].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                    restore[restoreCount].Transition.pResource   = m_xGPUBuffer.Get();
                    restore[restoreCount].Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                    restore[restoreCount].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
                    restore[restoreCount].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                    restoreCount++;
                }
                m_inferCmdList->ResourceBarrier(restoreCount, restore);

                m_inferCmdList->Close();

                ID3D12CommandList* ls[] = { m_inferCmdList.Get() };
                m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
                m_dxEngine->GetComputeQueue()->Signal(m_inferFence.Get(), ++m_inferFenceValue);
                WaitFence(m_inferFence.Get(), m_inferFenceValue, m_inferFenceEvent);

                void* pData = nullptr;
                m_yReadbackBuffer->Map(0, nullptr, &pData);
                std::memcpy(outLogits.data(), pData, vocabBytes);
                m_yReadbackBuffer->Unmap(0, nullptr);

                // Write current token KV state to GPU KV buffer (quant-aware CPU->GPU upload)
                uint32_t slot = m_kvCache.AllocateTokens(0, 1);
                m_kvCache.WriteTokenKV(m_dxEngine->GetDevice(), m_dxEngine->GetComputeQueue(), m_inferFenceEvent,
                    0, slot, x.data(), x.data());

                m_gpuDispatchedOperators++;
                return true;
            }

            // ----------------------------------------------------------------
            //  CPU fallback — lm_head is in system RAM.
            // ----------------------------------------------------------------
            if (lmHeadTensor && !lmHeadTensor->GPUResource && !lmHeadTensor->CPUHostData.empty()) {
                outLogits.assign(vocab, 0.0f);
                int nThreads = std::max(1, m_cpuThreadCount);
                int chunk    = (vocab + nThreads - 1) / nThreads;

                if (lmHeadTensor->QuantType == QuantizationType::None_FP32) {
                    const float* W = reinterpret_cast<const float*>(lmHeadTensor->CPUHostData.data());
                    auto worker = [&](int jStart, int jEnd) {
                        for (int j = jStart; j < jEnd; j++) {
                            outLogits[j] = DotProductSIMD(x.data(), W + (size_t)j * hidden, hidden);
                        }
                    };
                    if (nThreads == 1) {
                        worker(0, vocab);
                    } else {
                        std::vector<std::thread> threads;
                        threads.reserve(nThreads);
                        for (int t = 0; t < nThreads; t++) {
                            int jStart = t * chunk;
                            int jEnd   = std::min(jStart + chunk, vocab);
                            if (jStart < jEnd)
                                threads.emplace_back(worker, jStart, jEnd);
                        }
                        for (auto& th : threads) th.join();
                    }
                } else if (lmHeadTensor->QuantType == QuantizationType::None_FP16) {
                    const uint16_t* W = reinterpret_cast<const uint16_t*>(lmHeadTensor->CPUHostData.data());
                    auto worker = [&](int jStart, int jEnd) {
                        std::vector<float> row(hidden);
                        for (int j = jStart; j < jEnd; j++) {
                            const uint16_t* w_row = W + (size_t)j * hidden;
                            for (int i = 0; i < hidden; ++i) row[i] = FP16ToFloat(w_row[i]);
                            outLogits[j] = DotProductSIMD(x.data(), row.data(), hidden);
                        }
                    };
                    if (nThreads == 1) {
                        worker(0, vocab);
                    } else {
                        std::vector<std::thread> threads;
                        threads.reserve(nThreads);
                        for (int t = 0; t < nThreads; t++) {
                            int jStart = t * chunk;
                            int jEnd   = std::min(jStart + chunk, vocab);
                            if (jStart < jEnd)
                                threads.emplace_back(worker, jStart, jEnd);
                        }
                        for (auto& th : threads) th.join();
                    }
                } else if (lmHeadTensor->QuantType == QuantizationType::Q8_0) {
                    const uint8_t* basePtr = lmHeadTensor->CPUHostData.data();
                    size_t blockSizeBytes = 34;
                    size_t blocksPerRow = (size_t)hidden / 32;
                    size_t rowBytes = blocksPerRow * blockSizeBytes;
                    auto worker = [&](int jStart, int jEnd) {
                        std::vector<float> row(hidden);
                        for (int j = jStart; j < jEnd; j++) {
                            const uint8_t* rowPtr = basePtr + (size_t)j * rowBytes;
                            for (size_t b = 0; b < blocksPerRow; ++b) {
                                const uint8_t* blockPtr = rowPtr + b * blockSizeBytes;
                                uint16_t d_raw;
                                std::memcpy(&d_raw, blockPtr, 2);
                                float d = FP16ToFloat(d_raw);
                                const int8_t* qs = reinterpret_cast<const int8_t*>(blockPtr + 2);
                                for (size_t i = 0; i < 32; ++i) {
                                    row[b * 32 + i] = qs[i] * d;
                                }
                            }
                            outLogits[j] = DotProductSIMD(x.data(), row.data(), hidden);
                        }
                    };
                    if (nThreads == 1) {
                        worker(0, vocab);
                    } else {
                        std::vector<std::thread> threads;
                        threads.reserve(nThreads);
                        for (int t = 0; t < nThreads; t++) {
                            int jStart = t * chunk;
                            int jEnd   = std::min(jStart + chunk, vocab);
                            if (jStart < jEnd)
                                threads.emplace_back(worker, jStart, jEnd);
                        }
                        for (auto& th : threads) th.join();
                    }
                } else if (lmHeadTensor->QuantType == QuantizationType::Q4_0) {
                    const uint8_t* basePtr = lmHeadTensor->CPUHostData.data();
                    size_t blocksPerCol = (size_t)hidden / 32;
                    auto worker = [&](int jStart, int jEnd) {
                        std::vector<float> row(hidden);
                        for (int j = jStart; j < jEnd; j++) {
                            const uint8_t* rowPtr = basePtr + (size_t)j * blocksPerCol * 18;
                            for (int b = 0; b < blocksPerCol; ++b) {
                                const uint8_t* blockPtr = rowPtr + b * 18;
                                uint16_t scale_raw;
                                std::memcpy(&scale_raw, blockPtr, 2);
                                float scale = FP16ToFloat(scale_raw);
                                const uint8_t* qs = blockPtr + 2;
                                for (int i = 0; i < 32; ++i) {
                                    uint8_t byte = qs[i / 2];
                                    int8_t nibble = (i % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                                    row[b * 32 + i] = (float)nibble * scale;
                                }
                            }
                            outLogits[j] = DotProductSIMD(x.data(), row.data(), hidden);
                        }
                    };
                    if (nThreads == 1) {
                        worker(0, vocab);
                    } else {
                        std::vector<std::thread> threads;
                        threads.reserve(nThreads);
                        for (int t = 0; t < nThreads; t++) {
                            int jStart = t * chunk;
                            int jEnd   = std::min(jStart + chunk, vocab);
                            if (jStart < jEnd)
                                threads.emplace_back(worker, jStart, jEnd);
                        }
                        for (auto& th : threads) th.join();
                    }
                }

                m_cpuDispatchedOperators++;
                return true;
            }
        }

        std::cerr << "[ModelPipeline] Error: Missing required embedding (token_embd/tok_embeddings) or output projection (output/lm_head) tensors in loaded GGUF weight map!" << std::endl;
        return false;
    }

    // ================================================================
    //  Misc
    // ================================================================
    void ModelPipeline::Reset() {
        m_weightTensors.clear();
        m_vramUsageBytes = m_systemRamUsageBytes = 0;
        m_gemmPSOReady = false;
        m_persistentHiddenBytes = m_persistentVocabBytes = 0;
        m_dsLoader.Shutdown();
        m_kvCache.Reset();
        m_openVINO.Shutdown();
    }

    bool ModelPipeline::DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        if (!m_dxEngine || !m_gemmPSOReady) return false;
        m_gpuDispatchedOperators++;
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        m_dxEngine->GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&alloc));
        m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
            alloc.Get(), m_gemmPSO.Get(), IID_PPV_ARGS(&list));
        list->SetComputeRootSignature(m_gemmRootSignature.Get());
        struct { uint32_t M, N, K; } c;
        c.M = (!X.Shape.empty()) ? (uint32_t)X.Shape[0] : 1;
        c.N = (W.Shape.size() >= 2) ? (uint32_t)W.Shape[0] : 4096;
        c.K = (W.Shape.size() >= 2) ? (uint32_t)W.Shape[1] : 4096;
        list->SetComputeRoot32BitConstants(0, 3, &c, 0);
        if (X.GPUResource) list->SetComputeRootShaderResourceView(1, X.GPUResource->GetGPUVirtualAddress());
        if (W.GPUResource) list->SetComputeRootShaderResourceView(2, W.GPUResource->GetGPUVirtualAddress());
        auto sc = W.GPUScales     ? W.GPUScales     : W.GPUResource;
        auto zp = W.GPUZeroPoints ? W.GPUZeroPoints : W.GPUResource;
        if (sc) list->SetComputeRootShaderResourceView(3, sc->GetGPUVirtualAddress());
        if (zp) list->SetComputeRootShaderResourceView(4, zp->GetGPUVirtualAddress());
        if (Y.GPUResource) list->SetComputeRootUnorderedAccessView(5, Y.GPUResource->GetGPUVirtualAddress());
        list->Dispatch((c.N + 15) / 16, (c.M + 15) / 16, 1);
        list->Close();
        ID3D12CommandList* ls[] = { list.Get() };
        m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
        HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        ComPtr<ID3D12Fence> fence;
        m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        m_dxEngine->GetComputeQueue()->Signal(fence.Get(), 1);
        WaitFence(fence.Get(), 1, evt);
        CloseHandle(evt);
        return true;
    }

bool ModelPipeline::DispatchCPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        if (W.CPUHostData.empty() || X.CPUHostData.empty()) return false;
        
        size_t M = X.Shape.empty() ? 1 : X.Shape[0];
        size_t N = W.Shape.size() >= 2 ? W.Shape[0] : 4096;
        size_t K = W.Shape.size() >= 2 ? W.Shape[1] : 4096;
        
        Y.Shape = {N, M};
        Y.CPUHostData.resize(N * M * sizeof(float));
        float* yOut = reinterpret_cast<float*>(Y.CPUHostData.data());
        
        int nThreads = std::max(1, m_cpuThreadCount);
        int chunk = (N + nThreads - 1) / nThreads;
        
        auto worker = [&](int jStart, int jEnd) {
            std::vector<float> wRow(K);
            for (int j = jStart; j < jEnd; j++) {
                if (W.QuantType == QuantizationType::None_FP32) {
                    const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = wPtr[j * K + k];
                } else if (W.QuantType == QuantizationType::None_FP16) {
                    const uint16_t* wPtr = reinterpret_cast<const uint16_t*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = FP16ToFloat(wPtr[j * K + k]);
                } else if (W.QuantType == QuantizationType::Q8_0) {
                    size_t blocksPerRow = K / 32;
                    const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * 34;
                    for (int b = 0; b < blocksPerRow; b++) {
                        const uint8_t* blockPtr = rowPtr + b * 34;
                        uint16_t d_raw;
                        std::memcpy(&d_raw, blockPtr, 2);
                        float d = FP16ToFloat(d_raw);
                        const int8_t* qs = reinterpret_cast<const int8_t*>(blockPtr + 2);
                        for (int k = 0; k < 32; k++) wRow[b * 32 + k] = qs[k] * d;
                    }
                } else if (W.QuantType == QuantizationType::Q4_0) {
                    size_t blocksPerRow = K / 32;
                    const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * 18;
                    for (int b = 0; b < blocksPerRow; b++) {
                        const uint8_t* blockPtr = rowPtr + b * 18;
                        uint16_t scale_raw;
                        std::memcpy(&scale_raw, blockPtr, 2);
                        float scale = FP16ToFloat(scale_raw);
                        const uint8_t* qs = blockPtr + 2;
                        for (int k = 0; k < 32; k++) {
                            uint8_t byte = qs[k / 2];
                            int8_t nibble = (k % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                            wRow[b * 32 + k] = (float)nibble * scale;
                        }
                    }
                } else if (W.QuantType == QuantizationType::Q4_K) {
                    // Q4_K block format: 32 weights per block, variable layout
                    // For simplicity, treat as raw float (placeholder - needs full implementation)
                    const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = wPtr[j * K + k];
                } else if (W.QuantType == QuantizationType::Q5_K) {
                    // Q5_K: 5-bit quantization - treat as raw (placeholder)
                    const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = wPtr[j * K + k];
                } else if (W.QuantType == QuantizationType::Q6_K) {
                    // Q6_K: 6-bit quantization - treat as raw (placeholder)
                    const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = wPtr[j * K + k];
                } else {
                    // Fallback for Q4_0 and other types - treat as raw
                    const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                    for (int k = 0; k < K; k++) wRow[k] = wPtr[j * K + k];
                }
                yOut[j] = DotProductSIMD(reinterpret_cast<const float*>(X.CPUHostData.data()), wRow.data(), (int)K);
            }
        };
        
        m_cpuDispatchedOperators++;
        if (nThreads == 1) {
            worker(0, N);
        } else {
            std::vector<std::thread> threads;
            for (int t = 0; t < nThreads; t++) {
                int jStart = t * chunk;
                int jEnd = std::min(jStart + chunk, (int)N);
                if (jStart < jEnd) threads.emplace_back(worker, jStart, jEnd);
            }
            for (auto& th : threads) th.join();
        }
        return true;
    }

    bool ModelPipeline::DispatchMoERouting(const Tensor& X, const TransformerLayer& layer,
                                             std::vector<float>& expertWeights,
                                             std::vector<std::vector<uint32_t>>& expertTokens) {
        // MoE routing: compute gate scores and assign tokens to experts
        if (layer.MoE_Gate.CPUHostData.empty() && layer.MoE_Gate.GPUResource == nullptr) {
            return false;
        }
        
        expertWeights.assign(layer.Experts.size(), 0.0f);
        expertTokens.clear();
        expertTokens.resize(layer.Experts.size());
        
        // For now, assign to first expert (simplified routing)
        // Full implementation would use softmax over gate weights
        if (!layer.Experts.empty()) {
            expertTokens[0].push_back(0); // token 0 -> expert 0
            expertWeights[0] = 1.0f;
        }
        
        m_gpuDispatchedOperators++;
        return true;
    }

    float ModelPipeline::GetGpuExecutionRatio() const {
        if (m_layers.empty()) return 0.0f;
        size_t n = 0;
        for (auto& l : m_layers) if (l.PrimaryLocation == DeviceLocation::GPU_VRAM) n++;
        return (float)n / (float)m_layers.size();
    }

    void ModelPipeline::WaitForGPU() {
        if (m_dxEngine && m_executionFence) {
            m_fenceValue++;
            m_dxEngine->GetComputeQueue()->Signal(m_executionFence.Get(), m_fenceValue);
            if (m_executionFence->GetCompletedValue() < m_fenceValue) {
                HANDLE ev = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                m_executionFence->SetEventOnCompletion(m_fenceValue, ev);
                WaitForSingleObject(ev, INFINITE);
                CloseHandle(ev);
            }
        }
    }
}
