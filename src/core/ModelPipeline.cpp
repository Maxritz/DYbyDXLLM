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
#include <atomic>
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
            kvConfig.NumLayers = (uint32_t)m_config.NumLayers;
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
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_inferCmdAllocator)))) return false;
        if (FAILED(m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
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

        // Generic architecture-prefixed metadata: every GGUF stores its keys as
        // "<general.architecture>.<key>", so read them that way instead of
        // maintaining a hardcoded per-arch ladder. Works for llama, qwen2/3,
        // phi3, gemma*, minicpm, glm, and anything future.
        std::string arch = loader.GetMetadataString("general.architecture", "llama");
        std::cout << "[ModelPipeline] Arch: " << arch << std::endl;

        auto metaU32 = [&](const std::string& key, uint32_t def) -> uint32_t {
            if (loader.HasMetadata(arch + "." + key))
                return loader.GetMetadataUint32(arch + "." + key);
            if (loader.HasMetadata("llama." + key)) // some converters emit llama.* regardless
                return loader.GetMetadataUint32("llama." + key);
            return def;
        };
        auto metaF32 = [&](const std::string& key, float def) -> float {
            if (loader.HasMetadata(arch + "." + key))
                return loader.GetMetadataFloat(arch + "." + key, def);
            if (loader.HasMetadata("llama." + key))
                return loader.GetMetadataFloat("llama." + key, def);
            return def;
        };

        m_config.NumLayers  = metaU32("block_count",      m_config.NumLayers  ? (uint32_t)m_config.NumLayers  : 32);
        m_config.HiddenDim  = metaU32("embedding_length", m_config.HiddenDim  ? (uint32_t)m_config.HiddenDim  : 4096);
        m_config.NumHeads   = metaU32("attention.head_count", m_config.NumHeads ? (uint32_t)m_config.NumHeads : 32);
        m_config.NumKVHeads = metaU32("attention.head_count_kv", (uint32_t)m_config.NumHeads);
        m_config.IntermediateDim = metaU32("feed_forward_length", (uint32_t)(m_config.HiddenDim * 4));
        m_config.RopeFreqBase = metaF32("rope.freq_base", 10000.0f);
        m_config.RmsNormEps   = metaF32("attention.layer_norm_rms_epsilon", 1e-5f);

        // Rope style: llama-family rotates adjacent pairs (NORM); most others
        // (qwen2/3, phi3, gemma, stablelm) use the NEOX half-split rotation.
        m_config.RopeNeox = !(arch.find("llama") != std::string::npos ||
                              arch.find("mistral") != std::string::npos ||
                              arch.find("minicpm") != std::string::npos);

        // head_dim: prefer explicit key (attention.key_length) over derived
        uint32_t keyLen = metaU32("attention.key_length", 0);
        // Some custom archs (laguna) store head_count as 0 or omit it: derive
        // from hidden/key_length so the forward pass still has valid dims.
        if (m_config.NumHeads == 0 && keyLen)
            m_config.NumHeads = m_config.HiddenDim / keyLen;
        if (m_config.NumHeads == 0) m_config.NumHeads = 32;
        if (m_config.NumKVHeads == 0) m_config.NumKVHeads = m_config.NumHeads;
        m_config.HeadDim = keyLen ? keyLen : (m_config.HiddenDim / m_config.NumHeads);

        // Auto-enable system RAM offload if model is likely larger than VRAM limit
        m_config.EnableSystemRamOffload = true;

        std::cout << "[ModelPipeline] layers=" << m_config.NumLayers
                  << " hidden=" << m_config.HiddenDim
                  << " heads=" << m_config.NumHeads << "/" << m_config.NumKVHeads
                  << " headDim=" << m_config.HeadDim
                  << " ffn=" << m_config.IntermediateDim
                  << " ropeBase=" << m_config.RopeFreqBase
                  << " rope=" << (m_config.RopeNeox ? "neox" : "norm")
                  << " eps=" << m_config.RmsNormEps << std::endl;

        // GGUF stores dims in ggml ne-order: Shape[0]=hidden (ne0), Shape[1]=vocab (ne1)
        const GgufTensor* tokEmb = loader.GetTensor("token_embd.weight");
        if (tokEmb && tokEmb->Shape.size() >= 2)
            m_config.VocabSize = (size_t)tokEmb->Shape[1];

        // Ensure KV cache is re-initialized with correct dimensions
        if (m_dxEngine) {
            KVCacheConfig kvConfig;
            kvConfig.MaxSequenceLength = 2048;
            kvConfig.BatchSize = 1;
            kvConfig.NumLayers = (uint32_t)m_config.NumLayers;
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
                D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator));
            m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
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
                case GgmlType::BF16:    t.QuantType = QuantizationType::BF16;    break;
                case GgmlType::IQ4_NL:  t.QuantType = QuantizationType::IQ4_NL;  break;
                case GgmlType::IQ4_XS:  t.QuantType = QuantizationType::IQ4_XS;  break;
                default:
                    std::cerr << "[ModelPipeline] WARNING: tensor " << name
                              << " has unsupported ggml type " << (uint32_t)tensor.Type
                              << " — inference using it will fail cleanly." << std::endl;
                    t.QuantType = QuantizationType::None_FP16;
                    break;
            }

            size_t sizeInBytes = tensor.DataSize;
            DeviceLocation loc = DeviceLocation::GPU_VRAM;

            // token_embd: CPU lookup table. output/lm_head + output_norm: kept in
            // CPU RAM because the CPU fallback has correct dequant for every quant,
            // while the GPU GEMM shader expects a custom INT4 format no GGUF quant
            // matches — dispatching it produced garbage logits AND a per-token
            // full-vocab GPU readback (the hang/overuse path).
            if (!m_dxEngine || name.find("token_embd") != std::string::npos
                || name == "output.weight" || name == "lm_head.weight"
                || name == "output_norm.weight")
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
                // Keep a CPU copy alongside the VRAM upload: the CPU forward
                // pass (currently the only correct path) reads CPUHostData.
                // Costs RAM equal to model size until GPU kernels are correct.
                if (!loader.IsMetadataOnly() && tensor.DataPtr) {
                    t.CPUHostData.resize(sizeInBytes);
                    std::memcpy(t.CPUHostData.data(), tensor.DataPtr, sizeInBytes);
                    m_systemRamUsageBytes += sizeInBytes;
                }
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

            // Attention projections. FIRST names are the standard GGUF ones
            // (blk.N.attn_q.weight etc.) — the previous list omitted them
            // entirely, so attention never ran for any normal GGUF model.
            Tensor* q = getTensorAlt(base + "attn_q.weight", baseAlt + "self_attn.q_proj.weight", base + "attention.wq.weight");
            if (q) m_layers[l].Q_Proj = std::move(*q);
            Tensor* k = getTensorAlt(base + "attn_k.weight", baseAlt + "self_attn.k_proj.weight", base + "attention.wk.weight");
            if (k) m_layers[l].K_Proj = std::move(*k);
            Tensor* v = getTensorAlt(base + "attn_v.weight", baseAlt + "self_attn.v_proj.weight", base + "attention.wv.weight");
            if (v) m_layers[l].V_Proj = std::move(*v);

            Tensor* qkv = getTensorAlt(base + "attn_qkv.weight", baseAlt + "self_attn.qkv_proj.weight", base + "attention.query_key_value.weight");
            if (qkv) m_layers[l].QKV_Proj = std::move(*qkv);

            Tensor* o = getTensorAlt(base + "attn_output.weight", baseAlt + "self_attn.o_proj.weight", base + "attention.wo.weight");
            if (o) m_layers[l].O_Proj = std::move(*o);

            // Optional QKV biases (Qwen2) and Q/K head norms (Qwen3)
            Tensor* qb = getTensor(base + "attn_q.bias");
            if (qb) m_layers[l].Q_Bias = std::move(*qb);
            Tensor* kb = getTensor(base + "attn_k.bias");
            if (kb) m_layers[l].K_Bias = std::move(*kb);
            Tensor* vb = getTensor(base + "attn_v.bias");
            if (vb) m_layers[l].V_Bias = std::move(*vb);
            Tensor* qn = getTensor(base + "attn_q_norm.weight");
            if (qn) m_layers[l].Attn_Q_Norm = std::move(*qn);
            Tensor* kn = getTensor(base + "attn_k_norm.weight");
            if (kn) m_layers[l].Attn_K_Norm = std::move(*kn);

            // FFN projections (for non-MoE models) - support multiple naming conventions
            Tensor* gate = getTensorAlt(base + "ffn_gate.weight", baseAlt + "mlp.gate_proj.weight", base + "mlp.gate.weight");
            if (gate) m_layers[l].FFN_Gate_Proj = std::move(*gate);

            Tensor* up = getTensorAlt(base + "ffn_up.weight", baseAlt + "mlp.up_proj.weight", base + "mlp.up.weight");
            if (up) m_layers[l].FFN_Up_Proj = std::move(*up);

            Tensor* down = getTensorAlt(base + "ffn_down.weight", baseAlt + "mlp.down_proj.weight", base + "mlp.down.weight");
            if (down) m_layers[l].FFN_Down_Proj = std::move(*down);

            // RMSNorm weights - support multiple naming conventions
            Tensor* attnNorm = getTensorAlt(base + "attn_norm.weight", baseAlt + "input_layernorm.weight", base + "attention_norm.weight");
            if (attnNorm) m_layers[l].Attn_Norm = std::move(*attnNorm);

            Tensor* ffnNorm = getTensorAlt(base + "ffn_norm.weight", baseAlt + "post_attention_layernorm.weight", base + "mlp_norm.weight");
            if (ffnNorm) m_layers[l].FFN_Norm = std::move(*ffnNorm);
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
    //  Q4_K/Q5_K scale unpacking helper (K_SCALE_SIZE = 12)
    //  Reference layout (ggml-quants.c get_scale_min_k4): 8 six-bit
    //  (scale, min) pairs packed into 12 bytes. The previous version here
    //  used a made-up layout, so every Q4_K/Q5_K weight dequantized wrong.
    // ================================================================
    static void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
        if (j < 4) {
            d = q[j] & 63;
            m = q[j + 4] & 63;
        } else {
            d = (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4);
            m = (q[j + 4] >> 4)   | ((q[j] >> 6) << 4);
        }
    }

    // ================================================================
    //  Q4_K dequantization (QK_K = 256 weights per block)
    //  Block format: dm[4B] + scales[12B] + qs[128B]
    // ================================================================
    static void DequantizeQ4_K(const uint8_t* blockPtr, float* outVec) {
        uint16_t dm_raw[2];
        std::memcpy(dm_raw, blockPtr, 4);
        float dall = FP16ToFloat(dm_raw[0]);
        float dmin = FP16ToFloat(dm_raw[1]);
        const uint8_t* scales = blockPtr + 4;
        const uint8_t* qs = blockPtr + 4 + 12;

        for (int i = 0; i < 256; ++i) {
            int il = i / 64;
            int in = i % 64;
            int is = 2 * il + (in >= 32 ? 1 : 0);
            int off = in & 31;
            int qsi = 32 * il + off;

            uint8_t sc, mn;
            get_scale_min_k4(is, scales, sc, mn);

            uint8_t q = qs[qsi];
            uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0x0F);
            outVec[i] = dall * sc * qv - dmin * mn;
        }
    }

    // ================================================================
    //  Q5_K dequantization (QK_K = 256 weights per block)
    //  Block format: dm[4B] + scales[12B] + qh[32B] + qs[128B]
    // ================================================================
    static void DequantizeQ5_K(const uint8_t* blockPtr, float* outVec) {
        uint16_t dm_raw[2];
        std::memcpy(dm_raw, blockPtr, 4);
        float dall = FP16ToFloat(dm_raw[0]);
        float dmin = FP16ToFloat(dm_raw[1]);
        const uint8_t* scales = blockPtr + 4;
        const uint8_t* qh = blockPtr + 4 + 12;
        const uint8_t* qs = blockPtr + 4 + 12 + 32;

        for (int i = 0; i < 256; ++i) {
            int il = i / 64;
            int in = i % 64;
            int is = 2 * il + (in >= 32 ? 1 : 0);
            int ir = (in & 31) / 2;
            int iq = in & 1;

            uint8_t sc, mn;
            get_scale_min_k4(is, scales, sc, mn);

            uint8_t q = qs[32 * il + 2 * ir + iq];
            uint8_t h = qh[2 * ir + iq];
            uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0x0F);
            uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

            outVec[i] = dall * sc * ((qv + ((h & hm) ? 16 : 0))) - dmin * mn;
        }
    }

    // ================================================================
    //  Q6_K dequantization (QK_K = 256 weights per block)
    //  Block format: ql[128B] + qh[64B] + scales[16B] + d[2B]
    //  Scale index follows the reference: sc[8*half + l/16 + 2*group]
    //  (the old code dropped the +2*group term -> wrong scales for 3/4 of
    //  every block).
    // ================================================================
    static void DequantizeQ6_K(const uint8_t* blockPtr, float* outVec) {
        const uint8_t* ql = blockPtr;
        const uint8_t* qh = blockPtr + 128;
        const int8_t* sc = reinterpret_cast<const int8_t*>(blockPtr + 128 + 64);
        uint16_t d_raw;
        std::memcpy(&d_raw, blockPtr + 128 + 64 + 16, 2);
        float d = FP16ToFloat(d_raw);

        float* y = outVec;
        for (int n = 0; n < 2; ++n) {           // two 128-element halves
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l]      & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l]      >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }

    // ================================================================
    //  Q2_K dequantization (256/block): scales[16] + qs[64] + d + dmin
    // ================================================================
    static void DequantizeQ2_K(const uint8_t* blockPtr, float* outVec) {
        const uint8_t* scales = blockPtr;
        const uint8_t* qs = blockPtr + 16;
        uint16_t dr[2];
        std::memcpy(dr, blockPtr + 16 + 64, 4);
        float d    = FP16ToFloat(dr[0]);
        float dmin = FP16ToFloat(dr[1]);

        float* y = outVec;
        int is = 0;
        const uint8_t* q = qs;
        for (int n = 0; n < 2; ++n) {           // two 128-element halves
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * (sc & 0xF), ml = dmin * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * (sc & 0xF); ml = dmin * (sc >> 4);
                for (int l = 16; l < 32; ++l) *y++ = dl * ((q[l] >> shift) & 3) - ml;
                shift += 2;
            }
            q += 32;
        }
    }

    // ================================================================
    //  Q3_K dequantization (256/block): hmask[32] + qs[64] + scales[12] + d
    // ================================================================
    static void DequantizeQ3_K(const uint8_t* blockPtr, float* outVec) {
        const uint8_t* hm = blockPtr;
        const uint8_t* qs = blockPtr + 32;
        uint16_t d_raw;
        std::memcpy(&d_raw, blockPtr + 32 + 64 + 12, 2);
        float d_all = FP16ToFloat(d_raw);

        // Unpack 16 six-bit scales from 12 bytes (reference kmask logic)
        const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
        uint32_t aux[4];
        std::memcpy(aux, blockPtr + 32 + 64, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t* scales = reinterpret_cast<const int8_t*>(aux);

        float* y = outVec;
        int is = 0;
        uint8_t m = 1;
        const uint8_t* q = qs;
        for (int n = 0; n < 2; ++n) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                dl = d_all * (scales[is++] - 32);
                for (int l = 16; l < 32; ++l)
                    *y++ = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
                shift += 2;
                m <<= 1;
            }
            q += 32;
        }
    }

    // IQ4 non-linear codebook (shared by IQ4_NL and IQ4_XS)
    static const int8_t kIQ4NLValues[16] = {
        -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
    };

    // IQ4_NL (32/block): d[2] + qs[16]
    static void DequantizeIQ4_NL(const uint8_t* blockPtr, float* outVec) {
        uint16_t d_raw;
        std::memcpy(&d_raw, blockPtr, 2);
        float d = FP16ToFloat(d_raw);
        const uint8_t* qs = blockPtr + 2;
        for (int j = 0; j < 16; ++j) {
            outVec[j]      = d * kIQ4NLValues[qs[j] & 0xF];
            outVec[j + 16] = d * kIQ4NLValues[qs[j] >> 4];
        }
    }

    // IQ4_XS (256/block): d[2] + scales_h[2] + scales_l[4] + qs[128]
    static void DequantizeIQ4_XS(const uint8_t* blockPtr, float* outVec) {
        uint16_t d_raw, scales_h;
        std::memcpy(&d_raw, blockPtr, 2);
        std::memcpy(&scales_h, blockPtr + 2, 2);
        const uint8_t* scales_l = blockPtr + 4;
        const uint8_t* qs = blockPtr + 8;
        float d = FP16ToFloat(d_raw);

        float* y = outVec;
        for (int ib = 0; ib < 8; ++ib) {        // eight 32-element sub-blocks
            int ls = ((scales_l[ib / 2] >> (4 * (ib & 1))) & 0xF)
                   | (((scales_h >> (2 * ib)) & 3) << 4);
            float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j]      = dl * kIQ4NLValues[qs[j] & 0xF];
                y[j + 16] = dl * kIQ4NLValues[qs[j] >> 4];
            }
            y  += 32;
            qs += 16;
        }
    }

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
    //  Unified per-row dequantization — one switch, every supported quant.
    //  Row layout is ggml row-major: row `rowIdx` of a (N x K) weight holds
    //  K contiguous elements starting at rowIdx * rowStrideBytes.
    // ================================================================
    static size_t RowStrideBytes(QuantizationType qt, size_t rowLen) {
        switch (qt) {
            case QuantizationType::None_FP32: return rowLen * 4;
            case QuantizationType::None_FP16: return rowLen * 2;
            case QuantizationType::BF16:      return rowLen * 2;
            case QuantizationType::Q8_0:      return (rowLen / 32) * 34;
            case QuantizationType::Q4_0:      return (rowLen / 32) * 18;
            case QuantizationType::Q4_1:      return (rowLen / 32) * 20;
            case QuantizationType::Q5_0:      return (rowLen / 32) * 22;
            case QuantizationType::Q5_1:      return (rowLen / 32) * 24;
            case QuantizationType::Q2_K:      return (rowLen / 256) * 84;
            case QuantizationType::Q3_K:      return (rowLen / 256) * 110;
            case QuantizationType::Q4_K:      return (rowLen / 256) * 144;
            case QuantizationType::Q5_K:      return (rowLen / 256) * 176;
            case QuantizationType::Q6_K:      return (rowLen / 256) * 210;
            case QuantizationType::Q8_K:      return (rowLen / 256) * 292;
            case QuantizationType::IQ4_NL:    return (rowLen / 32) * 18;
            case QuantizationType::IQ4_XS:    return (rowLen / 256) * 136;
            default:                          return 0;
        }
    }

    bool ModelPipeline::DequantRow(const Tensor& W, size_t rowIdx, size_t rowLen, float* out) const {
        if (W.CPUHostData.empty() || rowLen == 0) return false;
        size_t stride = RowStrideBytes(W.QuantType, rowLen);
        if (stride == 0) return false; // unsupported quant type
        size_t offset = rowIdx * stride;
        if (offset + stride > W.CPUHostData.size()) return false;
        const uint8_t* row = W.CPUHostData.data() + offset;

        switch (W.QuantType) {
            case QuantizationType::None_FP32:
                std::memcpy(out, row, rowLen * 4);
                return true;
            case QuantizationType::None_FP16: {
                const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
                for (size_t i = 0; i < rowLen; ++i) out[i] = FP16ToFloat(h[i]);
                return true;
            }
            case QuantizationType::BF16: {
                const uint16_t* h = reinterpret_cast<const uint16_t*>(row);
                for (size_t i = 0; i < rowLen; ++i) {
                    union { uint32_t u; float f; } w;
                    w.u = (uint32_t)h[i] << 16;
                    out[i] = w.f;
                }
                return true;
            }
            case QuantizationType::Q8_0: {
                for (size_t b = 0; b < rowLen / 32; ++b) {
                    const uint8_t* p = row + b * 34;
                    uint16_t dr; std::memcpy(&dr, p, 2);
                    float d = FP16ToFloat(dr);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(p + 2);
                    for (int j = 0; j < 32; ++j) out[b * 32 + j] = d * qs[j];
                }
                return true;
            }
            case QuantizationType::Q4_0: {
                // Reference layout: elems 0..15 = low nibbles, 16..31 = high
                // nibbles (the old code interleaved them -> permuted weights)
                for (size_t b = 0; b < rowLen / 32; ++b) {
                    const uint8_t* p = row + b * 18;
                    uint16_t dr; std::memcpy(&dr, p, 2);
                    float d = FP16ToFloat(dr);
                    const uint8_t* qs = p + 2;
                    float* y = out + b * 32;
                    for (int j = 0; j < 16; ++j) {
                        y[j]      = d * (int)((qs[j] & 0x0F) - 8);
                        y[j + 16] = d * (int)((qs[j] >>   4) - 8);
                    }
                }
                return true;
            }
            case QuantizationType::Q4_1: {
                for (size_t b = 0; b < rowLen / 32; ++b) {
                    const uint8_t* p = row + b * 20;
                    uint16_t dr[2]; std::memcpy(dr, p, 4);
                    float d = FP16ToFloat(dr[0]), m = FP16ToFloat(dr[1]);
                    const uint8_t* qs = p + 4;
                    float* y = out + b * 32;
                    for (int j = 0; j < 16; ++j) {
                        y[j]      = d * (qs[j] & 0x0F) + m;
                        y[j + 16] = d * (qs[j] >>   4) + m;
                    }
                }
                return true;
            }
            case QuantizationType::Q5_0: {
                for (size_t b = 0; b < rowLen / 32; ++b) {
                    const uint8_t* p = row + b * 22;
                    uint16_t dr; std::memcpy(&dr, p, 2);
                    float d = FP16ToFloat(dr);
                    uint32_t qh; std::memcpy(&qh, p + 2, 4);
                    const uint8_t* qs = p + 6;
                    float* y = out + b * 32;
                    for (int j = 0; j < 16; ++j) {
                        int x0 = (qs[j] & 0x0F) | (((qh >> j)        & 1) << 4);
                        int x1 = (qs[j] >>   4) | (((qh >> (j + 16)) & 1) << 4);
                        y[j]      = d * (x0 - 16);
                        y[j + 16] = d * (x1 - 16);
                    }
                }
                return true;
            }
            case QuantizationType::Q5_1: {
                for (size_t b = 0; b < rowLen / 32; ++b) {
                    const uint8_t* p = row + b * 24;
                    uint16_t dr[2]; std::memcpy(dr, p, 4);
                    float d = FP16ToFloat(dr[0]), m = FP16ToFloat(dr[1]);
                    uint32_t qh; std::memcpy(&qh, p + 4, 4);
                    const uint8_t* qs = p + 8;
                    float* y = out + b * 32;
                    for (int j = 0; j < 16; ++j) {
                        int x0 = (qs[j] & 0x0F) | (((qh >> j)        & 1) << 4);
                        int x1 = (qs[j] >>   4) | (((qh >> (j + 16)) & 1) << 4);
                        y[j]      = d * x0 + m;
                        y[j + 16] = d * x1 + m;
                    }
                }
                return true;
            }
            case QuantizationType::Q2_K:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeQ2_K(row + b * 84, out + b * 256);
                return true;
            case QuantizationType::Q3_K:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeQ3_K(row + b * 110, out + b * 256);
                return true;
            case QuantizationType::Q4_K:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeQ4_K(row + b * 144, out + b * 256);
                return true;
            case QuantizationType::Q5_K:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeQ5_K(row + b * 176, out + b * 256);
                return true;
            case QuantizationType::Q6_K:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeQ6_K(row + b * 210, out + b * 256);
                return true;
            case QuantizationType::Q8_K: {
                for (size_t b = 0; b < rowLen / 256; ++b) {
                    const uint8_t* p = row + b * 292;
                    float d; std::memcpy(&d, p, 4);
                    const int8_t* qs = reinterpret_cast<const int8_t*>(p + 4);
                    for (int j = 0; j < 256; ++j) out[b * 256 + j] = d * qs[j];
                }
                return true;
            }
            case QuantizationType::IQ4_NL:
                for (size_t b = 0; b < rowLen / 32; ++b)
                    DequantizeIQ4_NL(row + b * 18, out + b * 32);
                return true;
            case QuantizationType::IQ4_XS:
                for (size_t b = 0; b < rowLen / 256; ++b)
                    DequantizeIQ4_XS(row + b * 136, out + b * 256);
                return true;
            default:
                return false;
        }
    }

    // ================================================================
    //  MatVec: y[0..N) = W(N x K) @ x[0..K)   (multithreaded, any quant)
    // ================================================================
    bool ModelPipeline::MatVec(const Tensor& W, const float* x, float* y, size_t N, size_t K) const {
        if (W.CPUHostData.empty() || N == 0 || K == 0) return false;

        // Thread count scales with work: ~500K MACs per thread minimum, so
        // small projections don't pay 32x thread-spawn overhead.
        size_t totalOps = N * K;
        int nThreads = (int)std::min<size_t>(m_cpuThreadCount,
                        std::max<size_t>(1, totalOps / 500000));
        int chunk = (int)((N + nThreads - 1) / nThreads);

        std::atomic<bool> failed{false};
        auto worker = [&](int jStart, int jEnd) {
            std::vector<float> wRow(K);
            for (int j = jStart; j < jEnd; ++j) {
                if (!DequantRow(W, (size_t)j, K, wRow.data())) {
                    if (!failed.exchange(true))
                        std::cerr << "[ModelPipeline] MatVec failed on tensor '" << W.Name
                                  << "' quant=" << (int)W.QuantType
                                  << " N=" << N << " K=" << K
                                  << " bytes=" << W.CPUHostData.size() << std::endl;
                    return;
                }
                y[j] = DotProductSIMD(x, wRow.data(), (int)K);
            }
        };

        if (nThreads == 1) {
            worker(0, (int)N);
        } else {
            std::vector<std::thread> threads;
            threads.reserve(nThreads);
            for (int t = 0; t < nThreads; ++t) {
                int jStart = t * chunk;
                int jEnd = std::min<int>(jStart + chunk, (int)N);
                if (jStart < jEnd) threads.emplace_back(worker, jStart, jEnd);
            }
            for (auto& th : threads) th.join();
        }
        return !failed;
    }

    // ================================================================
    //  RunInferenceStep — CPU-correct llama-family decode step.
    //  Processes inputTokenIds.back() at position m_cpuSeqLen, appends to the
    //  CPU KV cache, and returns full-vocab logits. Residual connections,
    //  RoPE (NORM or NEOX), GQA, optional QKV biases (Qwen2), optional Q/K
    //  head norms (Qwen3), fused QKV (Phi-3), fused gate+up FFN (Phi-3),
    //  and the final output_norm are all applied.
    // ================================================================
    bool ModelPipeline::RunInferenceStep(uint32_t batchSize,
                                          const std::vector<int32_t>& inputTokenIds,
                                          uint32_t currentSequenceOffset,
                                          std::vector<float>& outLogits) {
        (void)currentSequenceOffset;
        if (batchSize == 0 || inputTokenIds.empty()) return false;

        const size_t hidden = m_config.HiddenDim;
        const size_t nHeads = m_config.NumHeads;
        const size_t nKV    = m_config.NumKVHeads ? m_config.NumKVHeads : nHeads;
        const size_t headDim = m_config.HeadDim;
        const size_t qDim  = nHeads * headDim;
        const size_t kvDim = nKV * headDim;
        const float  eps   = m_config.RmsNormEps;
        size_t vocab = m_config.VocabSize;
        if (hidden == 0 || nHeads == 0 || headDim == 0) return false;

        auto findTensor = [&](std::initializer_list<const char*> names) -> const Tensor* {
            for (const char* n : names) {
                auto it = m_weightTensors.find(n);
                if (it != m_weightTensors.end()) return &it->second;
            }
            return nullptr;
        };

        const Tensor* embedTensor = findTensor({"token_embd.weight", "tok_embeddings.weight",
                                                "model.embed_tokens.weight"});
        const Tensor* lmHeadTensor = findTensor({"output.weight", "lm_head.weight",
                                                 "embed_tokens.weight"});
        if (!lmHeadTensor) lmHeadTensor = embedTensor; // tied embeddings
        const Tensor* outNormTensor = findTensor({"output_norm.weight", "model.norm.weight"});

        if (!embedTensor || embedTensor->CPUHostData.empty() ||
            !lmHeadTensor || lmHeadTensor->CPUHostData.empty()) {
            std::cerr << "[ModelPipeline] Missing token_embd/output weights (or no CPU copy)." << std::endl;
            return false;
        }
        if (vocab == 0 && embedTensor->Shape.size() >= 2)
            vocab = (size_t)embedTensor->Shape[1];
        if (vocab == 0) return false;

        int tokId = inputTokenIds.back();
        if (tokId < 0 || tokId >= (int)vocab) { tokId %= (int)vocab; if (tokId < 0) tokId = 0; }

        const uint32_t pos = m_cpuSeqLen;
        const uint32_t maxSeq = 2048;
        if (pos >= maxSeq) {
            std::cerr << "[ModelPipeline] Context window full (" << maxSeq << ")." << std::endl;
            return false;
        }

        // ---- Embedding lookup (any quant) ----
        std::vector<float> x(hidden, 0.0f);
        if (!DequantRow(*embedTensor, (size_t)tokId, hidden, x.data())) {
            std::cerr << "[ModelPipeline] Embedding dequant failed (unsupported quant?)." << std::endl;
            return false;
        }

        // ---- Helpers ----
        auto rmsNormApply = [&](const float* in, const Tensor* normT, float* out, size_t n) {
            float sq = 0.0f;
            for (size_t i = 0; i < n; ++i) sq += in[i] * in[i];
            float inv = 1.0f / std::sqrt(sq / (float)n + eps);
            if (normT && !normT->CPUHostData.empty()) {
                std::vector<float> w(n);
                if (DequantRow(*normT, 0, n, w.data())) {
                    for (size_t i = 0; i < n; ++i) out[i] = in[i] * inv * w[i];
                    return;
                }
            }
            for (size_t i = 0; i < n; ++i) out[i] = in[i] * inv;
        };

        auto addBias = [&](const Tensor* biasT, float* v, size_t n) {
            if (!biasT || biasT->CPUHostData.empty()) return;
            std::vector<float> b(n);
            if (DequantRow(*biasT, 0, n, b.data()))
                for (size_t i = 0; i < n; ++i) v[i] += b[i];
        };

        // RoPE base frequency (per-layer rope/norm helpers live in the layer
        // loop below so they use each layer's actual head dimensions)
        const float freqBase = m_config.RopeFreqBase > 0 ? m_config.RopeFreqBase : 10000.0f;

        // ---- Ensure CPU KV cache sized ----
        if (m_cpuKCache.size() != m_config.NumLayers) {
            m_cpuKCache.assign(m_config.NumLayers, {});
            m_cpuVCache.assign(m_config.NumLayers, {});
        }

        std::vector<float> attnIn(hidden), q, k, v;
        std::vector<float> attnOut, proj(hidden), ffnIn(hidden);

        // ---- Transformer layers ----
        for (size_t l = 0; l < m_layers.size(); ++l) {
            const auto& L = m_layers[l];
            const bool hasSeparate = !L.Q_Proj.CPUHostData.empty() &&
                                     !L.K_Proj.CPUHostData.empty() &&
                                     !L.V_Proj.CPUHostData.empty();
            const bool hasFused = !L.QKV_Proj.CPUHostData.empty();
            const bool hasAttn = (hasSeparate || hasFused) && !L.O_Proj.CPUHostData.empty();

            // ---------- Attention ----------
            if (hasAttn) {
                // Trust the tensor shapes over the config: some archs report
                // head counts that don't match the actual projection sizes.
                size_t qDimL  = qDim, kvDimL = kvDim;
                if (hasSeparate) {
                    if (L.Q_Proj.Shape.size() >= 2) qDimL  = (size_t)L.Q_Proj.Shape[1];
                    if (L.K_Proj.Shape.size() >= 2) kvDimL = (size_t)L.K_Proj.Shape[1];
                }
                size_t headDimL = headDim;
                if (qDimL % headDimL != 0) headDimL = qDimL / nHeads;
                if (headDimL == 0 || qDimL % headDimL != 0 || kvDimL % headDimL != 0) {
                    std::cerr << "[ModelPipeline] Layer " << l << ": incompatible attention dims"
                              << " q=" << qDimL << " kv=" << kvDimL << " headDim=" << headDimL
                              << " — skipping layer." << std::endl;
                    continue;
                }
                const size_t nHeadsL = qDimL / headDimL;
                const size_t nKVL    = kvDimL / headDimL;
                const float invSqrtD = 1.0f / std::sqrt((float)headDimL);

                q.resize(qDimL); k.resize(kvDimL); v.resize(kvDimL);
                attnOut.resize(qDimL);

                rmsNormApply(x.data(), L.Attn_Norm.CPUHostData.empty() ? nullptr : &L.Attn_Norm,
                             attnIn.data(), hidden);

                if (hasFused) {
                    std::vector<float> qkv(qDimL + 2 * kvDimL);
                    if (!MatVec(L.QKV_Proj, attnIn.data(), qkv.data(), qDimL + 2 * kvDimL, hidden)) return false;
                    std::memcpy(q.data(), qkv.data(),                   qDimL  * sizeof(float));
                    std::memcpy(k.data(), qkv.data() + qDimL,           kvDimL * sizeof(float));
                    std::memcpy(v.data(), qkv.data() + qDimL + kvDimL,  kvDimL * sizeof(float));
                } else {
                    if (!MatVec(L.Q_Proj, attnIn.data(), q.data(), qDimL,  hidden)) return false;
                    if (!MatVec(L.K_Proj, attnIn.data(), k.data(), kvDimL, hidden)) return false;
                    if (!MatVec(L.V_Proj, attnIn.data(), v.data(), kvDimL, hidden)) return false;
                    addBias(L.Q_Bias.CPUHostData.empty() ? nullptr : &L.Q_Bias, q.data(), qDimL);
                    addBias(L.K_Bias.CPUHostData.empty() ? nullptr : &L.K_Bias, k.data(), kvDimL);
                    addBias(L.V_Bias.CPUHostData.empty() ? nullptr : &L.V_Bias, v.data(), kvDimL);
                }

                // Per-head RMS norms need the layer's actual headDim
                {
                    auto headNormL = [&](float* vec, size_t numHeads_, const Tensor* normT) {
                        if (!normT || normT->CPUHostData.empty()) return;
                        std::vector<float> w(headDimL);
                        if (!DequantRow(*normT, 0, headDimL, w.data())) return;
                        for (size_t h = 0; h < numHeads_; ++h) {
                            float* vv = vec + h * headDimL;
                            float sq = 0.0f;
                            for (size_t i = 0; i < headDimL; ++i) sq += vv[i] * vv[i];
                            float inv = 1.0f / std::sqrt(sq / (float)headDimL + eps);
                            for (size_t i = 0; i < headDimL; ++i) vv[i] = vv[i] * inv * w[i];
                        }
                    };
                    headNormL(q.data(), nHeadsL, L.Attn_Q_Norm.CPUHostData.empty() ? nullptr : &L.Attn_Q_Norm);
                    headNormL(k.data(), nKVL,    L.Attn_K_Norm.CPUHostData.empty() ? nullptr : &L.Attn_K_Norm);
                }

                // RoPE with the layer's actual headDim
                {
                    size_t half = headDimL / 2;
                    auto ropeL = [&](float* vec, size_t numHeads_) {
                        for (size_t h = 0; h < numHeads_; ++h) {
                            float* vv = vec + h * headDimL;
                            for (size_t i = 0; i < half; ++i) {
                                float theta = (float)pos * std::pow(freqBase, -2.0f * (float)i / (float)headDimL);
                                float c = std::cos(theta), s = std::sin(theta);
                                if (m_config.RopeNeox) {
                                    float a = vv[i], b = vv[i + half];
                                    vv[i]        = a * c - b * s;
                                    vv[i + half] = a * s + b * c;
                                } else {
                                    float a = vv[2 * i], b = vv[2 * i + 1];
                                    vv[2 * i]     = a * c - b * s;
                                    vv[2 * i + 1] = a * s + b * c;
                                }
                            }
                        }
                    };
                    ropeL(q.data(), nHeadsL);
                    ropeL(k.data(), nKVL);
                }

                // Append this position's K/V to the CPU cache
                auto& KC = m_cpuKCache[l];
                auto& VC = m_cpuVCache[l];
                KC.resize((size_t)(pos + 1) * kvDimL);
                VC.resize((size_t)(pos + 1) * kvDimL);
                std::memcpy(KC.data() + (size_t)pos * kvDimL, k.data(), kvDimL * sizeof(float));
                std::memcpy(VC.data() + (size_t)pos * kvDimL, v.data(), kvDimL * sizeof(float));

                // Causal attention over cached positions [0, pos]
                const size_t groupSize = nHeadsL / (nKVL ? nKVL : 1);
                std::vector<float> scores(pos + 1);
                for (size_t h = 0; h < nHeadsL; ++h) {
                    const size_t kvh = groupSize ? h / groupSize : 0;
                    const float* qh = q.data() + h * headDimL;
                    float maxScore = -1e30f;
                    for (uint32_t t = 0; t <= pos; ++t) {
                        const float* kt = KC.data() + (size_t)t * kvDimL + kvh * headDimL;
                        float s = 0.0f;
                        for (size_t d = 0; d < headDimL; ++d) s += qh[d] * kt[d];
                        s *= invSqrtD;
                        scores[t] = s;
                        if (s > maxScore) maxScore = s;
                    }
                    float z = 0.0f;
                    for (uint32_t t = 0; t <= pos; ++t) {
                        scores[t] = std::exp(scores[t] - maxScore);
                        z += scores[t];
                    }
                    float invZ = z > 0.0f ? 1.0f / z : 0.0f;
                    float* oh = attnOut.data() + h * headDimL;
                    std::memset(oh, 0, headDimL * sizeof(float));
                    for (uint32_t t = 0; t <= pos; ++t) {
                        const float p = scores[t] * invZ;
                        const float* vt = VC.data() + (size_t)t * kvDimL + kvh * headDimL;
                        for (size_t d = 0; d < headDimL; ++d) oh[d] += p * vt[d];
                    }
                }

                // Output projection + residual
                if (!MatVec(L.O_Proj, attnOut.data(), proj.data(), hidden, qDimL)) return false;
                for (size_t i = 0; i < hidden; ++i) x[i] += proj[i];
            }

            // ---------- FFN ----------
            const bool hasDown = !L.FFN_Down_Proj.CPUHostData.empty();
            const bool hasUp   = !L.FFN_Up_Proj.CPUHostData.empty();
            const bool hasGate = !L.FFN_Gate_Proj.CPUHostData.empty();
            if (hasDown && hasUp) {
                rmsNormApply(x.data(), L.FFN_Norm.CPUHostData.empty() ? nullptr : &L.FFN_Norm,
                             ffnIn.data(), hidden);

                size_t upRows = L.FFN_Up_Proj.Shape.size() >= 2 ? (size_t)L.FFN_Up_Proj.Shape[1]
                                                                : m_config.IntermediateDim;
                size_t inter = m_config.IntermediateDim ? m_config.IntermediateDim : upRows;

                std::vector<float> act(inter);
                if (!hasGate && upRows == 2 * inter) {
                    // Phi-3 style fused gate+up: first half gate, second half up
                    std::vector<float> gu(upRows);
                    if (!MatVec(L.FFN_Up_Proj, ffnIn.data(), gu.data(), upRows, hidden)) return false;
                    for (size_t i = 0; i < inter; ++i)
                        act[i] = (gu[i] / (1.0f + std::exp(-gu[i]))) * gu[i + inter];
                } else if (hasGate) {
                    std::vector<float> gate(inter), up(inter);
                    if (!MatVec(L.FFN_Gate_Proj, ffnIn.data(), gate.data(), inter, hidden)) return false;
                    if (!MatVec(L.FFN_Up_Proj,   ffnIn.data(), up.data(),   inter, hidden)) return false;
                    for (size_t i = 0; i < inter; ++i)
                        act[i] = (gate[i] / (1.0f + std::exp(-gate[i]))) * up[i];
                } else {
                    // No gate (GPT-style MLP): up -> silu -> down
                    if (!MatVec(L.FFN_Up_Proj, ffnIn.data(), act.data(), inter, hidden)) return false;
                    for (size_t i = 0; i < inter; ++i)
                        act[i] = act[i] / (1.0f + std::exp(-act[i]));
                }

                if (!MatVec(L.FFN_Down_Proj, act.data(), proj.data(), hidden, inter)) return false;
                for (size_t i = 0; i < hidden; ++i) x[i] += proj[i];
            }
        }

        m_cpuSeqLen = pos + 1;

        // ---- Final norm + lm_head ----
        std::vector<float> finalX(hidden);
        if (outNormTensor && !outNormTensor->CPUHostData.empty())
            rmsNormApply(x.data(), outNormTensor, finalX.data(), hidden);
        else
            finalX = x;

        outLogits.resize(vocab);
        if (!MatVec(*lmHeadTensor, finalX.data(), outLogits.data(), vocab, hidden)) {
            std::cerr << "[ModelPipeline] lm_head matvec failed (unsupported quant?)." << std::endl;
            return false;
        }

        m_cpuDispatchedOperators++;
        return true;
    }

    // ================================================================
    //  Misc
    // ================================================================
void ModelPipeline::Reset() {
         m_weightTensors.clear();
         m_cpuKCache.clear();
         m_cpuVCache.clear();
         m_cpuSeqLen = 0;
         m_vramUsageBytes = m_systemRamUsageBytes = 0;
         m_gemmPSOReady = false;
         m_flashAttnPSO.Reset();
         m_flashAttnRootSig.Reset();
         m_persistentHiddenBytes = m_persistentVocabBytes = 0;
         m_dsLoader.Shutdown();
         m_kvCache.Reset();
         m_openVINO.Shutdown();
     }

    bool ModelPipeline::DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        if (!m_dxEngine || !m_gemmPSOReady) return false;
        // Never dispatch with an unbound root descriptor: if any of X/W/Y lacks a
        // GPU resource the shader would read/write wild VAs → device removal.
        // Fall back to the CPU path (correct dequant) instead.
        if (!X.GPUResource || !W.GPUResource || !Y.GPUResource)
            return DispatchCPUMatrixMultiply(X, W, Y);
        m_gpuDispatchedOperators++;
        ComPtr<ID3D12CommandAllocator> alloc;
        ComPtr<ID3D12GraphicsCommandList> list;
        m_dxEngine->GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
        m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
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

        // ggml ne-order: W.Shape[0] = row length = input dim (K),
        //                W.Shape[1] = row count  = output dim (N).
        // The previous N/K swap made every dot product read K=IntermediateDim
        // floats from an X vector holding only HiddenDim -> OOB read -> segfault.
        size_t M = 1; // X is always a single activation vector here
        size_t N = W.Shape.size() >= 2 ? W.Shape[1] : 4096;
        size_t K = W.Shape.size() >= 2 ? W.Shape[0] : 4096;
        
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
                            wRow[b * 32 + k] = ((float)nibble - 8.0f) * scale;
                        }
                    }
                } else if (W.QuantType == QuantizationType::Q4_K) {
                    size_t blocksPerRow = K / 256;
                    size_t blockBytes = 4 + 12 + 128; // dm + scales + qs
                    const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                    for (int b = 0; b < blocksPerRow; b++) {
                        DequantizeQ4_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                    }
                } else if (W.QuantType == QuantizationType::Q5_K) {
                    size_t blocksPerRow = K / 256;
                    size_t blockBytes = 4 + 12 + 32 + 128; // dm + scales + qh + qs
                    const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                    for (int b = 0; b < blocksPerRow; b++) {
                        DequantizeQ5_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                    }
                } else if (W.QuantType == QuantizationType::Q6_K) {
                    size_t blocksPerRow = K / 256;
                    size_t blockBytes = 128 + 64 + 16 + 2; // ql + qh + scales + d
                    const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                    for (int b = 0; b < blocksPerRow; b++) {
                        DequantizeQ6_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                    }
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

     // ================================================================
     //  QKV-specific matrix multiply (for QKV projection output shape handling)
     // ================================================================
     bool ModelPipeline::DispatchCPUMatrixMultiplyQKV(const Tensor& X, const Tensor& W, Tensor& Y) {
         if (W.CPUHostData.empty() || X.CPUHostData.empty()) return false;

         // ggml ne-order: Shape[0] = input dim (K, row length), Shape[1] = output rows (N)
         size_t K = X.Shape.empty() ? 1 : X.Shape[0];
         size_t N = W.Shape.size() >= 2 ? W.Shape[1] : 4096;

         Y.Shape = {N, 1};
         Y.CPUHostData.resize(N * sizeof(float));
         float* yOut = reinterpret_cast<float*>(Y.CPUHostData.data());

         int nThreads = std::max(1, m_cpuThreadCount);
         int chunk = (N + nThreads - 1) / nThreads;

         auto worker = [&](int jStart, int jEnd) {
             std::vector<float> wRow(K);
             for (int j = jStart; j < jEnd; j++) {
                 if (W.QuantType == QuantizationType::None_FP32) {
                     const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                     for (int k = 0; k < (int)K; k++) wRow[k] = wPtr[j * K + k];
                 } else if (W.QuantType == QuantizationType::None_FP16) {
                     const uint16_t* wPtr = reinterpret_cast<const uint16_t*>(W.CPUHostData.data());
                     for (int k = 0; k < (int)K; k++) wRow[k] = FP16ToFloat(wPtr[j * K + k]);
                 } else if (W.QuantType == QuantizationType::Q8_0) {
                     size_t blocksPerRow = K / 32;
                     const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * 34;
                     for (size_t b = 0; b < blocksPerRow; ++b) {
                         const uint8_t* blockPtr = rowPtr + b * 34;
                         uint16_t d_raw;
                         std::memcpy(&d_raw, blockPtr, 2);
                         float d = FP16ToFloat(d_raw);
                         const int8_t* qs = reinterpret_cast<const int8_t*>(blockPtr + 2);
                         for (size_t k = 0; k < 32; k++) wRow[b * 32 + k] = qs[k] * d;
                     }
                 } else if (W.QuantType == QuantizationType::Q4_0) {
                     size_t blocksPerRow = K / 32;
                     const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * 18;
                     for (int b = 0; b < (int)blocksPerRow; b++) {
                         const uint8_t* blockPtr = rowPtr + b * 18;
                         uint16_t scale_raw;
                         std::memcpy(&scale_raw, blockPtr, 2);
                         float scale = FP16ToFloat(scale_raw);
                         const uint8_t* qs = blockPtr + 2;
                         for (int k = 0; k < 32; k++) {
                             uint8_t byte = qs[k / 2];
                             int8_t nibble = (k % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                             wRow[b * 32 + k] = ((float)nibble - 8.0f) * scale;
                         }
                     }
                 } else if (W.QuantType == QuantizationType::Q4_K) {
                     size_t blocksPerRow = K / 256;
                     size_t blockBytes = 4 + 12 + 128;
                     const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                     for (int b = 0; b < (int)blocksPerRow; b++) {
                         DequantizeQ4_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                     }
                 } else if (W.QuantType == QuantizationType::Q5_K) {
                     size_t blocksPerRow = K / 256;
                     size_t blockBytes = 4 + 12 + 32 + 128;
                     const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                     for (int b = 0; b < (int)blocksPerRow; b++) {
                         DequantizeQ5_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                     }
                 } else if (W.QuantType == QuantizationType::Q6_K) {
                     size_t blocksPerRow = K / 256;
                     size_t blockBytes = 128 + 64 + 16 + 2;
                     const uint8_t* rowPtr = W.CPUHostData.data() + j * blocksPerRow * blockBytes;
                     for (int b = 0; b < (int)blocksPerRow; b++) {
                         DequantizeQ6_K(rowPtr + b * blockBytes, wRow.data() + b * 256);
                     }
                 } else {
                     const float* wPtr = reinterpret_cast<const float*>(W.CPUHostData.data());
                     for (int k = 0; k < (int)K; k++) wRow[k] = wPtr[j * K + k];
                 }
                 yOut[j] = DotProductSIMD(reinterpret_cast<const float*>(X.CPUHostData.data()), wRow.data(), (int)K);
             }
         };

         m_cpuDispatchedOperators++;
         if (nThreads == 1) {
             worker(0, (int)N);
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

// ================================================================
     //  FlashAttention CPU (masked, causal, online softmax)
     // ================================================================
     // Dequantise one head-slot from a raw byte cache readback, matching
     // KVCacheManager::QuantizeVector's encode exactly. dst must hold headDim floats.
     static void DequantHeadSlotCPU(const uint8_t* slot, uint32_t headDim,
                                     KVCacheQuantType qt, float* dst) {
         switch (qt) {
             case KVCacheQuantType::None_FP32: {
                 std::memcpy(dst, slot, headDim * sizeof(float));
                 break;
             }
             case KVCacheQuantType::None_FP16: {
                 const uint16_t* h = reinterpret_cast<const uint16_t*>(slot);
                 for (uint32_t d = 0; d < headDim; ++d) dst[d] = FP16ToFloat(h[d]);
                 break;
             }
             case KVCacheQuantType::FP8: {
                 for (uint32_t d = 0; d < headDim; ++d) {
                     float scaled = (float)slot[d] / 127.5f - 1.0f;
                     dst[d] = scaled * 448.0f;
                 }
                 break;
             }
             case KVCacheQuantType::INT8: {
                 float scale; std::memcpy(&scale, slot, sizeof(float));
                 const int8_t* q = reinterpret_cast<const int8_t*>(slot + sizeof(float));
                 for (uint32_t d = 0; d < headDim; ++d) dst[d] = (float)q[d] * scale;
                 break;
             }
             case KVCacheQuantType::INT4: {
                 float scale; std::memcpy(&scale, slot, sizeof(float));
                 const uint8_t* q = slot + sizeof(float);
                 for (uint32_t d = 0; d < headDim; ++d) {
                     uint8_t byte = q[d / 2];
                     uint8_t nibble = (d % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
                     dst[d] = ((float)nibble - 8.0f) * scale;
                 }
                 break;
             }
             default:
                 std::memset(dst, 0, headDim * sizeof(float));
                 break;
         }
     }

     // CPU fallback FlashAttention. Reads the actual cached K/V history back from GPU
     // (the cache is always GPU-resident regardless of which path handles matmuls),
     // dequantises per head-slot, and computes real causal attention over it.
     // Q/K/V parameters: Q is the current token's query; K/V are the current token's
     // own key/value (already written into m_kvCache by the caller before this runs).
     void ModelPipeline::ComputeFlashAttentionCPU(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                                                  std::vector<float>& out, int seqLen, int nHeads, int headDim, uint32_t layerIdx) {
         out.assign((size_t)nHeads * headDim, 0.0f);
         float invSqrtD = 1.0f / std::sqrt((float)headDim);

if (!m_dxEngine || seqLen <= 0) { (void)K; (void)V; return; }

          ComPtr<ID3D12Fence> attnLocalFence;
          uint64_t attnLocalFenceValue = 0;
          HANDLE attnLocalEvent = nullptr;
          if (FAILED(m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&attnLocalFence)))) return;
          attnLocalEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
          if (!attnLocalEvent) return;

          const auto& cfg = m_kvCache.GetConfig();
         size_t headStride = m_kvCache.GetHeadStrideBytes();
         size_t perHeadBufBytes = (size_t)cfg.MaxSequenceLength * headStride;
         // Layer-major then head-major: this layer's region starts at layerIdx*NumHeads heads in.
         size_t layerHeadBase = (size_t)layerIdx * cfg.NumHeads;

         // Read back only the [0, seqLen) prefix actually in use, per head, for K and V.
         auto readbackHead = [&](ID3D12Resource* gpuBuf, uint32_t h, std::vector<uint8_t>& dst) -> bool {
             size_t readBytes = (size_t)seqLen * headStride;
             size_t headBase  = (layerHeadBase + h) * perHeadBufBytes; // batchIdx=0
             dst.resize(readBytes);

             D3D12_HEAP_PROPERTIES rbProps = {}; rbProps.Type = D3D12_HEAP_TYPE_READBACK;
             D3D12_RESOURCE_DESC rbDesc = {};
             rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
             rbDesc.Width = readBytes; rbDesc.Height = 1; rbDesc.DepthOrArraySize = 1;
             rbDesc.MipLevels = 1; rbDesc.Format = DXGI_FORMAT_UNKNOWN;
             rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR; rbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

             ComPtr<ID3D12Resource> readback;
             if (FAILED(m_dxEngine->GetDevice()->CreateCommittedResource(&rbProps, D3D12_HEAP_FLAG_NONE, &rbDesc,
                     D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback)))) return false;

             ComPtr<ID3D12CommandAllocator> alloc;
             ComPtr<ID3D12GraphicsCommandList> list;
             m_dxEngine->GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
             m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr, IID_PPV_ARGS(&list));

             // gpuBuf is UAV-resident (WriteTokenKV always leaves it that way); must
             // transition UAV -> COPY_SOURCE before reading, then back to UAV afterward
             // so the next WriteTokenKV call (which expects UAV state) still works.
             D3D12_RESOURCE_BARRIER toCS = {};
             toCS.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
             toCS.Transition.pResource   = gpuBuf;
             toCS.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
             toCS.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
             toCS.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
             list->ResourceBarrier(1, &toCS);

             list->CopyBufferRegion(readback.Get(), 0, gpuBuf, headBase, readBytes);

             D3D12_RESOURCE_BARRIER toUAV = {};
             toUAV.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
             toUAV.Transition.pResource   = gpuBuf;
             toUAV.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
             toUAV.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
             toUAV.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
             list->ResourceBarrier(1, &toUAV);

list->Close();
              ID3D12CommandList* ls[] = { list.Get() };
              m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
              m_dxEngine->GetComputeQueue()->Signal(attnLocalFence.Get(), ++attnLocalFenceValue);
              WaitFence(attnLocalFence.Get(), attnLocalFenceValue, attnLocalEvent);

             void* p = nullptr;
             readback->Map(0, nullptr, &p);
             std::memcpy(dst.data(), p, readBytes);
             readback->Unmap(0, nullptr);
             return true;
         };

         std::vector<float> kVec(headDim), vVec(headDim), scores((size_t)seqLen);
         std::vector<uint8_t> kRaw, vRaw;

         for (int h = 0; h < nHeads; ++h) {
             int hBase = h * headDim;
             if (!readbackHead(m_kvCache.GetKeyBuffer(),   (uint32_t)h, kRaw)) continue;
             if (!readbackHead(m_kvCache.GetValueBuffer(), (uint32_t)h, vRaw)) continue;

             // ---- scores over real per-position K (no causal mask needed: seqLen only
             // ever contains this token and strictly earlier ones) ----
             float runningMax = -1e30f;
             for (int kPos = 0; kPos < seqLen; ++kPos) {
                 DequantHeadSlotCPU(kRaw.data() + (size_t)kPos * headStride, (uint32_t)headDim, cfg.QuantType, kVec.data());
                 float score = 0.0f; // was incorrectly initialised to -1e30f, swallowing the real dot product
                 for (int d = 0; d < headDim; ++d)
                     score += Q[hBase + d] * kVec[d];
                 score *= invSqrtD;
                 scores[kPos] = score;
                 runningMax = std::max(runningMax, score);
             }

             // ---- softmax + weighted V accumulation over real per-position V ----
             float runningZ = 0.0f;
             std::vector<float> acc(headDim, 0.0f);
             for (int kPos = 0; kPos < seqLen; ++kPos) {
                 float p = std::exp(scores[kPos] - runningMax);
                 runningZ += p;
                 DequantHeadSlotCPU(vRaw.data() + (size_t)kPos * headStride, (uint32_t)headDim, cfg.QuantType, vVec.data());
                 for (int d = 0; d < headDim; ++d)
                     acc[d] += p * vVec[d];
             }

             float invZ = (runningZ > 0.0f) ? (1.0f / runningZ) : 0.0f;
             for (int d = 0; d < headDim; ++d)
                 out[hBase + d] = acc[d] * invZ;
         }
     }

     // ================================================================
     //  GPU FlashAttention dispatch (using TurboKernels.hlsl FusedFlashAttentionKernel)
     // ================================================================
bool ModelPipeline::DispatchGPUFlashAttention(const std::vector<float>& Q, const std::vector<float>& K, const std::vector<float>& V,
                                                    std::vector<float>& out, int seqLen, int nHeads, int headDim, uint32_t layerIdx) {
         if (!m_dxEngine) return false;
         // K and V here are the CURRENT token's freshly computed vectors. The caller must
         // have already written them into m_kvCache (WriteTokenKV) BEFORE calling this, so
         // that seqLen (the cache's current fill length) includes this token's own slot.
         // Historical K/V come from m_kvCache directly; K/V parameters are otherwise unused.
         (void)K; (void)V;

         size_t qBytes = (size_t)nHeads * headDim * sizeof(float);
         ComPtr<ID3D12Resource> qBuffer, outBuffer;
         ComPtr<ID3D12Resource> qUpload;

         ComPtr<ID3D12Device> dev = m_dxEngine->GetDevice();

         D3D12_HEAP_PROPERTIES defProps = {};
         defProps.Type = D3D12_HEAP_TYPE_DEFAULT;
         D3D12_HEAP_PROPERTIES upProps = {};
         upProps.Type = D3D12_HEAP_TYPE_UPLOAD;
         D3D12_RESOURCE_DESC bufDesc = {};
         bufDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
         bufDesc.Width = qBytes;
         bufDesc.Height = 1;
         bufDesc.DepthOrArraySize = 1;
         bufDesc.MipLevels = 1;
         bufDesc.Format = DXGI_FORMAT_UNKNOWN;
         bufDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
         bufDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

         // Q buffer (copied to, then used as SRV) - only the current token, no K/V buffers
         // needed here anymore; the shader reads m_kvCache's buffers directly as UAVs.
         dev->CreateCommittedResource(&defProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
             D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&qBuffer));

         // Output buffer needs UAV
         bufDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
         dev->CreateCommittedResource(&defProps, D3D12_HEAP_FLAG_NONE, &bufDesc,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&outBuffer));

         D3D12_RESOURCE_DESC upDesc = {};
         upDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
         upDesc.Width = qBytes;
         upDesc.Height = 1;
         upDesc.DepthOrArraySize = 1;
         upDesc.MipLevels = 1;
         upDesc.Format = DXGI_FORMAT_UNKNOWN;
         upDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
         upDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

         dev->CreateCommittedResource(&upProps, D3D12_HEAP_FLAG_NONE, &upDesc,
             D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&qUpload));

         // Upload Q to upload heap
         void* p;
         qUpload->Map(0, nullptr, &p);
         std::memcpy(p, Q.data(), qBytes);
         qUpload->Unmap(0, nullptr);

         // Compile FlashAttention shader if needed
         if (!m_flashAttnPSO) {
             BuildFlashAttentionPipeline();
         }

         ComPtr<ID3D12CommandAllocator> alloc;
         ComPtr<ID3D12GraphicsCommandList> list;
         dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
         dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(),
             m_flashAttnPSO.Get(), IID_PPV_ARGS(&list));

         // Must match AttentionConstants in TurboKernels.hlsl exactly (10 x 32-bit values).
         struct {
             uint32_t BatchSize, NumHeads, HeadDim, SeqLen;
             float    InvSqrtD;
             uint32_t MaxSeqLen, QuantType, HeadStrideBytes;
             uint32_t NumLayers, LayerIdx;
         } attnConsts;
         attnConsts.BatchSize       = 1;
         attnConsts.NumHeads        = (uint32_t)nHeads;
         attnConsts.HeadDim         = (uint32_t)headDim;
         attnConsts.SeqLen          = (uint32_t)seqLen;
         attnConsts.InvSqrtD        = 1.0f / std::sqrt((float)headDim);
         attnConsts.MaxSeqLen       = m_kvCache.GetConfig().MaxSequenceLength;
         attnConsts.QuantType       = (uint32_t)m_kvCache.GetConfig().QuantType;
         attnConsts.HeadStrideBytes = (uint32_t)m_kvCache.GetHeadStrideBytes();
         attnConsts.NumLayers       = m_kvCache.GetConfig().NumLayers;
         attnConsts.LayerIdx        = layerIdx;

         list->SetPipelineState(m_flashAttnPSO.Get());
         list->SetComputeRootSignature(m_flashAttnRootSig.Get());
         list->SetComputeRoot32BitConstants(0, 10, &attnConsts, 0);
         list->SetComputeRootShaderResourceView(1, qBuffer->GetGPUVirtualAddress());
         list->SetComputeRootUnorderedAccessView(2, m_kvCache.GetKeyBuffer()->GetGPUVirtualAddress());
         list->SetComputeRootUnorderedAccessView(3, m_kvCache.GetValueBuffer()->GetGPUVirtualAddress());
         list->SetComputeRootUnorderedAccessView(4, outBuffer->GetGPUVirtualAddress());

         // Copy Q from upload heap to default heap
         list->CopyBufferRegion(qBuffer.Get(), 0, qUpload.Get(), 0, qBytes);

         // Transition Q buffer: COPY_DEST -> NON_PIXEL_SHADER_RESOURCE
         // (m_kvCache's K/V buffers are already UAV-resident; WriteTokenKV always
         // leaves them in UNORDERED_ACCESS state, matching what this dispatch needs.)
         D3D12_RESOURCE_BARRIER toSRV = {};
         toSRV.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
         toSRV.Transition.pResource = qBuffer.Get();
         toSRV.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
         toSRV.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
         list->ResourceBarrier(1, &toSRV);

         // Decode-only: exactly one query token per dispatch (see kernel comment).
         list->Dispatch(1, (uint32_t)nHeads, 1);
         list->Close();

         ID3D12CommandList* ls[] = { list.Get() };
         m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls);
         m_dxEngine->GetComputeQueue()->Signal(m_inferFence.Get(), ++m_inferFenceValue);
         WaitFence(m_inferFence.Get(), m_inferFenceValue, m_inferFenceEvent);

// Read back output
          ComPtr<ID3D12Resource> readback;
          D3D12_HEAP_PROPERTIES rbProps = {};
          rbProps.Type = D3D12_HEAP_TYPE_READBACK;
          D3D12_RESOURCE_DESC rbDesc = {};
          rbDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
          rbDesc.Width = qBytes;
          rbDesc.Height = 1;
          rbDesc.DepthOrArraySize = 1;
          rbDesc.MipLevels = 1;
          rbDesc.Format = DXGI_FORMAT_UNKNOWN;
          rbDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
          rbDesc.Flags = D3D12_RESOURCE_FLAG_NONE;

          dev->CreateCommittedResource(&rbProps, D3D12_HEAP_FLAG_NONE, &rbDesc,
              D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&readback));

         ComPtr<ID3D12CommandAllocator> alloc2;
         ComPtr<ID3D12GraphicsCommandList> list2;
         dev->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc2));
         dev->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc2.Get(), nullptr, IID_PPV_ARGS(&list2));

         D3D12_RESOURCE_BARRIER toCopy = {};
         toCopy.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
         toCopy.Transition.pResource = outBuffer.Get();
         toCopy.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
         toCopy.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
         toCopy.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;

         list2->ResourceBarrier(1, &toCopy);
         list2->CopyBufferRegion(readback.Get(), 0, outBuffer.Get(), 0, qBytes);
         list2->Close();
         ID3D12CommandList* ls2[] = { list2.Get() };
         m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, ls2);
         m_dxEngine->GetComputeQueue()->Signal(m_inferFence.Get(), ++m_inferFenceValue);
         WaitFence(m_inferFence.Get(), m_inferFenceValue, m_inferFenceEvent);

         out.resize((size_t)nHeads * headDim);
         readback->Map(0, nullptr, &p);
         std::memcpy(out.data(), p, qBytes);
         readback->Unmap(0, nullptr);

         m_gpuDispatchedOperators += 2;
         return true;
     }

     // ================================================================
     //  Build FlashAttention PSO
     // ================================================================
     bool ModelPipeline::BuildFlashAttentionPipeline() {
         if (!m_dxEngine) return false;

         ComPtr<ID3DBlob> shaderBlob;
         if (!m_dxEngine->CompileComputeShader(L"shaders/TurboKernels.hlsl", "FusedFlashAttentionKernel", &shaderBlob)) {
             std::cerr << "[ModelPipeline][FlashAttn] Shader compile failed." << std::endl;
             return false;
         }

         // Root param layout MUST match TurboKernels.hlsl's FusedFlashAttentionKernel
         // register declarations exactly:
         //   b0 = AttentionConstants (10 x 32-bit values, see AttentionConstants struct)
         //   t0 = QueryBuffer   (SRV)
         //   u1 = KeyBuffer     (UAV, RWByteAddressBuffer - bound straight from KVCacheManager)
         //   u2 = ValueBuffer   (UAV, RWByteAddressBuffer - bound straight from KVCacheManager)
         //   u0 = AttnOutput    (UAV, RWStructuredBuffer<float>)
         // Previous version bound SRVs at t1/t2/t3 while the shader declared t0/t1/t2,
         // which silently shifted every buffer by one slot.
         D3D12_ROOT_PARAMETER rootParams[5] = {};
         rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
         rootParams[0].Constants.ShaderRegister = 0;
         rootParams[0].Constants.Num32BitValues = 10; // see AttentionConstants
         rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

         rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
         rootParams[1].Descriptor.ShaderRegister = 0; // t0 = QueryBuffer
         rootParams[1].Descriptor.RegisterSpace = 0;
         rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

         rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
         rootParams[2].Descriptor.ShaderRegister = 1; // u1 = KeyBuffer
         rootParams[2].Descriptor.RegisterSpace = 0;
         rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

         rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
         rootParams[3].Descriptor.ShaderRegister = 2; // u2 = ValueBuffer
         rootParams[3].Descriptor.RegisterSpace = 0;
         rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

         rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
         rootParams[4].Descriptor.ShaderRegister = 0; // u0 = AttnOutput
         rootParams[4].Descriptor.RegisterSpace = 0;
         rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

         D3D12_ROOT_SIGNATURE_DESC rsDesc = {};
         rsDesc.NumParameters = 5;
         rsDesc.pParameters = rootParams;
         rsDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

         ComPtr<ID3DBlob> sig, err;
         HRESULT hr = D3D12SerializeRootSignature(&rsDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
         if (FAILED(hr)) {
             std::cerr << "[ModelPipeline][FlashAttn] Root sig error: " << (err ? (char*)err->GetBufferPointer() : "unknown") << std::endl;
             return false;
         }

         if (FAILED(m_dxEngine->GetDevice()->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_flashAttnRootSig)))) {
             return false;
         }

         ComPtr<ID3D12PipelineState> pso;
         if (!m_dxEngine->CreateComputePipelineState(shaderBlob.Get(), m_flashAttnRootSig.Get(), &pso)) {
             return false;
         }
         m_flashAttnPSO = pso;

         std::cout << "[ModelPipeline][FlashAttn] PSO compiled." << std::endl;
         return true;
     }
 }
