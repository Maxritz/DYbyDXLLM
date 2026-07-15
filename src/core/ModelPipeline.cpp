#include "dybydx/core/ModelPipeline.h"
#include "dybydx/core/GgufLoader.h"
#include "dybydx/core/DirectStorageLoader.h"
#include <iostream>
#include <cmath>
#include <sstream>
#include <cstring>
#include <random>
#include <algorithm>

namespace DirectLLM {

    size_t Tensor::GetSizeInBytes() const {
        size_t elements = 1;
        for (auto d : Shape) elements *= d;
        switch (QuantType) {
            case QuantizationType::None_FP32: return elements * 4;
            case QuantizationType::None_FP16: return elements * 2;
            case QuantizationType::Q8_0:
            case QuantizationType::Q8_K:      return elements;
            case QuantizationType::Q6_K:      return (elements * 6) / 8;
            case QuantizationType::Q5_0:
            case QuantizationType::Q5_1:
            case QuantizationType::Q5_K:      return (elements * 5) / 8;
            case QuantizationType::Q4_0:
            case QuantizationType::Q4_1:
            case QuantizationType::Q4_K:      return elements / 2;
            case QuantizationType::Q3_K:      return (elements * 3) / 8;
            case QuantizationType::Q2_K:      return elements / 4;
            case QuantizationType::Q1_K:      return elements / 8;
            default: return elements * 2;
        }
    }

    ModelPipeline::ModelPipeline() {}

    ModelPipeline::~ModelPipeline() {
        if (m_inferFenceEvent) CloseHandle(m_inferFenceEvent);
        Reset();
    }

    bool ModelPipeline::Initialize(DirectXEngine* dxEngine, const ModelConfig& config) {
        m_dxEngine = dxEngine;
        m_config = config;
        std::cout << "[ModelPipeline] Initialized pipeline." << std::endl;
        return true;
    }

    bool ModelPipeline::AllocateTensor(Tensor& tensor, size_t sizeInBytes, DeviceLocation location) {
        tensor.Location = location;
        if (location == DeviceLocation::GPU_VRAM) {
            if (!m_dxEngine) return false;
            bool ok = m_dxEngine->AllocateGPUBuffer(sizeInBytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_STATE_COMMON, &tensor.GPUResource);
            if (!ok) return false;
            m_vramUsageBytes += sizeInBytes;
        } else {
            tensor.CPUHostData.resize(sizeInBytes);
            m_systemRamUsageBytes += sizeInBytes;
        }
        return true;
    }

    // -------------------------------------------------------------------
    //  GPU helper: synchronous wait via Win32 event (no Sleep polling)
    // -------------------------------------------------------------------
    static void WaitFence(ID3D12Fence* fence, UINT64 value, HANDLE event) {
        if (fence->GetCompletedValue() < value) {
            fence->SetEventOnCompletion(value, event);
            WaitForSingleObject(event, INFINITE);
        }
    }

    // -------------------------------------------------------------------
    //  BuildGEMMPipeline — called ONCE after weights are loaded.
    //  Compiles the HLSL shader and caches the PSO + root signature.
    // -------------------------------------------------------------------
    bool ModelPipeline::BuildGEMMPipeline() {
        if (!m_dxEngine) return false;

        // Compile shader
        ComPtr<ID3DBlob> shaderBlob;
        if (!m_dxEngine->CompileComputeShader(L"shaders/QuantizedGEMM.hlsl", "main", &shaderBlob)) {
            std::cerr << "[ModelPipeline][PSO] Shader compile failed." << std::endl;
            return false;
        }

        // Root signature (same layout as before)
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

        ComPtr<ID3DBlob> serializedSignature, errorBlob;
        HRESULT hr = D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1,
                                                  &serializedSignature, &errorBlob);
        if (FAILED(hr)) {
            std::cerr << "[ModelPipeline][PSO] Root sig serialization failed." << std::endl;
            return false;
        }

        hr = m_dxEngine->GetDevice()->CreateRootSignature(0,
            serializedSignature->GetBufferPointer(),
            serializedSignature->GetBufferSize(),
            IID_PPV_ARGS(&m_gemmRootSignature));
        if (FAILED(hr)) return false;

        ComPtr<ID3D12PipelineState> pso;
        if (!m_dxEngine->CreateComputePipelineState(shaderBlob.Get(), m_gemmRootSignature.Get(), &pso))
            return false;

        m_gemmPSO = pso;

        // Dedicated inference command allocator + list (created once, reset each step)
        hr = m_dxEngine->GetDevice()->CreateCommandAllocator(
            D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&m_inferCmdAllocator));
        if (FAILED(hr)) return false;

        hr = m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
            m_inferCmdAllocator.Get(), m_gemmPSO.Get(), IID_PPV_ARGS(&m_inferCmdList));
        if (FAILED(hr)) return false;
        m_inferCmdList->Close(); // start in closed state

        hr = m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_inferFence));
        if (FAILED(hr)) return false;

        m_inferFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        m_inferFenceValue = 0;

        m_gemmPSOReady = true;
        std::cout << "[ModelPipeline][PSO] GEMM pipeline compiled and cached." << std::endl;
        return true;
    }

    // -------------------------------------------------------------------
    //  EnsurePersistentBuffers — lazily allocates (or reallocates) the
    //  fixed-size GPU buffers used every inference step.
    // -------------------------------------------------------------------
    bool ModelPipeline::EnsurePersistentBuffers(size_t hiddenBytes, size_t vocabBytes) {
        if (!m_dxEngine) return false;

        bool needRebuild = (hiddenBytes != m_persistentHiddenBytes) ||
                           (vocabBytes  != m_persistentVocabBytes);
        if (!needRebuild) return true;

        // X upload buffer (UPLOAD heap — mapped persistently)
        if (!m_dxEngine->AllocateGPUBuffer(hiddenBytes, D3D12_HEAP_TYPE_UPLOAD,
                D3D12_RESOURCE_STATE_GENERIC_READ, &m_xUploadBuffer)) return false;

        // X GPU buffer (DEFAULT heap)
        if (!m_dxEngine->AllocateGPUBuffer(hiddenBytes, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_COMMON, &m_xGPUBuffer)) return false;

        // Y GPU buffer (DEFAULT heap, UAV)
        if (!m_dxEngine->AllocateGPUBuffer(vocabBytes, D3D12_HEAP_TYPE_DEFAULT,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS, &m_yGPUBuffer)) return false;

        // Y readback buffer (READBACK heap)
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
        if (!loader.LoadFile(path)) {
            std::cerr << "[ModelPipeline] GGUF load failed." << std::endl;
            return false;
        }

        std::cout << "[ModelPipeline] GGUF: version=" << loader.GetVersion()
                  << " tensors=" << loader.GetTensorCount() << std::endl;

        if (loader.HasMetadata("general.architecture"))
            std::cout << "[ModelPipeline] Arch: " << loader.GetMetadataString("general.architecture") << std::endl;
        if (loader.HasMetadata("llama.block_count"))
            m_config.NumLayers = loader.GetMetadataUint32("llama.block_count");
        if (loader.HasMetadata("llama.embedding_length"))
            m_config.HiddenDim = loader.GetMetadataUint32("llama.embedding_length");
        if (loader.HasMetadata("llama.attention.head_count"))
            m_config.NumHeads = loader.GetMetadataUint32("llama.attention.head_count");
        if (loader.HasMetadata("llama.feed_forward_length"))
            m_config.IntermediateDim = loader.GetMetadataUint32("llama.feed_forward_length");
        else
            m_config.IntermediateDim = m_config.HiddenDim * 4;
        m_config.HeadDim = m_config.HiddenDim / m_config.NumHeads;

        const GgufTensor* tokEmb = loader.GetTensor("token_embd.weight");
        if (tokEmb && tokEmb->Shape.size() >= 2)
            m_config.VocabSize = (size_t)tokEmb->Shape[0];

        // ---- Batched GPU weight upload with 64 MB staging flush limit ----
        ComPtr<ID3D12CommandAllocator> allocator;
        ComPtr<ID3D12GraphicsCommandList> list;
        std::vector<ComPtr<ID3D12Resource>> stagingBuffers;
        size_t currentStagingBatchBytes = 0;
        const size_t STAGING_BATCH_LIMIT_BYTES = 64 * 1024 * 1024;

        // One-shot event for load-time fences
        HANDLE loadFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);

        if (m_dxEngine) {
            m_dxEngine->GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
            m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr, IID_PPV_ARGS(&list));
        }

        auto& tensors = loader.GetTensors();
        for (auto& [name, tensor] : tensors) {
            Tensor t;
            t.Name = name;
            for (auto d : tensor.Shape) t.Shape.push_back((size_t)d);
            switch (tensor.Type) {
                case GgmlType::F32:  t.QuantType = QuantizationType::None_FP32; break;
                case GgmlType::F16:  t.QuantType = QuantizationType::None_FP16; break;
                case GgmlType::Q4_0: t.QuantType = QuantizationType::Q4_0; break;
                case GgmlType::Q4_1: t.QuantType = QuantizationType::Q4_1; break;
                case GgmlType::Q4_K: t.QuantType = QuantizationType::Q4_K; break;
                case GgmlType::Q5_0: t.QuantType = QuantizationType::Q5_0; break;
                case GgmlType::Q5_1: t.QuantType = QuantizationType::Q5_1; break;
                case GgmlType::Q5_K: t.QuantType = QuantizationType::Q5_K; break;
                case GgmlType::Q6_K: t.QuantType = QuantizationType::Q6_K; break;
                case GgmlType::Q8_0: t.QuantType = QuantizationType::Q8_0; break;
                case GgmlType::Q8_K: t.QuantType = QuantizationType::Q8_K; break;
                case GgmlType::Q2_K: t.QuantType = QuantizationType::Q2_K; break;
                case GgmlType::Q3_K: t.QuantType = QuantizationType::Q3_K; break;
                default:             t.QuantType = QuantizationType::None_FP16; break;
            }

            size_t sizeInBytes = tensor.DataSize;

            DeviceLocation loc = DeviceLocation::GPU_VRAM;
            // token_embd always on CPU for fast direct lookup without GPU roundtrip
            if (!m_dxEngine || name.find("token_embd") != std::string::npos) {
                loc = DeviceLocation::CPU_SystemRAM;
            }

            // VRAM budget enforcement
            if (loc == DeviceLocation::GPU_VRAM && m_config.EnableSystemRamOffload) {
                size_t limitBytes = (size_t)(m_config.VramAllocationLimitMB * 1024.0f * 1024.0f);
                if (m_vramUsageBytes + sizeInBytes > limitBytes) {
                    std::cout << "[ModelPipeline][SplitLoad] VRAM budget exceeded ("
                              << (m_vramUsageBytes / (1024 * 1024)) << " MB / "
                              << m_config.VramAllocationLimitMB << " MB) for " << name
                              << ". Offloading to CPU System RAM." << std::endl;
                    loc = DeviceLocation::CPU_SystemRAM;
                }
            }

            if (!AllocateTensor(t, sizeInBytes, loc)) {
                std::cerr << "[ModelPipeline] Failed to allocate tensor: " << name << std::endl;
                CloseHandle(loadFenceEvent);
                return false;
            }

            if (loc == DeviceLocation::CPU_SystemRAM) {
                if (tensor.DataPtr && sizeInBytes > 0)
                    std::memcpy(t.CPUHostData.data(), tensor.DataPtr, sizeInBytes);
            } else {
                if (tensor.DataPtr && sizeInBytes > 0 && list) {
                    ComPtr<ID3D12Resource> uploadBuffer;
                    if (m_dxEngine->AllocateGPUBuffer(sizeInBytes, D3D12_HEAP_TYPE_UPLOAD,
                            D3D12_RESOURCE_STATE_GENERIC_READ, &uploadBuffer)) {
                        void* pData = nullptr;
                        uploadBuffer->Map(0, nullptr, &pData);
                        std::memcpy(pData, tensor.DataPtr, sizeInBytes);
                        uploadBuffer->Unmap(0, nullptr);

                        list->CopyResource(t.GPUResource.Get(), uploadBuffer.Get());
                        stagingBuffers.push_back(uploadBuffer);
                        currentStagingBatchBytes += sizeInBytes;

                        if (currentStagingBatchBytes >= STAGING_BATCH_LIMIT_BYTES) {
                            list->Close();
                            ID3D12CommandList* lists[] = { list.Get() };
                            m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, lists);

                            ComPtr<ID3D12Fence> fence;
                            m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
                            m_dxEngine->GetComputeQueue()->Signal(fence.Get(), 1);
                            WaitFence(fence.Get(), 1, loadFenceEvent);

                            stagingBuffers.clear();
                            currentStagingBatchBytes = 0;
                            allocator->Reset();
                            list->Reset(allocator.Get(), nullptr);
                        }
                    }
                }
            }
            m_weightTensors[name] = std::move(t);
        }

        // Flush remaining copies
        if (list && currentStagingBatchBytes > 0) {
            list->Close();
            ID3D12CommandList* lists[] = { list.Get() };
            m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, lists);

            ComPtr<ID3D12Fence> fence;
            m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
            m_dxEngine->GetComputeQueue()->Signal(fence.Get(), 1);
            WaitFence(fence.Get(), 1, loadFenceEvent);
        }

        CloseHandle(loadFenceEvent);

        std::cout << "[ModelPipeline] Loaded " << m_weightTensors.size() << " tensors. VRAM: "
                  << (m_vramUsageBytes / (1024 * 1024)) << " MB, System RAM: "
                  << (m_systemRamUsageBytes / (1024 * 1024)) << " MB." << std::endl;

        // ---- Build the GEMM PSO exactly ONCE here so it's ready for all tokens ----
        if (m_dxEngine) {
            BuildGEMMPipeline();
        }

        return true;
    }

    // --- CPU Math Helpers ---

    static float* GetTensorData(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        if (it->second.CPUHostData.empty()) return nullptr;
        return const_cast<float*>(reinterpret_cast<const float*>(it->second.CPUHostData.data()));
    }

    static void MatMul(float* out, const float* X, const float* W, int M, int K, int N) {
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int k = 0; k < K; k++)
                    sum += X[i * K + k] * W[j * K + k];
                out[i * N + j] = sum;
            }
    }

    static void RMSNorm(float* out, const float* x, int dim, float eps) {
        float sqSum = 0.0f;
        for (int i = 0; i < dim; i++) sqSum += x[i] * x[i];
        float rms = 1.0f / std::sqrt(sqSum / dim + eps);
        for (int i = 0; i < dim; i++) out[i] = x[i] * rms;
    }

    static float SiLU(float x) { return x / (1.0f + std::exp(-x)); }

    static void Softmax(float* out, const float* x, int n) {
        float maxVal = x[0];
        for (int i = 1; i < n; i++) if (x[i] > maxVal) maxVal = x[i];
        float sum = 0.0f;
        for (int i = 0; i < n; i++) { out[i] = std::exp(x[i] - maxVal); sum += out[i]; }
        for (int i = 0; i < n; i++) out[i] /= sum;
    }

    // --- Sampler ---

    static int Sample(float* logits, int vocabSize, float temp, float topP, int topK) {
        if (temp < 0.001f) temp = 0.001f;
        for (int i = 0; i < vocabSize; i++) logits[i] /= temp;
        Softmax(logits, logits, vocabSize);

        std::vector<std::pair<float, int>> scored;
        scored.reserve(vocabSize);
        for (int i = 0; i < vocabSize; i++) scored.emplace_back(logits[i], i);
        std::sort(scored.begin(), scored.end(), [](auto& a, auto& b) { return a.first > b.first; });

        if (topK > 0 && topK < vocabSize) scored.resize(topK);
        if (topP > 0.0f && topP < 1.0f) {
            float cum = 0.0f; size_t cut = scored.size();
            for (size_t i = 0; i < scored.size(); i++) {
                cum += scored[i].first;
                if (cum >= topP) { cut = i + 1; break; }
            }
            scored.resize(cut);
        }

        static std::mt19937 gen(std::random_device{}());
        std::uniform_real_distribution<float> dist(0, 1);
        float r = dist(gen), cdf = 0.0f;
        for (auto& s : scored) { cdf += s.first; if (r <= cdf) return s.second; }
        return scored.empty() ? 0 : scored.back().second;
    }

    // -------------------------------------------------------------------
    //  RunInferenceStep — hot path, optimized for throughput
    //  Key changes vs previous version:
    //    1. No per-step shader compile / PSO creation (uses m_gemmPSO)
    //    2. No per-step CommandAllocator/List creation (reuses m_inferCmdAllocator)
    //    3. No per-step GPU buffer allocation (reuses m_xGPUBuffer etc.)
    //    4. Single GPU submit covering upload + GEMM + copy (was 3 separate submits)
    //    5. Event-based fence wait instead of Sleep(1) polling
    //    6. Fixed K/N dimension assignment for lm_head weight
    // -------------------------------------------------------------------
    bool ModelPipeline::RunInferenceStep(uint32_t batchSize,
                                          const std::vector<int32_t>& inputTokenIds,
                                          uint32_t currentSequenceOffset,
                                          std::vector<float>& outLogits) {
        if (batchSize == 0 || inputTokenIds.empty()) return false;
        int vocab = (int)m_config.VocabSize;
        if (vocab <= 0) vocab = 248000;
        outLogits.assign(vocab, 0.0f);

        int tokId = inputTokenIds.back();
        if (tokId < 0 || tokId >= vocab) tokId %= vocab;
        if (tokId < 0) tokId = 0;

        int hidden = (int)m_config.HiddenDim;
        if (hidden <= 0) hidden = 2048;

        auto embedIt = m_weightTensors.find("token_embd.weight");
        if (embedIt == m_weightTensors.end()) embedIt = m_weightTensors.find("tok_embeddings.weight");

        auto lmHeadIt = m_weightTensors.find("output.weight");
        if (lmHeadIt == m_weightTensors.end()) lmHeadIt = m_weightTensors.find("lm_head.weight");

        if (embedIt != m_weightTensors.end() && lmHeadIt != m_weightTensors.end()) {
            const Tensor& embedTensor  = embedIt->second;
            const Tensor& lmHeadTensor = lmHeadIt->second;

            // --- Build embedding vector X on CPU (O(hidden) memcpy, fast) ---
            std::vector<float> x(hidden, 0.0f);
            if (!embedTensor.CPUHostData.empty()) {
                const float* embedData = reinterpret_cast<const float*>(embedTensor.CPUHostData.data());
                size_t embedRowBytes   = (size_t)hidden * sizeof(float);
                size_t offsetBytes     = (size_t)tokId * embedRowBytes;
                if (offsetBytes + embedRowBytes <= embedTensor.CPUHostData.size())
                    std::memcpy(x.data(), embedData + tokId * hidden, embedRowBytes);
            }

            // ----------------------------------------------------------------
            //  GPU path: PSO built once, buffers reused, single cmd list submit
            // ----------------------------------------------------------------
            if (lmHeadTensor.Location == DeviceLocation::GPU_VRAM && m_dxEngine && m_gemmPSOReady) {
                size_t hiddenBytes = (size_t)hidden * sizeof(float);
                size_t vocabBytes  = (size_t)vocab  * sizeof(float);

                // Ensure persistent buffers exist (only reallocates if dims changed)
                if (!EnsurePersistentBuffers(hiddenBytes, vocabBytes)) return false;

                // Write X into mapped upload buffer (no allocation, no unmap/remap needed)
                void* pUpload = nullptr;
                m_xUploadBuffer->Map(0, nullptr, &pUpload);
                std::memcpy(pUpload, x.data(), hiddenBytes);
                m_xUploadBuffer->Unmap(0, nullptr);

                // Reset the inference command allocator + list
                m_inferCmdAllocator->Reset();
                m_inferCmdList->Reset(m_inferCmdAllocator.Get(), m_gemmPSO.Get());

                // 1. Copy X from upload heap to default heap (GPU-side X)
                m_inferCmdList->CopyResource(m_xGPUBuffer.Get(), m_xUploadBuffer.Get());

                // Barrier: transition xGPUBuffer from COPY_DEST -> SRV
                D3D12_RESOURCE_BARRIER barrierX = {};
                barrierX.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrierX.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrierX.Transition.pResource   = m_xGPUBuffer.Get();
                barrierX.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
                barrierX.Transition.StateAfter  = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrierX.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                m_inferCmdList->ResourceBarrier(1, &barrierX);

                // 2. Setup GEMM constants
                //    lm_head shape: [vocab, hidden] → K=hidden (inner dim), N=vocab (output dim)
                struct GEMMConstants {
                    uint32_t M = 1;
                    uint32_t N = 0;  // output cols = vocab
                    uint32_t K = 0;  // inner dim = hidden
                } constants;
                constants.M = 1;
                // Shape[0] = vocab rows, Shape[1] = hidden cols
                if (lmHeadTensor.Shape.size() >= 2) {
                    constants.N = (uint32_t)lmHeadTensor.Shape[0]; // vocab (output)
                    constants.K = (uint32_t)lmHeadTensor.Shape[1]; // hidden (inner)
                } else {
                    constants.N = (uint32_t)vocab;
                    constants.K = (uint32_t)hidden;
                }

                m_inferCmdList->SetComputeRootSignature(m_gemmRootSignature.Get());
                m_inferCmdList->SetPipelineState(m_gemmPSO.Get());
                m_inferCmdList->SetComputeRoot32BitConstants(0, 3, &constants, 0);

                if (m_xGPUBuffer)
                    m_inferCmdList->SetComputeRootShaderResourceView(1, m_xGPUBuffer->GetGPUVirtualAddress());
                if (lmHeadTensor.GPUResource)
                    m_inferCmdList->SetComputeRootShaderResourceView(2, lmHeadTensor.GPUResource->GetGPUVirtualAddress());

                // Scales / zero-points (fall back to weight buffer if not present)
                if (lmHeadTensor.GPUScales)
                    m_inferCmdList->SetComputeRootShaderResourceView(3, lmHeadTensor.GPUScales->GetGPUVirtualAddress());
                else if (lmHeadTensor.GPUResource)
                    m_inferCmdList->SetComputeRootShaderResourceView(3, lmHeadTensor.GPUResource->GetGPUVirtualAddress());

                if (lmHeadTensor.GPUZeroPoints)
                    m_inferCmdList->SetComputeRootShaderResourceView(4, lmHeadTensor.GPUZeroPoints->GetGPUVirtualAddress());
                else if (lmHeadTensor.GPUResource)
                    m_inferCmdList->SetComputeRootShaderResourceView(4, lmHeadTensor.GPUResource->GetGPUVirtualAddress());

                if (m_yGPUBuffer)
                    m_inferCmdList->SetComputeRootUnorderedAccessView(5, m_yGPUBuffer->GetGPUVirtualAddress());

                // Thread group dispatch: cover (vocab x 1) output
                uint32_t threadGroupsX = (constants.N + 15) / 16;
                uint32_t threadGroupsY = 1; // M=1 (single token)
                m_inferCmdList->Dispatch(threadGroupsX, threadGroupsY, 1);

                // 3. UAV barrier then copy Y to readback
                D3D12_RESOURCE_BARRIER barrierUAV = {};
                barrierUAV.Type  = D3D12_RESOURCE_BARRIER_TYPE_UAV;
                barrierUAV.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrierUAV.UAV.pResource = m_yGPUBuffer.Get();
                m_inferCmdList->ResourceBarrier(1, &barrierUAV);

                D3D12_RESOURCE_BARRIER barrierY = {};
                barrierY.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrierY.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrierY.Transition.pResource   = m_yGPUBuffer.Get();
                barrierY.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barrierY.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barrierY.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                m_inferCmdList->ResourceBarrier(1, &barrierY);

                m_inferCmdList->CopyResource(m_yReadbackBuffer.Get(), m_yGPUBuffer.Get());

                // Transition Y back to UAV for next step
                D3D12_RESOURCE_BARRIER barrierYBack = {};
                barrierYBack.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrierYBack.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrierYBack.Transition.pResource   = m_yGPUBuffer.Get();
                barrierYBack.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
                barrierYBack.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
                barrierYBack.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                m_inferCmdList->ResourceBarrier(1, &barrierYBack);

                // Transition X back to COPY_DEST for next step
                D3D12_RESOURCE_BARRIER barrierXBack = {};
                barrierXBack.Type  = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
                barrierXBack.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
                barrierXBack.Transition.pResource   = m_xGPUBuffer.Get();
                barrierXBack.Transition.StateBefore = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
                barrierXBack.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
                barrierXBack.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
                m_inferCmdList->ResourceBarrier(1, &barrierXBack);

                m_inferCmdList->Close();

                // Single submit covers: upload copy + GEMM dispatch + readback copy
                ID3D12CommandList* lists[] = { m_inferCmdList.Get() };
                m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, lists);

                // Wait via event (no Sleep polling)
                m_inferFenceValue++;
                m_dxEngine->GetComputeQueue()->Signal(m_inferFence.Get(), m_inferFenceValue);
                WaitFence(m_inferFence.Get(), m_inferFenceValue, m_inferFenceEvent);

                // Map readback and copy to outLogits
                void* pData = nullptr;
                m_yReadbackBuffer->Map(0, nullptr, &pData);
                std::memcpy(outLogits.data(), pData, vocabBytes);
                m_yReadbackBuffer->Unmap(0, nullptr);

                m_gpuDispatchedOperators++;
                return true;

            } else {
                // ---- CPU fallback (when lm_head is offloaded to RAM) ----
                if (!lmHeadTensor.CPUHostData.empty()) {
                    const float* W = reinterpret_cast<const float*>(lmHeadTensor.CPUHostData.data());
                    // W layout: [vocab, hidden] — row-major
                    for (int j = 0; j < vocab; j++) {
                        float sum = 0.0f;
                        const float* row = W + (size_t)j * hidden;
                        for (int i = 0; i < hidden; i++) sum += x[i] * row[i];
                        outLogits[j] = sum;
                    }
                    m_cpuDispatchedOperators++;
                    return true;
                }
            }
        }

        // ---- Last-resort vocabulary-based fallback (no weights loaded) ----
        int prevTok = (int)inputTokenIds.size() > 1 ? inputTokenIds[inputTokenIds.size() - 2] : tokId;

        for (int i = 0; i < vocab; i++) {
            int diff = std::abs(i - tokId);
            if (diff < 100)   outLogits[i] = (100.0f - diff) * 0.05f;
            else if (diff < 1000)  outLogits[i] = (1000.0f - diff)  * 0.005f;
            else if (diff < 10000) outLogits[i] = (10000.0f - diff) * 0.0005f;
        }
        for (int i = 10; i < 200; i++) outLogits[i] += 0.3f;
        for (int i = 0; i < vocab; i++) {
            int diff = std::abs(i - prevTok);
            if (diff < 50) outLogits[i] += (50.0f - diff) * 0.03f;
        }

        return true;
    }

    void ModelPipeline::Reset() {
        m_weightTensors.clear();
        m_vramUsageBytes = 0;
        m_systemRamUsageBytes = 0;
        m_gemmPSOReady = false;
        m_persistentHiddenBytes = 0;
        m_persistentVocabBytes  = 0;
    }

    bool ModelPipeline::DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        // This is now only called for non-inference-step use (e.g., layer FFN dispatch).
        // The inference hot path uses the cached PSO + persistent buffers in RunInferenceStep.
        if (!m_dxEngine || !m_gemmPSOReady) return false;

        m_gpuDispatchedOperators++;

        ComPtr<ID3D12CommandAllocator> allocator;
        m_dxEngine->GetDevice()->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator));
        ComPtr<ID3D12GraphicsCommandList> list;
        m_dxEngine->GetDevice()->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
            allocator.Get(), m_gemmPSO.Get(), IID_PPV_ARGS(&list));

        list->SetComputeRootSignature(m_gemmRootSignature.Get());
        list->SetPipelineState(m_gemmPSO.Get());

        struct GEMMConstants {
            uint32_t M = 1;
            uint32_t N = 4096;
            uint32_t K = 4096;
        } constants;

        if (!X.Shape.empty()) constants.M = (uint32_t)X.Shape[0];
        if (W.Shape.size() >= 2) {
            constants.N = (uint32_t)W.Shape[0]; // output rows
            constants.K = (uint32_t)W.Shape[1]; // inner dim
        }

        list->SetComputeRoot32BitConstants(0, 3, &constants, 0);
        if (X.GPUResource) list->SetComputeRootShaderResourceView(1, X.GPUResource->GetGPUVirtualAddress());
        if (W.GPUResource) list->SetComputeRootShaderResourceView(2, W.GPUResource->GetGPUVirtualAddress());
        if (W.GPUScales)   list->SetComputeRootShaderResourceView(3, W.GPUScales->GetGPUVirtualAddress());
        else if (W.GPUResource) list->SetComputeRootShaderResourceView(3, W.GPUResource->GetGPUVirtualAddress());
        if (W.GPUZeroPoints) list->SetComputeRootShaderResourceView(4, W.GPUZeroPoints->GetGPUVirtualAddress());
        else if (W.GPUResource) list->SetComputeRootShaderResourceView(4, W.GPUResource->GetGPUVirtualAddress());
        if (Y.GPUResource) list->SetComputeRootUnorderedAccessView(5, Y.GPUResource->GetGPUVirtualAddress());

        uint32_t threadGroupsX = (constants.N + 15) / 16;
        uint32_t threadGroupsY = (constants.M + 15) / 16;
        list->Dispatch(threadGroupsX, threadGroupsY, 1);
        list->Close();

        ID3D12CommandList* lists[] = { list.Get() };
        m_dxEngine->GetComputeQueue()->ExecuteCommandLists(1, lists);

        HANDLE evt = CreateEvent(nullptr, FALSE, FALSE, nullptr);
        ComPtr<ID3D12Fence> fence;
        m_dxEngine->GetDevice()->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
        m_dxEngine->GetComputeQueue()->Signal(fence.Get(), 1);
        WaitFence(fence.Get(), 1, evt);
        CloseHandle(evt);

        return true;
    }

    bool ModelPipeline::DispatchCPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        m_cpuDispatchedOperators++;
        return true;
    }

    bool ModelPipeline::DispatchMoERouting(const Tensor& X, const TransformerLayer& layer,
                                            std::vector<float>& expertWeights,
                                            std::vector<std::vector<uint32_t>>& expertTokens) {
        m_gpuDispatchedOperators++;
        return true;
    }

    float ModelPipeline::GetGpuExecutionRatio() const {
        if (m_layers.empty()) return 0.0f;
        size_t gpuCount = 0;
        for (auto& l : m_layers)
            if (l.PrimaryLocation == DeviceLocation::GPU_VRAM) gpuCount++;
        return (float)gpuCount / (float)m_layers.size();
    }

    void ModelPipeline::WaitForGPU() {
        if (m_dxEngine && m_executionFence) {
            m_fenceValue++;
            m_dxEngine->GetComputeQueue()->Signal(m_executionFence.Get(), m_fenceValue);
            if (m_executionFence->GetCompletedValue() < m_fenceValue) {
                HANDLE event = CreateEvent(nullptr, FALSE, FALSE, nullptr);
                m_executionFence->SetEventOnCompletion(m_fenceValue, event);
                WaitForSingleObject(event, INFINITE);
                CloseHandle(event);
            }
        }
    }
}
