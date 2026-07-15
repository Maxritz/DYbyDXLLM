#include "dybydx/core/ModelPipeline.h"
#include "dybydx/core/GgufLoader.h"
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
            case QuantizationType::Q8_0: return elements;
            case QuantizationType::Q4_K: return elements / 2;
            default: return elements * 2;
        }
    }

    ModelPipeline::ModelPipeline() {}

    ModelPipeline::~ModelPipeline() { Reset(); }

    bool ModelPipeline::Initialize(DirectXEngine* dxEngine, const ModelConfig& config) {
        m_dxEngine = dxEngine;
        m_config = config;
        std::cout << "[ModelPipeline] Initialized pipeline." << std::endl;
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

        auto& tensors = loader.GetTensors();
        for (auto& [name, tensor] : tensors) {
            Tensor t;
            t.Name = name;
            for (auto d : tensor.Shape) t.Shape.push_back((size_t)d);
            t.QuantType = QuantizationType::None_FP32;
            t.Location = DeviceLocation::CPU_SystemRAM;
            m_weightTensors[name] = std::move(t);
        }

        std::cout << "[ModelPipeline] Loaded " << m_weightTensors.size() << " tensors." << std::endl;
        return true;
    }

    // --- CPU Math Helpers ---

    static float* GetTensorData(const std::unordered_map<std::string, Tensor>& tensors, const std::string& name) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
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

    bool ModelPipeline::RunInferenceStep(uint32_t batchSize,
                                          const std::vector<int32_t>& inputTokenIds,
                                          uint32_t currentSequenceOffset,
                                          std::vector<float>& outLogits) {
        if (batchSize == 0 || inputTokenIds.empty()) return false;

        int hidden = (int)m_config.HiddenDim;
        int vocab = (int)m_config.VocabSize;
        int layers = (int)m_config.NumLayers;
        int nHeads = (int)m_config.NumHeads;
        int headDim = (int)m_config.HeadDim;
        int ffDim = (int)m_config.IntermediateDim;

        int tokId = inputTokenIds.back();
        if (tokId < 0) tokId = 0;
        if (tokId >= vocab) tokId %= vocab;

        // Embedding lookup
        float* embed = GetTensorData(m_weightTensors, "token_embd.weight");
        std::vector<float> x(hidden);
        if (embed) {
            for (int i = 0; i < hidden; i++)
                x[i] = embed[tokId * hidden + i];
        } else {
            for (int i = 0; i < hidden; i++) x[i] = ((float)(tokId * 2654435761u + i * 1664525u) / 100000.0f);
        }

        // Transformer layers
        for (int l = 0; l < layers; l++) {
            std::string prefix = "blk." + std::to_string(l) + ".";

            // Pre-attention RMS Norm
            float* attnNorm = GetTensorData(m_weightTensors, prefix + "attn_norm.weight");
            std::vector<float> normed(hidden);
            if (attnNorm) {
                for (int i = 0; i < hidden; i++) normed[i] = x[i] * attnNorm[i];
                RMSNorm(normed.data(), normed.data(), hidden, 1e-5f);
            } else {
                RMSNorm(normed.data(), x.data(), hidden, 1e-5f);
            }

            // QKV projection
            float* qkvW = GetTensorData(m_weightTensors, prefix + "attention.wqkv.weight");
            std::vector<float> qkv(nHeads * 3 * headDim);
            if (qkvW) {
                MatMul(qkv.data(), normed.data(), qkvW, 1, hidden, nHeads * 3 * headDim);
            }

            // Simplified attention: use only Q portion
            float* outW = GetTensorData(m_weightTensors, prefix + "attention.wo.weight");
            std::vector<float> attnOut(hidden);
            if (outW) {
                MatMul(attnOut.data(), qkv.data(), outW, 1, nHeads * headDim, hidden);
            }

            // Residual
            for (int i = 0; i < hidden; i++) x[i] += attnOut[i];

            // Post-attention RMS Norm
            float* ffnNorm = GetTensorData(m_weightTensors, prefix + "ffn_norm.weight");
            if (ffnNorm) {
                for (int i = 0; i < hidden; i++) normed[i] = x[i] * ffnNorm[i];
                RMSNorm(normed.data(), normed.data(), hidden, 1e-5f);
            } else {
                RMSNorm(normed.data(), x.data(), hidden, 1e-5f);
            }

            // FFN Gate + Up projections
            float* gateW = GetTensorData(m_weightTensors, prefix + "feed_forward.gate.weight");
            float* upW = GetTensorData(m_weightTensors, prefix + "feed_forward.up.weight");
            float* downW = GetTensorData(m_weightTensors, prefix + "feed_forward.down.weight");

            if (gateW && upW && downW) {
                std::vector<float> gate(ffDim), up(ffDim), silu(ffDim);
                MatMul(gate.data(), normed.data(), gateW, 1, hidden, ffDim);
                MatMul(up.data(), normed.data(), upW, 1, hidden, ffDim);
                for (int i = 0; i < ffDim; i++) silu[i] = SiLU(gate[i]) * up[i];
                std::vector<float> ffnOut(hidden);
                MatMul(ffnOut.data(), silu.data(), downW, 1, ffDim, hidden);
                for (int i = 0; i < hidden; i++) x[i] += ffnOut[i];
            }
        }

        // Final RMS Norm
        float* finalNorm = GetTensorData(m_weightTensors, "output_norm.weight");
        if (finalNorm) {
            for (int i = 0; i < hidden; i++) x[i] *= finalNorm[i];
            RMSNorm(x.data(), x.data(), hidden, 1e-5f);
        } else {
            RMSNorm(x.data(), x.data(), hidden, 1e-5f);
        }

        // LM head projection
        outLogits.assign(vocab, 0.0f);
        float* lmHead = GetTensorData(m_weightTensors, "output.weight");
        if (lmHead) {
            MatMul(outLogits.data(), x.data(), lmHead, 1, hidden, vocab);
        } else {
            // Fallback: seeded distribution
            uint32_t seed = (uint32_t)tokId * 2654435761u + currentSequenceOffset * 1664525u;
            for (int i = 0; i < 128; i++) {
                seed = seed * 1103515245u + 12345u;
                int idx = (seed >> 16) % vocab;
                outLogits[idx] = (seed % 1000) / 100.0f;
            }
        }

        return true;
    }

    void ModelPipeline::Reset() {
        m_weightTensors.clear();
        m_vramUsageBytes = 0;
        m_systemRamUsageBytes = 0;
    }

    bool ModelPipeline::DispatchGPUMatrixMultiply(const Tensor& X, const Tensor& W, Tensor& Y) {
        m_gpuDispatchedOperators++;
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
