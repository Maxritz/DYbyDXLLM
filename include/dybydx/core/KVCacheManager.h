// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    enum class KVCacheQuantType {
        None_FP32 = 0,
        None_FP16,
        FP8,
        INT8,
        INT4
    };

    struct KVCacheConfig {
        uint32_t MaxSequenceLength;
        uint32_t BatchSize;
        uint32_t NumHeads;
        uint32_t HeadDimension;
        KVCacheQuantType QuantType = KVCacheQuantType::None_FP16;
    };

    class KVCacheManager {
    public:
        KVCacheManager();
        ~KVCacheManager();

        bool Initialize(ID3D12Device* device, const KVCacheConfig& config);
        void Reset();

        // KV cache allocation and append pointers
        ID3D12Resource* GetKeyBuffer() const { return m_keyBuffer.Get(); }
        ID3D12Resource* GetValueBuffer() const { return m_valueBuffer.Get(); }

        // Track sequences
        uint32_t AllocateTokens(uint32_t batchIdx, uint32_t numTokens);
        uint32_t GetSequenceLength(uint32_t batchIdx) const { return m_sequenceLengths[batchIdx]; }
        void ReleaseSequence(uint32_t batchIdx);

        const KVCacheConfig& GetConfig() const { return m_config; }

    private:
        ComPtr<ID3D12Resource> m_keyBuffer;
        ComPtr<ID3D12Resource> m_valueBuffer;

        KVCacheConfig m_config;
        std::vector<uint32_t> m_sequenceLengths;

        size_t m_bufferSizeInBytes = 0;
    };
}
