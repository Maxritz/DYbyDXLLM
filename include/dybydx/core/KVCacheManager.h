// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <wrl/client.h>
#include <vector>
#include <cstdint>

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
        uint32_t MaxSequenceLength = 2048;
        uint32_t BatchSize         = 1;
        uint32_t NumHeads          = 32;
        uint32_t HeadDimension     = 128;
        KVCacheQuantType QuantType = KVCacheQuantType::None_FP16;
    };

    class KVCacheManager {
    public:
        KVCacheManager();
        ~KVCacheManager();

        bool Initialize(ID3D12Device* device, const KVCacheConfig& config);
        void Reset();

        // Accessors
        ID3D12Resource* GetKeyBuffer()   const { return m_keyBuffer.Get(); }
        ID3D12Resource* GetValueBuffer() const { return m_valueBuffer.Get(); }
        const KVCacheConfig& GetConfig() const { return m_config; }

        // Sequence tracking
        uint32_t AllocateTokens(uint32_t batchIdx, uint32_t numTokens);
        uint32_t GetSequenceLength(uint32_t batchIdx) const;
        void     ReleaseSequence(uint32_t batchIdx);

        // Write one token's key and value vectors into the GPU KV buffer.
        // keyData / valueData are float32 input vectors of length (NumHeads * HeadDimension).
        // They are quantized on-CPU to the configured QuantType before the GPU upload.
        // seqPos is the position within the sequence (0-indexed).
        bool WriteTokenKV(ID3D12Device*       device,
                          ID3D12CommandQueue* queue,
                          HANDLE              fenceEvent,
                          uint32_t            batchIdx,
                          uint32_t            seqPos,
                          const float*        keyData,
                          const float*        valueData);

    private:
        // GPU buffers (DEFAULT heap, UAV)
        ComPtr<ID3D12Resource> m_keyBuffer;
        ComPtr<ID3D12Resource> m_valueBuffer;

        // Per-batch sequence lengths
        std::vector<uint32_t> m_sequenceLengths;

        KVCacheConfig m_config;
        size_t        m_bufferSizeInBytes  = 0;
        size_t        m_tokenStrideBytes   = 0; // bytes per token slot per batch in the buffer

        // Helpers
        size_t BytesPerElement() const;
        // Quantise float32 vector -> packed bytes for the configured QuantType.
        // Returns the number of bytes written into dst.
        size_t QuantizeVector(const float* src, uint8_t* dst, uint32_t numElements) const;
    };
}
