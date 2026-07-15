// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/KVCacheManager.h"
#include <iostream>

namespace DirectLLM {

    KVCacheManager::KVCacheManager() {
        m_config = {};
    }

    KVCacheManager::~KVCacheManager() {
        Reset();
    }

    bool KVCacheManager::Initialize(ID3D12Device* device, const KVCacheConfig& config) {
        m_config = config;
        m_sequenceLengths.assign(config.BatchSize, 0);

        // Size computation based on cache quantization type
        size_t numElements = (size_t)config.BatchSize * config.MaxSequenceLength * config.NumHeads * config.HeadDimension;
        
        float bytesPerElement = 2.0f;
        switch (config.QuantType) {
            case KVCacheQuantType::None_FP32: bytesPerElement = 4.0f; break;
            case KVCacheQuantType::None_FP16: bytesPerElement = 2.0f; break;
            case KVCacheQuantType::FP8:
            case KVCacheQuantType::INT8:      bytesPerElement = 1.0f; break;
            case KVCacheQuantType::INT4:      bytesPerElement = 0.5f; break;
        }
        m_bufferSizeInBytes = (size_t)(numElements * bytesPerElement);

        // Allocating large Default Heap Buffers (GPU high-speed memory)
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width = m_bufferSizeInBytes;
        desc.Height = 1;
        desc.DepthOrArraySize = 1;
        desc.MipLevels = 1;
        desc.Format = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count = 1;
        desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        // Key Buffer
        HRESULT hr = device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_keyBuffer)
        );
        if (FAILED(hr)) return false;

        // Value Buffer
        hr = device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            nullptr,
            IID_PPV_ARGS(&m_valueBuffer)
        );
        if (FAILED(hr)) return false;

        std::cout << "[KVCacheManager] Allocated Key and Value Buffers on GPU. Size per buffer: " 
                  << (m_bufferSizeInBytes / (1024 * 1024)) << " MB" << std::endl;

        return true;
    }

    uint32_t KVCacheManager::AllocateTokens(uint32_t batchIdx, uint32_t numTokens) {
        if (batchIdx >= m_sequenceLengths.size()) return 0;

        uint32_t currentLen = m_sequenceLengths[batchIdx];
        if (currentLen + numTokens > m_config.MaxSequenceLength) {
            // Out of bounds or wrap-around / Eviction triggers (Ring buffer fallback)
            std::cout << "[KVCacheManager][WARN] Out of token budget. Triggering FIFO eviction sequence." << std::endl;
            m_sequenceLengths[batchIdx] = m_config.MaxSequenceLength / 2; // Evict 50%
            return m_sequenceLengths[batchIdx];
        }

        m_sequenceLengths[batchIdx] += numTokens;
        return currentLen; // Returns offset to start writing new token KVs
    }

    void KVCacheManager::ReleaseSequence(uint32_t batchIdx) {
        if (batchIdx < m_sequenceLengths.size()) {
            m_sequenceLengths[batchIdx] = 0;
        }
    }

    void KVCacheManager::Reset() {
        m_keyBuffer.Reset();
        m_valueBuffer.Reset();
        m_sequenceLengths.clear();
    }
}
