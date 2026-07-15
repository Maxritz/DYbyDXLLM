// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/KVCacheManager.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

namespace DirectLLM {

    // ----------------------------------------------------------------
    //  Float16 conversion helper (software, no intrinsics needed)
    // ----------------------------------------------------------------
    static uint16_t FloatToFP16(float v) {
        uint32_t bits;
        std::memcpy(&bits, &v, 4);
        uint32_t sign     = (bits >> 31) & 0x1;
        int32_t  exponent = ((bits >> 23) & 0xFF) - 127;
        uint32_t mantissa = bits & 0x7FFFFF;

        if (exponent > 15)  return (uint16_t)((sign << 15) | 0x7C00); // inf
        if (exponent < -14) return (uint16_t)(sign << 15);             // underflow -> 0

        uint16_t h_exp = (uint16_t)(exponent + 15) << 10;
        uint16_t h_man = (uint16_t)(mantissa >> 13);
        return (uint16_t)((sign << 15) | h_exp | h_man);
    }

    // Very simple FP8 (E4M3) approximation via scale-and-clamp
    static uint8_t FloatToFP8(float v) {
        // Scale to [-448, 448] range of E4M3, clamp, then quantize to 8-bit
        float scaled = std::fmax(-448.f, std::fmin(448.f, v));
        // Map to [0,255]: 0 = -448, 128 = 0, 255 = +447
        return (uint8_t)std::fmax(0.f, std::fmin(255.f, (scaled / 448.f + 1.f) * 127.5f));
    }

    // ----------------------------------------------------------------
    KVCacheManager::KVCacheManager() { m_config = {}; }
    KVCacheManager::~KVCacheManager() { Reset(); }

    size_t KVCacheManager::BytesPerElement() const {
        switch (m_config.QuantType) {
            case KVCacheQuantType::None_FP32: return 4;
            case KVCacheQuantType::None_FP16: return 2;
            case KVCacheQuantType::FP8:
            case KVCacheQuantType::INT8:      return 1;
            case KVCacheQuantType::INT4:      return 0; // special: 0.5 bytes, handled separately
            default:                          return 2;
        }
    }

    bool KVCacheManager::Initialize(ID3D12Device* device, const KVCacheConfig& config) {
        m_config = config;
        m_sequenceLengths.assign(config.BatchSize, 0);

        uint32_t headsTimeDim = config.NumHeads * config.HeadDimension;

        // INT4: 2 elements per byte
        size_t elementBytes = (m_config.QuantType == KVCacheQuantType::INT4)
                              ? (headsTimeDim / 2)
                              : (headsTimeDim * BytesPerElement());

        // Token stride = bytes needed per sequence position (one K or V vector)
        m_tokenStrideBytes = elementBytes;

        // Total: batch * maxSeq * token_stride
        m_bufferSizeInBytes = (size_t)config.BatchSize
                            * config.MaxSequenceLength
                            * m_tokenStrideBytes;

        if (m_bufferSizeInBytes == 0) {
            std::cerr << "[KVCache] Buffer size zero - check config." << std::endl;
            return false;
        }

        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = D3D12_HEAP_TYPE_DEFAULT;

        D3D12_RESOURCE_DESC desc = {};
        desc.Dimension          = D3D12_RESOURCE_DIMENSION_BUFFER;
        desc.Width              = m_bufferSizeInBytes;
        desc.Height             = 1;
        desc.DepthOrArraySize   = 1;
        desc.MipLevels          = 1;
        desc.Format             = DXGI_FORMAT_UNKNOWN;
        desc.SampleDesc.Count   = 1;
        desc.Layout             = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        desc.Flags              = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;

        HRESULT hr = device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_keyBuffer));
        if (FAILED(hr)) { std::cerr << "[KVCache] Key buffer alloc failed hr=" << std::hex << hr << std::endl; return false; }

        hr = device->CreateCommittedResource(
            &heapProps, D3D12_HEAP_FLAG_NONE, &desc,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr,
            IID_PPV_ARGS(&m_valueBuffer));
        if (FAILED(hr)) { std::cerr << "[KVCache] Value buffer alloc failed hr=" << std::hex << hr << std::endl; return false; }

        const char* qtName = "?";
        switch (m_config.QuantType) {
            case KVCacheQuantType::None_FP32: qtName = "fp32"; break;
            case KVCacheQuantType::None_FP16: qtName = "fp16"; break;
            case KVCacheQuantType::FP8:       qtName = "fp8";  break;
            case KVCacheQuantType::INT8:      qtName = "int8"; break;
            case KVCacheQuantType::INT4:      qtName = "int4"; break;
        }
        std::cout << "[KVCache] Allocated K+V buffers. quant=" << qtName
                  << " size_per_buf=" << (m_bufferSizeInBytes / (1024*1024)) << " MB"
                  << " maxSeq=" << config.MaxSequenceLength << std::endl;
        return true;
    }

    // ----------------------------------------------------------------
    //  QuantizeVector
    //  Converts float32 src[numElements] -> packed bytes in dst[]
    //  according to the configured QuantType.
    //  Returns number of bytes written.
    // ----------------------------------------------------------------
    size_t KVCacheManager::QuantizeVector(const float* src, uint8_t* dst, uint32_t n) const {
        switch (m_config.QuantType) {

        case KVCacheQuantType::None_FP32:
            std::memcpy(dst, src, n * sizeof(float));
            return n * sizeof(float);

        case KVCacheQuantType::None_FP16: {
            uint16_t* out = reinterpret_cast<uint16_t*>(dst);
            for (uint32_t i = 0; i < n; i++) out[i] = FloatToFP16(src[i]);
            return n * sizeof(uint16_t);
        }

        case KVCacheQuantType::FP8: {
            for (uint32_t i = 0; i < n; i++) dst[i] = FloatToFP8(src[i]);
            return n;
        }

        case KVCacheQuantType::INT8: {
            // Per-vector symmetric quantization: scale = max(|x|) / 127
            float absMax = 1e-8f;
            for (uint32_t i = 0; i < n; i++) absMax = std::max(absMax, std::fabs(src[i]));
            float scale = absMax / 127.f;
            // Store scale as float32 in first 4 bytes, then n INT8 values
            std::memcpy(dst, &scale, sizeof(float));
            int8_t* out = reinterpret_cast<int8_t*>(dst + sizeof(float));
            for (uint32_t i = 0; i < n; i++)
                out[i] = (int8_t)std::max(-127.f, std::min(127.f, std::round(src[i] / scale)));
            return sizeof(float) + n;
        }

        case KVCacheQuantType::INT4: {
            // Per-vector symmetric quantization: scale = max(|x|) / 7
            float absMax = 1e-8f;
            for (uint32_t i = 0; i < n; i++) absMax = std::max(absMax, std::fabs(src[i]));
            float scale = absMax / 7.f;
            std::memcpy(dst, &scale, sizeof(float));
            uint8_t* out = dst + sizeof(float);
            for (uint32_t i = 0; i < n; i += 2) {
                int lo = (int)std::max(-7.f, std::min(7.f, std::round(src[i]     / scale))) + 8;
                int hi = (i+1 < n)
                       ? (int)std::max(-7.f, std::min(7.f, std::round(src[i+1]  / scale))) + 8
                       : 8;
                out[i/2] = (uint8_t)((hi << 4) | (lo & 0x0F));
            }
            return sizeof(float) + (n / 2);
        }

        default:
            return 0;
        }
    }

    // ----------------------------------------------------------------
    //  WriteTokenKV
    //  1. Quantize key and value float32 vectors on CPU.
    //  2. Upload quantized bytes via UPLOAD heap.
    //  3. Transition KV buffer: UAV -> COPY_DEST.
    //  4. CopyBufferRegion to correct offset in the GPU KV buffer.
    //  5. Transition back: COPY_DEST -> UAV.
    //  6. Submit and wait on fence event.
    // ----------------------------------------------------------------
    bool KVCacheManager::WriteTokenKV(ID3D12Device*       device,
                                       ID3D12CommandQueue* queue,
                                       HANDLE              fenceEvent,
                                       uint32_t            batchIdx,
                                       uint32_t            seqPos,
                                       const float*        keyData,
                                       const float*        valueData) {
        if (!m_keyBuffer || !m_valueBuffer) return false;
        if (batchIdx >= m_config.BatchSize)  return false;
        if (seqPos   >= m_config.MaxSequenceLength) return false;

        uint32_t n = m_config.NumHeads * m_config.HeadDimension;

        // Quantize key and value on CPU
        // Worst case: FP32 scale(4) + n bytes; allocate generously
        size_t quantBufSize = sizeof(float) + n * sizeof(float);
        std::vector<uint8_t> quantKey(quantBufSize, 0);
        std::vector<uint8_t> quantVal(quantBufSize, 0);

        size_t keyBytes = QuantizeVector(keyData,   quantKey.data(), n);
        size_t valBytes = QuantizeVector(valueData, quantVal.data(), n);

        if (keyBytes == 0 || valBytes == 0) return false;

        // Offset into the KV buffer: [batchIdx][seqPos] * tokenStride
        size_t slotOffset = ((size_t)batchIdx * m_config.MaxSequenceLength + seqPos)
                           * m_tokenStrideBytes;

        if (slotOffset + keyBytes > m_bufferSizeInBytes) {
            std::cerr << "[KVCache] WriteTokenKV: offset out of bounds at seqPos=" << seqPos << std::endl;
            return false;
        }

        // Allocate a temporary upload buffer large enough for both K and V
        size_t uploadSize = std::max(keyBytes, valBytes);

        D3D12_HEAP_PROPERTIES uploadHeap = {};
        uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
        D3D12_RESOURCE_DESC uploadDesc = {};
        uploadDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        uploadDesc.Width            = uploadSize;
        uploadDesc.Height           = 1;
        uploadDesc.DepthOrArraySize = 1;
        uploadDesc.MipLevels        = 1;
        uploadDesc.Format           = DXGI_FORMAT_UNKNOWN;
        uploadDesc.SampleDesc.Count = 1;
        uploadDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

        ComPtr<ID3D12Resource> uploadBuf;
        HRESULT hr = device->CreateCommittedResource(
            &uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&uploadBuf));
        if (FAILED(hr)) return false;

        // Helper: perform one key or value write
        auto writeBuffer = [&](const uint8_t* quantData, size_t numBytes,
                               ID3D12Resource* gpuBuf) -> bool {
            // Map upload buffer and copy quantized data
            void* pMap = nullptr;
            if (FAILED(uploadBuf->Map(0, nullptr, &pMap))) return false;
            std::memcpy(pMap, quantData, numBytes);
            uploadBuf->Unmap(0, nullptr);

            // Create a fresh command allocator + list for this write
            ComPtr<ID3D12CommandAllocator> alloc;
            if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    IID_PPV_ARGS(&alloc)))) return false;
            ComPtr<ID3D12GraphicsCommandList> list;
            if (FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE,
                    alloc.Get(), nullptr, IID_PPV_ARGS(&list)))) return false;

            // Transition GPU buffer: UAV -> COPY_DEST
            D3D12_RESOURCE_BARRIER toCD = {};
            toCD.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toCD.Transition.pResource   = gpuBuf;
            toCD.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            toCD.Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
            toCD.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            list->ResourceBarrier(1, &toCD);

            // Copy just the quantized token slot
            list->CopyBufferRegion(gpuBuf, slotOffset, uploadBuf.Get(), 0, numBytes);

            // Transition back: COPY_DEST -> UAV
            D3D12_RESOURCE_BARRIER toUAV = {};
            toUAV.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toUAV.Transition.pResource   = gpuBuf;
            toUAV.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            toUAV.Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            toUAV.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            list->ResourceBarrier(1, &toUAV);

            list->Close();

            ID3D12CommandList* ls[] = { list.Get() };
            queue->ExecuteCommandLists(1, ls);

            // Signal and wait (event-based)
            ComPtr<ID3D12Fence> fence;
            device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
            queue->Signal(fence.Get(), 1);
            if (fence->GetCompletedValue() < 1) {
                fence->SetEventOnCompletion(1, fenceEvent);
                WaitForSingleObject(fenceEvent, INFINITE);
            }
            return true;
        };

        if (!writeBuffer(quantKey.data(), keyBytes, m_keyBuffer.Get()))   return false;
        if (!writeBuffer(quantVal.data(), valBytes, m_valueBuffer.Get())) return false;

        return true;
    }

    uint32_t KVCacheManager::AllocateTokens(uint32_t batchIdx, uint32_t numTokens) {
        if (batchIdx >= m_sequenceLengths.size()) return 0;
        uint32_t cur = m_sequenceLengths[batchIdx];
        if (cur + numTokens > m_config.MaxSequenceLength) {
            std::cout << "[KVCache] Sequence full at pos=" << cur << ". Evicting oldest 50%." << std::endl;
            uint32_t half = m_config.MaxSequenceLength / 2;
            m_sequenceLengths[batchIdx] = half;
            return half;
        }
        m_sequenceLengths[batchIdx] += numTokens;
        return cur;
    }

    uint32_t KVCacheManager::GetSequenceLength(uint32_t batchIdx) const {
        if (batchIdx >= m_sequenceLengths.size()) return 0;
        return m_sequenceLengths[batchIdx];
    }

    void KVCacheManager::ReleaseSequence(uint32_t batchIdx) {
        if (batchIdx < m_sequenceLengths.size())
            m_sequenceLengths[batchIdx] = 0;
    }

    void KVCacheManager::Reset() {
        m_keyBuffer.Reset();
        m_valueBuffer.Reset();
        m_sequenceLengths.clear();
        m_bufferSizeInBytes = 0;
        m_tokenStrideBytes  = 0;
    }
}
