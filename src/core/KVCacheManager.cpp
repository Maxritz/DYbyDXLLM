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
         float scaled = std::fmax(-448.f, std::fmin(448.f, v));
         return (uint8_t)std::fmax(0.f, std::fmin(255.f, (scaled / 448.f + 1.f) * 127.5f));
     }

    // ----------------------------------------------------------------
    KVCacheManager::KVCacheManager() : m_kvUploadBufferSize(0) {}
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
        if (!device) {
            std::cerr << "[KVCache] Warning: No device provided - CPU-only mode" << std::endl;
            m_config = config;
            m_sequenceLengths.assign(config.BatchSize, 0);
            return true;
        }
        m_config = config;
        m_device = device; // Store device for persistent resource creation
        m_sequenceLengths.assign(config.BatchSize, 0);

        // Head-major layout: each (batch, head, seqPos) slot holds one head's
        // quantised HeadDimension-length vector, matching QuantizeVector(n=HeadDimension).
        // This must mirror QuantizeVector's byte layout exactly (see that function for
        // the embedded-scale format used by INT8/INT4).
        uint32_t headDim = config.HeadDimension;
        switch (m_config.QuantType) {
            case KVCacheQuantType::None_FP32:
                m_headStrideBytes = (size_t)headDim * 4;
                break;
            case KVCacheQuantType::None_FP16:
                m_headStrideBytes = (size_t)headDim * 2;
                break;
            case KVCacheQuantType::FP8:
                m_headStrideBytes = (size_t)headDim;
                break;
            case KVCacheQuantType::INT8:
                m_headStrideBytes = sizeof(float) + (size_t)headDim;
                break;
            case KVCacheQuantType::INT4:
                m_headStrideBytes = sizeof(float) + (size_t)((headDim + 1) / 2);
                break;
            default:
                m_headStrideBytes = (size_t)headDim * 2;
                break;
        }

        // Total: batch * numLayers * numHeads * maxSeq * headStride
        m_bufferSizeInBytes = (size_t)config.BatchSize
                            * config.NumLayers
                            * config.NumHeads
                            * config.MaxSequenceLength
                            * m_headStrideBytes;

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

        hr = m_device->CreateCommittedResource(
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
                                         uint32_t            batchIdx,
                                         uint32_t            layerIdx,
                                         uint32_t            seqPos,
                                         const float*        keyData,
                                         const float*        valueData) {
        // Skip GPU writes in CPU-only mode (no GPU buffers)
        if (!m_keyBuffer || !m_valueBuffer) {
            return true; // No-op in CPU-only mode
        }
        if (!m_device) return false;
        if (batchIdx >= m_config.BatchSize)  return false;
        if (layerIdx >= m_config.NumLayers)  return false;
        if (seqPos   >= m_config.MaxSequenceLength) return false;

        uint32_t headDim  = m_config.HeadDimension;
        uint32_t numHeads = m_config.NumHeads;
        size_t totalBytes = (size_t)numHeads * m_headStrideBytes;

        // Private fence + event (lazy init): sole signaler = monotonic values
        if (!m_kvFence) {
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_kvFence))))
                return false;
            m_kvFenceValue = 0;
        }
        if (!m_kvFenceEvent) {
            m_kvFenceEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
            if (!m_kvFenceEvent) return false;
        }

        // Upload buffer holds K bytes then V bytes so one submit covers both
        size_t uploadBytes = totalBytes * 2;
        if (!m_kvUploadBuffer || m_kvUploadBufferSize < uploadBytes) {
            if (m_kvUploadBuffer) m_kvUploadBuffer.Reset();
            D3D12_HEAP_PROPERTIES uploadHeap = {};
            uploadHeap.Type = D3D12_HEAP_TYPE_UPLOAD;
            D3D12_RESOURCE_DESC uploadDesc = {};
            uploadDesc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            uploadDesc.Width            = uploadBytes;
            uploadDesc.Height           = 1;
            uploadDesc.DepthOrArraySize = 1;
            uploadDesc.MipLevels        = 1;
            uploadDesc.Format           = DXGI_FORMAT_UNKNOWN;
            uploadDesc.SampleDesc.Count = 1;
            uploadDesc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            uploadDesc.Flags            = D3D12_RESOURCE_FLAG_NONE;

            HRESULT hr = device->CreateCommittedResource(
                &uploadHeap, D3D12_HEAP_FLAG_NONE, &uploadDesc,
                D3D12_RESOURCE_STATE_GENERIC_READ, nullptr, IID_PPV_ARGS(&m_kvUploadBuffer));
            if (FAILED(hr)) return false;
            m_kvUploadBufferSize = uploadBytes;
        }

        // Quantise each head's HeadDimension-length vector separately (per-head scale)
        std::vector<uint8_t> quantKey(totalBytes, 0);
        std::vector<uint8_t> quantVal(totalBytes, 0);
        for (uint32_t h = 0; h < numHeads; ++h) {
            QuantizeVector(keyData   + (size_t)h * headDim,
                            quantKey.data() + (size_t)h * m_headStrideBytes, headDim);
            QuantizeVector(valueData + (size_t)h * headDim,
                            quantVal.data() + (size_t)h * m_headStrideBytes, headDim);
        }

        // Persistent command allocator/list
        if (!m_kvAlloc) {
            device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&m_kvAlloc));
        }
        if (!m_kvCmdList) {
            device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                m_kvAlloc.Get(), nullptr, IID_PPV_ARGS(&m_kvCmdList));
            m_kvCmdList->Close();
        }

        // Single upload map: K bytes at offset 0, V bytes at offset totalBytes
        {
            void* pMap = nullptr;
            if (FAILED(m_kvUploadBuffer->Map(0, nullptr, &pMap))) return false;
            std::memcpy(static_cast<uint8_t*>(pMap),              quantKey.data(), totalBytes);
            std::memcpy(static_cast<uint8_t*>(pMap) + totalBytes, quantVal.data(), totalBytes);
            D3D12_RANGE written{ 0, uploadBytes };
            m_kvUploadBuffer->Unmap(0, &written);
        }

        // Previous call's fence wait guarantees the GPU is done with this
        // allocator, so resetting here is safe and keeps its memory bounded.
        // Skip on the very first use: RDNA4 returns E_FAIL for Reset() on a
        // fresh allocator (same workaround as the Notllama DX12 backend).
        if (m_kvAllocUsed) {
            if (FAILED(m_kvAlloc->Reset())) return false;
        }
        if (FAILED(m_kvCmdList->Reset(m_kvAlloc.Get(), nullptr))) return false;

        // One command list covers K and V: barrier both to COPY_DEST, copy
        // both, barrier both back to UAV, one submit, one fence wait.
        ID3D12Resource* bufs[2]    = { m_keyBuffer.Get(), m_valueBuffer.Get() };
        size_t          srcBase[2] = { 0, totalBytes };

        D3D12_RESOURCE_BARRIER toCD[2] = {};
        for (int i = 0; i < 2; ++i) {
            toCD[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toCD[i].Transition.pResource   = bufs[i];
            toCD[i].Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            toCD[i].Transition.StateAfter  = D3D12_RESOURCE_STATE_COPY_DEST;
            toCD[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        m_kvCmdList->ResourceBarrier(2, toCD);

        for (int i = 0; i < 2; ++i) {
            for (uint32_t h = 0; h < numHeads; ++h) {
                size_t dstOffset = (((batchIdx * m_config.NumLayers + layerIdx) * numHeads + h)
                                    * m_config.MaxSequenceLength + seqPos) * m_headStrideBytes;
                size_t srcOffset = srcBase[i] + (size_t)h * m_headStrideBytes;
                if (dstOffset + m_headStrideBytes > m_bufferSizeInBytes) continue;
                m_kvCmdList->CopyBufferRegion(bufs[i], dstOffset, m_kvUploadBuffer.Get(), srcOffset, m_headStrideBytes);
            }
        }

        D3D12_RESOURCE_BARRIER toUAV[2] = {};
        for (int i = 0; i < 2; ++i) {
            toUAV[i].Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            toUAV[i].Transition.pResource   = bufs[i];
            toUAV[i].Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
            toUAV[i].Transition.StateAfter  = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            toUAV[i].Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        }
        m_kvCmdList->ResourceBarrier(2, toUAV);

        if (FAILED(m_kvCmdList->Close())) return false;

        ID3D12CommandList* ls[] = { m_kvCmdList.Get() };
        queue->ExecuteCommandLists(1, ls);
        m_kvAllocUsed = true;
        queue->Signal(m_kvFence.Get(), ++m_kvFenceValue);
        if (m_kvFence->GetCompletedValue() < m_kvFenceValue) {
            m_kvFence->SetEventOnCompletion(m_kvFenceValue, m_kvFenceEvent);
            if (WaitForSingleObject(m_kvFenceEvent, 10000) != WAIT_OBJECT_0) {
                std::cerr << "[KVCache] Fence wait timed out - GPU queue stalled." << std::endl;
                return false;
            }
        }
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
        m_kvAlloc.Reset();
        m_kvCmdList.Reset();
        m_kvUploadBuffer.Reset();
        m_kvFence.Reset();
        if (m_kvFenceEvent) { CloseHandle(m_kvFenceEvent); m_kvFenceEvent = nullptr; }
        m_kvFenceValue = 0;
        m_kvAllocUsed  = false;
        m_sequenceLengths.clear();
        m_bufferSizeInBytes = 0;
        m_headStrideBytes   = 0;
        m_kvUploadBufferSize = 0;
    }
}
