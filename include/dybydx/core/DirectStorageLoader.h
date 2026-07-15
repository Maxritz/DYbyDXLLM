// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <cstdint>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    // DirectStorage load request — supports both CPU-memory source and file source.
    //
    // SOURCE_MEMORY (default):  reads from a CPU pointer already in RAM.
    //   Advantage: works even when GgufLoader has already mapped the file.
    //   Batch-enqueues all tensors before a single Submit+Wait — avoids the
    //   per-tensor CopyResource + fence overhead.
    //
    // SOURCE_FILE (streaming):  reads directly NVMe→GPU bypassing CPU memory.
    //   Requires absolute file offset per tensor.  Best used when GgufLoader
    //   is operating in "metadata-only" mode (future).
    class DirectStorageLoader {
    public:
        DirectStorageLoader();
        ~DirectStorageLoader();

        bool Initialize(ID3D12Device* device);
        void Shutdown();

        // Enqueue one tensor from a CPU memory pointer directly to a GPU default-heap buffer.
        // Call Submit() once all tensors are enqueued to fire the DMA pipeline.
        bool EnqueueMemoryToGPU(const void*      srcCpuPtr,
                                 uint64_t         sizeInBytes,
                                 ID3D12Resource*  dstGPUBuffer);

        // Enqueue one tensor from a raw file offset directly to GPU (true bypass path).
        // absoluteFileOffset = GetTensorDataBaseOffset() + tensor.FileOffset
        bool EnqueueFileToGPU(const std::wstring& filePath,
                               uint64_t            absoluteFileOffset,
                               uint64_t            sizeInBytes,
                               ID3D12Resource*     dstGPUBuffer);

        // Submit all queued requests and block until all GPU copies are complete.
        // signalFence / signalValue are used to synchronise with the compute queue.
        bool SubmitAndWait(ID3D12Fence* signalFence, UINT64 signalValue, HANDLE waitEvent);

        bool IsInitialized() const { return m_initialized; }
        size_t GetPendingCount() const { return m_pendingCount; }

    private:
        ComPtr<IDStorageFactory> m_dsFactory;
        ComPtr<IDStorageQueue>   m_dsQueueGPU;   // SOURCE_MEMORY or SOURCE_FILE → GPU
        ComPtr<IDStorageFile>    m_currentFile;
        std::wstring             m_currentFilePath;

        bool   m_initialized  = false;
        size_t m_pendingCount = 0;
    };
}
