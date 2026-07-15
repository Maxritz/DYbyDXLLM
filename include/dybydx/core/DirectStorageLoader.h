// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <d3d12.h>
#include <dstorage.h>
#include <wrl/client.h>
#include <string>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    class DirectStorageLoader {
    public:
        DirectStorageLoader();
        ~DirectStorageLoader();

        bool Initialize(ID3D12Device* device);
        void Shutdown();

        // Enqueue direct load from NVMe disk straight to high-speed Default VRAM Heap
        bool QueueDirectLoad(const std::wstring& filePath, 
                             uint64_t fileOffset, 
                             uint64_t sizeInBytes, 
                             ID3D12Resource* destinationGPUBuffer,
                             ID3D12Fence* signalFence,
                             UINT64 signalFenceValue);

    private:
        ComPtr<IDStorageFactory> m_dsFactory;
        ComPtr<IDStorageQueue> m_dsQueue;
        ComPtr<IDStorageFile> m_currentFile;
        bool m_initialized = false;
    };
}
