// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/DirectStorageLoader.h"
#include <iostream>

#pragma comment(lib, "dstorage.lib")

namespace DirectLLM {

    DirectStorageLoader::DirectStorageLoader() : m_initialized(false) {}

    DirectStorageLoader::~DirectStorageLoader() {
        Shutdown();
    }

    bool DirectStorageLoader::Initialize(ID3D12Device* device) {
        std::cout << "[DirectStorage] Initializing Win11 DirectStorage Factory..." << std::endl;

        // Create factory
        HRESULT hr = DStorageGetFactory(IID_PPV_ARGS(&m_dsFactory));
        if (FAILED(hr)) {
            std::cout << "[DirectStorage][ERROR] Failed to fetch DirectStorage factory. Fallback to CPU file IO." << std::endl;
            return false;
        }

        // Configure high priority Queue targeting Default VRAM Heap
        DSTORAGE_QUEUE_DESC queueDesc = {};
        queueDesc.Capacity = DSTORAGE_MAX_QUEUE_CAPACITY;
        queueDesc.Priority = DSTORAGE_PRIORITY_HIGH;
        queueDesc.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        queueDesc.Device = device;

        hr = m_dsFactory->CreateQueue(&queueDesc, IID_PPV_ARGS(&m_dsQueue));
        if (FAILED(hr)) {
            std::cout << "[DirectStorage][ERROR] Failed to initialize DirectStorage queue context." << std::endl;
            return false;
        }

        m_initialized = true;
        std::cout << "[DirectStorage] Pipeline successfully created. GPU GDeflate decompression active." << std::endl;
        return true;
    }

    bool DirectStorageLoader::QueueDirectLoad(const std::wstring& filePath, 
                                             uint64_t fileOffset, 
                                             uint64_t sizeInBytes, 
                                             ID3D12Resource* destinationGPUBuffer,
                                             ID3D12Fence* signalFence,
                                             UINT64 signalFenceValue) {
        if (!m_initialized) return false;

        // Open weight binary file natively with Win11 DirectStorage bypass
        HRESULT hr = m_dsFactory->OpenFile(filePath.c_str(), IID_PPV_ARGS(&m_currentFile));
        if (FAILED(hr)) {
            std::cout << "[DirectStorage][ERROR] Unable to open weights file: " << std::string(filePath.begin(), filePath.end()) << std::endl;
            return false;
        }

        // Configure high speed read request structure
        DSTORAGE_REQUEST request = {};
        request.Options.SourceType = DSTORAGE_REQUEST_SOURCE_FILE;
        request.Options.DestinationType = DSTORAGE_REQUEST_DESTINATION_MULTIPLE_SUBRESOURCES;
        request.Options.CompressionFormat = DSTORAGE_COMPRESSION_FORMAT_GDEFLATE; // Decompress weights on-the-fly inside the GPU stream
        
        request.Source.File.Source = m_currentFile.Get();
        request.Source.File.Offset = fileOffset;
        request.Source.File.Size = sizeInBytes;

        request.Destination.MultipleSubresources.Resource = destinationGPUBuffer;
        request.Destination.MultipleSubresources.FirstSubresource = 0;

        // Enqueue high speed read request
        m_dsQueue->EnqueueRequest(&request);

        // Append GPU Fence to release compute queue waiting states when files have been completely read & decompressed in VRAM
        m_dsQueue->EnqueueSignal(signalFence, signalFenceValue);

        // Fire asynchronous PCI-e direct load pipeline
        m_dsQueue->Submit();
        std::cout << "[DirectStorage] Direct load submitted. Enqueued weight load block of size " 
                  << (sizeInBytes / (1024 * 1024)) << " MB directly to GPU memory." << std::endl;

        return true;
    }

    void DirectStorageLoader::Shutdown() {
        m_dsQueue.Reset();
        m_currentFile.Reset();
        m_dsFactory.Reset();
        m_initialized = false;
    }
}
