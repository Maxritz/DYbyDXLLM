// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    class IntelOpenVINOInterop {
    public:
        IntelOpenVINOInterop();
        ~IntelOpenVINOInterop();

        // Bind shared DX12 contexts
        bool InitializeWithSharedDevice(ID3D12Device* d3d12Device);
        
        // Load model network using OpenVINO NPU/GPU plugins
        bool LoadModelIR(const std::string& xmlPath, const std::string& binPath, const std::string& deviceTarget);
        
        // Execute matrix tensor operation utilizing a shared DirectX 12 pointer address (zero-copy interop)
        bool ExecuteSharedOperator(ID3D12Resource* d3d12BufferIn, ID3D12Resource* d3d12BufferOut, size_t numElements);

        void Shutdown();

    private:
        bool m_hasIntelDevice = false;
        void* m_ovCore = nullptr;          // Mock OpenVINO Core reference
        void* m_ovCompiledModel = nullptr;  // Compiled inference graph
        ID3D12Device* m_sharedDevice = nullptr;
    };
}
