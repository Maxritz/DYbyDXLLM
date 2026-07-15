// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <memory>

using Microsoft::WRL::ComPtr;

// Enable Microsoft DirectX Agility SDK exports
extern "C" {
    __declspec(dllexport) extern const UINT D3D12SDKVersion;
    __declspec(dllexport) extern const char* D3D12SDKPath;

    __declspec(dllexport) void* CreateEngine();
    __declspec(dllexport) int InitializeEngine(void* engine, int forceWarp);
    __declspec(dllexport) void DestroyEngine(void* engine);
    __declspec(dllexport) const char* GetDeviceInfo(void* engine);
}

namespace DirectLLM {

    enum class HardwareVendor {
        Unknown = 0,
        AMD,
        NVIDIA,
        Intel,
        Microsoft_Fallback
    };

    struct DeviceCapabilities {
        std::wstring DeviceName;
        HardwareVendor Vendor;
        size_t DedicatedVRAM;
        bool SupportsSM66;
        bool SupportsFP16;
        bool SupportsInt8;
    };

    class DirectXEngine {
    public:
        DirectXEngine();
        ~DirectXEngine();

        bool Initialize(bool forceWarp = false);
        void Shutdown();

        // Queue & Execution
        ID3D12Device* GetDevice() { return m_device.Get(); }
        ID3D12CommandQueue* GetComputeQueue() { return m_computeQueue.Get(); }
        ID3D12CommandQueue* GetCopyQueue() { return m_copyQueue.Get(); }

        // Pipeline Utilities
        bool CompileComputeShader(const std::wstring& shaderPath, const std::string& entryPoint, ID3DBlob** outShaderBlob);
        bool CreateComputePipelineState(ID3DBlob* shaderBlob, ID3D12RootSignature* rootSignature, ID3D12PipelineState** outPSO);

        // Memory Management
        bool AllocateGPUBuffer(size_t sizeInBytes, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, ID3D12Resource** outResource);

        // Capability query
        const DeviceCapabilities& GetCaps() const { return m_caps; }

        // Trace API
        void LogTrace(const std::string& moduleName, const std::string& message);

    private:
        ComPtr<IDXGIFactory6> m_dxgiFactory;
        ComPtr<IDXGIAdapter4> m_adapter;
        ComPtr<ID3D12Device8> m_device;
        
        ComPtr<ID3D12CommandQueue> m_computeQueue;
        ComPtr<ID3D12CommandQueue> m_copyQueue;
        ComPtr<ID3D12Fence> m_computeFence;
        ComPtr<ID3D12Fence> m_copyFence;
        
        UINT64 m_computeFenceValue = 0;
        UINT64 m_copyFenceValue = 0;

        DeviceCapabilities m_caps;
        void QueryDeviceCapabilities();
        HardwareVendor ResolveVendor(UINT vendorId);
    };
}
