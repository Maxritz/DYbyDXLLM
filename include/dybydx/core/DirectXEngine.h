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
    __declspec(dllexport) int  InitializeEngine(void* engine, int forceWarp);
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
        std::wstring   DeviceName;
        HardwareVendor Vendor          = HardwareVendor::Unknown;
        size_t         DedicatedVRAM   = 0;
        bool           SupportsSM66    = false;
        bool           SupportsFP16    = false;
        bool           SupportsInt8    = false;
        // PCIe Resizable BAR / Smart Access Memory
        // When true, GPU_UPLOAD heap maps directly to VRAM.
        // CPU writes go NVMe->VRAM without a staging copy.
        // Works on: AMD RDNA2+ with SAM, NVIDIA Ampere+ ReBAR, Intel Arc.
        bool           SupportsReBAR   = false;
    };

    class DirectXEngine {
    public:
        DirectXEngine();
        ~DirectXEngine();

        bool Initialize(bool forceWarp = false);
        void Shutdown();

        // Queues
        ID3D12Device*       GetDevice()       { return m_device.Get(); }
        ID3D12CommandQueue* GetComputeQueue() { return m_computeQueue.Get(); }
        ID3D12CommandQueue* GetCopyQueue()    { return m_copyQueue.Get(); }

        // Pipeline
        bool CompileComputeShader(const std::wstring& shaderPath,
                                   const std::string&  entryPoint,
                                   ID3DBlob**          outShaderBlob);
        bool CreateComputePipelineState(ID3DBlob*            shaderBlob,
                                         ID3D12RootSignature* rootSignature,
                                         ID3D12PipelineState** outPSO);

        // Memory
        // Standard allocation: UPLOAD / DEFAULT / READBACK
        bool AllocateGPUBuffer(size_t              sizeInBytes,
                               D3D12_HEAP_TYPE     heapType,
                               D3D12_RESOURCE_STATES initialState,
                               ID3D12Resource**    outResource);

        // ReBAR allocation: GPU_UPLOAD heap (CPU-writable VRAM).
        // Falls back to UPLOAD heap if ReBAR is not supported.
        // Use instead of UPLOAD+CopyResource when SupportsReBAR() is true.
        bool AllocateReBarBuffer(size_t           sizeInBytes,
                                 ID3D12Resource** outResource);

        // Capabilities
        const DeviceCapabilities& GetCaps() const { return m_caps; }
        bool SupportsReBAR() const { return m_caps.SupportsReBAR; }

        void LogTrace(const std::string& moduleName, const std::string& message);

    private:
        ComPtr<IDXGIFactory6>      m_dxgiFactory;
        ComPtr<IDXGIAdapter4>      m_adapter;
        ComPtr<ID3D12Device8>      m_device;
        ComPtr<ID3D12CommandQueue> m_computeQueue;
        ComPtr<ID3D12CommandQueue> m_copyQueue;
        ComPtr<ID3D12Fence>        m_computeFence;
        ComPtr<ID3D12Fence>        m_copyFence;
        UINT64                     m_computeFenceValue = 0;
        UINT64                     m_copyFenceValue    = 0;
        DeviceCapabilities         m_caps;
        void QueryDeviceCapabilities();
        HardwareVendor ResolveVendor(UINT vendorId);
    };
}
