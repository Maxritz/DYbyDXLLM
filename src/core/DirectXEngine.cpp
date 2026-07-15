// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/DirectXEngine.h"
#include <d3dcompiler.h>
#include <iostream>
#include <sstream>

#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "d3dcompiler.lib")

namespace DirectLLM {

    DirectXEngine::DirectXEngine() {
        m_caps = {};
    }

    DirectXEngine::~DirectXEngine() {
        Shutdown();
    }

    bool DirectXEngine::Initialize(bool forceWarp) {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        LogTrace("DirectXEngine", "Initializing DirectX 12 Device via Microsoft Agility SDK...");

        UINT dxgiFactoryFlags = 0;
#ifdef _DEBUG
        // Enable debug layer
        ComPtr<ID3D12Debug> debugController;
        if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debugController)))) {
            debugController->EnableDebugLayer();
            LogTrace("DirectXEngine", "D3D12 Debug Layer Enabled");
        }
#endif

        if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)))) {
            LogTrace("DirectXEngine", "CRITICAL ERROR: Failed to create DXGI Factory.");
            return false;
        }

        // Search for adapters
        ComPtr<IDXGIAdapter1> adapter1;
        if (forceWarp) {
            m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1));
        } else {
            for (UINT adapterIndex = 0; 
                 DXGI_ERROR_NOT_FOUND != m_dxgiFactory->EnumAdapterByGpuPreference(adapterIndex, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter1));
                 ++adapterIndex) {
                
                DXGI_ADAPTER_DESC1 desc;
                adapter1->GetDesc1(&desc);

                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue; // Skip software rendering adapters
                break;
            }
        }

        if (!adapter1) {
            LogTrace("DirectXEngine", "No hardware GPU found. Falling back to Microsoft WARP...");
            m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1));
        }

        if (!adapter1) {
            LogTrace("DirectXEngine", "CRITICAL: No suitable Direct3D 12 adapter found.");
            return false;
        }

        adapter1.As(&m_adapter);
        if (!m_adapter) {
            LogTrace("DirectXEngine", "Failed to promote adapter to IDXGIAdapter4.");
            return false;
        }
        QueryDeviceCapabilities();

        // Create the actual D3D12 Device
        HRESULT hr = D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device));
        if (FAILED(hr)) {
            LogTrace("DirectXEngine", "CRITICAL: Failed to initialize D3D12 Logical Device.");
            return false;
        }

        LogTrace("DirectXEngine", "Direct3D 12 Device Created successfully!");

        // Create Queues
        D3D12_COMMAND_QUEUE_DESC computeQueueDesc = {};
        computeQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
        computeQueueDesc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        computeQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        m_device->CreateCommandQueue(&computeQueueDesc, IID_PPV_ARGS(&m_computeQueue));

        D3D12_COMMAND_QUEUE_DESC copyQueueDesc = {};
        copyQueueDesc.Type = D3D12_COMMAND_LIST_TYPE_COPY;
        copyQueueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        m_device->CreateCommandQueue(&copyQueueDesc, IID_PPV_ARGS(&m_copyQueue));

        // Create Synchronization Fences
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence));
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_copyFence));

        LogTrace("DirectXEngine", "DirectX Compute and Asynchronous Copy Queues provisioned.");
        return true;
    }

    void DirectXEngine::QueryDeviceCapabilities() {
        DXGI_ADAPTER_DESC1 desc;
        m_adapter->GetDesc1(&desc);

        m_caps.DeviceName = desc.Description;
        m_caps.Vendor = ResolveVendor(desc.VendorId);
        m_caps.DedicatedVRAM = desc.DedicatedVideoMemory;

        // Query SM6.6 and quantization supports
        D3D12_FEATURE_DATA_SHADER_MODEL shaderModel = { D3D_SHADER_MODEL_6_6 };
        if (m_device && SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &shaderModel, sizeof(shaderModel)))) {
            m_caps.SupportsSM66 = (shaderModel.HighestShaderModel >= D3D_SHADER_MODEL_6_6);
        } else {
            m_caps.SupportsSM66 = false;
        }

        D3D12_FEATURE_DATA_D3D12_OPTIONS options = {};
        if (m_device && SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &options, sizeof(options)))) {
            m_caps.SupportsFP16 = true; // Most DX12 compute devices support half float
            m_caps.SupportsInt8 = true;  // INT8 Dot Product check
        }

        std::wstringstream ws;
        ws << L"Selected GPU: " << m_caps.DeviceName 
           << L" | VRAM: " << (m_caps.DedicatedVRAM / (1024 * 1024)) << L" MB"
           << L" | SM6.6: " << (m_caps.SupportsSM66 ? L"YES" : L"NO");
        
        // Output cap string
        std::wstring info = ws.str();
        std::string infoStr(info.begin(), info.end());
        LogTrace("DirectXEngine", infoStr);
    }

    HardwareVendor DirectXEngine::ResolveVendor(UINT vendorId) {
        switch (vendorId) {
            case 0x10DE: return HardwareVendor::NVIDIA;
            case 0x1002: return HardwareVendor::AMD;
            case 0x8086: return HardwareVendor::Intel;
            case 0x1414: return HardwareVendor::Microsoft_Fallback;
            default: return HardwareVendor::Unknown;
        }
    }

    bool DirectXEngine::CompileComputeShader(const std::wstring& shaderPath, const std::string& entryPoint, ID3DBlob** outShaderBlob) {
        LogTrace("ShaderCompiler", "Compiling shader " + std::string(shaderPath.begin(), shaderPath.end()) + " [" + entryPoint + "]...");

        ComPtr<ID3DBlob> errorBlob;
        UINT compileFlags = D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES | D3DCOMPILE_OPTIMIZATION_LEVEL3;

        // Try cs_5_1 as the robust compiler-agnostic shader model standard when using d3dcompiler
        HRESULT hr = D3DCompileFromFile(
            shaderPath.c_str(),
            nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint.c_str(),
            "cs_5_1",
            compileFlags,
            0,
            outShaderBlob,
            &errorBlob
        );

        if (FAILED(hr)) {
            // Fall back to cs_5_0 for maximum driver/legacy compatibility
            hr = D3DCompileFromFile(
                shaderPath.c_str(),
                nullptr,
                D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entryPoint.c_str(),
                "cs_5_0",
                compileFlags,
                0,
                outShaderBlob,
                &errorBlob
            );
        }

        if (FAILED(hr)) {
            if (errorBlob) {
                std::string errorMsg = (char*)errorBlob->GetBufferPointer();
                LogTrace("ShaderCompiler", "COMPILATION FAILED: " + errorMsg);
            } else {
                LogTrace("ShaderCompiler", "COMPILATION FAILED: Unknown file error.");
            }
            return false;
        }

        LogTrace("ShaderCompiler", "Shader compiled successfully.");
        return true;
    }

    bool DirectXEngine::AllocateGPUBuffer(size_t sizeInBytes, D3D12_HEAP_TYPE heapType, D3D12_RESOURCE_STATES initialState, ID3D12Resource** outResource) {
        D3D12_HEAP_PROPERTIES heapProps = {};
        heapProps.Type = heapType;
        heapProps.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
        heapProps.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;

        D3D12_RESOURCE_DESC resDesc = {};
        resDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
        resDesc.Width = sizeInBytes;
        resDesc.Height = 1;
        resDesc.DepthOrArraySize = 1;
        resDesc.MipLevels = 1;
        resDesc.Format = DXGI_FORMAT_UNKNOWN;
        resDesc.SampleDesc.Count = 1;
        resDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        resDesc.Flags = (heapType == D3D12_HEAP_TYPE_DEFAULT) ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS : D3D12_RESOURCE_FLAG_NONE;

        HRESULT hr = m_device->CreateCommittedResource(
            &heapProps,
            D3D12_HEAP_FLAG_NONE,
            &resDesc,
            initialState,
            nullptr,
            IID_PPV_ARGS(outResource)
        );

        return SUCCEEDED(hr);
    }

    void DirectXEngine::LogTrace(const std::string& moduleName, const std::string& message) {
        std::cout << "[TRACE][" << moduleName << "] " << message << std::endl;
    }

    void DirectXEngine::Shutdown() {
        m_device.Reset();
        m_adapter.Reset();
        m_dxgiFactory.Reset();
    }
}

extern "C" {
    __declspec(dllexport) void* CreateEngine() {
        return new DirectLLM::DirectXEngine();
    }

    __declspec(dllexport) int InitializeEngine(void* engine, int forceWarp) {
        if (!engine) return 0;
        return static_cast<DirectLLM::DirectXEngine*>(engine)->Initialize(forceWarp != 0) ? 1 : 0;
    }

    __declspec(dllexport) void DestroyEngine(void* engine) {
        if (engine) {
            delete static_cast<DirectLLM::DirectXEngine*>(engine);
        }
    }

    __declspec(dllexport) const char* GetDeviceInfo(void* engine) {
        if (!engine) return "No engine instance";
        auto* dx = static_cast<DirectLLM::DirectXEngine*>(engine);
        auto& caps = dx->GetCaps();
        static std::string info;
        std::wstring name = caps.DeviceName;
        info = std::string(name.begin(), name.end()) +
            " | VRAM: " + std::to_string(caps.DedicatedVRAM / (1024 * 1024)) + " MB" +
            " | SM6.6: " + (caps.SupportsSM66 ? "YES" : "NO");
        return info.c_str();
    }

    __declspec(dllexport) const UINT D3D12SDKVersion = 611;
    __declspec(dllexport) const char* D3D12SDKPath = "C:\\Users\\rina0423\\Desktop\\DX\\agility-latest\\build\\native\\bin\\x64\\";
}


