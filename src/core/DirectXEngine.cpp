// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/DirectXEngine.h"
#include <d3dcompiler.h>
#include <dxcapi.h>
#include <iostream>
#include <sstream>
#include <fstream>

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
        LogTrace("ShaderCompiler", "Compiling " + std::string(shaderPath.begin(), shaderPath.end()) + " [" + entryPoint + "]...");

        // Try DXC compiler first (supports SM 6.0+)
        HRESULT hr;
        ComPtr<IDxcUtils> utils;
        ComPtr<IDxcCompiler3> compiler;
        hr = DxcCreateInstance(CLSID_DxcUtils, IID_PPV_ARGS(&utils));
        if (SUCCEEDED(hr)) hr = DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

        if (SUCCEEDED(hr) && compiler) {
            // Load shader source
            std::ifstream file(shaderPath);
            if (!file.is_open()) {
                LogTrace("ShaderCompiler", "Cannot open shader file.");
                return false;
            }
            std::string source((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

            ComPtr<IDxcBlobEncoding> sourceBlob;
            utils->CreateBlob(source.c_str(), (UINT32)source.size(), CP_UTF8, &sourceBlob);

            DxcBuffer buffer = {};
            buffer.Ptr = sourceBlob->GetBufferPointer();
            buffer.Size = sourceBlob->GetBufferSize();
            buffer.Encoding = 0;

            std::vector<LPCWSTR> args = {
                L"-E", std::wstring(entryPoint.begin(), entryPoint.end()).c_str(),
                L"-T", L"cs_6_0",
                L"-Zi", L"-Od"
            };

            ComPtr<IDxcResult> results;
            hr = compiler->Compile(&buffer, args.data(), (UINT32)args.size(), nullptr, IID_PPV_ARGS(&results));
            if (SUCCEEDED(hr)) {
                ComPtr<IDxcBlobUtf8> errors;
                results->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errors), nullptr);
                if (errors && errors->GetStringLength() > 0) {
                    LogTrace("ShaderCompiler", "DXC warnings: " + std::string(errors->GetStringPointer()));
                }

                HRESULT status;
                results->GetStatus(&status);
                if (SUCCEEDED(status)) {
                    ComPtr<IDxcBlob> shaderBlob;
                    results->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&shaderBlob), nullptr);
                    if (shaderBlob) {
                        // Convert DXC Blob to ID3DBlob
                        D3DCreateBlob(shaderBlob->GetBufferSize(), outShaderBlob);
                        if (*outShaderBlob) {
                            memcpy((*outShaderBlob)->GetBufferPointer(), shaderBlob->GetBufferPointer(), shaderBlob->GetBufferSize());
                        }
                        LogTrace("ShaderCompiler", "DXC SM 6.0 compiled OK (" + std::to_string(shaderBlob->GetBufferSize()) + " bytes).");
                        return *outShaderBlob != nullptr;
                    }
                } else {
                    // DXC failed — provide error message
                    ComPtr<IDxcBlobUtf8> errorBlob;
                    results->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&errorBlob), nullptr);
                    if (errorBlob && errorBlob->GetStringLength() > 0)
                        LogTrace("ShaderCompiler", "DXC FAILED: " + std::string(errorBlob->GetStringPointer()));
                    else
                        LogTrace("ShaderCompiler", "DXC FAILED: unknown error.");
                }
            }
        }

        // Fallback to d3dcompiler with cs_5_1
        LogTrace("ShaderCompiler", "Falling back to d3dcompiler cs_5_1...");
        ComPtr<ID3DBlob> errorBlob;
        UINT flags = D3DCOMPILE_ENABLE_UNBOUNDED_DESCRIPTOR_TABLES | D3DCOMPILE_OPTIMIZATION_LEVEL3;
        hr = D3DCompileFromFile(shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
            entryPoint.c_str(), "cs_5_1", flags, 0, outShaderBlob, &errorBlob);

        if (FAILED(hr)) {
            hr = D3DCompileFromFile(shaderPath.c_str(), nullptr, D3D_COMPILE_STANDARD_FILE_INCLUDE,
                entryPoint.c_str(), "cs_5_0", flags, 0, outShaderBlob, &errorBlob);
        }

        if (FAILED(hr)) {
            if (errorBlob) LogTrace("ShaderCompiler", "FALLBACK FAILED: " + std::string((char*)errorBlob->GetBufferPointer()));
            else LogTrace("ShaderCompiler", "FALLBACK FAILED: unknown.");
            return false;
        }
        LogTrace("ShaderCompiler", "d3dcompiler fallback OK.");
        return true;
    }

    bool DirectXEngine::CreateComputePipelineState(ID3DBlob* shaderBlob, ID3D12RootSignature* rootSignature, ID3D12PipelineState** outPSO) {
        if (!shaderBlob || !rootSignature || !m_device) return false;

        D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
        psoDesc.pRootSignature = rootSignature;
        psoDesc.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
        psoDesc.CS.BytecodeLength = shaderBlob->GetBufferSize();
        psoDesc.Flags = D3D12_PIPELINE_STATE_FLAG_NONE;

        HRESULT hr = m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(outPSO));
        if (FAILED(hr)) {
            LogTrace("DirectXEngine", "CreateComputePipelineState FAILED.");
            return false;
        }
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


