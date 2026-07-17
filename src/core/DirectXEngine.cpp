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

    DirectXEngine::DirectXEngine() { m_caps = {}; }
    DirectXEngine::~DirectXEngine() { Shutdown(); }

    bool DirectXEngine::Initialize(bool forceWarp) {
        CoInitializeEx(nullptr, COINIT_MULTITHREADED);
        LogTrace("DirectXEngine", "Initializing DirectX 12 via Agility SDK...");

        UINT dxgiFactoryFlags = 0;
        // Previously this was gated behind #ifdef _DEBUG, which silently did nothing in
        // Release builds - setting $env:D3D12_DEBUG=1 had no effect at all, since no code
        // ever read that variable. That meant a bad resource binding or root signature
        // mismatch skipped validation entirely and only surfaced as a raw, undiagnosable
        // access violation deep in D3D12Core.dll instead of a clear D3D12 validation
        // message naming the actual call and resource at fault.
        bool wantDebugLayer = false;
        {
            char buf[8] = {};
            size_t len = 0;
            if (getenv_s(&len, buf, sizeof(buf), "D3D12_DEBUG") == 0 && len > 0) {
                wantDebugLayer = (buf[0] != '0');
            }
        }
#ifdef _DEBUG
        wantDebugLayer = true; // always on for debug builds regardless of env var
#endif
        if (wantDebugLayer) {
            ComPtr<ID3D12Debug> dbg;
            if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dbg)))) {
                dbg->EnableDebugLayer();
                LogTrace("DirectXEngine", "D3D12 Debug Layer enabled.");

                // GPU-based validation catches invalid resource states/bindings that
                // object-level validation alone can miss (exactly the class of bug that
                // produces a bare access violation instead of a clean error message).
                ComPtr<ID3D12Debug1> dbg1;
                if (SUCCEEDED(dbg.As(&dbg1))) {
                    dbg1->SetEnableGPUBasedValidation(TRUE);
                    LogTrace("DirectXEngine", "D3D12 GPU-Based Validation enabled.");
                }
            } else {
                LogTrace("DirectXEngine", "D3D12_DEBUG requested but debug interface unavailable (Graphics Tools optional feature not installed?).");
            }
        }

        if (FAILED(CreateDXGIFactory2(dxgiFactoryFlags, IID_PPV_ARGS(&m_dxgiFactory)))) {
            LogTrace("DirectXEngine", "CRITICAL: Failed to create DXGI Factory.");
            return false;
        }

        ComPtr<IDXGIAdapter1> adapter1;
        if (forceWarp) {
            m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1));
        } else {
            for (UINT i = 0;
                 DXGI_ERROR_NOT_FOUND != m_dxgiFactory->EnumAdapterByGpuPreference(
                     i, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&adapter1));
                 ++i) {
                DXGI_ADAPTER_DESC1 d;
                adapter1->GetDesc1(&d);
                if (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) continue;
                break;
            }
        }
        if (!adapter1) {
            LogTrace("DirectXEngine", "No hardware GPU found, falling back to WARP.");
            m_dxgiFactory->EnumWarpAdapter(IID_PPV_ARGS(&adapter1));
        }
        if (!adapter1) { LogTrace("DirectXEngine", "CRITICAL: No D3D12 adapter."); return false; }

        adapter1.As(&m_adapter);
        if (!m_adapter) { LogTrace("DirectXEngine", "Adapter promotion failed."); return false; }

        // QueryDeviceCapabilities needs the device — create device first
        HRESULT hr = D3D12CreateDevice(m_adapter.Get(), D3D_FEATURE_LEVEL_12_0, IID_PPV_ARGS(&m_device));
        if (FAILED(hr)) { LogTrace("DirectXEngine", "CRITICAL: D3D12CreateDevice failed."); return false; }

        if (wantDebugLayer) {
            // Print debug layer messages straight to stderr instead of requiring a
            // debugger attached or DebugView to see them via OutputDebugString.
            ComPtr<ID3D12InfoQueue> infoQueue;
            if (SUCCEEDED(m_device.As(&infoQueue))) {
                infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
                infoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);

                ComPtr<ID3D12InfoQueue1> infoQueue1;
                if (SUCCEEDED(infoQueue.As(&infoQueue1))) {
                    DWORD cookie = 0;
                    infoQueue1->RegisterMessageCallback(
                        [](D3D12_MESSAGE_CATEGORY, D3D12_MESSAGE_SEVERITY severity,
                           D3D12_MESSAGE_ID, LPCSTR pDescription, void*) {
                            if (severity <= D3D12_MESSAGE_SEVERITY_WARNING) {
                                std::cerr << "[D3D12" << (severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ? "-CORRUPTION" :
                                              severity == D3D12_MESSAGE_SEVERITY_ERROR ? "-ERROR" : "-WARN")
                                          << "] " << pDescription << std::endl;
                            }
                        },
                        D3D12_MESSAGE_CALLBACK_FLAG_NONE, nullptr, &cookie);
                    LogTrace("DirectXEngine", "D3D12 InfoQueue message callback registered.");
                } else {
                    LogTrace("DirectXEngine", "ID3D12InfoQueue1 unavailable (older Agility SDK?) - debug messages will only reach a debugger/DebugView, not this console.");
                }
            }
        }

        // Now query capabilities (needs device for feature checks)
        QueryDeviceCapabilities();

        // Compute queue — DIRECT type, not COMPUTE: on AMD RDNA4 (RX 9070 XT,
        // driver 25.30.x) COMPUTE-type queues/lists cause spurious device removal.
        D3D12_COMMAND_QUEUE_DESC cqd = {};
        cqd.Type     = D3D12_COMMAND_LIST_TYPE_DIRECT;
        cqd.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        m_device->CreateCommandQueue(&cqd, IID_PPV_ARGS(&m_computeQueue));

        // Copy queue (async DMA, separate from compute)
        D3D12_COMMAND_QUEUE_DESC cpqd = {};
        cpqd.Type  = D3D12_COMMAND_LIST_TYPE_COPY;
        cpqd.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        m_device->CreateCommandQueue(&cpqd, IID_PPV_ARGS(&m_copyQueue));

        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_computeFence));
        m_device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&m_copyFence));

        LogTrace("DirectXEngine", "D3D12 device + queues ready.");
        return true;
    }

    void DirectXEngine::QueryDeviceCapabilities() {
        DXGI_ADAPTER_DESC1 desc;
        m_adapter->GetDesc1(&desc);
        m_caps.DeviceName    = desc.Description;
        m_caps.Vendor        = ResolveVendor(desc.VendorId);
        m_caps.DedicatedVRAM = desc.DedicatedVideoMemory;

        // Shader Model
        D3D12_FEATURE_DATA_SHADER_MODEL sm = { D3D_SHADER_MODEL_6_6 };
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &sm, sizeof(sm))))
            m_caps.SupportsSM66 = (sm.HighestShaderModel >= D3D_SHADER_MODEL_6_6);

        // FP16 / INT8
        D3D12_FEATURE_DATA_D3D12_OPTIONS opts = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS, &opts, sizeof(opts)))) {
            m_caps.SupportsFP16 = true;
            m_caps.SupportsInt8 = true;
        }

        // ---------------------------------------------------------------
        //  PCIe Resizable BAR / Smart Access Memory detection
        //  D3D12_FEATURE_DATA_D3D12_OPTIONS16::GPUUploadHeapSupported
        //  Agility SDK 1.611+.  When true:
        //   - D3D12_HEAP_TYPE_GPU_UPLOAD resources live in VRAM
        //   - CPU Map()/writes go directly to VRAM over full BAR
        //   - Eliminates UPLOAD-heap staging copy for weight loads
        //   - Works on: AMD SAM (RDNA2+), NVIDIA ReBAR, Intel Arc
        // ---------------------------------------------------------------
        D3D12_FEATURE_DATA_D3D12_OPTIONS16 opts16 = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS16, &opts16, sizeof(opts16)))) {
            m_caps.SupportsReBAR = opts16.GPUUploadHeapSupported;
        }

        // Build capability string
        std::wstring name = m_caps.DeviceName;
        std::string nameStr(name.begin(), name.end());
        LogTrace("DirectXEngine",
            "GPU: "        + nameStr +
            " | VRAM: "   + std::to_string(m_caps.DedicatedVRAM / (1024*1024)) + " MB" +
            " | SM6.6: "  + (m_caps.SupportsSM66  ? "YES" : "NO") +
            " | ReBAR: "  + (m_caps.SupportsReBAR  ? "YES (GPU_UPLOAD active)" : "NO (staging path)"));
    }

    HardwareVendor DirectXEngine::ResolveVendor(UINT vendorId) {
        switch (vendorId) {
            case 0x10DE: return HardwareVendor::NVIDIA;
            case 0x1002: return HardwareVendor::AMD;
            case 0x8086: return HardwareVendor::Intel;
            case 0x1414: return HardwareVendor::Microsoft_Fallback;
            default:     return HardwareVendor::Unknown;
        }
    }

    // -----------------------------------------------------------------------
    //  AllocateReBarBuffer
    //  Creates a CPU-writable, GPU-fast buffer.
    //  With ReBAR: GPU_UPLOAD heap = direct VRAM writes, no staging copy.
    //  Without ReBAR: falls back to standard UPLOAD heap (still works).
    // -----------------------------------------------------------------------
    bool DirectXEngine::AllocateReBarBuffer(size_t sizeInBytes, ID3D12Resource** outResource) {
        D3D12_HEAP_PROPERTIES hp = {};
        if (m_caps.SupportsReBAR) {
            // True ReBAR: CPU-writable VRAM
            hp.Type = D3D12_HEAP_TYPE_GPU_UPLOAD;
        } else {
            // Fallback: standard upload heap (DRAM, not VRAM)
            hp.Type = D3D12_HEAP_TYPE_UPLOAD;
        }

        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width            = sizeInBytes;
        rd.Height           = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.Format           = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags            = D3D12_RESOURCE_FLAG_NONE;

        // GPU_UPLOAD starts in GENERIC_READ; UPLOAD also uses GENERIC_READ
        HRESULT hr = m_device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd,
            D3D12_RESOURCE_STATE_GENERIC_READ, nullptr,
            IID_PPV_ARGS(outResource));
        return SUCCEEDED(hr);
    }

    bool DirectXEngine::AllocateGPUBuffer(size_t sizeInBytes, D3D12_HEAP_TYPE heapType,
                                           D3D12_RESOURCE_STATES initialState,
                                           ID3D12Resource** outResource) {
        D3D12_HEAP_PROPERTIES hp = {};
        hp.Type = heapType;

        D3D12_RESOURCE_DESC rd = {};
        rd.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
        rd.Width            = sizeInBytes;
        rd.Height           = 1;
        rd.DepthOrArraySize = 1;
        rd.MipLevels        = 1;
        rd.Format           = DXGI_FORMAT_UNKNOWN;
        rd.SampleDesc.Count = 1;
        rd.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
        rd.Flags = (heapType == D3D12_HEAP_TYPE_DEFAULT)
                   ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                   : D3D12_RESOURCE_FLAG_NONE;

        HRESULT hr = m_device->CreateCommittedResource(
            &hp, D3D12_HEAP_FLAG_NONE, &rd, initialState, nullptr,
            IID_PPV_ARGS(outResource));
        return SUCCEEDED(hr);
    }

    bool DirectXEngine::CompileComputeShader(const std::wstring& shaderPath,
                                              const std::string&  entryPoint,
                                              ID3DBlob**          outShaderBlob) {
        LogTrace("ShaderCompiler", "Compiling " +
            std::string(shaderPath.begin(), shaderPath.end()) + " [" + entryPoint + "]...");

        ComPtr<IDxcUtils>     utils;
        ComPtr<IDxcCompiler3> compiler;
        HRESULT hr = DxcCreateInstance(CLSID_DxcUtils,     IID_PPV_ARGS(&utils));
        if (SUCCEEDED(hr)) DxcCreateInstance(CLSID_DxcCompiler, IID_PPV_ARGS(&compiler));

        if (SUCCEEDED(hr) && compiler) {
            std::ifstream file(shaderPath);
            if (!file.is_open()) { LogTrace("ShaderCompiler", "Cannot open shader."); return false; }
            std::string src((std::istreambuf_iterator<char>(file)), {});

            ComPtr<IDxcBlobEncoding> blob;
            utils->CreateBlob(src.c_str(), (UINT32)src.size(), CP_UTF8, &blob);

            DxcBuffer buf = { blob->GetBufferPointer(), blob->GetBufferSize(), 0 };

            std::wstring ep(entryPoint.begin(), entryPoint.end());
            LPCWSTR args[] = { L"-E", ep.c_str(), L"-T", L"cs_6_6", L"-O3" };

            ComPtr<IDxcResult> res;
            hr = compiler->Compile(&buf, args, 5, nullptr, IID_PPV_ARGS(&res));
            if (SUCCEEDED(hr)) {
                HRESULT status;
                res->GetStatus(&status);
                if (SUCCEEDED(status)) {
                    ComPtr<IDxcBlob> sb;
                    res->GetOutput(DXC_OUT_OBJECT, IID_PPV_ARGS(&sb), nullptr);
                    if (sb) {
                        D3DCreateBlob(sb->GetBufferSize(), outShaderBlob);
                        if (*outShaderBlob)
                            memcpy((*outShaderBlob)->GetBufferPointer(),
                                   sb->GetBufferPointer(), sb->GetBufferSize());
                        LogTrace("ShaderCompiler", "DXC SM6 OK (" +
                            std::to_string(sb->GetBufferSize()) + " bytes).");
                        return *outShaderBlob != nullptr;
                    }
                } else {
                    ComPtr<IDxcBlobUtf8> err;
                    res->GetOutput(DXC_OUT_ERRORS, IID_PPV_ARGS(&err), nullptr);
                    if (err && err->GetStringLength() > 0)
                        LogTrace("ShaderCompiler", "DXC FAILED: " + std::string(err->GetStringPointer()));
                }
            }
        }

        // Fallback: legacy d3dcompiler cs_5_1
        LogTrace("ShaderCompiler", "Falling back to d3dcompiler cs_5_1...");
        ComPtr<ID3DBlob> errBlob;
        UINT flags = D3DCOMPILE_OPTIMIZATION_LEVEL3;
        hr = D3DCompileFromFile(shaderPath.c_str(), nullptr,
            D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint.c_str(), "cs_5_1",
            flags, 0, outShaderBlob, &errBlob);
        if (FAILED(hr)) {
            hr = D3DCompileFromFile(shaderPath.c_str(), nullptr,
                D3D_COMPILE_STANDARD_FILE_INCLUDE, entryPoint.c_str(), "cs_5_0",
                flags, 0, outShaderBlob, &errBlob);
        }
        if (FAILED(hr)) {
            if (errBlob) LogTrace("ShaderCompiler",
                "FALLBACK FAILED: " + std::string((char*)errBlob->GetBufferPointer()));
            return false;
        }
        LogTrace("ShaderCompiler", "d3dcompiler fallback OK.");
        return true;
    }

    bool DirectXEngine::CreateComputePipelineState(ID3DBlob* shaderBlob,
                                                    ID3D12RootSignature* rootSignature,
                                                    ID3D12PipelineState** outPSO) {
        if (!shaderBlob || !rootSignature || !m_device) return false;
        D3D12_COMPUTE_PIPELINE_STATE_DESC d = {};
        d.pRootSignature    = rootSignature;
        d.CS.pShaderBytecode = shaderBlob->GetBufferPointer();
        d.CS.BytecodeLength  = shaderBlob->GetBufferSize();
        HRESULT hr = m_device->CreateComputePipelineState(&d, IID_PPV_ARGS(outPSO));
        if (FAILED(hr)) { LogTrace("DirectXEngine", "CreateComputePipelineState FAILED."); return false; }
        return true;
    }

    void DirectXEngine::LogTrace(const std::string& mod, const std::string& msg) {
        std::cout << "[" << mod << "] " << msg << std::endl;
    }

    void DirectXEngine::Shutdown() {
        m_computeQueue.Reset();
        m_copyQueue.Reset();
        m_computeFence.Reset();
        m_copyFence.Reset();
        m_device.Reset();
        m_adapter.Reset();
        m_dxgiFactory.Reset();
    }
}

extern "C" {
    __declspec(dllexport) void* CreateEngine() {
        return new DirectLLM::DirectXEngine();
    }
    __declspec(dllexport) int InitializeEngine(void* e, int forceWarp) {
        if (!e) return 0;
        return static_cast<DirectLLM::DirectXEngine*>(e)->Initialize(forceWarp != 0) ? 1 : 0;
    }
    __declspec(dllexport) void DestroyEngine(void* e) {
        if (e) delete static_cast<DirectLLM::DirectXEngine*>(e);
    }
    __declspec(dllexport) const char* GetDeviceInfo(void* e) {
        if (!e) return "No engine";
        auto* dx = static_cast<DirectLLM::DirectXEngine*>(e);
        auto& c  = dx->GetCaps();
        static std::string s;
        std::wstring n = c.DeviceName;
        s = std::string(n.begin(), n.end()) +
            " | VRAM: " + std::to_string(c.DedicatedVRAM / (1024*1024)) + " MB" +
            " | ReBAR: " + (c.SupportsReBAR ? "YES" : "NO");
        return s.c_str();
    }
    __declspec(dllexport) const UINT D3D12SDKVersion = 611;
    __declspec(dllexport) const char* D3D12SDKPath =
        "./agility-sdk/bin/x64/";
}
