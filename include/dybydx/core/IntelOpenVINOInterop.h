// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace DirectLLM {

    // IntelOpenVINOInterop
    // -----------------------------------------------------------------------
    // Hardware-agnostic OpenVINO runtime wrapper.
    //
    // Device priority probed at InitializeWithSharedDevice():
    //   1. "NPU"  — Intel AI Boost (Core Ultra 135H, 135U, etc.)
    //   2. "GPU"  — Intel Arc / integrated Xe via OpenCL plugin
    //   3. "CPU"  — any x86 machine (Zen3, Zen4, Core Ultra, etc.)
    //
    // On a Zen3 desktop with an AMD GPU there is no NPU and no Intel GPU,
    // so the CPU plugin activates automatically.  Same binary everywhere.
    //
    // ExecuteSharedOperator():
    //   Reads the input from a D3D12 GPU buffer (READBACK copy),
    //   wraps it in an ov::Tensor, runs the compiled infer request,
    //   then writes the output back to a D3D12 GPU buffer (UPLOAD copy).
    //   This is not zero-copy but is functionally correct on all vendors.
    //   True zero-copy D3D12<->OpenCL interop can be layered on later
    //   via ov::intel_gpu::ocl::D3DContext when targeting Intel GPU plugin.
    class IntelOpenVINOInterop {
    public:
        IntelOpenVINOInterop();
        ~IntelOpenVINOInterop();

        // Probe available OpenVINO devices and bind the D3D12 device.
        // Returns true even if only CPU plugin is available.
        bool InitializeWithSharedDevice(ID3D12Device* d3d12Device);

        // Load an OpenVINO IR model (.xml + .bin) and compile it.
        // deviceTarget is auto-selected if empty ("" -> best available).
        bool LoadModelIR(const std::string& xmlPath,
                         const std::string& binPath,
                         const std::string& deviceTarget = "");

        // Run one inference step.
        // Reads numElements floats from d3d12BufferIn (via readback),
        // runs inference, writes numElements floats to d3d12BufferOut.
        bool ExecuteSharedOperator(ID3D12Resource* d3d12BufferIn,
                                   ID3D12Resource* d3d12BufferOut,
                                   size_t          numElements,
                                   ID3D12Device*   device,
                                   ID3D12CommandQueue* queue,
                                   HANDLE          fenceEvent);

        // Returns the device that will be used for inference.
        const std::string& GetActiveDevice() const { return m_activeDevice; }

        // Returns all available OpenVINO device strings.
        std::vector<std::string> GetAvailableDevices() const;

        bool IsInitialized() const { return m_initialized; }

        void Shutdown();

    private:
        void*  m_ovCore          = nullptr;   // ov::Core*
        void*  m_ovCompiledModel = nullptr;   // ov::CompiledModel*
        void*  m_ovInferRequest  = nullptr;   // ov::InferRequest*

        ID3D12Device* m_sharedDevice = nullptr;
        std::string   m_activeDevice;
        bool          m_initialized  = false;
        bool          m_hasModel     = false;
    };
}
