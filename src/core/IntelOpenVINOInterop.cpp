// DirectLLM C++ Core - (C) 2026 DirectLLM Team
// OpenVINO stub when SDK not available - falls back to CPU-only operation.
#include "dybydx/core/IntelOpenVINOInterop.h"
#include <iostream>
#include <cstring>
#include <vector>

namespace DirectLLM {

    IntelOpenVINOInterop::IntelOpenVINOInterop()
        : m_sharedDevice(nullptr), m_initialized(false), m_hasModel(false) {}

    IntelOpenVINOInterop::~IntelOpenVINOInterop() { Shutdown(); }

    std::vector<std::string> IntelOpenVINOInterop::GetAvailableDevices() const {
        return { "CPU" };
    }

    bool IntelOpenVINOInterop::InitializeWithSharedDevice(ID3D12Device* d3d12Device) {
        m_sharedDevice = d3d12Device;
        std::cout << "[OpenVINO] Stub mode - No OpenVINO SDK found. CPU-only path active." << std::endl;
        m_activeDevice = "CPU";
        m_initialized = true;
        return true;
    }

    bool IntelOpenVINOInterop::LoadModelIR(const std::string& xmlPath,
                                           const std::string& binPath,
                                           const std::string& deviceTarget) {
        std::cerr << "[OpenVINO] Stub mode - LoadModelIR unavailable without OpenVINO SDK" << std::endl;
        return false;
    }

    bool IntelOpenVINOInterop::ExecuteSharedOperator(
            ID3D12Resource*     d3d12BufferIn,
            ID3D12Resource*     d3d12BufferOut,
            size_t              numElements,
            ID3D12Device*       device,
            ID3D12CommandQueue* queue,
            HANDLE              fenceEvent) {
        return false;
    }

    void IntelOpenVINOInterop::Shutdown() {
        m_sharedDevice = nullptr;
        m_initialized = false;
        m_hasModel = false;
    }

} // namespace DirectLLM