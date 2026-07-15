// DirectLLM C++ Core - (C) 2026 DirectLLM Team
// Hardware-agnostic OpenVINO runtime interop.
// Device priority: NPU (Core Ultra) > GPU (Intel Arc) > CPU (any x86)
// Zen3 desktop: no NPU, no Intel GPU -> CPU plugin activates automatically.
// Same binary on all machines.
//
// ExecuteSharedOperator pipeline
// -----------------------------------------------------------------------
//  1. Create a READBACK heap buffer the same size as the input.
//  2. Record a COMPUTE command list:
//       transition input D3D12 buffer UAV->COPY_SOURCE,
//       CopyResource into the readback buffer,
//       restore to UAV.
//  3. Submit + signal fence; wait until GPU is done.
//  4. Map the readback buffer to a CPU float* pointer.
//  5. Wrap that pointer in an ov::Tensor (zero-copy on the CPU side).
//  6. Call infer_request.infer() synchronously.
//  7. Retrieve the output ov::Tensor.
//  8. Create an UPLOAD heap buffer; map it; memcpy output data in.
//  9. Record a second COMPUTE command list:
//       transition output D3D12 buffer UAV->COPY_DEST,
//       CopyResource from upload buffer,
//       restore to UAV.
// 10. Submit + signal a second fence; wait until GPU is done.
//
// This is not zero-copy D3D12<->OpenCL interop (that requires the
// ov::intel_gpu::ocl::D3DContext path, only available for Intel GPU
// plugin). This implementation is functionally correct on AMD, NVIDIA,
// Intel GPU and NPU — every buffer state is explicitly managed.

#include "dybydx/core/IntelOpenVINOInterop.h"
#include <openvino/openvino.hpp>
#include <iostream>
#include <cstring>
#include <vector>
#include <algorithm>

namespace DirectLLM {

    // -----------------------------------------------------------------------
    IntelOpenVINOInterop::IntelOpenVINOInterop()
        : m_ovCore(nullptr), m_ovCompiledModel(nullptr), m_ovInferRequest(nullptr),
          m_sharedDevice(nullptr), m_initialized(false), m_hasModel(false) {}

    IntelOpenVINOInterop::~IntelOpenVINOInterop() { Shutdown(); }

    // -----------------------------------------------------------------------
    //  GetAvailableDevices
    // -----------------------------------------------------------------------
    std::vector<std::string> IntelOpenVINOInterop::GetAvailableDevices() const {
        if (!m_ovCore) return {};
        try {
            return static_cast<ov::Core*>(m_ovCore)->get_available_devices();
        } catch (...) { return {}; }
    }

    // -----------------------------------------------------------------------
    //  InitializeWithSharedDevice
    //  Probes OpenVINO device availability.
    //  Priority: NPU > GPU > CPU.
    //  Returns true even if only CPU is available.
    // -----------------------------------------------------------------------
    bool IntelOpenVINOInterop::InitializeWithSharedDevice(ID3D12Device* d3d12Device) {
        m_sharedDevice = d3d12Device;
        try {
            ov::Core* core = new ov::Core();
            m_ovCore = core;

            auto devices = core->get_available_devices();

            std::cout << "[OpenVINO] Available devices:";
            for (const auto& d : devices) std::cout << " " << d;
            std::cout << std::endl;

            // Priority: NPU > GPU (any variant) > CPU
            m_activeDevice = "CPU"; // CPU is always present
            bool foundNPU = false;
            bool foundGPU = false;
            for (const auto& d : devices) {
                if (d == "NPU" && !foundNPU) {
                    m_activeDevice = "NPU";
                    foundNPU = true;
                } else if (d.rfind("GPU", 0) == 0 && !foundNPU && !foundGPU) {
                    // Prefer GPU.0 over GPU.1 etc.; take the first GPU encountered
                    m_activeDevice = d;
                    foundGPU = true;
                }
            }

            std::cout << "[OpenVINO] Selected device: " << m_activeDevice << std::endl;
            m_initialized = true;
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "[OpenVINO] Init failed: " << ex.what() << std::endl;
            return false;
        }
    }

    // -----------------------------------------------------------------------
    //  LoadModelIR
    //  Reads an OpenVINO IR pair (.xml + .bin) and compiles it for the
    //  selected (or caller-supplied) device with appropriate performance hints.
    // -----------------------------------------------------------------------
    bool IntelOpenVINOInterop::LoadModelIR(const std::string& xmlPath,
                                            const std::string& binPath,
                                            const std::string& deviceTarget) {
        if (!m_ovCore) return false;
        auto* core   = static_cast<ov::Core*>(m_ovCore);
        std::string target = deviceTarget.empty() ? m_activeDevice : deviceTarget;

        try {
            std::cout << "[OpenVINO] Loading IR: " << xmlPath
                      << "  (compiling for " << target << ")" << std::endl;

            auto model = core->read_model(xmlPath, binPath);

            ov::AnyMap props;
            if (target == "NPU") {
                // NPU: latency mode with fixed shapes gives best throughput on Core Ultra.
                props[ov::hint::performance_mode.name()] =
                    ov::hint::PerformanceMode::LATENCY;
            } else if (target.rfind("GPU", 0) == 0) {
                // Intel GPU plugin: throughput for batched workloads, latency for decode.
                props[ov::hint::performance_mode.name()] =
                    ov::hint::PerformanceMode::LATENCY;
            } else {
                // CPU: latency mode, use all hardware threads (0 == auto).
                props[ov::hint::performance_mode.name()] =
                    ov::hint::PerformanceMode::LATENCY;
                props[ov::inference_num_threads.name()] = 0;
            }

            auto* compiled = new ov::CompiledModel(
                core->compile_model(model, target, props));
            m_ovCompiledModel = compiled;

            auto* req = new ov::InferRequest(compiled->create_infer_request());
            m_ovInferRequest = req;

            m_hasModel = true;
            std::cout << "[OpenVINO] Compiled on " << target << " — OK." << std::endl;
            return true;
        } catch (const std::exception& ex) {
            std::cerr << "[OpenVINO] LoadModelIR failed: " << ex.what() << std::endl;
            return false;
        }
    }

    // -----------------------------------------------------------------------
    //  Internal helpers
    // -----------------------------------------------------------------------
    namespace {

        // Create a fresh command allocator + list of the given type.
        static bool MakeCmdList(ID3D12Device* device,
                                D3D12_COMMAND_LIST_TYPE type,
                                ComPtr<ID3D12CommandAllocator>&    outAlloc,
                                ComPtr<ID3D12GraphicsCommandList>& outList) {
            if (FAILED(device->CreateCommandAllocator(type, IID_PPV_ARGS(&outAlloc))))
                return false;
            if (FAILED(device->CreateCommandList(0, type,
                    outAlloc.Get(), nullptr, IID_PPV_ARGS(&outList))))
                return false;
            return true;
        }

        // Signal fence value 1 on the queue and wait until it completes.
        // Always creates a fresh fence so signal value 1 is meaningful.
        static bool FlushQueue(ID3D12Device*       device,
                               ID3D12CommandQueue* queue,
                               HANDLE              fenceEvent) {
            ComPtr<ID3D12Fence> fence;
            if (FAILED(device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                    IID_PPV_ARGS(&fence))))
                return false;
            if (FAILED(queue->Signal(fence.Get(), 1)))
                return false;
            if (fence->GetCompletedValue() < 1) {
                fence->SetEventOnCompletion(1, fenceEvent);
                WaitForSingleObject(fenceEvent, INFINITE);
            }
            return true;
        }

        // Create a committed buffer on a specified heap type.
        static bool MakeCommittedBuffer(ID3D12Device*          device,
                                        size_t                 byteSize,
                                        D3D12_HEAP_TYPE        heapType,
                                        D3D12_RESOURCE_STATES  initialState,
                                        ComPtr<ID3D12Resource>& outBuf) {
            D3D12_HEAP_PROPERTIES heap = {};
            heap.Type = heapType;

            D3D12_RESOURCE_DESC desc = {};
            desc.Dimension        = D3D12_RESOURCE_DIMENSION_BUFFER;
            desc.Alignment        = 0;
            desc.Width            = byteSize;
            desc.Height           = 1;
            desc.DepthOrArraySize = 1;
            desc.MipLevels        = 1;
            desc.Format           = DXGI_FORMAT_UNKNOWN;
            desc.SampleDesc.Count = 1;
            desc.Layout           = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
            desc.Flags            = D3D12_RESOURCE_FLAG_NONE;

            return SUCCEEDED(device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &desc,
                initialState, nullptr, IID_PPV_ARGS(&outBuf)));
        }

        // Emit a single transition barrier.
        static void Transition(ID3D12GraphicsCommandList* list,
                               ID3D12Resource*            res,
                               D3D12_RESOURCE_STATES      before,
                               D3D12_RESOURCE_STATES      after) {
            D3D12_RESOURCE_BARRIER b = {};
            b.Type                   = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
            b.Flags                  = D3D12_RESOURCE_BARRIER_FLAG_NONE;
            b.Transition.pResource   = res;
            b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            b.Transition.StateBefore = before;
            b.Transition.StateAfter  = after;
            list->ResourceBarrier(1, &b);
        }

    } // anonymous namespace

    // -----------------------------------------------------------------------
    //  ExecuteSharedOperator
    // -----------------------------------------------------------------------
    bool IntelOpenVINOInterop::ExecuteSharedOperator(
            ID3D12Resource*     d3d12BufferIn,
            ID3D12Resource*     d3d12BufferOut,
            size_t              numElements,
            ID3D12Device*       device,
            ID3D12CommandQueue* queue,
            HANDLE              fenceEvent) {

        if (!m_hasModel || !m_ovInferRequest) {
            return true; // No model loaded, skip silently
        }
        if (!d3d12BufferIn || !d3d12BufferOut || !device || !queue || !fenceEvent) {
            std::cerr << "[OpenVINO] ExecuteSharedOperator: null argument." << std::endl;
            return false;
        }

        const size_t inputBytes = numElements * sizeof(float);

        // ====================================================================
        // PHASE 1 — GPU input buffer -> CPU (READBACK)
        // ====================================================================

        // 1a. Allocate readback buffer (D3D12_HEAP_TYPE_READBACK starts in COPY_DEST).
        ComPtr<ID3D12Resource> readbackBuf;
        if (!MakeCommittedBuffer(device, inputBytes,
                D3D12_HEAP_TYPE_READBACK,
                D3D12_RESOURCE_STATE_COPY_DEST, readbackBuf)) {
            std::cerr << "[OpenVINO] Failed to create readback buffer." << std::endl;
            return false;
        }

        // 1b. Record: UAV -> COPY_SOURCE, copy, restore.
        ComPtr<ID3D12CommandAllocator>    alloc1;
        ComPtr<ID3D12GraphicsCommandList> list1;
        if (!MakeCmdList(device, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc1, list1)) {
            std::cerr << "[OpenVINO] Failed to create copy-in command list." << std::endl;
            return false;
        }

        Transition(list1.Get(), d3d12BufferIn,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        list1->CopyResource(readbackBuf.Get(), d3d12BufferIn);
        Transition(list1.Get(), d3d12BufferIn,
                   D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

        if (FAILED(list1->Close())) return false;

        {
            ID3D12CommandList* cls[] = { list1.Get() };
            queue->ExecuteCommandLists(1, cls);
        }

        if (!FlushQueue(device, queue, fenceEvent)) {
            std::cerr << "[OpenVINO] FlushQueue (readback) failed." << std::endl;
            return false;
        }

        // 1c. Map the readback buffer — valid after the GPU copy completed.
        void* pMappedIn = nullptr;
        D3D12_RANGE readRange{ 0, inputBytes };
        if (FAILED(readbackBuf->Map(0, &readRange, &pMappedIn))) {
            std::cerr << "[OpenVINO] Map(readback) failed." << std::endl;
            return false;
        }
        const float* inputPtr = reinterpret_cast<const float*>(pMappedIn);

        // ====================================================================
        // PHASE 2 — Run OpenVINO inference
        // ====================================================================

        auto* req     = static_cast<ov::InferRequest*>(m_ovInferRequest);
        auto* compiled = static_cast<ov::CompiledModel*>(m_ovCompiledModel);

        bool inferOk = false;
        size_t outBytes = 0;
        ComPtr<ID3D12Resource> uploadBuf;

        try {
            // Wrap the mapped CPU memory in an ov::Tensor — no copy.
            // Shape [1, numElements] matches the most common single-vector operator.
            ov::Shape inputShape{ 1, numElements };
            ov::Tensor inputTensor(ov::element::f32, inputShape,
                                   const_cast<float*>(inputPtr));
            req->set_input_tensor(inputTensor);

            req->infer(); // synchronous on CPU/GPU/NPU

            // Retrieve output tensor.
            ov::Tensor outputTensor = req->get_output_tensor();
            const float* outPtr = outputTensor.data<float>();
            outBytes             = outputTensor.get_size() * sizeof(float);

            // Unmap input before we're done with the CPU-side buffer.
            D3D12_RANGE writeRange{ 0, 0 }; // we never wrote back to the readback buf
            readbackBuf->Unmap(0, &writeRange);
            pMappedIn = nullptr;

            // ================================================================
            // PHASE 3 — CPU output -> GPU output buffer (UPLOAD -> CopyResource)
            // ================================================================

            // 3a. Create UPLOAD buffer and fill it.
            if (!MakeCommittedBuffer(device, outBytes,
                    D3D12_HEAP_TYPE_UPLOAD,
                    D3D12_RESOURCE_STATE_GENERIC_READ, uploadBuf)) {
                std::cerr << "[OpenVINO] Failed to create upload buffer." << std::endl;
                return false;
            }

            void* pMappedOut = nullptr;
            if (FAILED(uploadBuf->Map(0, nullptr, &pMappedOut))) {
                std::cerr << "[OpenVINO] Map(upload) failed." << std::endl;
                return false;
            }
            std::memcpy(pMappedOut, outPtr, outBytes);
            D3D12_RANGE written{ 0, outBytes };
            uploadBuf->Unmap(0, &written);

            // 3b. Record: UAV -> COPY_DEST, copy from upload, restore.
            ComPtr<ID3D12CommandAllocator>    alloc2;
            ComPtr<ID3D12GraphicsCommandList> list2;
            if (!MakeCmdList(device, D3D12_COMMAND_LIST_TYPE_COMPUTE, alloc2, list2)) {
                std::cerr << "[OpenVINO] Failed to create copy-out command list." << std::endl;
                return false;
            }

            Transition(list2.Get(), d3d12BufferOut,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                       D3D12_RESOURCE_STATE_COPY_DEST);
            list2->CopyResource(d3d12BufferOut, uploadBuf.Get());
            Transition(list2.Get(), d3d12BufferOut,
                       D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_UNORDERED_ACCESS);

            if (FAILED(list2->Close())) return false;

            {
                ID3D12CommandList* cls2[] = { list2.Get() };
                queue->ExecuteCommandLists(1, cls2);
            }

            if (!FlushQueue(device, queue, fenceEvent)) {
                std::cerr << "[OpenVINO] FlushQueue (upload) failed." << std::endl;
                return false;
            }

            inferOk = true;
        } catch (const std::exception& ex) {
            std::cerr << "[OpenVINO] ExecuteSharedOperator infer failed: "
                      << ex.what() << std::endl;
        }

        // Safety: make sure we always unmap if an exception fired after mapping.
        if (pMappedIn) {
            D3D12_RANGE writeRangeNone{ 0, 0 };
            readbackBuf->Unmap(0, &writeRangeNone);
        }

        return inferOk;
    }

    // -----------------------------------------------------------------------
    //  Shutdown
    // -----------------------------------------------------------------------
    void IntelOpenVINOInterop::Shutdown() {
        if (m_ovInferRequest) {
            delete static_cast<ov::InferRequest*>(m_ovInferRequest);
            m_ovInferRequest = nullptr;
        }
        if (m_ovCompiledModel) {
            delete static_cast<ov::CompiledModel*>(m_ovCompiledModel);
            m_ovCompiledModel = nullptr;
        }
        if (m_ovCore) {
            delete static_cast<ov::Core*>(m_ovCore);
            m_ovCore = nullptr;
        }
        m_sharedDevice = nullptr;
        m_initialized  = false;
        m_hasModel     = false;
    }

} // namespace DirectLLM
