// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#include "dybydx/core/AdvancedVendorOptimizations.h"
#include <iostream>
#include <cmath>
#include <cstring>
#include <vector>

namespace DirectLLM {

    AdvancedVendorOptimizations::AdvancedVendorOptimizations() {}

    AdvancedVendorOptimizations::~AdvancedVendorOptimizations() {
        m_attnPSO.Reset();
        m_attnRootSignature.Reset();
        m_moePSO.Reset();
        m_moeRootSignature.Reset();
        m_tqPSO.Reset();
        m_tqRootSignature.Reset();
    }

    bool AdvancedVendorOptimizations::Initialize(DirectXEngine* dxEngine, const OffloadConfig& offloadConfig) {
        m_dxEngine = dxEngine;
        m_config = offloadConfig;
        if (m_dxEngine) {
            m_device = m_dxEngine->GetDevice();
        }

        DetectHardwarePipeline();
        LogOptimizationSpecs();

        if (m_dxEngine && m_device) {
            if (!CompileAndBuildPipelines()) {
                std::cerr << "[AdvancedOptimizations] Failed to compile and build TurboKernels pipelines via DXC." << std::endl;
                return false;
            }
        }

        return true;
    }

    void AdvancedVendorOptimizations::DetectHardwarePipeline() {
        if (!m_device) return;

        D3D12_FEATURE_DATA_ARCHITECTURE arch = {};
        if (SUCCEEDED(m_device->CheckFeatureSupport(D3D12_FEATURE_ARCHITECTURE, &arch, sizeof(arch)))) {
            m_activePipeline = OptimizationPipelineType::StandardD3D12Compute;
        } else {
            m_activePipeline = OptimizationPipelineType::StandardD3D12Compute;
        }
    }

    bool AdvancedVendorOptimizations::CompileAndBuildPipelines() {
        if (!m_dxEngine) return false;

        ComPtr<ID3DBlob> dflashBlob;
        ComPtr<ID3DBlob> dsparkBlob;
        ComPtr<ID3DBlob> tqBlob;

        std::wstring shaderPath = L"shaders/TurboKernels.hlsl";

        // Compile compute shaders using the DXC SM6 pipeline (built-in to DirectXEngine)
        if (!m_dxEngine->CompileComputeShader(shaderPath, "FusedFlashAttentionKernel", &dflashBlob)) return false;
        if (!m_dxEngine->CompileComputeShader(shaderPath, "SparseMoERoutingKernel", &dsparkBlob)) return false;
        if (!m_dxEngine->CompileComputeShader(shaderPath, "TurboQuantDequantKernel", &tqBlob)) return false;

        // 1. FLASH ATTENTION PIPELINE
        // NOTE: this root signature must stay in sync with shaders/TurboKernels.hlsl's
        // FusedFlashAttentionKernel register declarations (see also
        // ModelPipeline::BuildFlashAttentionPipeline, which is the copy actually used
        // during real inference - this one is a separate, currently-unused pipeline that
        // was left stale after the kernel's registers changed, silently failing to build
        // and falling back rather than crashing, but still worth keeping correct).
        {
            D3D12_ROOT_PARAMETER rootParams[5] = {};
            // 32-bit constants (attnConfig) - AttentionConstants is now 10 x uint32
            rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParams[0].Constants.ShaderRegister = 0;
            rootParams[0].Constants.Num32BitValues = 10;
            rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // SRV (QueryBuffer only - t0)
            rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[1].Descriptor.ShaderRegister = 0;
            rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // UAV (KeyBuffer - u1)
            rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[2].Descriptor.ShaderRegister = 1;
            rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // UAV (ValueBuffer - u2)
            rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[3].Descriptor.ShaderRegister = 2;
            rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // UAV (AttnOutput - u0)
            rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[4].Descriptor.ShaderRegister = 0;
            rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
            rootSigDesc.NumParameters = 5;
            rootSigDesc.pParameters = rootParams;
            rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            ComPtr<ID3DBlob> sig, err;
            if (FAILED(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) return false;
            if (FAILED(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_attnRootSignature)))) return false;

            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.pRootSignature = m_attnRootSignature.Get();
            psoDesc.CS = { dflashBlob->GetBufferPointer(), dflashBlob->GetBufferSize() };
            if (FAILED(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_attnPSO)))) return false;
        }

        // 2. MOE ROUTING PIPELINE
        {
            D3D12_ROOT_PARAMETER rootParams[5] = {};
            // CBV (moeConst) - matches HLSL register(b1)
            rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParams[0].Constants.ShaderRegister = 1;
            rootParams[0].Constants.Num32BitValues = 4;
            rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // SRVs (HiddenStates, GateWeights) - matches HLSL registers(t3,t4)
            rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[1].Descriptor.ShaderRegister = 3;
            rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[2].Descriptor.ShaderRegister = 4;
            rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // UAVs (ExpertIds, ExpertWeights) - matches HLSL registers(u1,u2)
            rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[3].Descriptor.ShaderRegister = 1;
            rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[4].Descriptor.ShaderRegister = 2;
            rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
            rootSigDesc.NumParameters = 5;
            rootSigDesc.pParameters = rootParams;
            rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            ComPtr<ID3DBlob> sig, err;
            if (FAILED(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) return false;
            if (FAILED(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_moeRootSignature)))) return false;

            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.pRootSignature = m_moeRootSignature.Get();
            psoDesc.CS = { dsparkBlob->GetBufferPointer(), dsparkBlob->GetBufferSize() };
            if (FAILED(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_moePSO)))) return false;
        }

        // 3. TURBO QUANT DEQUANT PIPELINE
        {
            D3D12_ROOT_PARAMETER rootParams[5] = {};
            // CBV (tqConst) - matches HLSL register(b2)
            rootParams[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
            rootParams[0].Constants.ShaderRegister = 2;
            rootParams[0].Constants.Num32BitValues = 3;
            rootParams[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // SRVs (TQ_PackedWeights, TQ_Scales, TQ_ZeroPoints) - matches HLSL registers(t5,t6,t7)
            rootParams[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[1].Descriptor.ShaderRegister = 5;
            rootParams[1].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParams[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[2].Descriptor.ShaderRegister = 6;
            rootParams[2].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            rootParams[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
            rootParams[3].Descriptor.ShaderRegister = 7;
            rootParams[3].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            // UAV (TQ_Output) - matches HLSL register(u3)
            rootParams[4].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
            rootParams[4].Descriptor.ShaderRegister = 3;
            rootParams[4].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

            D3D12_ROOT_SIGNATURE_DESC rootSigDesc = {};
            rootSigDesc.NumParameters = 5;
            rootSigDesc.pParameters = rootParams;
            rootSigDesc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

            ComPtr<ID3DBlob> sig, err;
            if (FAILED(D3D12SerializeRootSignature(&rootSigDesc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err))) return false;
            if (FAILED(m_device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(), IID_PPV_ARGS(&m_tqRootSignature)))) return false;

            D3D12_COMPUTE_PIPELINE_STATE_DESC psoDesc = {};
            psoDesc.pRootSignature = m_tqRootSignature.Get();
            psoDesc.CS = { tqBlob->GetBufferPointer(), tqBlob->GetBufferSize() };
            if (FAILED(m_device->CreateComputePipelineState(&psoDesc, IID_PPV_ARGS(&m_tqPSO)))) return false;
        }

        return true;
    }

    bool AdvancedVendorOptimizations::DispatchFusedAttention(ID3D12GraphicsCommandList* cmdList,
                                                             ID3D12Resource* query,
                                                             ID3D12Resource* key,
                                                             ID3D12Resource* value,
                                                             ID3D12Resource* output,
                                                             uint32_t batchSize,
                                                             uint32_t numHeads,
                                                             uint32_t headDim,
                                                             uint32_t seqLen) {
        if (!cmdList || !query || !key || !value || !output) return false;
        if (!m_attnPSO || !m_attnRootSignature) return false;

        cmdList->SetComputeRootSignature(m_attnRootSignature.Get());
        cmdList->SetPipelineState(m_attnPSO.Get());

        struct {
            uint32_t  BatchSize;
            uint32_t  NumHeads;
            uint32_t  HeadDim;
            uint32_t  SeqLen;
            float     InvSqrtD;
        } constants;
        constants.BatchSize = batchSize;
        constants.NumHeads = numHeads;
        constants.HeadDim = headDim;
        constants.SeqLen = seqLen;
        constants.InvSqrtD = 1.0f / std::sqrt((float)headDim);

        cmdList->SetComputeRoot32BitConstants(0, 5, &constants, 0);
        cmdList->SetComputeRootShaderResourceView(1, query->GetGPUVirtualAddress());
        cmdList->SetComputeRootShaderResourceView(2, key->GetGPUVirtualAddress());
        cmdList->SetComputeRootShaderResourceView(3, value->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(4, output->GetGPUVirtualAddress());

        cmdList->Dispatch(seqLen, numHeads, batchSize);

        return true;
    }

    bool AdvancedVendorOptimizations::DispatchMoEExpertLayers(ID3D12GraphicsCommandList* cmdList,
                                                              ID3D12Resource* hiddenStates,
                                                              ID3D12Resource* gateWeights,
                                                              ID3D12Resource* expertIds,
                                                              ID3D12Resource* expertWeights,
                                                              uint32_t numTokens,
                                                              uint32_t numExperts,
                                                              uint32_t activeK,
                                                              uint32_t hiddenDim) {
        if (!cmdList || !hiddenStates || !gateWeights || !expertIds || !expertWeights) return false;
        if (!m_moePSO || !m_moeRootSignature) return false;

        cmdList->SetComputeRootSignature(m_moeRootSignature.Get());
        cmdList->SetPipelineState(m_moePSO.Get());

        struct {
            uint32_t NumTokens;
            uint32_t NumExperts;
            uint32_t ActiveK;
            uint32_t HiddenDim;
        } constants;
        constants.NumTokens = numTokens;
        constants.NumExperts = numExperts;
        constants.ActiveK = activeK;
        constants.HiddenDim = hiddenDim;

        cmdList->SetComputeRoot32BitConstants(0, 4, &constants, 0);
        cmdList->SetComputeRootShaderResourceView(1, hiddenStates->GetGPUVirtualAddress());
        cmdList->SetComputeRootShaderResourceView(2, gateWeights->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(3, expertIds->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(4, expertWeights->GetGPUVirtualAddress());

        cmdList->Dispatch(numTokens, 1, 1);

        return true;
    }

    bool AdvancedVendorOptimizations::DispatchTurboQuantDequant(ID3D12GraphicsCommandList* cmdList,
                                                                ID3D12Resource* packedWeights,
                                                                ID3D12Resource* scales,
                                                                ID3D12Resource* zeroPoints,
                                                                ID3D12Resource* output,
                                                                uint32_t numRows,
                                                                uint32_t numCols,
                                                                uint32_t groupSize) {
        if (!cmdList || !packedWeights || !scales || !zeroPoints || !output) return false;
        if (!m_tqPSO || !m_tqRootSignature) return false;

        cmdList->SetComputeRootSignature(m_tqRootSignature.Get());
        cmdList->SetPipelineState(m_tqPSO.Get());

        struct {
            uint32_t NumRows;
            uint32_t NumCols;
            uint32_t GroupSize;
        } constants;
        constants.NumRows = numRows;
        constants.NumCols = numCols;
        constants.GroupSize = groupSize;

        cmdList->SetComputeRoot32BitConstants(0, 3, &constants, 0);
        cmdList->SetComputeRootShaderResourceView(1, packedWeights->GetGPUVirtualAddress());
        cmdList->SetComputeRootShaderResourceView(2, scales->GetGPUVirtualAddress());
        cmdList->SetComputeRootShaderResourceView(3, zeroPoints->GetGPUVirtualAddress());
        cmdList->SetComputeRootUnorderedAccessView(4, output->GetGPUVirtualAddress());

        uint32_t totalWeights = numRows * numCols;
        uint32_t totalPacked = (totalWeights + 7) / 8;
        uint32_t groups = (totalPacked + 63) / 64;
        cmdList->Dispatch(groups, 1, 1);

        return true;
    }

    void AdvancedVendorOptimizations::LogOptimizationSpecs() {
        std::cout << "[Optimizations] Registered pipelines:" << std::endl;
        std::cout << "  - dflash (Fused Attention registers loading CS)" << std::endl;
        std::cout << "  - dspark (Dynamic sparse expert routers CS)" << std::endl;
        std::cout << "  - turboquant (Register-level bit-shifting block decompressions CS)" << std::endl;
        
        switch (m_activePipeline) {
            case OptimizationPipelineType::AMDRDNA_Wave32:
                std::cout << "  - Active optimization profile: AMD RDNA Wave32 Mode." << std::endl;
                break;
            case OptimizationPipelineType::AMDCDNA_MatrixCore:
                std::cout << "  - Active optimization profile: AMD CDNA Matrix Core Instructions." << std::endl;
                break;
            case OptimizationPipelineType::NVIDIACUDA_Native:
                std::cout << "  - Active optimization profile: NVIDIA Tensor Core interop pipelines." << std::endl;
                break;
            case OptimizationPipelineType::IntelXMX_DPAS:
                std::cout << "  - Active optimization profile: Intel Xe Matrix Extensions (XMX)." << std::endl;
                break;
            default:
                std::cout << "  - Active optimization profile: Standard DirectX 12 Compute." << std::endl;
                break;
        }
    }
}