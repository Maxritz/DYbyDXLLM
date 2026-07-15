# DybyDx (dy/dx — *do you by DirectX?*)

A DirectX 12 / Microsoft Agility SDK-based quantized local LLM inference engine for Windows 10/11, with CPU inference fallback, DXC shader compilation, DirectStorage GPU streaming, and Intel OpenVINO interop.

**Repository:** [github.com/Maxritz/DYbyDXLLM](https://github.com/Maxritz/DYbyDXLLM)

---

## Build

```bat
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Produces both `DybyDx.exe` (C++ CLI) and `DirectLLM.Core.dll` (C# P/Invoke DLL).

## Usage

```
DybyDx.exe -m model.gguf -p "Your prompt" -n 128
```

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model` | required | Path to GGUF quantized model |
| `-p, --prompt` | `"Hello"` | Input prompt text |
| `-n, --n-predict` | 64 | Number of tokens to generate |
| `--temperature` | 0.8 | Sampling temperature (0 = greedy) |
| `--top-p` | 0.95 | Nucleus sampling threshold |
| `--top-k` | 40 | Top-k sampling (0 = disabled) |
| `-h, --help` | | Show help |

## Architecture

```
├── include/dybydx/core/       # Public headers
│   ├── DirectXEngine.h        # D3D12 device, queues, DXC shader compilation
│   ├── ModelPipeline.h        # CPU inference pipeline with GGUF weights
│   ├── GgufLoader.h           # GGUF format parser
│   ├── BpeTokenizer.h         # Tokenizer with GGUF vocabulary
│   ├── DirectStorageLoader.h  # DirectStorage 1.2 GPU streaming
│   ├── KVCacheManager.h       # GPU KV cache ring buffer
│   └── IntelOpenVINOInterop.h # DX12 shared-memory OpenVINO interop
├── src/core/                  # Implementation
├── shaders/                   # HLSL SM 6.6 compute shaders
│   ├── QuantizedGEMM.hlsl     # INT4 quantized matrix multiply
│   ├── DirectMLLinAlg.hlsl    # INT8 DP4A kernel
│   └── TurboKernels.hlsl      # Fused FlashAttention, MoE, TurboQuant
├── cli/main.cpp               # C++ CLI entry point
└── tools/DirectLLM.CLI/       # C# P/Invoke harness (.NET 10)
```

## Features

- **GGUF model loading** — reads model architecture, vocabulary, tensor shapes from GGUF
- **CPU inference** — full 32-layer transformer pipeline (RMS Norm, QKV, SiLU FFN, LM head)
- **Vocabulary from model** — loads 248,000+ real tokens from GGUF metadata
- **DXC shader compilation** — compiles HLSL SM 6.6 shaders to DXIL via DXC
- **Sampler** — temperature, top-p (nucleus), top-k sampling on CPU logits
- **Prompt-dependent output** — output varies based on input token IDs
- **DirectStorage 1.2** — GDeflate decompression pipeline for GPU weight streaming
- **OpenVINO interop** — DXGI shared-memory for Intel NPU/GPU offloading

## Prerequisites

- Visual Studio 2022 with C++20
- Windows SDK 10.0+
- CMake >= 3.20
- DirectX 12 Agility SDK
- DirectStorage SDK
- DXC Compiler
- Vulkan SDK (for `glslc`)
- .NET 10 SDK (for C# CLI)
