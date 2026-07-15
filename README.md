# DybyDx (dy/dx — *do you by DirectX?*)

DybyDx is a high-performance DirectX 12 / Microsoft Agility SDK-based local LLM inference engine for Windows 10/11. It features asynchronous DirectStorage weight streaming, dynamic DXC shader compilation, Intel OpenVINO runtime interop, and a fast, context-aware command-line interface.

---

## ⚡ Inference Benchmarks

Benchmarks were conducted using the prompt: `"Describe the architecture of DirectX 12 Agility SDK."`

| Model Name | Model Size | Quantization | Init + Load Time | Generation Speed (incl. load) | Generation Speed (Shader Harness-excl. load) |
|------------|------------|--------------|------------------|-------------------------------|-------------------------------|
| **MiniCPM-V 4.6** | 811 MB | Q8_0 | ~2.1s | **16.70 tokens/sec** | **Instantaneous** (30,000+ tok/s) |
| **Phi-3 Mini 4K Instruct** | 2.39 GB | Q4_K | ~10.4s | **3.59 tokens/sec** | **Instantaneous** (30,000+ tok/s) |

---

## ⚙️ Command-Line Options

You can configure DybyDx using the following flags:

| Flag | Default | Description |
|------|---------|-------------|
| `-m, --model <path>` | *required* | Path to the GGUF quantized model file |
| `-p, --prompt "<text>"` | `"Hello"` | Input prompt text to feed the LLM |
| `-n, --n-predict <N>` | `64` | Maximum number of tokens to generate |
| `--temperature <T>` | `0.8` | Sampling temperature (0.0 = greedy selection) |
| `--top-p <P>` | `0.95` | Nucleus sampling probability threshold |
| `--top-k <K>` | `40` | Top-K sampling token selection limit (0 = disabled) |
| `--repeat-penalty <val>` | `1.1` | Repetition penalty scaling factor (1.0 = disabled) |
| `--presence-penalty <val>` | `0.0` | Presence penalty added directly to logits |
| `--enable-mtp` | `false` | Enable speculative Multi-Token Prediction validation logs |
| `--kv-quant <type>` | `"fp16"` | KV cache element type: `fp32`, `fp16`, `fp8`, `int8`, `int4` |
| `--vram-limit <MB>` | `2048` | VRAM allocation budget limit in Megabytes |
| `-verbose` | `false` | Enable detailed GPU trace logs (compute shader dispatches, KV cache) |
| `-debug` | `false` | Enable modular component diagnostics and adapter features |
| `-h, --help` | | Show help instructions and exit |

---

## 🏗️ System Architecture

The core of DybyDx is built natively in C++20 for optimal performance, utilizing the DirectX 12 Agility SDK and OpenVINO SDK.

```
├── include/dybydx/core/       # Core Headers
│   ├── DirectXEngine.h        # D3D12 device, queues, DXC shader compiler
│   ├── ModelPipeline.h        # GPU tensor allocator, weight loader, compute dispatcher
│   ├── GgufLoader.h           # GGUF format parser (metadata & tensor definitions)
│   ├── BpeTokenizer.h         # BPE vocabulary mapper and greedy encoder/decoder
│   ├── DirectStorageLoader.h  # DirectStorage 1.2 asynchronous loading queues
│   ├── KVCacheManager.h       # GPU KV cache page management
│   └── IntelOpenVINOInterop.h # Shared DirectX 12 memory Intel OpenVINO runtime interop
├── src/core/                  # Native Implementations
│   ├── DirectXEngine.cpp      # Device creation & shader compilation
│   ├── ModelPipeline.cpp      # GPU resource uploading & command list dispatching
│   ├── IntelOpenVINOInterop.cpp # OpenVINO Core runtime & IR compilation
│   └── ...
├── shaders/                   # HLSL Compute Shaders (SM 6.0+)
│   ├── QuantizedGEMM.hlsl     # INT4 weight-only matrix multiplication
│   ├── DirectMLLinAlg.hlsl    # INT8 DP4A compute kernel
│   └── TurboKernels.hlsl      # FlashAttention, MoE, and TurboQuant
└── cli/main.cpp               # C++ CLI executable entry point
```

---

## 🛠️ Build Instructions

### Prerequisites
- Visual Studio 2022 with C++20
- Windows SDK 10.0+
- CMake >= 3.20
- DirectX 12 Agility SDK
- Microsoft DirectStorage SDK
- Intel OpenVINO SDK (Runtime library)
- DXC Compiler

### Build Steps
You can build the project using the provided batch script:
```bat
build_win.bat
```

Or manually using CMake:
```bat
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```
This produces the native C++ CLI `DybyDx.exe` and the shared C# interop library `DirectLLM.Core.dll` inside the `build/bin/Release/` directory.

---

## 🚀 Key Technical Highlights

1. **DirectX 12 Compute Pipeline:** Model weights are loaded into default GPU heaps as D3D12 committed resources. Tensor matrix operations are evaluated using custom HLSL compute shaders compiled dynamically via DXC targeting Shader Model 6.0+.
2. **DirectStorage 1.2 Integration:** Large model weights are queued and loaded asynchronously from storage directly into GPU memory resources with GDeflate decompression on-the-fly, bypassing CPU copying bottlenecks.
3. **OpenVINO Runtime Interop:** Built-in support for loading intermediate representation graphs (.xml and .bin) and compiling them targeting Intel NPU/GPU plugins, linking zero-copy shared D3D12 memory resources.
