# DybyDx (dy/dx)

> **dy/dx** — *do you by DirectX?*

A DirectX 12 / Microsoft Agility SDK-based quantized local LLM inference engine for Windows 10/11.

**Repository:** [github.com/Maxritz/DYbyDXLLM](https://github.com/Maxritz/DYbyDXLLM)

---

## Project Structure

```
DybyDx/
├── include/dybydx/core/      # Public headers
│   ├── DirectXEngine.h       # D3D12 device, queues, fences, shader compilation
│   ├── KVCacheManager.h      # Ring buffer KV cache
│   ├── BpeTokenizer.h        # BPE tokenizer
│   ├── MultiFormatTokenizer.h # SentencePiece / WordPiece wrapper
│   ├── DirectStorageLoader.h  # DirectStorage GPU streaming
│   ├── IntelOpenVINOInterop.h # NPU/GPU zero-copy interop
│   ├── AdvancedVendorOptimizations.h  # Fused attention, MoE, vendor dispatch
│   └── ModelPipeline.h       # Split-layer model loader, tensor management, inference step
├── src/core/                  # Implementation
│   ├── DirectXEngine.cpp
│   ├── KVCacheManager.cpp
│   ├── BpeTokenizer.cpp
│   ├── DirectStorageLoader.cpp
│   ├── IntelOpenVINOInterop.cpp
│   ├── AdvancedVendorOptimizations.cpp
│   └── ModelPipeline.cpp
├── shaders/                   # HLSL SM 6.6 compute shaders
│   ├── QuantizedGEMM.hlsl     # INT4 weight-only matrix multiply
│   ├── DirectMLLinAlg.hlsl    # INT8 DP4A dot-product
│   └── TurboKernels.hlsl      # Fused FlashAttention, MoE, TurboQuant
├── cli/main.cpp               # CLI entry point
├── tools/DirectLLM.CLI/       # C# P/Invoke harness (.NET 8)
│   └── DirectLLM.CLI.csproj
├── CMakeLists.txt
└── README.md
```

## Build

```bat
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

**Output:** `build/bin/Release/DybyDx.exe`

## Usage

```
DybyDx.exe [options]
```

### Model Options

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--model` | `-m` | `PATH` | (required) | Path to quantized model weights (GGUF/bin) |
| `--prompt` | `-p` | `TEXT` | (none) | Input prompt text |
| `--file` | `-f` | `PATH` | (none) | Read prompt from file |
| `--system-prompt` | | `TEXT` | (none) | System prompt prefix |

### Generation / Sampling

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--n-predict` | `-n` | `N` | `512` | Tokens to generate |
| `--ctx-size` | `-c` | `N` | `4096` | Context window size |
| `--batch-size` | `-b` | `N` | `2048` | Batch size |
| `--temperature` | | `T` | `0.8` | Sampling temperature (0 = greedy) |
| `--top-p` | | `P` | `0.95` | Nucleus sampling threshold |
| `--top-k` | | `K` | `40` | Top-k sampling (0 = disabled) |
| `--repeat-penalty` | | `P` | `1.1` | Token repeat penalty |
| `--seed` | | `N` | random | RNG seed for reproducibility |

### Performance & Memory

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--gpu-layers` | `-ngl` | `N` | `-1` | Layers in GPU VRAM (-1 = all, 0 = CPU) |
| `--threads` | `-t` | `N` | auto | CPU thread count |
| `--flash-attn` | | | on | Enable Flash Attention |
| `--no-flash-attn` | | | | Disable Flash Attention |

### Diagnostics & Output

| Switch | Short | Argument | Default | Description |
|--------|-------|----------|---------|-------------|
| `--verbose` | | | false | Granular shader/KV tracing |
| `--verbose-prompt` | | | false | Print prompt before generation |
| `--debug` | | | false | Diagnostic component listing |
| `--trace` | | `MODULES` | (all) | Module filter (DirectXEngine,QuantGEMM,KVCache,...) |
| `--color` | | | true | Colored terminal output |
| `--no-color` | | | | Disable colored output |
| `-h` | `--help` | | | Show help |

### Traceable Modules

`DirectXEngine`, `ShaderCompiler`, `QuantGEMM`, `KVCache`, `Tokenizer`, `Pipeline`, `WeightLoader`

### Examples

```bat
# Basic generation
DybyDx.exe -m model.bin -p "What is the capital of France?" -n 128

# With custom sampling
DybyDx.exe -m model.bin -p "Write a function" -n 256 --temperature 1.2 --top-p 0.9

# GPU offloading (first 20 layers on GPU)
DybyDx.exe -m model.bin -ngl 20 -p "Hello" -n 64

# CPU only
DybyDx.exe -m model.bin -ngl 0 -p "Hello" -n 64

# Full diagnostics with module tracing
DybyDx.exe -m model.bin -p "Explain DX12" -verbose -debug -trace QuantGEMM,KVCache

# Read prompt from file
DybyDx.exe -m model.bin -f prompt.txt -n 128 --temperature 0.6
```

## Prerequisites

- Visual Studio 2022 with C++20
- Windows SDK 10.0+
- CMake >= 3.20
- Vulkan SDK 1.4.350.0 (for glslc)
- Agility SDK + DirectStorage SDK (paths in CMakeLists.txt)
