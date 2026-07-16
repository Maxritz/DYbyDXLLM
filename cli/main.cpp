#include "dybydx/core/DirectXEngine.h"
#include "dybydx/core/BpeTokenizer.h"
#include "dybydx/core/KVCacheManager.h"
#include "dybydx/core/ModelPipeline.h"
#include "dybydx/core/GgufLoader.h"

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <random>
#include <algorithm>

using namespace DirectLLM;

struct CliOptions {
    bool debug = false;
    bool verbose = false;
    std::string modelPath;
    std::string prompt = "Hello, what is DirectX 12?";
    int nPredict = 64;
    float temperature = 0.8f;
    float topP = 0.95f;
    int topK = 40;
    float repeatPenalty = 1.1f;
    float presencePenalty = 0.0f;
    bool enableMtp = false;
    std::string kvQuant = "fp16";
    float vramLimit = 2048.0f;
    bool forceStream = false;
    bool metadataOnly = false;
    std::string outputPath; // Path for generated essay
};

static std::string timestamp() {
    auto now = std::chrono::system_clock::now();
    auto t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    std::tm bt;
    localtime_s(&bt, &t);
    std::ostringstream ss;
    ss << std::put_time(&bt, "%H:%M:%S") << "." << std::setw(3) << std::setfill('0') << ms.count();
    return ss.str();
}

static void log(const std::string& module, const std::string& msg, const std::string& type = "INFO") {
    std::cout << "[" << timestamp() << "][" << type << "] [" << module << "] " << msg << std::endl;
}

static void printHelp() {
    std::cout << "DybyDx v1.0.0 - DX12 LLM Inference Engine" << std::endl;
    std::cout << "Usage: DybyDx.exe [options]" << std::endl;
    std::cout << "  -m, --model <path>       GGUF model file" << std::endl;
    std::cout << "  -p, --prompt \"text\"      Input prompt" << std::endl;
    std::cout << "  -n, --n-predict <N>      Tokens to generate (default 64)" << std::endl;
    std::cout << "  --temperature <T>        Sampling temp (default 0.8)" << std::endl;
    std::cout << "  --top-p <P>              Nucleus sampling (default 0.95)" << std::endl;
    std::cout << "  --top-k <K>              Top-K sampling (default 40)" << std::endl;
    std::cout << "  --repeat-penalty <val>   Repetition penalty multiplier (default 1.1)" << std::endl;
    std::cout << "  --presence-penalty <val> Presence penalty (default 0.0)" << std::endl;
    std::cout << "  --enable-mtp             Enable Multi-Token Prediction (speculative decoding)" << std::endl;
    std::cout << "  --kv-quant <type>        KV cache quantization: fp32, fp16, fp8, int8, int4 (default fp16)" << std::endl;
    std::cout << "  --vram-limit <MB>        VRAM allocation limit in MB (default 2048)" << std::endl;
    std::cout << "  --force-stream           Force streaming decode even on dense models" << std::endl;
    std::cout << "  --metadata-only          Skip weight loading; read GGUF metadata only (for 100 GB+ models)" << std::endl;
    std::cout << "  -verbose                 Detailed tracing" << std::endl;
    std::cout << "  -h, --help               This help" << std::endl;
}

static CliOptions parseArgs(int argc, char* argv[]) {
    CliOptions opts;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help" || arg == "-help") { printHelp(); exit(0); }
        else if (arg == "-verbose" || arg == "--verbose") opts.verbose = true;
        else if (arg == "-debug" || arg == "--debug") opts.debug = true;
        else if ((arg == "-m" || arg == "--model") && i + 1 < argc) opts.modelPath = argv[++i];
        else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) opts.prompt = argv[++i];
        else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) opts.nPredict = std::stoi(argv[++i]);
        else if (arg == "--temperature" && i + 1 < argc) opts.temperature = std::stof(argv[++i]);
        else if (arg == "--top-p" && i + 1 < argc) opts.topP = std::stof(argv[++i]);
        else if (arg == "--top-k" && i + 1 < argc) opts.topK = std::stoi(argv[++i]);
        else if (arg == "--repeat-penalty" && i + 1 < argc) opts.repeatPenalty = std::stof(argv[++i]);
        else if (arg == "--presence-penalty" && i + 1 < argc) opts.presencePenalty = std::stof(argv[++i]);
        else if (arg == "--enable-mtp") opts.enableMtp = true;
        else if (arg == "--kv-quant" && i + 1 < argc) opts.kvQuant = argv[++i];
        else if (arg == "--vram-limit" && i + 1 < argc) opts.vramLimit = std::stof(argv[++i]);
        else if (arg == "--force-stream") opts.forceStream = true;
        else if (arg == "--metadata-only") opts.metadataOnly = true;
        else if (arg == "--output" && i + 1 < argc) opts.outputPath = argv[++i];
    }
    return opts;
}

// Same O(n) sampler as ModelPipeline for local main sampling
static int LocalSample(float* logits, int vocabSize, float temp, float topP, int topK) {
    if (temp < 0.001f) temp = 0.001f;
    for (int i = 0; i < vocabSize; i++) logits[i] /= temp;

    std::vector<std::pair<float, int>> scored;
    scored.reserve(vocabSize);
    for (int i = 0; i < vocabSize; i++) scored.emplace_back(logits[i], i);

    int effectiveK = (topK > 0 && topK < vocabSize) ? topK : vocabSize;
    std::nth_element(scored.begin(), scored.begin() + effectiveK, scored.end(),
                     [](auto& a, auto& b) { return a.first > b.first; });
    scored.resize(effectiveK);
    std::sort(scored.begin(), scored.end(),
              [](auto& a, auto& b) { return a.first > b.first; });

    float maxVal = scored[0].first;
    float sum = 0.0f;
    for (auto& s : scored) { s.first = std::exp(s.first - maxVal); sum += s.first; }
    for (auto& s : scored) s.first /= sum;

    if (topP > 0.0f && topP < 1.0f) {
        float cum = 0.0f; size_t cut = scored.size();
        for (size_t i = 0; i < scored.size(); i++) {
            cum += scored[i].first;
            if (cum >= topP) { cut = i + 1; break; }
        }
        scored.resize(cut);
    }

    static std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    float r = dist(gen), cdf = 0.f;
    for (auto& s : scored) { cdf += s.first; if (r <= cdf) return s.second; }
    return scored.empty() ? 0 : scored.back().second;
}

int main(int argc, char* argv[]) {
    auto opts = parseArgs(argc, argv);
    auto loadStartTime = std::chrono::high_resolution_clock::now();

    log("CLI", "DybyDx v1.0.0 starting");

    // 1. DirectX 12 Engine
    log("DX12", "Initializing...");
    DirectXEngine engine;
    bool engineOk = engine.Initialize(false);
    if (!engineOk) { engineOk = engine.Initialize(true); }
    if (engineOk) {
        auto& caps = engine.GetCaps();
        std::wstring name = caps.DeviceName;
        std::string nameStr(name.begin(), name.end());
        log("DX12", nameStr + " | VRAM: " +
            std::to_string(caps.DedicatedVRAM / (1024 * 1024)) + " MB" +
            " | ReBAR: " + (caps.SupportsReBAR ? "YES" : "NO"));
    } else {
        log("DX12", "No device.", "WARN");
    }

    // 2. Load model — REQUIRED. Abort if not found or fails to parse.
    if (opts.modelPath.empty()) {
        log("CLI", "No --model specified. Aborting.", "ERROR");
        return 1;
    }

    GgufLoader ggufLoader;
    bool modelLoaded = false;

    {
        if (opts.metadataOnly) {
            log("Model", "Parsing metadata only from " + opts.modelPath + "...");
            modelLoaded = ggufLoader.LoadMetadataOnly(opts.modelPath);
        } else {
            log("Model", "Loading " + opts.modelPath + "...");
            modelLoaded = ggufLoader.LoadFile(opts.modelPath);
        }
        if (!modelLoaded) {
            log("Model", "FAILED to open/parse: " + opts.modelPath, "ERROR");
            log("Model", "Check the path exists and is a valid GGUF file. Aborting.", "ERROR");
            return 1;   // <-- hard stop: no fake output
        }
        log("Model", "GGUF v" + std::to_string(ggufLoader.GetVersion()) +
            " | " + std::to_string(ggufLoader.GetTensorCount()) + " tensors");
        if (ggufLoader.HasMetadata("general.architecture"))
            log("Model", "Arch: " + ggufLoader.GetMetadataString("general.architecture"));
        if (ggufLoader.HasMetadata("llama.embedding_length"))
            log("Model", "Dim: " + std::to_string(ggufLoader.GetMetadataUint32("llama.embedding_length")));
    }

    // 3. Tokenizer
    BpeTokenizer tokenizer;
    std::string modelArch = "";
    if (modelLoaded) {
        tokenizer.LoadFromGGUF(ggufLoader);
        if (ggufLoader.HasMetadata("general.architecture"))
            modelArch = ggufLoader.GetMetadataString("general.architecture");
    } else {
        tokenizer.LoadVocabulary("vocab.txt");
    }

    // Wrap prompt in instruct template so the model generates proper text
    std::string formattedPrompt = tokenizer.ApplyChatTemplate(opts.prompt, modelArch);
    log("Tokenizer", "Chat template arch: " + (modelArch.empty() ? "(none)" : modelArch));

    auto promptTokens = tokenizer.Encode(formattedPrompt);
    log("Tokenizer", std::to_string(promptTokens.size()) + " tokens from prompt");

    // 4. Pipeline Config
    ModelPipeline pipeline;
    ModelConfig config;
    
    if (opts.kvQuant == "fp32") config.CacheQuantType = KVCacheQuantType::None_FP32;
    else if (opts.kvQuant == "fp16") config.CacheQuantType = KVCacheQuantType::None_FP16;
    else if (opts.kvQuant == "fp8") config.CacheQuantType = KVCacheQuantType::FP8;
    else if (opts.kvQuant == "int8") config.CacheQuantType = KVCacheQuantType::INT8;
    else if (opts.kvQuant == "int4") config.CacheQuantType = KVCacheQuantType::INT4;
    
    config.VramAllocationLimitMB = opts.vramLimit;
    config.ForceStreamMode = opts.forceStream;
    config.UseMetadataOnlyLoad = opts.metadataOnly;

    if (modelLoaded) {
        if (ggufLoader.HasMetadata("general.architecture")) {
            std::string arch = ggufLoader.GetMetadataString("general.architecture");
            std::string archLower;
            for (char c : arch) archLower += tolower(c);
            if (archLower.find("laguna") != std::string::npos) {
                config.Arch = ModelArchitecture::Laguna;
            } else if (archLower.find("llama") != std::string::npos) {
                config.Arch = ModelArchitecture::Llama;
            } else if (archLower.find("phi") != std::string::npos) {
                config.Arch = ModelArchitecture::Phi;
            } else if (archLower.find("gemma") != std::string::npos) {
                config.Arch = ModelArchitecture::Gemma;
            } else if (archLower.find("deepseek") != std::string::npos) {
                config.Arch = ModelArchitecture::DeepSeek;
            }
        }

        // Extract config based on architecture - support all variants
        auto getMeta = [&](const std::string& key) -> uint32_t {
            return ggufLoader.HasMetadata(key) ? ggufLoader.GetMetadataUint32(key) : 0;
        };

        // Block count - try all arch variants
        if (getMeta("llama.block_count")) config.NumLayers = getMeta("llama.block_count");
        else if (getMeta("qwen2.block_count")) config.NumLayers = getMeta("qwen2.block_count");
        else if (getMeta("qwen35.block_count")) config.NumLayers = getMeta("qwen35.block_count");
        else if (getMeta("phi3.block_count")) config.NumLayers = getMeta("phi3.block_count");
        else if (getMeta("gemma2.block_count")) config.NumLayers = getMeta("gemma2.block_count");
        else if (getMeta("gemma4.block_count")) config.NumLayers = getMeta("gemma4.block_count");

        // Hidden dim
        if (getMeta("llama.embedding_length")) config.HiddenDim = getMeta("llama.embedding_length");
        else if (getMeta("qwen2.embedding_length")) config.HiddenDim = getMeta("qwen2.embedding_length");
        else if (getMeta("qwen35.embedding_length")) config.HiddenDim = getMeta("qwen35.embedding_length");
        else if (getMeta("phi3.embedding_length")) config.HiddenDim = getMeta("phi3.embedding_length");
        else if (getMeta("gemma2.embedding_length")) config.HiddenDim = getMeta("gemma2.embedding_length");
        else if (getMeta("gemma4.embedding_length")) config.HiddenDim = getMeta("gemma4.embedding_length");

        // Attention heads
        if (getMeta("llama.attention.head_count")) config.NumHeads = getMeta("llama.attention.head_count");
        else if (getMeta("qwen2.attention.head_count")) config.NumHeads = getMeta("qwen2.attention.head_count");
        else if (getMeta("qwen35.attention.head_count")) config.NumHeads = getMeta("qwen35.attention.head_count");
        else if (getMeta("phi3.attention.head_count")) config.NumHeads = getMeta("phi3.attention.head_count");
        else if (getMeta("gemma2.attention.head_count")) config.NumHeads = getMeta("gemma2.attention.head_count");
        else if (getMeta("gemma4.attention.head_count")) config.NumHeads = getMeta("gemma4.attention.head_count");

        // Feed forward length (intermediate dim)
        if (getMeta("llama.feed_forward_length")) config.IntermediateDim = getMeta("llama.feed_forward_length");
        else if (getMeta("qwen2.feed_forward_length")) config.IntermediateDim = getMeta("qwen2.feed_forward_length");
        else if (getMeta("qwen35.feed_forward_length")) config.IntermediateDim = getMeta("qwen35.feed_forward_length");
        else if (getMeta("phi3.feed_forward_length")) config.IntermediateDim = getMeta("phi3.feed_forward_length");
        else if (getMeta("gemma2.feed_forward_length")) config.IntermediateDim = getMeta("gemma2.feed_forward_length");
        else if (getMeta("gemma4.feed_forward_length")) config.IntermediateDim = getMeta("gemma4.feed_forward_length");

        if (config.IntermediateDim == 0) config.IntermediateDim = config.HiddenDim * 4;
        config.HeadDim = config.HiddenDim / config.NumHeads;

        // MoE check
        if (ggufLoader.HasMetadata("llama.expert_count") || ggufLoader.HasMetadata("deepseek.expert_count")) {
            config.Type = ModelType::MixtureOfExperts;
            config.NumExperts = ggufLoader.HasMetadata("llama.expert_count") ?
                                ggufLoader.GetMetadataUint32("llama.expert_count") :
                                ggufLoader.GetMetadataUint32("deepseek.expert_count");
            config.ActiveExpertsK = ggufLoader.HasMetadata("llama.expert_used_count") ?
                                    ggufLoader.GetMetadataUint32("llama.expert_used_count") : 2;
        }

        // Vocab size from any embedding tensor
        for (const auto& [name, tensor] : ggufLoader.GetTensors()) {
            if (name.find("embed") != std::string::npos || name.find("tok_embeddings") != std::string::npos ||
                name.find("output") != std::string::npos || name.find("lm_head") != std::string::npos) {
                if (tensor.Shape.size() >= 1) {
                    config.VocabSize = std::max(config.VocabSize, (size_t)tensor.Shape[0]);
                }
            }
        }
    }

    // Force Stream dense check warning
    if (config.ForceStreamMode && config.Type == ModelType::Dense) {
        log("Model", "WARNING: --force-stream enabled on a dense model. Generation will be slow!", "WARN");
    }

    pipeline.Initialize(engineOk ? &engine : nullptr, config);
    if (modelLoaded) {
        pipeline.LoadModelWeights(std::wstring(opts.modelPath.begin(), opts.modelPath.end()));
    }

    auto loadEndTime = std::chrono::high_resolution_clock::now();
    float loadMs = std::chrono::duration<float, std::milli>(loadEndTime - loadStartTime).count();
    log("CLI", "Model loaded & pipeline initialized in " + std::to_string((int)loadMs) + "ms");

    // 5. Inference loop timing setup
    log("Pipeline", "Generating " + std::to_string(opts.nPredict) + " tokens...");
    std::cout << "\n--- OUTPUT ---\n";

    std::vector<float> logits(config.VocabSize > 0 ? config.VocabSize : 32000);
    std::vector<int32_t> tokens = promptTokens;
    int genCount = 0;

std::string lowerPrompt = opts.prompt;
for (char &c : lowerPrompt) c = tolower(c);

    auto genStartTime = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < opts.nPredict; step++) {
        // Run real GPU or fallback CPU/NPU inference step
        auto stepStart = std::chrono::high_resolution_clock::now();
        bool ok = pipeline.RunInferenceStep(1, tokens, (uint32_t)step, logits);
        if (!ok) break;

        // Apply sampling parameter options (Temperature, Repetition Penalty, Presence Penalty) to logits
        if (opts.temperature > 0.0f) {
            for (float& l : logits) {
                l /= opts.temperature;
            }
        }

        for (int32_t tok : tokens) {
            if (tok >= 0 && tok < (int32_t)logits.size()) {
                if (logits[tok] > 0.0f) {
                    logits[tok] /= opts.repeatPenalty;
                } else {
                    logits[tok] *= opts.repeatPenalty;
                }
                logits[tok] -= opts.presencePenalty;
            }
        }

        // Sample next token from logits using temperature, top-p, top-k
        int32_t nextToken = LocalSample(logits.data(), config.VocabSize, opts.temperature, opts.topP, opts.topK);

        // Stop cleanly on any EOS token
        if (tokenizer.IsEosToken(nextToken)) {
            log("Pipeline", "EOS token hit at step " + std::to_string(step) + " — stopping.");
            break;
        }

        std::string word = tokenizer.Decode({nextToken});

        genCount++;
        std::cout << word << std::flush;
        tokens.push_back(nextToken);

        // Verbose detailed prints
        if (opts.verbose) {
            auto stepEnd = std::chrono::high_resolution_clock::now();
            float latency = std::chrono::duration<float, std::milli>(stepEnd - stepStart).count();
            std::stringstream ss1;
            ss1 << std::fixed << std::setprecision(2) << latency;
            log("QuantGEMM", "Token " + std::to_string(step) + " - Dispatch Compute Shader. Latency: " + ss1.str() + "ms (measured)", "TRACE");
        }

        } // end inference loop
    std::cout << "\n" << std::endl;

    // Write essay to file if requested
    if (!opts.outputPath.empty()) {
        std::ofstream outFile(opts.outputPath);
        if (outFile) {
            // Reconstruct essay from generated tokens
            std::string essay = tokenizer.Decode(tokens);
            outFile << essay;
        }
    }

    auto genEndTime = std::chrono::high_resolution_clock::now();
    float genMs = std::chrono::duration<float, std::milli>(genEndTime - genStartTime).count();
    
    log("Pipeline", "Init/Load time: " + std::to_string((int)loadMs) + "ms");
    log("Pipeline", std::to_string(genCount) + " tokens generated in " +
        std::to_string((int)genMs) + "ms (" +
        std::to_string(genCount / (genMs / 1000.0f)) + " tok/s)");

    return 0;
}
