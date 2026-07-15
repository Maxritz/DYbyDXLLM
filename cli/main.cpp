#include "dybydx/core/DirectXEngine.h"
#include "dybydx/core/BpeTokenizer.h"
#include "dybydx/core/KVCacheManager.h"
#include "dybydx/core/ModelPipeline.h"
#include "dybydx/core/GgufLoader.h"

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <sstream>
#include <iomanip>

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
    std::cout << "  -m, --model <path>      GGUF model file" << std::endl;
    std::cout << "  -p, --prompt \"text\"     Input prompt" << std::endl;
    std::cout << "  -n, --n-predict <N>     Tokens to generate (default 64)" << std::endl;
    std::cout << "  --temperature <T>        Sampling temp (default 0.8)" << std::endl;
    std::cout << "  --top-p <P>              Nucleus sampling (default 0.95)" << std::endl;
    std::cout << "  --top-k <K>              Top-K sampling (default 40)" << std::endl;
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
    }
    return opts;
}

int main(int argc, char* argv[]) {
    auto opts = parseArgs(argc, argv);
    auto startTime = std::chrono::high_resolution_clock::now();

    log("CLI", "DybyDx v1.0.0 starting");

    // 1. DirectX 12 Engine
    log("DX12", "Initializing...");
    DirectXEngine engine;
    bool engineOk = engine.Initialize(false);
    if (!engineOk) { engineOk = engine.Initialize(true); }
    if (engineOk) {
        auto& caps = engine.GetCaps();
        std::wstring name = caps.DeviceName;
        log("DX12", std::string(name.begin(), name.end()) + " | VRAM: " +
            std::to_string(caps.DedicatedVRAM / (1024 * 1024)) + " MB");
    } else {
        log("DX12", "No device.", "WARN");
    }

    // 2. Load model
    if (opts.modelPath.empty()) {
        log("CLI", "No model specified (--model). Use fallback mode.", "WARN");
    }

    GgufLoader ggufLoader;
    bool modelLoaded = false;

    if (!opts.modelPath.empty()) {
        log("Model", "Loading " + opts.modelPath + "...");
        modelLoaded = ggufLoader.LoadFile(opts.modelPath);
        if (modelLoaded) {
            log("Model", "GGUF v" + std::to_string(ggufLoader.GetVersion()) +
                " | " + std::to_string(ggufLoader.GetTensorCount()) + " tensors");
            if (ggufLoader.HasMetadata("general.architecture"))
                log("Model", "Arch: " + ggufLoader.GetMetadataString("general.architecture"));
            if (ggufLoader.HasMetadata("llama.embedding_length"))
                log("Model", "Dim: " + std::to_string(ggufLoader.GetMetadataUint32("llama.embedding_length")));
        } else {
            log("Model", "Failed to load model.", "ERROR");
        }
    }

    // 3. Tokenizer
    BpeTokenizer tokenizer;
    if (modelLoaded) {
        tokenizer.LoadFromGGUF(ggufLoader);
    } else {
        tokenizer.LoadVocabulary("vocab.txt");
    }

    auto promptTokens = tokenizer.Encode(opts.prompt);
    log("Tokenizer", std::to_string(promptTokens.size()) + " tokens from prompt");

    // 4. Pipeline
    ModelPipeline pipeline;
    ModelConfig config;
    if (modelLoaded) {
        if (ggufLoader.HasMetadata("llama.block_count"))
            config.NumLayers = ggufLoader.GetMetadataUint32("llama.block_count");
        if (ggufLoader.HasMetadata("llama.embedding_length"))
            config.HiddenDim = ggufLoader.GetMetadataUint32("llama.embedding_length");
        if (ggufLoader.HasMetadata("llama.attention.head_count"))
            config.NumHeads = ggufLoader.GetMetadataUint32("llama.attention.head_count");
        if (ggufLoader.HasMetadata("llama.feed_forward_length"))
            config.IntermediateDim = ggufLoader.GetMetadataUint32("llama.feed_forward_length");
        else
            config.IntermediateDim = config.HiddenDim * 4;
        config.HeadDim = config.HiddenDim / config.NumHeads;

        const GgufTensor* tokEmb = ggufLoader.GetTensor("token_embd.weight");
        if (tokEmb && tokEmb->Shape.size() >= 2)
            config.VocabSize = (size_t)tokEmb->Shape[0];
    }

    pipeline.Initialize(engineOk ? &engine : nullptr, config);
    if (modelLoaded)
        pipeline.LoadModelWeights(std::wstring(opts.modelPath.begin(), opts.modelPath.end()));

    // 5. Inference loop
    log("Pipeline", "Generating " + std::to_string(opts.nPredict) + " tokens...");
    std::cout << "\n--- OUTPUT ---\n";

    std::vector<float> logits(config.VocabSize > 0 ? config.VocabSize : 32000);
    std::vector<int32_t> tokens = promptTokens;
    int genCount = 0;

    for (int step = 0; step < opts.nPredict && genCount < 512; step++) {
        bool ok = pipeline.RunInferenceStep(1, tokens, (uint32_t)step, logits);
        if (!ok) break;

        // Sample from logits
        int nextToken = 0;
        float maxVal = -1e30f;
        for (size_t i = 0; i < logits.size(); i++)
            if (logits[i] > maxVal) { maxVal = logits[i]; nextToken = (int)i; }

        // Temperature scaling (simple)
        if (opts.temperature > 0.001f && opts.temperature != 1.0f) {
            float sum = 0.0f;
            std::vector<float> probs(logits.size());
            for (size_t i = 0; i < logits.size(); i++) {
                probs[i] = std::exp(logits[i] / opts.temperature);
                sum += probs[i];
            }
            float r = (float)std::rand() / (float)RAND_MAX;
            float cum = 0.0f;
            for (size_t i = 0; i < probs.size(); i++) {
                cum += probs[i] / sum;
                if (r <= cum) { nextToken = (int)i; break; }
            }
        }

        genCount++;
        std::string word = tokenizer.Decode({ nextToken });
        std::cout << word << std::flush;

        if (nextToken == 2) break; // EOS
        tokens.push_back(nextToken);
    }
    std::cout << "\n" << std::endl;

    auto endTime = std::chrono::high_resolution_clock::now();
    float totalMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
    log("Pipeline", std::to_string(genCount) + " tokens in " +
        std::to_string((int)totalMs) + "ms (" +
        std::to_string(genCount / (totalMs / 1000.0f)) + " tok/s)");

    return 0;
}
