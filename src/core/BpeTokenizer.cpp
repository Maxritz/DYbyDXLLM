#include "dybydx/core/BpeTokenizer.h"
#include "dybydx/core/GgufLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace DirectLLM {

    BpeTokenizer::BpeTokenizer() {
        m_encoder["<pad>"] = 0;  m_decoder[0] = "<pad>";
        m_encoder["<s>"]   = 1;  m_decoder[1] = "<s>";
        m_encoder["</s>"]  = 2;  m_decoder[2] = "</s>";
        m_encoder["<unk>"] = 3;  m_decoder[3] = "<unk>";
    }

    BpeTokenizer::~BpeTokenizer() {}

    bool BpeTokenizer::LoadFromGGUF(const GgufLoader& loader) {
        // 1. Load the token table
        for (auto& [key, val] : loader.GetAllMetadata()) {
            if (key == "tokenizer.ggml.tokens" && val.Type == GgufType::ARRAY) {
                for (size_t i = 0; i < val.ArrayValues.size(); i++) {
                    auto& v = val.ArrayValues[i];
                    if (v.Type == GgufType::STRING && !v.StringValue.empty()) {
                        int32_t id = (int32_t)(i);
                        m_encoder[v.StringValue] = id;
                        m_decoder[id] = v.StringValue;
                    }
                }
            }
            if (key == "tokenizer.ggml.scores" && val.Type == GgufType::ARRAY) {
                for (size_t i = 0; i < val.ArrayValues.size(); i++) {
                    auto& v = val.ArrayValues[i];
                    if (v.Type == GgufType::FLOAT32 && v.Data.size() == 4) {
                        float score = *reinterpret_cast<const float*>(v.Data.data());
                        m_bpeRanks[std::to_string(i)] = (int64_t)(score * 1000);
                    }
                }
            }
        }

        // 2. Read BOS / EOS token IDs from GGUF metadata
        if (loader.HasMetadata("tokenizer.ggml.bos_token_id")) {
            m_bosId = (int32_t)loader.GetMetadataUint32("tokenizer.ggml.bos_token_id");
        }
        if (loader.HasMetadata("tokenizer.ggml.eos_token_id")) {
            m_eosId = (int32_t)loader.GetMetadataUint32("tokenizer.ggml.eos_token_id");
        }
        // Collect all EOS-equivalent ids (some models have multiple)
        m_eosIds.clear();
        m_eosIds.push_back(m_eosId);
        // Common end-of-turn / end-of-text tokens to also stop on
        static const std::vector<std::string> kEosStrings = {
            "</s>", "<|endoftext|>", "<|end|>", "<end_of_turn>",
            "<|im_end|>", "<eot_id>", "<|eom_id|>", "<|end_of_sentence|>"
        };
        for (auto& s : kEosStrings) {
            auto it = m_encoder.find(s);
            if (it != m_encoder.end()) {
                int32_t id = it->second;
                if (std::find(m_eosIds.begin(), m_eosIds.end(), id) == m_eosIds.end())
                    m_eosIds.push_back(id);
            }
        }

        std::cout << "[Tokenizer] Loaded " << m_encoder.size()
                  << " tokens from GGUF metadata."
                  << " BOS=" << m_bosId << " EOS=" << m_eosId << std::endl;
        return m_encoder.size() > 4;
    }

    bool BpeTokenizer::LoadVocabulary(const std::string& vocabPath) {
        std::ifstream file(vocabPath);
        if (!file.is_open()) {
            std::cout << "[Tokenizer] No vocab file at " << vocabPath << ", using fallback." << std::endl;
            return false;
        }
        std::string line;
        while (std::getline(file, line)) {
            auto pos = line.find(' ');
            if (pos != std::string::npos) {
                std::string word = line.substr(0, pos);
                int32_t id = std::stoi(line.substr(pos + 1));
                m_encoder[word] = id;
                m_decoder[id] = word;
            }
        }
        std::cout << "[Tokenizer] Loaded " << m_encoder.size() << " tokens from " << vocabPath << std::endl;
        return true;
    }

    // -------------------------------------------------------------------------
    //  ApplyChatTemplate
    //  Wraps the raw user prompt in the correct instruct template for each arch.
    //  The result is the full string to pass to Encode().
    // -------------------------------------------------------------------------
    std::string BpeTokenizer::ApplyChatTemplate(const std::string& userPrompt,
                                                  const std::string& arch) const {
        // qwen2 / qwen25 / qwen35 (MiniCPM uses qwen35 arch)
        if (arch == "qwen2" || arch == "qwen25" || arch == "qwen35" ||
            arch.find("qwen") != std::string::npos) {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                   "<|im_start|>user\n" + userPrompt + "<|im_end|>\n"
                   "<|im_start|>assistant\n";
        }
        // gemma / gemma2 / gemma4
        if (arch.find("gemma") != std::string::npos) {
            return "<bos><start_of_turn>user\n" + userPrompt + "<end_of_turn>\n"
                   "<start_of_turn>model\n";
        }
        // phi3 / phi
        if (arch.find("phi") != std::string::npos) {
            return "<|user|>\n" + userPrompt + "<|end|>\n<|assistant|>\n";
        }
        // llama3 / llama
        if (arch.find("llama") != std::string::npos) {
            return "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
                   + userPrompt +
                   "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        }
        // deepseek / mistral: simple [INST] format
        return "[INST] " + userPrompt + " [/INST]";
    }

    bool BpeTokenizer::IsEosToken(int32_t id) const {
        return std::find(m_eosIds.begin(), m_eosIds.end(), id) != m_eosIds.end();
    }

    std::vector<int32_t> BpeTokenizer::Encode(const std::string& text) {
        std::vector<int32_t> tokens;
        // BOS is already included in the template string for most models;
        // add it here only when the template doesn't encode it as a literal token
        // (i.e., it's not already in the text as a named token).
        if (m_encoder.find("<bos>") == m_encoder.end() &&
            text.find("<bos>") == std::string::npos) {
            tokens.push_back(m_bosId);
        }
        size_t i = 0;
        while (i < text.length()) {
            std::string best;
            size_t bestLen = 1;
            // Greedy longest-match
            for (size_t len = text.length() - i; len >= 1; len--) {
                std::string sub = text.substr(i, len);
                if (m_encoder.find(sub) != m_encoder.end()) {
                    best = sub; bestLen = len; break;
                }
            }
            if (!best.empty()) {
                tokens.push_back(m_encoder.at(best));
                i += bestLen;
            } else {
                char byte[2] = { text[i], '\0' };
                auto it = m_encoder.find(std::string(byte));
                tokens.push_back(it != m_encoder.end() ? it->second : m_encoder.at("<unk>"));
                i++;
            }
        }
        return tokens;
    }

    std::string BpeTokenizer::Decode(const std::vector<int32_t>& ids) {
        std::string text;
        for (int32_t id : ids) {
            if (IsEosToken(id)) break;
            if (id == m_bosId || id == m_padId) continue;
            auto it = m_decoder.find(id);
            if (it != m_decoder.end()) {
                const std::string& tok = it->second;
                // Strip Ġ (GPT-2 space marker U+0120) -> space
                std::string cleaned;
                for (size_t k = 0; k < tok.size(); ) {
                    unsigned char c = (unsigned char)tok[k];
                    if (c == 0xC4 && k + 1 < tok.size() && (unsigned char)tok[k+1] == 0xA0) {
                        cleaned += ' '; k += 2;
                    } else {
                        cleaned += tok[k]; k++;
                    }
                }
                // Skip special tokens in output (e.g. <|im_start|>, <unused*>)
                if (cleaned.size() > 1 && cleaned.front() == '<' && cleaned.back() == '>')
                    continue;
                text += cleaned;
            }
        }
        return text;
    }
}
