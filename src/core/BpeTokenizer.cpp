#include "dybydx/core/BpeTokenizer.h"
#include "dybydx/core/GgufLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace DirectLLM {

    BpeTokenizer::BpeTokenizer() {
        m_encoder["<pad>"] = 0;  m_decoder[0] = "<pad>";
        m_encoder["<s>"] = 1;    m_decoder[1] = "<s>";
        m_encoder["</s>"] = 2;   m_decoder[2] = "</s>";
        m_encoder["<unk>"] = 3;  m_decoder[3] = "<unk>";
    }

    BpeTokenizer::~BpeTokenizer() {}

    bool BpeTokenizer::LoadFromGGUF(const GgufLoader& loader) {
        // Read tokenizer tokens from GGUF metadata
        const GgufValue* tokensVal = nullptr;
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
        std::cout << "[Tokenizer] Loaded " << m_encoder.size() << " tokens from GGUF metadata." << std::endl;
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

    std::vector<int32_t> BpeTokenizer::Encode(const std::string& text) {
        std::vector<int32_t> tokens;
        tokens.push_back(m_bosId);
        size_t i = 0;
        while (i < text.length()) {
            std::string best;
            size_t bestLen = 1;
            for (size_t len = text.length() - i; len >= 1; len--) {
                std::string sub = text.substr(i, len);
                if (m_encoder.find(sub) != m_encoder.end()) {
                    best = sub; bestLen = len; break;
                }
            }
            if (!best.empty()) {
                tokens.push_back(m_encoder[best]);
                i += bestLen;
            } else {
                char byte[2] = { text[i], '\0' };
                auto it = m_encoder.find(std::string(byte));
                tokens.push_back(it != m_encoder.end() ? it->second : m_encoder["<unk>"]);
                i++;
            }
        }
        return tokens;
    }

    std::string BpeTokenizer::Decode(const std::vector<int32_t>& ids) {
        std::string text;
        for (int32_t id : ids) {
            if (id == m_eosId) break;
            if (id == m_bosId || id == m_padId) continue;
            auto it = m_decoder.find(id);
            if (it != m_decoder.end()) text += it->second;
        }
        return text;
    }
}
