// DirectLLM C++ Core - (C) 2026 DirectLLM Team
#pragma once

#include <string>
#include <vector>
#include <unordered_map>

namespace DirectLLM {

    class BpeTokenizer {
    public:
        BpeTokenizer();
        ~BpeTokenizer();

        bool LoadVocabulary(const std::string& vocabPath);
        
        std::vector<int32_t> Encode(const std::string& text);
        std::string Decode(const std::vector<int32_t>& ids);

        int32_t GetBosTokenId() const { return m_bosId; }
        int32_t GetEosTokenId() const { return m_eosId; }
        int32_t GetPadTokenId() const { return m_padId; }

    private:
        std::unordered_map<std::string, int32_t> m_encoder;
        std::unordered_map<int32_t, std::string> m_decoder;
        std::unordered_map<std::string, int64_t> m_bpeRanks;

        int32_t m_bosId = 1;
        int32_t m_eosId = 2;
        int32_t m_padId = 0;

        std::vector<std::string> SplitWords(const std::string& text);
        std::string CleanByteFallback(const std::string& token);
    };
}
